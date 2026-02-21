"""
Order Book Tracker — WebSocket de profundidad de mercado (Binance Futures).

Suscribe a múltiples streams `{symbol}@depth20@250ms` simultáneamente,
mantiene el estado actual del book en memoria, detecta paredes y spoofing,
y expone una API simple para el whale scanner en tiempo real.

Uso:
    tracker = OrderBookTracker(symbols=["BTCUSDT", "ETHUSDT"])
    await tracker.start()
    signals = tracker.get_signals("BTCUSDT")
    await tracker.stop()

O como task independiente dentro de un event loop:
    asyncio.create_task(tracker.start())
"""
import asyncio
import json
import logging
import time
from collections import deque
from typing import Dict, List, Optional

try:
    import websockets
except ImportError:
    websockets = None  # Se valida en runtime

from .whale_math_ob import (
    orderbook_imbalance,
    detect_large_walls,
    spoofing_score,
)

logger = logging.getLogger("OrderBookTracker")

# Binance Futures WebSocket base
WS_BASE = "wss://fstream.binance.com/stream?streams="
# Máx símbolos por conexión WebSocket (límite Binance: 200 streams/conn)
MAX_SYMBOLS_PER_CONN = 50
# Historial de snapshots para detección de spoofing
SNAPSHOT_HISTORY = 12   # últimos 3 segundos (a 250ms cadencia = 12 snapshots)


class OrderBookTracker:
    """
    Tracker asíncrono del order book para múltiples pares.

    - Conecta vía WebSocket a Binance Futures depth streams
    - Mantiene el estado del book (top 20 bid/ask) en memoria
    - Calcula imbalance, paredes y spoofing en tiempo real
    - Reconexión automática con backoff exponencial
    """

    def __init__(self, symbols: List[str], depth_levels: int = 20):
        self.symbols = [s.lower() for s in symbols]
        self.depth_levels = depth_levels

        # Estado del book: {symbol: {'bids': [[p,q],...], 'asks': [[p,q],...], 'ts': float}}
        self._books: Dict[str, dict] = {}
        # Historial de snapshots por símbolo para spoofing
        self._snapshots: Dict[str, deque] = {
            s: deque(maxlen=SNAPSHOT_HISTORY) for s in self.symbols
        }
        # Caché de señales calculadas (se recalculan cada N actualizaciones)
        self._signals_cache: Dict[str, dict] = {}
        self._update_counts: Dict[str, int] = {s: 0 for s in self.symbols}

        self._running = False
        self._tasks: List[asyncio.Task] = []

    def get_signals(self, symbol: str) -> dict:
        """
        Retorna las señales del order book para un símbolo.

        Returns:
        {
          'imbalance': float,      # [-1, +1]: positivo = presión compradora
          'bid_volume': float,
          'ask_volume': float,
          'has_bid_wall': bool,
          'has_ask_wall': bool,
          'bid_walls': list,       # [{'price', 'qty', 'multiple'}]
          'ask_walls': list,
          'bid_spoof': int,
          'ask_spoof': int,
          'spoof_signal': bool,
          'stale': bool,           # True si los datos tienen >5s de antigüedad
        }
        """
        sym = symbol.lower()
        cached = self._signals_cache.get(sym)
        if cached:
            stale = (time.time() - cached.get('ts', 0)) > 5.0
            cached['stale'] = stale
            return cached
        return self._empty_signals()

    def get_book_snapshot(self, symbol: str) -> Optional[dict]:
        """Retorna el snapshot más reciente del book (crudo)."""
        return self._books.get(symbol.lower())

    @property
    def active_symbols(self) -> List[str]:
        """Símbolos con datos de book activos."""
        now = time.time()
        return [s for s, b in self._books.items() if (now - b.get('ts', 0)) < 10.0]

    # ============================================================
    # LOOP PRINCIPAL
    # ============================================================

    async def start(self):
        """Inicia todas las conexiones WebSocket."""
        if websockets is None:
            logger.error("❌ 'websockets' library not installed. Run: pip install websockets")
            return

        self._running = True
        logger.info(f"📡 OrderBookTracker iniciando para {len(self.symbols)} pares...")

        # Dividir en grupos de MAX_SYMBOLS_PER_CONN
        chunks = [
            self.symbols[i:i + MAX_SYMBOLS_PER_CONN]
            for i in range(0, len(self.symbols), MAX_SYMBOLS_PER_CONN)
        ]

        self._tasks = [
            asyncio.create_task(self._ws_loop(chunk))
            for chunk in chunks
        ]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self):
        """Detiene todas las conexiones."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        logger.info("🛑 OrderBookTracker detenido.")

    # ============================================================
    # WEBSOCKET LOOP
    # ============================================================

    async def _ws_loop(self, symbols: List[str]):
        """Loop principal de conexión con reconexión automática."""
        backoff = 1.0

        while self._running:
            streams = "/".join(f"{s}@depth20@250ms" for s in symbols)
            url = WS_BASE + streams

            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info(f"✅ WS conectado: {len(symbols)} depth streams")
                    backoff = 1.0  # Reset backoff on success

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            if "data" in msg:
                                self._process_depth(msg["data"])
                        except Exception as e:
                            logger.debug(f"Parse error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"⚠️ WS disconnected ({e}). Reconectando en {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    # ============================================================
    # PROCESAMIENTO
    # ============================================================

    def _process_depth(self, data: dict):
        """Procesa un mensaje de depth stream."""
        stream = data.get("e", "")  # event type
        if stream != "depthUpdate":
            # depth20 no tiene event type, viene directo con bids/asks
            pass

        # Binance depth20 format: {"e":"depthUpdate","s":"BTCUSDT","b":[[p,q]...],"a":[[p,q]...]}
        # o en combined stream: {"stream":"btcusdt@depth20@250ms","data":{...}}
        sym = data.get("s", "").lower()
        if not sym:
            return

        bids_raw = data.get("b", [])
        asks_raw = data.get("a", [])

        if not bids_raw and not bids_raw:
            return

        ts = time.time()
        book = {
            'bids': [[float(p), float(q)] for p, q in bids_raw if float(q) > 0],
            'asks': [[float(p), float(q)] for p, q in asks_raw if float(q) > 0],
            'ts': ts,
        }

        self._books[sym] = book

        # Agregar snapshot al historial para spoofing
        if sym in self._snapshots:
            self._snapshots[sym].append({'ts': ts, 'bids': book['bids'], 'asks': book['asks']})

        # Recalcular señales cada 4 actualizaciones
        self._update_counts[sym] = self._update_counts.get(sym, 0) + 1
        if self._update_counts[sym] % 4 == 0:
            self._recalculate_signals(sym, book)

    def _recalculate_signals(self, sym: str, book: dict):
        """Recalcula las señales para un símbolo usando el book actual."""
        try:
            bids = book['bids']
            asks = book['asks']

            # 1. Imbalance
            imbalance, bid_vol, ask_vol = orderbook_imbalance(bids, asks, levels=10)

            # 2. Paredes
            walls = detect_large_walls(bids, asks, levels=20, multiplier=5.0)

            # 3. Spoofing (sobre el historial de snapshots)
            snapshots = list(self._snapshots.get(sym, []))
            bid_spoof, ask_spoof, spoof_signal = spoofing_score(
                snapshots, levels=5, min_appearances=2, max_life_s=3.0
            )

            self._signals_cache[sym] = {
                'imbalance':    round(imbalance, 4),
                'bid_volume':   round(bid_vol, 2),
                'ask_volume':   round(ask_vol, 2),
                'has_bid_wall': walls['has_bid_wall'],
                'has_ask_wall': walls['has_ask_wall'],
                'bid_walls':    walls['bid_walls'],
                'ask_walls':    walls['ask_walls'],
                'bid_spoof':    bid_spoof,
                'ask_spoof':    ask_spoof,
                'spoof_signal': spoof_signal,
                'stale':        False,
                'ts':           book['ts'],
            }
        except Exception as e:
            logger.debug(f"Signal calc error {sym}: {e}")

    @staticmethod
    def _empty_signals() -> dict:
        return {
            'imbalance': 0.0, 'bid_volume': 0.0, 'ask_volume': 0.0,
            'has_bid_wall': False, 'has_ask_wall': False,
            'bid_walls': [], 'ask_walls': [],
            'bid_spoof': 0, 'ask_spoof': 0, 'spoof_signal': False,
            'stale': True, 'ts': 0.0,
        }
