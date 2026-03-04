"""
kline_buffer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reemplaza las llamadas REST de klines y ticker/24hr por WebSockets
persistentes de Binance Futures.

──────────────────────────────────────────────────────────────
KlineBuffer
  • Se suscribe a <symbol>@kline_15m para cada símbolo del universo.
  • Carga el historial inicial (200 velas) por REST una sola vez,
    luego lo mantiene actualizado en tiempo real con el stream WebSocket.
  • get_df(symbol) →  DataFrame listo para calculate_indicators(),
    sin hacer ninguna llamada REST.

MiniTickerStream
  • Suscribe al stream !miniTicker@arr (tick de todos los pares).
  • get_universe(min_vol_m) → lista de símbolos USDT filtrados por
    volumen 24h, sin ninguna llamada a ticker/24hr.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Dict, List, Optional

import aiohttp
import pandas as pd

logger = logging.getLogger("KlineBuffer")

# WebSocket base para Binance Futures
WS_BASE  = "wss://fstream.binance.com/stream"
REST_BASE = "https://fapi.binance.com"

# Cuántas velas guardar en el buffer (>= 200 para satisfy confirm_entry)
BUFFER_CANDLES = 220

# Intervalo de reconnect si el WebSocket se cae
RECONNECT_DELAY = 5

# ─── Columnas de DataFrame compatibles con market_data.get_klines() ───────────
_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
         'close_time', 'quote_asset_volume', 'trades',
         'taker_buy_base', 'taker_buy_quote', 'ignore']
_NUM  = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']


def _parse_rest_candles(raw: list) -> deque:
    """Convierte respuesta REST a deque de dicts."""
    buf = deque(maxlen=BUFFER_CANDLES)
    for c in raw:
        buf.append({
            'timestamp':          c[0],
            'open':               float(c[1]),
            'high':               float(c[2]),
            'low':                float(c[3]),
            'close':              float(c[4]),
            'volume':             float(c[5]),
            'close_time':         c[6],
            'quote_asset_volume': float(c[7]),
            'trades':             int(c[8]),
            'taker_buy_base':     float(c[9]),
            'taker_buy_quote':    float(c[10]),
            'ignore':             c[11],
        })
    return buf


def _buf_to_df(buf: deque) -> pd.DataFrame:
    """Convierte el buffer interno a DataFrame compatible con calculate_indicators()."""
    df = pd.DataFrame(list(buf))
    df['timestamp']      = pd.to_datetime(df['timestamp'], unit='ms')
    df['taker_buy_vol']  = df['taker_buy_base']
    return df


# ══════════════════════════════════════════════════════════════════════════════
# KLINE BUFFER
# ══════════════════════════════════════════════════════════════════════════════
class KlineBuffer:
    """
    Mantiene hasta BUFFER_CANDLES velas de 15m por símbolo en RAM.
    Acepta suscripciones en caliente (subscribe/unsubscribe).
    """

    def __init__(self, interval: str = "15m"):
        self.interval   = interval
        self._buffers:   Dict[str, deque]   = {}   # symbol → deque de candles
        self._ws_task:   Optional[asyncio.Task]  = None
        self._session:   Optional[aiohttp.ClientSession] = None
        self._symbols:   set = set()    # conjunto de símbolos suscritos
        self._pending_add:    set = set()  # backlog para agregar
        self._pending_remove: set = set()  # backlog para remover
        self._ready:     asyncio.Event = asyncio.Event()  # señal de que WS está vivo
        self._running    = False

    # ── API pública ────────────────────────────────────────────────────────────

    async def start(self):
        """Inicia el loop de WebSocket en background."""
        self._running = True
        self._session = aiohttp.ClientSession()
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("🟢 KlineBuffer iniciado")

    async def stop(self):
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
        if self._session:
            await self._session.close()

    async def subscribe(self, symbols: List[str]):
        """
        Suscribe nuevos símbolos. Carga el historial REST y los encola
        para el WebSocket. Usa Semaphore(3) para no saturar el API.
        """
        new_syms = [s for s in symbols if s not in self._symbols]
        if not new_syms:
            return

        sem = asyncio.Semaphore(3)  # Máx 3 requests REST simultáneos para historial

        async def _load_throttled(sym: str, idx: int):
            async with sem:
                await self._load_history(sym)
            if idx > 0 and idx % 10 == 0:
                await asyncio.sleep(0.8)  # Pausa cada 10 pares cargados

        await asyncio.gather(*[_load_throttled(s, i) for i, s in enumerate(new_syms)])

        self._pending_add.update(new_syms)
        self._symbols.update(new_syms)
        logger.info(f"📦 KlineBuffer: {len(new_syms)} símbolos nuevos suscritos (total={len(self._symbols)})")


    def get_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Devuelve el DataFrame de velas para el símbolo solicitado.
        Retorna None si no hay datos disponibles.
        """
        buf = self._buffers.get(symbol)
        if buf is None or len(buf) < 50:
            return None
        return _buf_to_df(buf)

    def symbols_ready(self) -> List[str]:
        """Lista de símbolos con suficientes velas (≥ 96) para análisis."""
        return [s for s, b in self._buffers.items() if len(b) >= 96]

    # ── Carga de historial REST (una sola vez por símbolo) ─────────────────────

    async def _load_history(self, symbol: str):
        url = f"{REST_BASE}/fapi/v1/klines"
        params = {'symbol': symbol, 'interval': self.interval, 'limit': BUFFER_CANDLES}
        try:
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 418:
                    logger.warning(f"⚠️ 418 cargando historial {symbol} — saltando")
                    return
                if resp.status == 200:
                    data = await resp.json()
                    self._buffers[symbol] = _parse_rest_candles(data)
        except Exception as e:
            logger.debug(f"KlineBuffer: error cargando {symbol}: {e}")

    # ── Loop principal de WebSocket ─────────────────────────────────────────────

    async def _ws_loop(self):
        """Mantiene la conexión WebSocket viva y reintenta si se cae."""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"KlineBuffer WS desconectado: {e} — reintentando en {RECONNECT_DELAY}s")
                self._ready.clear()
                await asyncio.sleep(RECONNECT_DELAY)

    async def _connect_and_listen(self):
        """Conecta al WebSocket combinado y escucha mensajes."""
        if not self._symbols:
            # Esperar a que haya suscripciones
            await asyncio.sleep(5)
            return

        streams = "/".join(
            f"{s.lower()}@kline_{self.interval}" for s in self._symbols
        )
        url = f"{WS_BASE}?streams={streams}"

        async with self._session.ws_connect(url, heartbeat=20) as ws:
            self._ready.set()
            logger.info(f"🔗 KlineBuffer WS conectado ({len(self._symbols)} streams)")

            async for msg in ws:
                if not self._running:
                    break

                # Procesar mensajes de datos
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_message(json.loads(msg.data))

                # Re-suscribir nuevos pares sin reconectar
                if self._pending_add:
                    new = list(self._pending_add)
                    self._pending_add.clear()
                    sub_msg = {
                        "method": "SUBSCRIBE",
                        "params": [f"{s.lower()}@kline_{self.interval}" for s in new],
                        "id": int(time.time())
                    }
                    await ws.send_json(sub_msg)

    def _handle_message(self, data: dict):
        """Procesa un mensaje de kline del WebSocket."""
        stream_data = data.get('data', data)  # puede venir envuelto o directo
        if stream_data.get('e') != 'kline':
            return

        k = stream_data['k']
        sym = k['s']
        candle = {
            'timestamp':          k['t'],
            'open':               float(k['o']),
            'high':               float(k['h']),
            'low':                float(k['l']),
            'close':              float(k['c']),
            'volume':             float(k['v']),
            'close_time':         k['T'],
            'quote_asset_volume': float(k['q']),
            'trades':             int(k['n']),
            'taker_buy_base':     float(k['V']),
            'taker_buy_quote':    float(k['Q']),
            'ignore':             '0',
        }

        if sym not in self._buffers:
            self._buffers[sym] = deque(maxlen=BUFFER_CANDLES)

        buf = self._buffers[sym]

        # Si la vela ya está en el buffer (misma timestamp), actualizar in-place
        if buf and buf[-1]['timestamp'] == candle['timestamp']:
            buf[-1] = candle
        else:
            # Nueva vela — si la anterior no estaba cerrada, ya se actualizó;
            # ahora empujamos la nueva
            buf.append(candle)


# ══════════════════════════════════════════════════════════════════════════════
# MINI TICKER STREAM  (reemplaza ticker/24hr REST)
# ══════════════════════════════════════════════════════════════════════════════
class MiniTickerStream:
    """
    Suscribe al stream !miniTicker@arr de Binance Futures.
    Mantiene el volumen 24h de todos los pares en memoria.
    get_universe() filtra por volumen sin ninguna llamada REST.
    """

    WS_URL = "wss://fstream.binance.com/ws/!miniTicker@arr"

    def __init__(self, min_vol_usdt_m: float = 50.0):
        self.min_vol      = min_vol_usdt_m * 1_000_000
        self._tickers:    Dict[str, float] = {}   # symbol → quoteVolume 24h
        self._ws_task:    Optional[asyncio.Task] = None
        self._session:    Optional[aiohttp.ClientSession] = None
        self._running     = False
        self._last_update = 0.0

    async def start(self):
        self._running = True
        self._session = aiohttp.ClientSession()
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("🟢 MiniTickerStream iniciado — esperando datos del mercado...")

    async def stop(self):
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
        if self._session:
            await self._session.close()

    def get_universe(self, blacklist: set = None) -> List[str]:
        """
        Retorna lista de símbolos USDT ordenados por volumen (mayor a menor).
        No hace ninguna llamada REST.
        """
        if not self._tickers:
            return []

        bl = blacklist or set()
        symbols = [
            sym for sym, vol in self._tickers.items()
            if sym.endswith('USDT') and sym not in bl and vol >= self.min_vol
        ]
        symbols.sort(key=lambda s: self._tickers[s], reverse=True)
        return symbols

    def is_ready(self) -> bool:
        """True si ya recibimos datos (tickers no vacío y actualizado recientemente)."""
        return len(self._tickers) > 0 and (time.time() - self._last_update) < 120

    async def wait_ready(self, timeout: float = 30.0):
        """Espera a que el stream tenga datos iniciales."""
        deadline = asyncio.get_event_loop().time() + timeout
        while not self.is_ready():
            if asyncio.get_event_loop().time() > deadline:
                logger.warning("MiniTickerStream: timeout esperando datos iniciales")
                return
            await asyncio.sleep(0.5)

    async def _ws_loop(self):
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"MiniTickerStream WS caído: {e} — reintentando en {RECONNECT_DELAY}s")
                await asyncio.sleep(RECONNECT_DELAY)

    async def _connect_and_listen(self):
        async with self._session.ws_connect(self.WS_URL, heartbeat=20) as ws:
            logger.info("🔗 MiniTickerStream WS conectado")
            async for msg in ws:
                if not self._running:
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        tickers = json.loads(msg.data)
                        for t in tickers:
                            sym = t.get('s', '')
                            vol = float(t.get('q', 0))   # quoteVolume 24h
                            if sym:
                                self._tickers[sym] = vol
                        self._last_update = time.time()
                    except Exception:
                        pass
