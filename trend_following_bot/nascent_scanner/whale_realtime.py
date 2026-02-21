"""
Whale Real-Time Scanner — Orquestador WebSocket asíncrono (v2 — FIX: pre-fill histórico)

Combina:
  1. Klines REST (pre-fill de buffer histórico al arrancar — FIX CRÍTICO)
  2. Klines WebSocket (velas 15m en tiempo real, mantiene buffer actualizado)
  3. AggTrades WebSocket (CVD real, trade a trade)
  4. OrderBookTracker (imbalance, paredes, spoofing)
  5. WhaleScanner (análisis L4 + L5)
  6. UniverseFilter (top-N pares por liquidez)

Inicio rápido:
    cd "c:\\Users\\Ajota\\Documents\\Nueva carpeta\\trend_following_bot"
    python -m nascent_scanner.whale_realtime

    # Con parámetros:
    python -m nascent_scanner.whale_realtime --top-n 30 --min-score 40 --verbose

Arquitectura:
    STARTUP:
      REST API → pre-fill 300 candles × N pares (concurrente, <5s)
      → escaneo inicial inmediato

    RUNTIME:
      WS kline → cierre de vela → añadir al buffer → escanear
      WS aggTrade → actualizar CVD real
      WS depth → actualizar OB signals
      heartbeat → scan periódico (60s)
"""
import asyncio
import json
import logging
import sys
import os
import argparse
import time
from collections import deque
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

from nascent_scanner.whale_scanner import WhaleScanner
from nascent_scanner.orderbook_tracker import OrderBookTracker
from nascent_scanner.universe_filter import get_liquid_universe_sync
from nascent_scanner.whale_math_ob import cvd_from_agg_trades

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("WhaleRealtime")

# ============================================================
# CONFIGURACIÓN
# ============================================================

BINANCE_REST      = "https://fapi.binance.com"
WS_FUTURES_BASE   = "wss://fstream.binance.com/stream?streams="
MAX_CANDLES_BUFFER = 320      # Candles en memoria por par
MAX_TRADES_BUFFER  = 5_000   # AggTrades en memoria por par
WHALE_SCAN_INTERVAL_S = 900  # Re-escanear en el heartbeat cada 15min (una vela)
MAX_SYMS_PER_CONN  = 50      # Límite Binance: 200 streams/conn
MIN_CANDLES_TO_SCAN = 96     # Mínimo de candles para escanear (96 = 24h de 15m)
PREFILL_CANDLES    = 300     # Candles históricos a descargar en startup
PREFILL_CONCURRENCY = 10     # Cuántos pares descargar en paralelo
LOG_MIN_SCORE      = 40      # Score mínimo para mostrar alertas


# ============================================================
# HELPERS DE DATOS
# ============================================================

def _kline_to_candle(k: dict) -> dict:
    """Convierte el dict de kline de Binance (WS) a formato del scanner."""
    return {
        'open':          float(k['o']),
        'high':          float(k['h']),
        'low':           float(k['l']),
        'close':         float(k['c']),
        'volume':        float(k['v']),
        'taker_buy_vol': float(k.get('Q', k.get('q', 0))),
        'timestamp':     int(k['t']),
    }


def _rest_kline_to_candle(row: list) -> dict:
    """Convierte una fila de klines REST al formato del scanner."""
    return {
        'open':          float(row[1]),
        'high':          float(row[2]),
        'low':           float(row[3]),
        'close':         float(row[4]),
        'volume':        float(row[5]),
        'taker_buy_vol': float(row[9]),   # taker_buy_base_asset_volume
        'timestamp':     int(row[0]),
    }


def _buffer_to_df(buf: deque) -> pd.DataFrame:
    """Convierte el deque de candles a DataFrame."""
    if len(buf) == 0:
        return pd.DataFrame()
    return pd.DataFrame(list(buf)).reset_index(drop=True)


def _fetch_historical_klines(symbol: str, limit: int = 300) -> list:
    """
    Descarga candles históricos via REST (síncrono, para llamar en executor).
    Returns list de dicts en formato del scanner.
    """
    try:
        resp = requests.get(
            f"{BINANCE_REST}/fapi/v1/klines",
            params={'symbol': symbol.upper(), 'interval': '15m', 'limit': limit},
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        # Excluir la última vela (puede estar incompleta)
        candles = [_rest_kline_to_candle(r) for r in rows[:-1]]
        return candles
    except Exception as e:
        logger.warning(f"Pre-fill REST error {symbol}: {e}")
        return []


# ============================================================
# ORQUESTADOR PRINCIPAL
# ============================================================

class WhaleRealtimeScanner:
    """
    Scanner de ballenas en tiempo real.

    FLUJO:
    1. startup() → descarga REST histórica simultánea para todos los pares
    2. Escaneo inmediato tras pre-fill
    3. start() → WebSocket streams para actualizaciones en tiempo real
    """

    def __init__(
        self,
        symbols: List[str],
        min_score: int = LOG_MIN_SCORE,
        enable_orderbook: bool = True,
        enable_aggtrades: bool = True,
        verbose: bool = False,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.sym_keys = [s.lower() for s in symbols]
        self.min_score = min_score
        self.enable_orderbook = enable_orderbook
        self.enable_aggtrades = enable_aggtrades
        self.verbose = verbose

        # Buffers de candles (pre-llenados antes de los streams)
        self.candle_buffers: Dict[str, deque] = {
            s.lower(): deque(maxlen=MAX_CANDLES_BUFFER) for s in symbols
        }
        self.trade_buffers: Dict[str, deque] = {
            s.lower(): deque(maxlen=MAX_TRADES_BUFFER) for s in symbols
        }
        self.current_candle: Dict[str, Optional[dict]] = {s.lower(): None for s in symbols}
        self.last_scan_ts: Dict[str, float] = {}

        # Contadores para logging
        self._scan_count = 0
        self._alert_count = 0

        self.ob_tracker = OrderBookTracker(symbols) if enable_orderbook else None
        self.whale_scanner = WhaleScanner()
        self._running = False

    # ============================================================
    # STARTUP: PRE-FILL HISTÓRICO (el fix crítico)
    # ============================================================

    async def _prefill_buffers(self):
        """
        Descarga PREFILL_CANDLES velas históricas para cada par via REST.
        Se ejecuta ANTES de conectar los WebSockets.
        Usa concurrencia controlada (PREFILL_CONCURRENCY pares a la vez).
        """
        total = len(self.symbols)
        logger.info(f"📥 Pre-llenando buffers históricos ({PREFILL_CANDLES} candles × {total} pares)...")

        sem = asyncio.Semaphore(PREFILL_CONCURRENCY)
        loop = asyncio.get_event_loop()
        filled = 0
        failed = 0

        async def _fill_one(sym: str):
            nonlocal filled, failed
            async with sem:
                candles = await loop.run_in_executor(
                    None, lambda: _fetch_historical_klines(sym, PREFILL_CANDLES)
                )
                if candles:
                    for c in candles:
                        self.candle_buffers[sym.lower()].append(c)
                    filled += 1
                else:
                    failed += 1

        await asyncio.gather(*[_fill_one(s) for s in self.symbols])

        logger.info(
            f"✅ Pre-fill completo: {filled}/{total} pares listos "
            f"({failed} fallaron) | Candles/par: ~{PREFILL_CANDLES}"
        )

        # Mostrar estado del buffer
        ready = sum(1 for s in self.sym_keys if len(self.candle_buffers[s]) >= MIN_CANDLES_TO_SCAN)
        logger.info(f"🟢 {ready}/{total} pares tienen ≥{MIN_CANDLES_TO_SCAN} candles → listos para escanear")

    async def _initial_scan(self):
        """Corre un escaneo completo inmediatamente después del pre-fill."""
        logger.info("🔍 Ejecutando escaneo inicial de todos los pares...")
        tasks = [self._run_whale_scan(s) for s in self.sym_keys]
        await asyncio.gather(*tasks)
        logger.info(f"✅ Escaneo inicial completo | {self._scan_count} escaneados | {self._alert_count} alertas")

    # ============================================================
    # START / STOP
    # ============================================================

    async def start(self):
        """Inicia el scanner: pre-fill → escaneo inicial → streams en vivo."""
        if not _WS_AVAILABLE:
            logger.error("❌ Instala websockets: pip install websockets")
            return

        self._running = True
        logger.info(f"🦈 WhaleRealtimeScanner v2 | {len(self.symbols)} pares")
        logger.info(f"   OB: {'✅' if self.enable_orderbook else '❌'}  "
                    f"AggTrades: {'✅' if self.enable_aggtrades else '❌'}  "
                    f"Score mínimo: {self.min_score}")

        # STEP 1: Pre-fill REST histórico
        await self._prefill_buffers()

        # STEP 2: Escaneo inicial inmediato
        await self._initial_scan()

        # STEP 3: Conectar streams en vivo
        sym_chunks = [
            self.symbols[i:i + MAX_SYMS_PER_CONN]
            for i in range(0, len(self.symbols), MAX_SYMS_PER_CONN)
        ]

        tasks = []
        for chunk in sym_chunks:
            tasks.append(asyncio.create_task(self._kline_loop(chunk)))

        if self.enable_aggtrades:
            for chunk in sym_chunks:
                tasks.append(asyncio.create_task(self._aggtrade_loop(chunk)))

        if self.enable_orderbook and self.ob_tracker:
            tasks.append(asyncio.create_task(self.ob_tracker.start()))

        tasks.append(asyncio.create_task(self._heartbeat_loop()))

        logger.info("📡 WebSocket streams activos. Analizando en cada cierre de vela...")
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        self._running = False
        if self.ob_tracker:
            await self.ob_tracker.stop()
        logger.info(f"🛑 Detenido. Total scans: {self._scan_count} | Alertas: {self._alert_count}")

    # ============================================================
    # KLINE STREAM
    # ============================================================

    async def _kline_loop(self, symbols: List[str]):
        streams = "/".join(f"{s.lower()}@kline_15m" for s in symbols)
        url = WS_FUTURES_BASE + streams
        backoff = 1.0

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    if self.verbose:
                        logger.info(f"📊 Kline WS [{len(symbols)} syms]: conectado")
                    backoff = 1.0
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            data = msg.get("data", msg)
                            if data.get("e") == "kline":
                                self._process_kline(data)
                        except Exception:
                            pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Kline WS: {e} → retry {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def _process_kline(self, data: dict):
        k = data.get("k", {})
        sym = k.get("s", "").lower()
        if sym not in self.candle_buffers:
            return

        candle = _kline_to_candle(k)
        is_closed = k.get("x", False)

        if is_closed:
            self.candle_buffers[sym].append(candle)
            self.current_candle[sym] = None
            asyncio.create_task(self._run_whale_scan(sym))
            if self.verbose:
                logger.info(f"  ✔ Vela cerrada: {sym.upper()} | buf={len(self.candle_buffers[sym])}")
        else:
            self.current_candle[sym] = candle

    # ============================================================
    # AGGTRADE STREAM (CVD Real)
    # ============================================================

    async def _aggtrade_loop(self, symbols: List[str]):
        streams = "/".join(f"{s.lower()}@aggTrade" for s in symbols)
        url = WS_FUTURES_BASE + streams
        backoff = 1.0

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    if self.verbose:
                        logger.info(f"💹 AggTrade WS [{len(symbols)} syms]: conectado")
                    backoff = 1.0
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            data = msg.get("data", msg)
                            if data.get("e") == "aggTrade":
                                sym = data.get("s", "").lower()
                                if sym in self.trade_buffers:
                                    self.trade_buffers[sym].append({
                                        'q': float(data.get("q", 0)),
                                        'm': data.get("m", False),
                                        'p': float(data.get("p", 0)),
                                        't': data.get("T", 0),
                                    })
                        except Exception:
                            pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"AggTrade WS: {e} → retry {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    # ============================================================
    # HEARTBEAT LOOP (scan periódico + status)
    # ============================================================

    async def _heartbeat_loop(self):
        """Muestra status cada 5min y re-escanea pares que no han sido escaneados."""
        tick = 0
        while self._running:
            await asyncio.sleep(60)
            tick += 1

            # Status cada 5 ticks (5min)
            if tick % 5 == 0:
                ready = sum(1 for s in self.sym_keys if len(self.candle_buffers[s]) >= MIN_CANDLES_TO_SCAN)
                logger.info(
                    f"💓 Heartbeat | pares listos: {ready}/{len(self.symbols)} | "
                    f"scans: {self._scan_count} | alertas: {self._alert_count}"
                )

            # Re-escanear pares que llevan más de WHALE_SCAN_INTERVAL_S sin escanear
            now = time.time()
            for sym in self.sym_keys:
                last = self.last_scan_ts.get(sym, 0)
                if now - last >= WHALE_SCAN_INTERVAL_S:
                    if len(self.candle_buffers[sym]) >= MIN_CANDLES_TO_SCAN:
                        await self._run_whale_scan(sym)

    # ============================================================
    # ANÁLISIS WHALE
    # ============================================================

    async def _run_whale_scan(self, sym: str):
        """Corre el análisis whale sobre el buffer actual de un símbolo."""
        buf = self.candle_buffers.get(sym, deque())
        n = len(buf)

        if n < MIN_CANDLES_TO_SCAN:
            if self.verbose:
                logger.debug(f"  ⏭ Skip {sym.upper()}: solo {n} candles (min={MIN_CANDLES_TO_SCAN})")
            return

        self.last_scan_ts[sym] = time.time()
        self._scan_count += 1

        df = _buffer_to_df(buf)

        # Añadir vela actual (no cerrada) si existe
        curr = self.current_candle.get(sym)
        if curr is not None:
            df = pd.concat([df, pd.DataFrame([curr])], ignore_index=True)

        # CVD real desde aggTrades
        trade_buf = list(self.trade_buffers.get(sym, []))
        if trade_buf:
            real_cvd, _ = cvd_from_agg_trades(trade_buf)
            df.at[len(df) - 1, '_real_cvd'] = real_cvd

        try:
            result = self.whale_scanner.analyze(df)
        except Exception as e:
            logger.debug(f"Whale scan error {sym}: {e}")
            return

        if result.get('rejected'):
            if self.verbose:
                logger.debug(f"  ⛔ Rejected {sym.upper()}: {result.get('reject_reason', '?')}")
            return

        total_score = result.get('total_score', 0)

        # Señales del order book
        ob_bonus = 0
        ob_notes = []
        if self.ob_tracker:
            ob_sigs = self.ob_tracker.get_signals(sym)
            if not ob_sigs['stale']:
                imb = ob_sigs['imbalance']
                if imb > 0.30:
                    ob_bonus += 30
                    ob_notes.append(f"OB_BULL({imb:+.2f})")
                elif imb < -0.30:
                    ob_bonus += 20
                    ob_notes.append(f"OB_BEAR({imb:+.2f})")
                if ob_sigs['has_bid_wall']:
                    walls = ob_sigs['bid_walls']
                    top = max(walls, key=lambda w: w['multiple'])
                    ob_bonus += 25
                    ob_notes.append(f"BID_WALL({top['multiple']:.1f}x)")
                if ob_sigs['has_ask_wall']:
                    walls = ob_sigs['ask_walls']
                    top = max(walls, key=lambda w: w['multiple'])
                    ob_bonus += 15
                    ob_notes.append(f"ASK_WALL({top['multiple']:.1f}x)")
                if ob_sigs['spoof_signal']:
                    ob_bonus += 20
                    ob_notes.append(f"SPOOF(b={ob_sigs['bid_spoof']})")

        total_score += ob_bonus

        if self.verbose:
            logger.info(
                f"  📊 {sym.upper()}: score={total_score} "
                f"(W={result.get('l4',{}).get('score',0)} "
                f"M={result.get('l5',{}).get('score',0)} "
                f"OB=+{ob_bonus}) dir={result.get('direction','?')}"
            )

        if total_score < self.min_score:
            return

        self._alert_count += 1
        confidence = self._confidence_label(total_score)
        self._print_alert(sym.upper(), result, total_score, confidence, ob_notes, ob_bonus)

    @staticmethod
    def _confidence_label(score: int) -> str:
        if score >= 250: return 'ULTRA'
        if score >= 160: return 'HIGH'
        if score >= 90:  return 'MEDIUM'
        return 'LOW'

    # ============================================================
    # ALERTAS
    # ============================================================

    def _print_alert(
        self,
        symbol: str,
        result: dict,
        total_score: int,
        confidence: str,
        ob_notes: List[str],
        ob_bonus: int,
    ):
        emoji_map = {'ULTRA': '🦈🦈', 'HIGH': '🐋', 'MEDIUM': '🐬', 'LOW': '🐟'}
        dir_emoji = {'LONG': '🟢', 'SHORT': '🔴', 'NEUTRAL': '⚪'}
        emoji = emoji_map.get(confidence, '─')
        d_emoji = dir_emoji.get(result.get('direction', ''), '⚪')

        ts = time.strftime("%H:%M:%S")
        l4 = result.get('l4', {'score': 0, 'reasons': []})
        l5 = result.get('l5', {'score': 0, 'reasons': []})

        lines = [
            f"\n{'='*60}",
            f"{ts} {emoji} [{confidence}] {symbol} | Score: {total_score} {d_emoji}{result.get('direction', '?')}",
            f"   🐋 WHALE ({l4['score']}):   {', '.join(l4['reasons'][:4]) or '—'}",
            f"   ⚠️  MANIP ({l5['score']}):   {', '.join(l5['reasons'][:3]) or '—'}",
        ]
        if ob_notes:
            lines.append(f"   📖 OB   (+{ob_bonus}):   {', '.join(ob_notes)}")
        lines.append(
            f"   CVD: {result.get('cvd_slope', 0):+.3f} | "
            f"Absorb: {result.get('absorption_count', 0)} candles | "
            f"Candles en buffer: {len(self.candle_buffers.get(symbol.lower(), []))}"
        )
        print("\n".join(lines), flush=True)


# ============================================================
# ENTRY POINT
# ============================================================

def build_arg_parser():
    p = argparse.ArgumentParser(description="Whale Real-Time Scanner v2 (Binance Futures)")
    p.add_argument("--top-n",     type=int,   default=50,   help="Nº de pares (default: 50)")
    p.add_argument("--min-vol",   type=float, default=50.0, help="Vol mínimo 24h en $M (default: 50)")
    p.add_argument("--min-score", type=int,   default=40,   help="Score mínimo para alerta (default: 40)")
    p.add_argument("--no-ob",     action="store_true",      help="Desactivar Order Book tracker")
    p.add_argument("--no-cvd",    action="store_true",      help="Desactivar AggTrades CVD real")
    p.add_argument("--verbose",   action="store_true",      help="Mostrar scores de todos los pares en cada scan")
    p.add_argument("--symbols",   type=str,   default=None, help="Símbolos separados por coma (override)")
    return p


async def _main():
    args = build_arg_parser().parse_args()

    if not _WS_AVAILABLE:
        print("❌ pip install websockets")
        return

    print("🦈 WHALE REAL-TIME SCANNER v2 — Binance Futures")
    print("=" * 60)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        print(f"📌 Símbolos manuales: {symbols}")
    else:
        print(f"🔍 Universo: top {args.top_n} pares (≥${args.min_vol}M vol/24h) ...")
        symbols = get_liquid_universe_sync(top_n=args.top_n, min_vol_usdt_m=args.min_vol)
        if not symbols:
            print("❌ No se pudo obtener el universo. Revisa conexión.")
            return
        print(f"✅ {len(symbols)} pares: {symbols[:6]}...")

    print(f"⚙️  Score mínimo: {args.min_score} | OB: {'OFF' if args.no_ob else 'ON'} | "
          f"CVD real: {'OFF' if args.no_cvd else 'ON'}")
    print("─" * 60)

    scanner = WhaleRealtimeScanner(
        symbols=symbols,
        min_score=args.min_score,
        enable_orderbook=not args.no_ob,
        enable_aggtrades=not args.no_cvd,
        verbose=args.verbose,
    )

    try:
        await scanner.start()
    except KeyboardInterrupt:
        await scanner.stop()


if __name__ == "__main__":
    asyncio.run(_main())
