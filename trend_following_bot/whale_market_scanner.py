"""
whale_market_scanner.py — Scanner de ballenas sobre todo el mercado de futuros Binance.

Itera TODOS los pares USDT en batches de 50, descarga 200 velas 15m por par,
calcula el whale_score y retorna los top-N con las señales más fuertes.

Ahora incluye contexto on-chain (OI, funding, long/short ratio, agg_trades)
y datos del libro de órdenes en tiempo real (ob_streamer) como señales extra.

Uso desde main.py:
    from whale_market_scanner import scan_whale_universe
    whale_picks = await scan_whale_universe(market, top_n=15, ob_streamer=self.ob_streamer)
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

import pandas as pd

from whale_math_core import whale_score

logger = logging.getLogger("WhaleMarketScanner")

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

BATCH_SIZE       = 50     # Pares por iteración
MIN_VOL_USDT_M   = 5.0    # Volumen mínimo 24h en millones USDT
MIN_SCORE        = 130    # Score mínimo: señales sólidas (HIGH+ en nuevo rango)
KLINES_LIMIT     = 200    # Velas a descargar por par
KLINES_INTERVAL  = '15m'  # Intervalo de las velas
CONCURRENCY      = 8      # Descargas paralelas dentro de cada batch
INTER_BATCH_WAIT = 0.3    # Segundos de pausa entre batches (evitar rate limit)

# Context fetch timeout (evitar que un endpoint lento bloquee el scan)
CONTEXT_TIMEOUT  = 4.0    # segundos

# Blacklist de pares con comportamiento errático / baja calidad
BLACKLIST = {
    'BTCDOMUSDT', 'DEFIUSDT', 'USDCUSDT', 'FDUSDUSDT',
    'BUSDUSDT', 'TUSDUSDT', 'USDPUSDT',
}


# ===================================================================
# HELPERS
# ===================================================================

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el DataFrame para que whale_math_core lo entienda."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    rename_map = {
        'taker_buy_base_asset_volume': 'taker_buy_vol',
        'taker_buy_quote_asset_volume': 'taker_buy_quote_vol',
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


def _make_pick(symbol: str, result: dict, quote_vol_m: float) -> Dict[str, Any]:
    """Crea el dict de pick en formato compatible con daily_watchlist del bot."""
    return {
        'symbol':        symbol,
        'score':         result['score'],
        'direction':     result['direction'],
        'reasons':       f"WHALE:{result['confidence']} | {', '.join(result['reasons'][:4])}",
        'layer':         'WHALE',
        'confidence':    result['confidence'],
        'cvd_slope':     result.get('cvd_slope', 0.0),
        'absorption':    result.get('absorption', 0),
        'vol_24h_m':     round(quote_vol_m, 1),
        'whale_reasons': result['reasons'],
    }


async def _safe(coro, default):
    """Ejecuta una coroutine con timeout, retornando default en caso de fallo."""
    try:
        return await asyncio.wait_for(coro, timeout=CONTEXT_TIMEOUT)
    except Exception:
        return default


# ===================================================================
# FILTRO DEL UNIVERSO
# ===================================================================

async def _get_universe(market) -> List[str]:
    """
    Obtiene todos los pares USDT activos con vol > MIN_VOL_USDT_M.
    """
    try:
        symbols = await market.get_trading_universe()
        if symbols:
            logger.info(f"🌍 Universo via MarketData: {len(symbols)} pares")
            return [s for s in symbols if s not in BLACKLIST]
    except Exception as e:
        logger.debug(f"get_trading_universe fallback: {e}")

    # Fallback REST con reintentos y backoff exponencial (maneja 418)
    import requests, time as _time
    BACKOFFS = [5, 10, 20]  # segundos entre reintentos
    for attempt, wait in enumerate(BACKOFFS, 1):
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/ticker/24hr",
                timeout=10,
            )
            if resp.status_code == 418:
                retry_after = int(resp.headers.get('Retry-After', wait))
                logger.warning(
                    f"⚠️ 418 Ban en ticker/24hr (intento {attempt}/3). "
                    f"Esperando {retry_after}s antes de reintentar..."
                )
                _time.sleep(retry_after)
                continue
            resp.raise_for_status()
            tickers = resp.json()
            min_vol = MIN_VOL_USDT_M * 1_000_000
            symbols = [
                t['symbol'] for t in tickers
                if t['symbol'].endswith('USDT')
                and t['symbol'] not in BLACKLIST
                and float(t.get('quoteVolume', 0)) >= min_vol
            ]
            symbols.sort(
                key=lambda s: float(next(
                    (t['quoteVolume'] for t in tickers if t['symbol'] == s), 0
                )),
                reverse=True,
            )
            logger.info(f"🌍 Universo via REST (intento {attempt}): {len(symbols)} pares")
            return symbols
        except Exception as e:
            logger.warning(f"🌍 Intento {attempt}/3 fallo: {e} — esperando {wait}s...")
            _time.sleep(wait)

    logger.error("❌ No se pudo obtener el universo tras 3 intentos. Retornando lista vacia.")
    return []



# ===================================================================
# DESCARGA Y ANÁLISIS POR BATCH
# ===================================================================

async def _analyze_batch(
    symbols: List[str],
    market,
    min_score: int,
    ob_streamer=None,
) -> List[Dict[str, Any]]:
    """
    Descarga klines, contexto on-chain y datos de OB, calcula whale_score.
    Retorna lista de picks con score >= min_score.
    """
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []

    async def _analyze_one(sym: str):
        async with sem:
            try:
                # ── 1. Klines (obligatorio) ──────────────────────────
                df = await market.get_klines(
                    sym, interval=KLINES_INTERVAL, limit=KLINES_LIMIT
                )
                if df is None or df.empty or len(df) < 96:
                    return

                df = _normalize_df(df)

                # Filtro de liquidez
                last_close = float(df['close'].iloc[-1]) if len(df) > 0 else 0
                vol_24h_candles = min(96, len(df))
                vol_24h_m = (
                    float(df['volume'].iloc[-vol_24h_candles:].sum()) * last_close / 1_000_000
                )
                if vol_24h_m < MIN_VOL_USDT_M:
                    return

                # ── 2. Contexto on-chain (Fases 1 y 4 — en paralelo) ─
                oi_coro      = market.get_open_interest(sym, period='1h')
                funding_coro = market.get_funding_rate(sym, limit=3)
                ls_coro      = market.get_long_short_ratio(sym, period='1h', limit=3)
                trades_coro  = market.get_agg_trades(sym, limit=300)

                oi_hist, funding, ls_data, agg_trades = await asyncio.gather(
                    _safe(oi_coro,      []),
                    _safe(funding_coro, []),
                    _safe(ls_coro,      []),
                    _safe(trades_coro,  []),
                )

                # ── 3. Order Book desde streamer (Fase 2) ────────────
                ob_bids, ob_asks = [], []
                if ob_streamer is not None:
                    book = ob_streamer.books.get(sym, {})
                    ob_bids = book.get('bids', [])
                    ob_asks = book.get('asks', [])

                # ── 4. Construir contexto ────────────────────────────
                context = {
                    'oi_history':   oi_hist,
                    'funding_list': funding,
                    'ls_data':      ls_data,
                    'ob_bids':      ob_bids,
                    'ob_asks':      ob_asks,
                    'agg_trades':   agg_trades,
                }

                # ── 5. Calcular whale score ───────────────────────────
                ws = whale_score(df, context=context)
                if ws['score'] < min_score:
                    return

                results.append(_make_pick(sym, ws, vol_24h_m))

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"Error analizando {sym}: {e}")

    await asyncio.gather(*[_analyze_one(s) for s in symbols])
    return results


# ===================================================================
# FUNCIÓN PÚBLICA PRINCIPAL
# ===================================================================

async def scan_whale_universe(
    market,
    top_n: int = 15,
    min_score: int = MIN_SCORE,
    verbose: bool = False,
    ob_streamer=None,
) -> List[Dict[str, Any]]:
    """
    Escanea TODO el mercado de futuros Binance en busca de ballenas.

    Args:
        market:     instancia de MarketData (del bot)
        top_n:      número de picks a retornar (default 15)
        min_score:  score mínimo para incluir un par
        verbose:    si True, loggea cada par analizado
        ob_streamer: instancia de OrderbookStreamer (opcional, activa Fase 2)

    Returns:
        lista de dicts con keys: symbol, score, direction, reasons, confidence,
        cvd_slope, absorption, vol_24h_m, whale_reasons → ordenada de mayor a menor score
    """
    t_start = time.time()
    ob_active = ob_streamer is not None
    logger.info(
        f"🐋 WHALE MARKET SCAN iniciando "
        f"(top={top_n}, min_score={min_score}, ob_streamer={'ON' if ob_active else 'OFF'})..."
    )

    # 1. Obtener universo
    universe = await _get_universe(market)
    if not universe:
        logger.warning("⚠️ Universo vacío — no se puede escanear")
        return []

    total = len(universe)
    logger.info(f"   Universo: {total} pares → {(total + BATCH_SIZE - 1) // BATCH_SIZE} batches de {BATCH_SIZE}")

    # 2. Dividir en batches y procesar
    batches = [universe[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    all_picks = []

    for idx, batch in enumerate(batches, 1):
        if verbose:
            logger.info(f"   Batch {idx}/{len(batches)}: analizando {len(batch)} pares...")
        else:
            logger.debug(f"   Batch {idx}/{len(batches)}: {len(batch)} pares")

        batch_picks = await _analyze_batch(batch, market, min_score, ob_streamer=ob_streamer)
        all_picks.extend(batch_picks)

        if verbose and batch_picks:
            for p in batch_picks:
                logger.info(
                    f"     🐋 {p['symbol']}: score={p['score']} "
                    f"({p['confidence']}) dir={p['direction']}"
                )

        if idx < len(batches):
            await asyncio.sleep(INTER_BATCH_WAIT)

    # 3. Ordenar y seleccionar top-N
    all_picks.sort(key=lambda x: x['score'], reverse=True)
    top_picks = all_picks[:top_n]

    elapsed = time.time() - t_start
    logger.info(
        f"✅ WHALE SCAN completo en {elapsed:.1f}s | "
        f"{len(all_picks)} candidatos → top {len(top_picks)} seleccionados"
    )

    if top_picks:
        logger.info("🐋 Top whale picks:")
        for p in top_picks[:5]:
            logger.info(
                f"   {p['symbol']:12s} score={p['score']:3d} [{p['confidence']}] "
                f"{p['direction']:7s} | {', '.join(p['whale_reasons'][:2])}"
            )
        if len(top_picks) > 5:
            logger.info(f"   ... +{len(top_picks) - 5} más")

    return top_picks
