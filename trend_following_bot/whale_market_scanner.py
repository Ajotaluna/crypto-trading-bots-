"""
whale_market_scanner.py — Scanner de ballenas sobre todo el mercado de futuros Binance.

Itera TODOS los pares USDT en batches de 50, descarga 200 velas 15m por par,
calcula el whale_score y retorna los top-N con las señales más fuertes.

Este módulo es completamente autónomo (sin nascent_scanner).

Uso desde main.py:
    from whale_market_scanner import scan_whale_universe
    whale_picks = await scan_whale_universe(market, top_n=15)
"""
import asyncio
import logging
import time
from typing import List, Dict, Any

import pandas as pd

from whale_math_core import whale_score

logger = logging.getLogger("WhaleMarketScanner")

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

BATCH_SIZE       = 50     # Pares por iteración (límite de streams Binance)
MIN_VOL_USDT_M   = 10.0   # Volumen mínimo 24h en millones USDT
MIN_SCORE        = 40     # Score mínimo para considerar un par
KLINES_LIMIT     = 200    # Velas a descargar por par
KLINES_INTERVAL  = '15m'  # Intervalo de las velas
CONCURRENCY      = 8      # Descargas paralelas dentro de cada batch
INTER_BATCH_WAIT = 0.3    # Segundos de pausa entre batches (evitar rate limit)

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

    # Renombrar taker buy si viene con nombre largo
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


# ===================================================================
# FILTRO DEL UNIVERSO
# ===================================================================

async def _get_universe(market) -> List[str]:
    """
    Obtiene todos los pares USDT activos con vol > MIN_VOL_USDT_M.
    Usa get_trading_universe() de MarketData si está disponible,
    cae en REST directo como fallback.
    """
    try:
        # Primero intentamos con el método del bot
        symbols = await market.get_trading_universe()
        if symbols:
            logger.info(f"🌍 Universo via MarketData: {len(symbols)} pares")
            return [s for s in symbols if s not in BLACKLIST]
    except Exception as e:
        logger.debug(f"get_trading_universe fallback: {e}")

    # Fallback REST directo
    try:
        import requests
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/ticker/24hr",
            timeout=10,
        )
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
        logger.info(f"🌍 Universo via REST directo: {len(symbols)} pares")
        return symbols
    except Exception as e:
        logger.error(f"Error obteniendo universo: {e}")
        return []


# ===================================================================
# DESCARGA Y ANÁLISIS POR BATCH
# ===================================================================

async def _analyze_batch(
    symbols: List[str],
    market,
    min_score: int,
) -> List[Dict[str, Any]]:
    """
    Descarga klines y calcula whale_score para un batch de símbolos.
    Retorna lista de picks con score >= min_score.
    """
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []

    async def _analyze_one(sym: str):
        async with sem:
            try:
                df = await market.get_klines(
                    sym, interval=KLINES_INTERVAL, limit=KLINES_LIMIT
                )
                if df is None or df.empty or len(df) < 96:
                    return

                df = _normalize_df(df)

                # Calcular volumen 24h en millones USDT
                last_close = float(df['close'].iloc[-1]) if len(df) > 0 else 0
                vol_24h_candles = min(96, len(df))
                vol_24h_m = (
                    float(df['volume'].iloc[-vol_24h_candles:].sum()) * last_close / 1_000_000
                )

                # Aplicar filtro de liquidez mínima
                if vol_24h_m < MIN_VOL_USDT_M:
                    return

                # Calcular whale score
                ws = whale_score(df)
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
) -> List[Dict[str, Any]]:
    """
    Escanea TODO el mercado de futuros Binance en busca de ballenas.

    Itera en batches de BATCH_SIZE (50), descarga 200 velas 15m por par,
    calcula whale_score y retorna los top_n con las señales más fuertes.

    Args:
        market:    instancia de MarketData (del bot)
        top_n:     número de picks a retornar (default 15)
        min_score: score mínimo para incluir un par (default 40)
        verbose:   si True, loggea cada par analizado

    Returns:
        list de dicts con keys: symbol, score, direction, reasons, confidence,
        cvd_slope, absorption, vol_24h_m, whale_reasons
        → último de mayor a menor score

    Ejemplo de pick retornado:
        {
          'symbol': 'SOLUSDT',
          'score': 160,
          'direction': 'LONG',
          'reasons': 'WHALE:HIGH | W_CVD_ACCUM, W_ABSORPTION(3v)',
          'layer': 'WHALE',
          'confidence': 'HIGH',
          ...
        }
    """
    t_start = time.time()
    logger.info(f"🐋 WHALE MARKET SCAN iniciando (top={top_n}, min_score={min_score})...")

    # 1. Obtener universo
    universe = await _get_universe(market)
    if not universe:
        logger.warning("⚠️ Universo vacío — no se puede escanear")
        return []

    total = len(universe)
    logger.info(f"   Universo: {total} pares → {(total + BATCH_SIZE - 1) // BATCH_SIZE} batches de {BATCH_SIZE}")

    # 2. Dividir en batches
    batches = [
        universe[i:i + BATCH_SIZE]
        for i in range(0, total, BATCH_SIZE)
    ]

    # 3. Procesar batch por batch
    all_picks = []
    for idx, batch in enumerate(batches, 1):
        if verbose:
            logger.info(f"   Batch {idx}/{len(batches)}: analizando {len(batch)} pares...")
        else:
            logger.debug(f"   Batch {idx}/{len(batches)}: {len(batch)} pares")

        batch_picks = await _analyze_batch(batch, market, min_score)
        all_picks.extend(batch_picks)

        if verbose and batch_picks:
            for p in batch_picks:
                logger.info(
                    f"     🐋 {p['symbol']}: score={p['score']} "
                    f"({p['confidence']}) dir={p['direction']}"
                )

        # Pausa breve entre batches para no saturar la API
        if idx < len(batches):
            await asyncio.sleep(INTER_BATCH_WAIT)

    # 4. Ordenar y seleccionar top-N
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
