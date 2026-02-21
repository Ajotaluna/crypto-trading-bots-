"""
Universe Filter — Selección inteligente del universo de pares para el Whale Scanner.

En tiempo real no tiene sentido analizar 300 pares simultáneamente.
Este módulo selecciona los pares más líquidos e institucionalmente relevantes.

Uso:
    from nascent_scanner.universe_filter import get_liquid_universe
    symbols = await get_liquid_universe(top_n=60, min_vol_usdt_m=50)
"""
import asyncio
import logging
import os
import json
import requests

logger = logging.getLogger("UniverseFilter")

# Defaults
BINANCE_FUTURES_TICKER = "https://fapi.binance.com/fapi/v1/ticker/24hr"
WHITELIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "whitelist.json"
)

# Pares a excluir siempre (muy volátiles, thin liquidity, leveraged tokens, etc.)
BLACKLIST_FRAGMENTS = [
    "DOWNUSDT", "UPUSDT", "BULLUSDT", "BEARUSDT",
    "BUSDUSDT", "USDCUSDT", "TUSDUSDT",
]


def _load_whitelist():
    """Carga el whitelist.json si existe."""
    try:
        if os.path.exists(WHITELIST_PATH):
            with open(WHITELIST_PATH, "r") as f:
                return set(json.load(f))
    except Exception:
        pass
    return set()


def _is_blacklisted(symbol):
    return any(frag in symbol for frag in BLACKLIST_FRAGMENTS)


def get_liquid_universe_sync(
    top_n: int = 60,
    min_vol_usdt_m: float = 50.0,
    use_whitelist: bool = True,
    timeout: int = 10,
) -> list:
    """
    Fetch síncrono del universo líquido desde Binance Futures 24h ticker.

    Args:
        top_n: número máximo de pares a retornar
        min_vol_usdt_m: volumen mínimo en millones de USDT/24h
        use_whitelist: si True, intersectar con whitelist.json
        timeout: timeout HTTP en segundos

    Returns:
        list de symbols ordenados por volumen descendente
    """
    try:
        resp = requests.get(BINANCE_FUTURES_TICKER, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"UniverseFilter: No se pudo obtener ticker 24h: {e}")
        return []

    whitelist = _load_whitelist() if use_whitelist else set()
    min_vol = min_vol_usdt_m * 1_000_000

    candidates = []
    for x in data:
        sym = x.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if _is_blacklisted(sym):
            continue
        # Si hay whitelist, solo incluir pares en ella
        if use_whitelist and whitelist and sym not in whitelist:
            continue
        try:
            vol = float(x.get("quoteVolume", 0))
        except Exception:
            continue
        if vol < min_vol:
            continue
        candidates.append((sym, vol))

    # Ordenar por volumen descendente
    candidates.sort(key=lambda x: x[1], reverse=True)
    result = [sym for sym, _ in candidates[:top_n]]

    logger.info(
        f"UniverseFilter: {len(result)} pares seleccionados "
        f"(min_vol=${min_vol_usdt_m}M, top_n={top_n})"
    )
    return result


async def get_liquid_universe(
    top_n: int = 60,
    min_vol_usdt_m: float = 50.0,
    use_whitelist: bool = True,
) -> list:
    """
    Versión asíncrona (no bloquea el event loop).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: get_liquid_universe_sync(top_n, min_vol_usdt_m, use_whitelist)
    )


def describe_universe(symbols: list) -> str:
    """Retorna un string descriptivo del universo seleccionado."""
    major = [s for s in symbols if s in (
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "SUIUSDT",
    )]
    altcoins = len(symbols) - len(major)
    return (
        f"{len(symbols)} pares: {len(major)} major + {altcoins} altcoins | "
        f"Primeros 5: {symbols[:5]}"
    )


if __name__ == "__main__":
    import sys
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    min_v = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    syms = get_liquid_universe_sync(top_n=top, min_vol_usdt_m=min_v)
    print(describe_universe(syms))
    for i, s in enumerate(syms, 1):
        print(f"  {i:3d}. {s}")
