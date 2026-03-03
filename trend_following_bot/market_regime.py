"""
market_regime.py — Filtro de Régimen de Mercado

Usa BTC como barómetro macro del mercado para determinar si el entorno
es favorable para operar tendencias o si hay que reducir/detener exposición.

Regímenes:
  BULL   → ADX > 22 + KF slope positivo  → operar LONG con prioridad, SHORT reducido
  BEAR   → ADX > 22 + KF slope negativo  → operar SHORT con prioridad, LONG reducido
  RANGING → ADX < 20                      → pausar nuevas entradas (el dinero se pierde aquí)
  VOLATILE → Volatilidad >2x normal        → reducir tamaño, entrar solo con señales ultra fuertes

Cómo funciona en el bot:
  1. Se calcula una vez por ciclo de macro scan (cada hora o al inicio del día)
  2. Devuelve SlotConfig: max_signals, long_allowed, short_allowed, size_multiplier
  3. El bot respeta esta config para todas las entradas del ciclo

Integración:
  - Backtest: se llama con el DataFrame alineado de BTC al índice actual
  - Live: se llama con los klines reales de BTC descargados desde Binance
"""

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════

REGIME_BULL     = "BULL"
REGIME_BEAR     = "BEAR"
REGIME_RANGING  = "RANGING"
REGIME_VOLATILE = "VOLATILE"

# Thresholds
ADX_TRENDING    = 22    # ADX ≥ 22 → hay tendencia definida
ADX_RANGING     = 18    # ADX < 18 → mercado en rango claro
VOLATILE_MULT   = 2.2   # Si ATR actual > 2.2x ATR promedio → mercado volátil
ATR_LOOKBACK    = 48    # Velas para calcular ATR promedio (12h en 15m)

# Configuración de slots por régimen
# max_signals: cuántas posiciones simultáneas permite
# long_bias / short_bias: multiplica el score de picks en esa dirección  
# size_mult: multiplica el tamaño de posición (1.0 = normal)

REGIME_CONFIG = {
    REGIME_BULL: {
        "max_signals":  5,
        "long_bias":    1.2,    # Favorece longs 20%
        "short_bias":   0.0,    # BLOQUEADO en BULL: shorts = score 0 = no entran
        "size_mult":    1.0,
        "description":  "Tendencia alcista - solo LONGs",
    },
    REGIME_BEAR: {
        "max_signals":  5,
        "long_bias":    0.0,    # BLOQUEADO en BEAR: longs = score 0 = no entran
        "short_bias":   1.2,    # Favorece shorts 20%
        "size_mult":    1.0,
        "description":  "Tendencia bajista - solo SHORTs",
    },
    REGIME_RANGING: {
        "max_signals":  2,      # Solo 2 posiciones max en rango
        "long_bias":    0.5,
        "short_bias":   0.5,
        "size_mult":    0.5,    # Mitad de tamaño
        "description":  "Mercado lateral — exposición reducida",
    },
    REGIME_VOLATILE: {
        "max_signals":  2,
        "long_bias":    0.8,
        "short_bias":   0.8,
        "size_mult":    0.6,    # 60% del tamaño normal
        "description":  "Alta volatilidad — tamaño reducido",
    },
}


# ═══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def detect_regime(btc_df: pd.DataFrame, candle_idx: int = -1) -> dict:
    """
    Detecta el régimen de mercado actual usando el DataFrame de BTC.

    Args:
        btc_df:     DataFrame con columnas OHLCV del BTC (ya con indicadores calculados
                    via calculate_indicators, o calculados aquí si no vienen).
        candle_idx: Índice de la vela "actual" (-1 = última vela disponible).

    Returns:
        dict con keys:
            regime      (str)   : BULL | BEAR | RANGING | VOLATILE
            adx         (float) : ADX actual
            kf_slope    (float) : Pendiente del filtro Kalman
            atr_ratio   (float) : ATR actual / ATR promedio
            config      (dict)  : Configuración de slots del régimen
            description (str)   : Descripción legible
    """
    result = {
        "regime":      REGIME_RANGING,
        "adx":         0.0,
        "kf_slope":    0.0,
        "atr_ratio":   1.0,
        "config":      REGIME_CONFIG[REGIME_RANGING],
        "description": "Sin datos — modo conservador",
    }

    try:
        # Slice hasta candle_idx
        if candle_idx == -1 or candle_idx >= len(btc_df):
            df = btc_df.copy()
        else:
            df = btc_df.iloc[:candle_idx + 1].copy()

        if len(df) < 50:
            return result

        close  = pd.to_numeric(df['close'],  errors='coerce').ffill()
        high   = pd.to_numeric(df['high'],   errors='coerce').ffill()
        low    = pd.to_numeric(df['low'],    errors='coerce').ffill()

        # ── ADX ─────────────────────────────────────────────────
        if 'adx' in df.columns and df['adx'].iloc[-1] > 0:
            adx = float(df['adx'].iloc[-1])
        else:
            adx = _calc_adx(high, low, close, period=14)

        # ── KF Slope ─────────────────────────────────────────────
        if 'kf_slope' in df.columns:
            kf_slope = float(df['kf_slope'].iloc[-1])
        else:
            kf_slope = _calc_kf_slope(close)

        # ── ATR Ratio ─────────────────────────────────────────────
        if 'atr' in df.columns and df['atr'].iloc[-1] > 0:
            atr_now = float(df['atr'].iloc[-1])
            atr_avg = float(df['atr'].iloc[-ATR_LOOKBACK:-1].mean())
        else:
            atr_now, atr_avg = _calc_atr_ratio(high, low, close)

        atr_ratio = atr_now / atr_avg if atr_avg > 0 else 1.0

        # ── Clasificación ─────────────────────────────────────────
        if atr_ratio > VOLATILE_MULT:
            regime = REGIME_VOLATILE

        elif adx >= ADX_TRENDING:
            regime = REGIME_BULL if kf_slope > 0 else REGIME_BEAR

        else:
            regime = REGIME_RANGING

        config = REGIME_CONFIG[regime]
        result = {
            "regime":      regime,
            "adx":         round(adx, 2),
            "kf_slope":    round(kf_slope, 8),
            "atr_ratio":   round(atr_ratio, 2),
            "config":      config,
            "description": config["description"],
        }

    except Exception as e:
        result["description"] = f"Error en detección: {e} — modo conservador"

    return result


# ═══════════════════════════════════════════════════════════════
# HELPERS: cálculos inline si los indicadores no están en el df
# ═══════════════════════════════════════════════════════════════

def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Calcula el ADX(14) sin dependencias externas."""
    try:
        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low  - close.shift(1)).abs(),
        }).max(axis=1)

        atr14    = tr.rolling(period).mean().replace(0, np.nan)
        plus_di  = 100 * (plus_dm.rolling(period).mean()  / atr14)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr14)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx      = dx.rolling(period).mean()
        return float(adx.iloc[-1]) if not adx.empty else 0.0
    except Exception:
        return 0.0


def _calc_kf_slope(close: pd.Series) -> float:
    """Kalman filter slope simple (Constant Velocity Model, q=0.01, r=8.0)."""
    try:
        prices = close.dropna().values[-200:]   # Últimas 200 velas
        if len(prices) < 20:
            return 0.0

        x = np.array([[prices[0]], [0.0]])
        P = np.eye(2)
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        R = np.array([[8.0]])
        Q = np.array([[0.01, 0], [0, 0.01]])

        for z in prices:
            x = F @ x
            P = F @ P @ F.T + Q
            y = z - (H @ x)[0, 0]
            S = (H @ P @ H.T + R)[0, 0]
            K = (P @ H.T / S)
            x = x + K * y
            P = (np.eye(2) - K @ H) @ P

        return float(x[1, 0])   # velocity component = slope
    except Exception:
        return 0.0


def _calc_atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
    """Calcula ATR actual y promedio."""
    try:
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low  - close.shift(1)).abs(),
        }).max(axis=1)
        atr = tr.rolling(14).mean()
        return float(atr.iloc[-1]), float(atr.iloc[-ATR_LOOKBACK:-1].mean())
    except Exception:
        return 1.0, 1.0


# ═══════════════════════════════════════════════════════════════
# INTEGRACIÓN CON LIVE BOT (async wrapper)
# ═══════════════════════════════════════════════════════════════

async def get_live_regime(market_data) -> dict:
    """
    Wrapper async para usar en main.py.
    Descarga klines de BTCUSDT y detecta el régimen actual.

    Args:
        market_data: instancia de MarketData con método get_klines()

    Returns:
        dict del régimen (mismo formato que detect_regime)
    """
    try:
        btc_df = await market_data.get_klines("BTCUSDT", interval="15m", limit=200)
        if btc_df is None or btc_df.empty:
            return {"regime": REGIME_RANGING, "config": REGIME_CONFIG[REGIME_RANGING],
                    "description": "BTC sin datos — conservador", "adx": 0, "kf_slope": 0, "atr_ratio": 1}
        return detect_regime(btc_df)
    except Exception as e:
        return {"regime": REGIME_RANGING, "config": REGIME_CONFIG[REGIME_RANGING],
                "description": f"Error BTC: {e}", "adx": 0, "kf_slope": 0, "atr_ratio": 1}
