"""
whale_math_core.py — Funciones matemáticas de detección de ballenas.

Módulo autónomo (sin dependencias de nascent_scanner).
Diseñado para operar directamente sobre DataFrames de klines REST estándar.

Columnas requeridas en el DataFrame:
    open, high, low, close, volume, taker_buy_vol (o taker_buy_base_asset_volume)

Uso:
    from whale_math_core import whale_score
    result = whale_score(df)
    # result = {'score': 140, 'direction': 'LONG', 'reasons': [...], 'confidence': 'HIGH'}
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

# ===================================================================
# CONSTANTES
# ===================================================================

CVD_WINDOW       = 96   # Candles para calcular CVD (= 24h en 15m)
CVD_SLOPE_WIN    = 48   # Ventana para la pendiente del CVD
ABSORB_WINDOW    = 48   # Ventana para absorción
SWEEP_WINDOW     = 48   # Ventana para liquidity sweeps
VOL_ZSCORE_WIN   = 96   # Ventana para z-score de volumen
VOLUME_CLOCK_WIN = 32   # Ventana para análisis de volume clock


# ===================================================================
# 1. CVD (Cumulative Volume Delta)
# ===================================================================

def _cvd(df: pd.DataFrame, window: int = CVD_WINDOW) -> pd.Series:
    """
    Calcula el CVD usando taker_buy_vol si disponible,
    si no usa aproximación por dirección de la vela.
    """
    data = df.tail(window).copy()

    if 'taker_buy_vol' in data.columns and data['taker_buy_vol'].sum() > 0:
        sell_vol = data['volume'] - data['taker_buy_vol']
        delta = data['taker_buy_vol'] - sell_vol
    elif 'taker_buy_base_asset_volume' in data.columns:
        sell_vol = data['volume'] - data['taker_buy_base_asset_volume']
        delta = data['taker_buy_base_asset_volume'] - sell_vol
    else:
        # Aproximación: delta proporcional al cuerpo de la vela
        body = data['close'] - data['open']
        rng  = (data['high'] - data['low']).replace(0, 1e-10)
        delta = data['volume'] * (body / rng)

    return delta.cumsum()


def cvd_slope(df: pd.DataFrame,
              window: int = CVD_WINDOW,
              slope_window: int = CVD_SLOPE_WIN) -> float:
    """
    Pendiente del CVD normalizada por precio.
    Positivo = presión compradora creciente, negativo = vendedora.
    """
    try:
        cvd_series = _cvd(df, window)
        if len(cvd_series) < slope_window:
            return 0.0
        recent = cvd_series.iloc[-slope_window:]
        x = np.arange(len(recent), dtype=float)
        slope = np.polyfit(x, recent.values, 1)[0]
        price = float(df['close'].iloc[-1]) if df['close'].iloc[-1] != 0 else 1.0
        return slope / price
    except Exception:
        return 0.0


# ===================================================================
# 2. ABSORCIÓN (alto volumen + rango pequeño)
# ===================================================================

def absorption_score(df: pd.DataFrame,
                     window: int = ABSORB_WINDOW,
                     vol_threshold: float = 2.0,
                     range_threshold: float = 0.4) -> Tuple[pd.Series, int]:
    """
    Detecta velas de absorción: alto volumen + rango pequeño.
    Retorna (serie_bool, conteo_de_absorción).
    """
    try:
        data = df.tail(window).copy()
        vol  = pd.to_numeric(data['volume'], errors='coerce').fillna(0)
        high = pd.to_numeric(data['high'],   errors='coerce').fillna(0)
        low  = pd.to_numeric(data['low'],    errors='coerce').fillna(0)

        if vol.std() == 0 or len(data) < 10:
            return pd.Series(dtype=bool), 0

        vol_mean   = vol.mean()
        rng        = (high - low).replace(0, 1e-10)
        range_mean = rng.mean()

        high_vol    = vol > vol_mean * vol_threshold
        small_range = rng < range_mean * range_threshold
        is_absorb   = high_vol & small_range

        return is_absorb, int(is_absorb.sum())
    except Exception:
        return pd.Series(dtype=bool), 0


# ===================================================================
# 3. LIQUIDITY SWEEPS (stop hunts)
# ===================================================================

def liquidity_sweep(df: pd.DataFrame,
                    window: int = SWEEP_WINDOW,
                    wick_body_ratio: float = 2.5) -> List[Dict]:
    """
    Detecta wick extremos que indican barridos de liquidez (stop hunts).
    Retorna lista de eventos {'type': 'BULLISH'|'BEARISH', 'idx', 'wick_ratio'}.
    """
    events = []
    try:
        data = df.tail(window).copy().reset_index(drop=True)
        for i in range(len(data)):
            c = data.iloc[i]
            try:
                o, h, l, cl = float(c['open']), float(c['high']), float(c['low']), float(c['close'])
            except Exception:
                continue

            body        = abs(cl - o)
            upper_wick  = h - max(o, cl)
            lower_wick  = min(o, cl) - l

            if body < 1e-10:
                continue

            # Bullish sweep: wick inferior largo + cierre alcista
            if lower_wick > body * wick_body_ratio and cl > o:
                events.append({'type': 'BULLISH', 'idx': i, 'wick_ratio': lower_wick / body})
            # Bearish sweep: wick superior largo + cierre bajista
            if upper_wick > body * wick_body_ratio and cl < o:
                events.append({'type': 'BEARISH', 'idx': i, 'wick_ratio': upper_wick / body})
    except Exception:
        pass
    return events


# ===================================================================
# 4. VOLUMEN Z-SCORE (spike detector)
# ===================================================================

def volume_zscore(df: pd.DataFrame, window: int = VOL_ZSCORE_WIN) -> float:
    """
    Z-score del volumen de las últimas 24 velas vs el historial.
    > 2.0 = spike significativo.
    """
    try:
        vol = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        if len(vol) < window + 24:
            return 0.0
        hist    = vol.iloc[-(window + 24):-24]
        current = vol.iloc[-24:].sum()
        # Agrupamos en bloques de 24 para comparar
        blocks  = [hist.iloc[i:i+24].sum() for i in range(0, len(hist)-24, 24)]
        if not blocks:
            return 0.0
        mean = np.mean(blocks)
        std  = np.std(blocks)
        return float((current - mean) / std) if std > 0 else 0.0
    except Exception:
        return 0.0


# ===================================================================
# 5. VOLUME CLOCK (acumulación en horario de ballena)
# ===================================================================

def volume_clock_bias(df: pd.DataFrame, window: int = VOLUME_CLOCK_WIN) -> float:
    """
    Compara volumen en las últimas 8 velas vs las 8 anteriores.
    >0 = aceleración reciente, <0 = desaceleración.
    """
    try:
        vol = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        if len(vol) < 16:
            return 0.0
        recent = vol.iloc[-8:].sum()
        prev   = vol.iloc[-16:-8].sum()
        return float((recent - prev) / prev) if prev > 0 else 0.0
    except Exception:
        return 0.0


# ===================================================================
# 6. WHALE SCORE — Función principal de entrada
# ===================================================================

def whale_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula el score de actividad de ballena sobre un DataFrame de klines.

    Args:
        df: DataFrame con columnas open, high, low, close, volume
            (y opcionalmente taker_buy_vol)

    Returns:
        {
          'score':      int,         # score total (0-300+)
          'direction':  str,         # 'LONG' | 'SHORT' | 'NEUTRAL'
          'reasons':    list[str],   # señales activas
          'confidence': str,         # 'ULTRA'|'HIGH'|'MEDIUM'|'LOW'|'NONE'
          'cvd_slope':  float,
          'absorption': int,         # número de velas de absorción
        }
    """
    score   = 0
    reasons = []

    # ─── Validación básica ───────────────────────────────────────
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in required) or len(df) < 96:
        return {'score': 0, 'direction': 'NEUTRAL', 'reasons': [],
                'confidence': 'NONE', 'cvd_slope': 0.0, 'absorption': 0}

    # ─── Señal 1: CVD slope ──────────────────────────────────────
    cv_slope = cvd_slope(df)
    if cv_slope > 0.05:
        score += 50
        reasons.append(f"W_CVD_ACCUM(+{cv_slope:.3f})")
    elif cv_slope > 0.02:
        score += 30
        reasons.append(f"W_CVD_RISING(+{cv_slope:.3f})")
    elif cv_slope < -0.05:
        score += 50
        reasons.append(f"W_CVD_DISTRIB({cv_slope:.3f})")
    elif cv_slope < -0.02:
        score += 30
        reasons.append(f"W_CVD_FALLING({cv_slope:.3f})")

    # ─── Señal 2: Absorción ──────────────────────────────────────
    _, abs_count = absorption_score(df)
    if abs_count >= 5:
        score += 60
        reasons.append(f"W_ABSORPTION_STRONG({abs_count}v)")
    elif abs_count >= 3:
        score += 40
        reasons.append(f"W_ABSORPTION({abs_count}v)")
    elif abs_count >= 1:
        score += 20
        reasons.append(f"W_ABSORPTION_WEAK({abs_count}v)")

    # ─── Señal 3: Liquidity sweeps ───────────────────────────────
    sweeps = liquidity_sweep(df)
    bull_sweeps = [e for e in sweeps if e['type'] == 'BULLISH']
    bear_sweeps = [e for e in sweeps if e['type'] == 'BEARISH']

    if len(bull_sweeps) >= 3:
        score += 50
        top = max(bull_sweeps, key=lambda e: e['wick_ratio'])
        reasons.append(f"W_BULL_SWEEP_X{len(bull_sweeps)}(ratio={top['wick_ratio']:.1f})")
    elif len(bull_sweeps) >= 1:
        score += 25
        reasons.append(f"W_BULL_SWEEP_X{len(bull_sweeps)}")

    if len(bear_sweeps) >= 3:
        score += 50
        top = max(bear_sweeps, key=lambda e: e['wick_ratio'])
        reasons.append(f"W_BEAR_SWEEP_X{len(bear_sweeps)}(ratio={top['wick_ratio']:.1f})")
    elif len(bear_sweeps) >= 1:
        score += 25
        reasons.append(f"W_BEAR_SWEEP_X{len(bear_sweeps)}")

    # ─── Señal 4: Volume z-score ─────────────────────────────────
    vol_z = volume_zscore(df)
    if vol_z >= 3.0:
        score += 40
        reasons.append(f"W_VOL_EXTREME(z={vol_z:.1f})")
    elif vol_z >= 2.0:
        score += 25
        reasons.append(f"W_VOL_SPIKE(z={vol_z:.1f})")
    elif vol_z >= 1.5:
        score += 10
        reasons.append(f"W_VOL_ELEVATED(z={vol_z:.1f})")

    # ─── Señal 5: Volume clock bias ──────────────────────────────
    vc_bias = volume_clock_bias(df)
    if vc_bias > 0.50:
        score += 20
        reasons.append(f"W_VOL_CLOCK_ACCEL({vc_bias:+.0%})")
    elif vc_bias < -0.30:
        score += 10
        reasons.append(f"W_VOL_CLOCK_DECAY({vc_bias:+.0%})")

    # ─── Bonus de convergencia ───────────────────────────────────
    # Absorción + CVD acumulador + sweep alcista = acumulación clásica
    if abs_count >= 2 and cv_slope > 0.02 and len(bull_sweeps) >= 1:
        score += 40
        reasons.append("W_CONVERGENCE_BULL")
    elif abs_count >= 2 and cv_slope < -0.02 and len(bear_sweeps) >= 1:
        score += 40
        reasons.append("W_CONVERGENCE_BEAR")

    # ─── Dirección ───────────────────────────────────────────────
    bull_pts = (
        (50 if cv_slope > 0.05 else 30 if cv_slope > 0.02 else 0) +
        len(bull_sweeps) * 15 +
        (20 if vc_bias > 0.50 else 0)
    )
    bear_pts = (
        (50 if cv_slope < -0.05 else 30 if cv_slope < -0.02 else 0) +
        len(bear_sweeps) * 15
    )

    if bull_pts > bear_pts + 10:
        direction = 'LONG'
    elif bear_pts > bull_pts + 10:
        direction = 'SHORT'
    else:
        direction = 'NEUTRAL'

    # ─── Confianza ────────────────────────────────────────────────
    if score >= 250:
        confidence = 'ULTRA'
    elif score >= 160:
        confidence = 'HIGH'
    elif score >= 90:
        confidence = 'MEDIUM'
    elif score >= 40:
        confidence = 'LOW'
    else:
        confidence = 'NONE'

    return {
        'score':      score,
        'direction':  direction,
        'reasons':    reasons,
        'confidence': confidence,
        'cvd_slope':  round(cv_slope, 4),
        'absorption': abs_count,
    }
