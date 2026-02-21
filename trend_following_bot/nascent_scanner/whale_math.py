"""
Whale Math — Estadísticas puras para rastreo de ballenas institucionales.
Stateless functions, sin side-effects. Usable desde cualquier capa.

Conceptos clave:
- CVD (Cumulative Volume Delta): diferencia acumulada entre volumen comprador y vendedor
- Absorption: velas de alto volumen con rango pequeño → ballenas absorbiendo la oferta/demanda
- Iceberg: órdenes grandes ejecutadas en porciones para no mover el precio
- Liquidity Sweep: movimiento brusco que barre stops y luego revierte
- Stop Hunt: ruptura falsa que elimina órdenes de stop-loss retail antes del movimiento real
"""

import numpy as np
import pandas as pd


# ============================================================
# CVD — CUMULATIVE VOLUME DELTA
# ============================================================

def cvd(df, window=96):
    """
    Cumulative Volume Delta (CVD).

    Aproximación con taker data:
      Delta = taker_buy_vol - taker_sell_vol
      CVD   = cumsum(Delta)

    Si no hay taker data, usa una aproximación por candles:
      Delta_approx = volume * (close - open) / (high - low + 1e-10)

    Returns: pd.Series con el CVD del último `window` candles.
    """
    data = df.iloc[-window:].copy()

    if 'taker_buy_vol' in data.columns:
        sell_vol = data['volume'] - data['taker_buy_vol']
        delta = data['taker_buy_vol'] - sell_vol
    else:
        # Aproximación: % del rango capturado por el cuerpo
        body = data['close'] - data['open']
        candle_range = (data['high'] - data['low']).replace(0, 1e-10)
        delta = data['volume'] * (body / candle_range)

    return delta.cumsum().reset_index(drop=True)


def cvd_slope(df, window=96, slope_window=48):
    """
    Pendiente normalizada del CVD en `slope_window` candles recientes.
    Positiva = presión compradora neta creciente.
    Returns: float
    """
    try:
        cvd_series = cvd(df, window=window)
        if len(cvd_series) < slope_window:
            return 0.0
        y = cvd_series.values[-slope_window:]
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        vol_mean = df['volume'].iloc[-slope_window:].mean()
        return slope / vol_mean if vol_mean > 0 else 0.0
    except Exception:
        return 0.0


def cvd_divergence(df, window=96):
    """
    Detecta divergencia entre CVD y precio.

    Divergencia alcista: precio baja pero CVD sube → acumulación oculta.
    Divergencia bajista: precio sube pero CVD baja → distribución oculta.

    Returns: (divergence_type, strength)
    - divergence_type: 'BULLISH', 'BEARISH', or None
    - strength: magnitud de la divergencia (0.0 a 1.0+)
    """
    try:
        data = df.iloc[-window:].copy()
        cvd_series = cvd(df, window=window)

        # Dividir en dos mitades
        mid = window // 2
        price_change = (data['close'].iloc[-1] - data['close'].iloc[mid]) / (data['close'].iloc[mid] + 1e-10)
        cvd_change = (cvd_series.iloc[-1] - cvd_series.iloc[mid]) / (abs(cvd_series.iloc[mid]) + 1e-10)

        # Divergencia alcista: precio cae, CVD sube
        if price_change < -0.005 and cvd_change > 0.1:
            strength = abs(cvd_change) + abs(price_change)
            return 'BULLISH', strength

        # Divergencia bajista: precio sube, CVD cae
        if price_change > 0.005 and cvd_change < -0.1:
            strength = abs(cvd_change) + abs(price_change)
            return 'BEARISH', strength

        return None, 0.0
    except Exception:
        return None, 0.0


# ============================================================
# ABSORPTION
# ============================================================

def absorption_score(df, window=48, vol_threshold=2.0, range_threshold=0.4):
    """
    Detecta velas de absorción: alto volumen con rango pequeño.

    Las ballenas absorben la oferta/demanda sin mover el precio,
    acumulando posición silenciosamente.

    Args:
        vol_threshold: múltiplo de la media de volumen para considerar "alto volumen"
        range_threshold: fracción del rango promedio para considerar "rango pequeño"

    Returns: (score 0.0-1.0, count de velas de absorción)
    """
    try:
        data = df.iloc[-window:].copy()
        vol_mean = data['volume'].mean()
        
        if vol_mean == 0:
            return 0.0, 0

        # Rango normalizado por precio (% rango de cada vela)
        candle_range = (data['high'] - data['low']) / (data['close'].replace(0, 1e-10)) * 100
        range_mean = candle_range.mean()

        # Criterios de absorción
        high_vol = data['volume'] > (vol_mean * vol_threshold)
        small_range = candle_range < (range_mean * range_threshold)
        is_absorption = high_vol & small_range

        count = int(is_absorption.sum())
        score = min(1.0, count / max(1, window * 0.05))  # Normar sobre 5% de velas
        return score, count
    except Exception:
        return 0.0, 0


def net_absorption_direction(df, window=48):
    """
    Determina la DIRECCIÓN de la absorción (¿quién está absorbiendo?).

    Si el cuerpo de las velas de absorción es positivo (cierre > apertura) → alcista.
    Si el cuerpo es negativo → bajista.

    Returns: ('BULLISH' | 'BEARISH' | 'NEUTRAL', strength_pct_bias)
    """
    try:
        data = df.iloc[-window:].copy()
        vol_mean = data['volume'].mean()
        if vol_mean == 0:
            return 'NEUTRAL', 0.0

        candle_range = (data['high'] - data['low']) / (data['close'].replace(0, 1e-10)) * 100
        range_mean = candle_range.mean()

        high_vol = data['volume'] > (vol_mean * 2.0)
        small_range = candle_range < (range_mean * 0.4)
        absorption_candles = data[high_vol & small_range]

        if len(absorption_candles) == 0:
            return 'NEUTRAL', 0.0

        bull_count = (absorption_candles['close'] > absorption_candles['open']).sum()
        bias = (bull_count / len(absorption_candles) - 0.5) * 2  # -1 a +1

        if bias > 0.3:
            return 'BULLISH', abs(bias)
        elif bias < -0.3:
            return 'BEARISH', abs(bias)
        return 'NEUTRAL', abs(bias)
    except Exception:
        return 'NEUTRAL', 0.0


# ============================================================
# ICEBERG ORDERS
# ============================================================

def iceberg_detection(df, window=48, price_stability=0.3, vol_growth=1.3):
    """
    Detecta órdenes iceberg institucionales.

    Patrón: precio lateralizado (estable) con volumen creciente en el tiempo.
    Esto indica que una ballena está ejecutando una orden grande en porciones,
    absorbiendo sin mover el precio.

    Args:
        price_stability: máximo movimiento de precio como % del rango para ser "estable"
        vol_growth: ratio de crecimiento del volumen en la segunda mitad vs primera

    Returns: (is_iceberg: bool, confidence: float 0.0-1.0)
    """
    try:
        data = df.iloc[-window:].copy()
        close = data['close']
        volume = data['volume']

        # 1. Precio estable: rango del período < price_stability * ATR
        price_range_pct = (close.max() - close.min()) / (close.mean() + 1e-10) * 100
        is_stable = price_range_pct < price_stability * 2  # <0.6% para un período de 12h

        # 2. Volumen creciendo durante la lateralización
        mid = window // 2
        vol_first = volume.iloc[:mid].mean()
        vol_second = volume.iloc[mid:].mean()
        vol_is_growing = (vol_second / (vol_first + 1e-10)) > vol_growth

        # 3. OBV con pendiente significativa durante precio plano
        from . import scanner_math as math
        obv_sl = math.obv_slope(df, window=window)
        obv_active = abs(obv_sl) > 0.05

        confidence_components = [is_stable, vol_is_growing, obv_active]
        confidence = sum(confidence_components) / len(confidence_components)
        is_iceberg = is_stable and vol_is_growing

        return is_iceberg, confidence
    except Exception:
        return False, 0.0


# ============================================================
# LIQUIDITY SWEEP + STOP HUNT
# ============================================================

def liquidity_sweep(df, window=48, wick_body_ratio=2.5):
    """
    Detecta barridas de liquidez (Liquidity Sweeps).

    Patrón: wick extremo (>2.5x el tamaño del cuerpo) que rompe un nivel
    y luego revierte. Las ballenas barren los stops del retail antes de
    ir en la dirección real.

    Returns: list of dicts con cada evento detectado:
      {'type': 'BULLISH'|'BEARISH', 'idx': int, 'wick_ratio': float}
    """
    try:
        data = df.iloc[-window:].copy().reset_index(drop=True)
        events = []

        for i in range(len(data)):
            candle = data.iloc[i]
            body = abs(candle['close'] - candle['open'])
            if body < 1e-10:
                continue

            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']

            # Bullish sweep (wick inferior largo + cierre alcista)
            if lower_wick > body * wick_body_ratio and candle['close'] > candle['open']:
                events.append({'type': 'BULLISH', 'idx': i, 'wick_ratio': lower_wick / body})

            # Bearish sweep (wick superior largo + cierre bajista)
            if upper_wick > body * wick_body_ratio and candle['close'] < candle['open']:
                events.append({'type': 'BEARISH', 'idx': i, 'wick_ratio': upper_wick / body})

        return events
    except Exception:
        return []


def stop_hunt_pattern(df, window=48, lookback=20):
    """
    Detecta patrones de stop-hunt.

    El precio hace un nuevo mínimo/máximo de los últimos `lookback` candles
    y luego revierte dentro del mismo candle o en el siguiente.
    Las ballenas activan los stops del retail para obtener liquidez.

    Returns: (bullish_hunts: int, bearish_hunts: int, last_hunt_type: str or None)
    """
    try:
        data = df.iloc[-window:].copy().reset_index(drop=True)
        bullish_hunts = 0
        bearish_hunts = 0
        last_hunt_type = None

        for i in range(lookback, len(data)):
            prev = data.iloc[i - lookback:i]
            candle = data.iloc[i]

            prev_low = prev['low'].min()
            prev_high = prev['high'].max()

            # Bullish stop-hunt: nuevo mínimo pero cierra arriba del prev_low
            if candle['low'] < prev_low and candle['close'] > prev_low:
                bullish_hunts += 1
                last_hunt_type = 'BULLISH'

            # Bearish stop-hunt: nuevo máximo pero cierra debajo del prev_high
            if candle['high'] > prev_high and candle['close'] < prev_high:
                bearish_hunts += 1
                last_hunt_type = 'BEARISH'

        return bullish_hunts, bearish_hunts, last_hunt_type
    except Exception:
        return 0, 0, None


def fake_breakout_score(df, window=48, lookback=24):
    """
    Detecta fake breakouts (trampas para retail).

    Un fake breakout ocurre cuando:
    1. El precio cierra SOBRE la resistencia (máximo de lookback candles)
    2. En el siguiente candle(s) vuelve POR DEBAJO de la resistencia

    Esto es distribución institucional: las ballenas venden a los retail
    que compran el "breakout".

    Returns: (count_bearish_fakes, count_bullish_fakes, latest_was_fake: bool)
    """
    try:
        data = df.iloc[-window:].copy().reset_index(drop=True)
        bearish_fakes = 0
        bullish_fakes = 0

        for i in range(lookback + 1, len(data) - 1):
            prev_window = data.iloc[i - lookback:i]
            resistance = prev_window['high'].max()
            support = prev_window['low'].min()

            candle = data.iloc[i]
            next_candle = data.iloc[i + 1]

            # Bearish fake breakout: cierra sobre resistencia, siguiente baja
            if candle['close'] > resistance and next_candle['close'] < resistance:
                bearish_fakes += 1

            # Bullish fake breakout (fake breakdown): cierra bajo soporte, siguiente sube
            if candle['close'] < support and next_candle['close'] > support:
                bullish_fakes += 1

        latest_candle = data.iloc[-1]
        latest_prev = data.iloc[-lookback - 1:-1]
        latest_was_fake = (
            latest_candle['close'] > latest_prev['high'].max() or
            latest_candle['close'] < latest_prev['low'].min()
        )

        return bearish_fakes, bullish_fakes, latest_was_fake
    except Exception:
        return 0, 0, False


# ============================================================
# VOLUME CLOCK — ANÁLISIS TEMPORAL
# ============================================================

def volume_clock(df, window=96):
    """
    Analiza la concentración temporal del volumen en 15m-candles.

    Las ballenas operan en ventanas horarias específicas:
    - Asian Close / London Open (04:00–06:00 UTC): acumulación silenciosa
    - NY Open (13:30–15:30 UTC): movimientos agresivos
    - NY Close (21:00–22:00 UTC): toma de ganancias / distribución

    Calcula el ratio de volumen en esas ventanas vs el resto.
    Requiere columna 'timestamp' en el DataFrame.

    Returns: (ny_ratio: float, asian_ratio: float, has_clock_signal: bool)
    """
    try:
        data = df.iloc[-window:].copy()
        if 'timestamp' not in data.columns:
            return 0.0, 0.0, False

        ts = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        hours_utc = ts.dt.hour + ts.dt.minute / 60

        total_vol = data['volume'].sum()
        if total_vol == 0:
            return 0.0, 0.0, False

        # NY Open window: 13:30 – 15:30 UTC
        ny_mask = (hours_utc >= 13.5) & (hours_utc <= 15.5)
        ny_vol = data.loc[ny_mask.values, 'volume'].sum()

        # Asian / London crossover: 04:00 – 07:00 UTC
        asian_mask = (hours_utc >= 4.0) & (hours_utc <= 7.0)
        asian_vol = data.loc[asian_mask.values, 'volume'].sum()

        ny_pct = ny_vol / total_vol
        asian_pct = asian_vol / total_vol

        # Expected % is proportional to time window / 24h
        ny_expected = 2.0 / 24   # 2h / 24h = 8.3%
        asian_expected = 3.0 / 24  # 3h / 24h = 12.5%

        ny_ratio = ny_pct / (ny_expected + 1e-10)
        asian_ratio = asian_pct / (asian_expected + 1e-10)

        # Signal: any session has 2x its expected share
        has_signal = ny_ratio > 2.0 or asian_ratio > 2.0

        return ny_ratio, asian_ratio, has_signal
    except Exception:
        return 0.0, 0.0, False


# ============================================================
# SMART MONEY INDEX (SMI)
# ============================================================

def smart_money_index(df, window=96):
    """
    Smart Money Index (SMI) — adaptado para crypto 15m candles.

    Concepto original: comparar el volumen de las primeras 30 min del día
    (retail: reaccionan a noticias) vs las últimas 30 min (smart money: executan).

    Adaptación cripto:
    - "First wave" = primeras 4 velas del día (apertura UTC 00:00)
    - "Last wave"  = últimas 4 velas del día (cierre UTC 23:00)

    Un SMI creciente mientras precio baja = smart money comprando al fondo.
    Un SMI cayendo mientras precio sube = smart money vendiendo en el techo.

    Returns: (smi_trend: float, smi_vs_price_divergence: bool)
    """
    try:
        data = df.iloc[-window:].copy()
        if 'timestamp' not in data.columns:
            return 0.0, False

        ts = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        hours_utc = ts.dt.hour

        # First wave: 00:00–01:00 UTC (4 candles de 15m)
        first_mask = hours_utc == 0
        first_wave_vol = data.loc[first_mask.values, 'volume'].reset_index(drop=True)

        # Last wave: 22:00–23:00 UTC
        last_mask = hours_utc == 22
        last_wave_vol = data.loc[last_mask.values, 'volume'].reset_index(drop=True)

        if len(first_wave_vol) == 0 or len(last_wave_vol) == 0:
            return 0.0, False

        # SMI = last_wave / first_wave ratio across days
        n_pairs = min(len(first_wave_vol), len(last_wave_vol))
        smi_ratios = []
        for i in range(n_pairs):
            f = float(first_wave_vol.iloc[i])
            l = float(last_wave_vol.iloc[i])
            if f > 0:
                smi_ratios.append(l / f)

        if not smi_ratios:
            return 0.0, False

        smi_series = pd.Series(smi_ratios)
        # Trend: positive = smart money vol growing relative to dumb money
        if len(smi_series) < 2:
            return 0.0, False

        smi_slope = smi_series.iloc[-1] - smi_series.iloc[0]

        # Divergence: price direction vs SMI direction
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / (data['close'].iloc[0] + 1e-10)
        divergence = (smi_slope > 0 and price_change < -0.005) or (smi_slope < 0 and price_change > 0.005)

        return smi_slope, divergence
    except Exception:
        return 0.0, False


# ============================================================
# LARGE TRADE DETECTOR
# ============================================================

def large_trade_ratio(df, window=96, threshold_mult=3.0):
    """
    Ratio de candles con volumen institucional (>threshold_mult × vol_mean).

    Las ballenas necesitan dividir sus órdenes, pero aún así dejan
    "huellas" en el volumen de ciertos candles.

    Returns: (ratio 0.0-1.0, count, avg_size_multiple)
    """
    try:
        vol = df['volume'].iloc[-window:]
        vol_mean = vol.mean()
        if vol_mean == 0:
            return 0.0, 0, 0.0

        threshold = vol_mean * threshold_mult
        large = vol[vol > threshold]
        count = len(large)
        ratio = count / len(vol)
        avg_mult = (large / vol_mean).mean() if len(large) > 0 else 0.0

        return ratio, count, avg_mult
    except Exception:
        return 0.0, 0, 0.0


# ============================================================
# PUMP & DUMP DETECTOR
# ============================================================

def pump_dump_score(df, window=96):
    """
    Detecta patrones de pump-and-dump / distribución institucional.

    Pump: subida rápida (>X%) en pocas velas con volumen extremo.
    Dump: posterior caída en volumen decreciente (el retail compró el techo).

    También detecta la fase de distribución:
    - Precio en máximos pero volumen ya cayendo
    - CVD bajista mientras precio se mantiene

    Returns: (pump_detected: bool, dump_phase: bool, pump_strength: float)
    """
    try:
        data = df.iloc[-window:].copy()
        close = data['close']
        volume = data['volume']

        # Dividir: primera mitad (pump phase) vs segunda mitad (dump/distribution)
        mid = window // 2
        first_half = data.iloc[:mid]
        second_half = data.iloc[mid:]

        # Pump: > 3% subida en primera mitad con volumen alto
        price_change_first = (first_half['close'].iloc[-1] - first_half['close'].iloc[0]) / (first_half['close'].iloc[0] + 1e-10) * 100
        vol_first = first_half['volume'].mean()
        vol_global_mean = volume.mean()
        pump_detected = price_change_first > 3.0 and vol_first > vol_global_mean * 1.5

        # Dump phase: segunda mitad el precio retrocede con volumen decayendo
        price_change_second = (second_half['close'].iloc[-1] - second_half['close'].iloc[0]) / (second_half['close'].iloc[0] + 1e-10) * 100
        vol_second = second_half['volume'].mean()
        dump_phase = price_change_second < -1.0 and vol_second < vol_first * 0.8

        pump_strength = abs(price_change_first) * (vol_first / (vol_global_mean + 1e-10))

        return pump_detected, dump_phase, pump_strength
    except Exception:
        return False, False, 0.0
