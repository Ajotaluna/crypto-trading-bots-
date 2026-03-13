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

    # Con contexto on-chain (opcional — activa señales extra):
    result = whale_score(df, context={
        'oi_history':   [...],   # output de market_data.get_open_interest()
        'funding_list': [...],   # output de market_data.get_funding_rate()
        'ls_data':      [...],   # output de market_data.get_long_short_ratio()
        'ob_bids':      [...],   # ob_streamer.books[sym]['bids']
        'ob_asks':      [...],   # ob_streamer.books[sym]['asks']
        'agg_trades':   [...],   # output de market_data.get_agg_trades()
    })
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

# ===================================================================
# CONSTANTES
# ===================================================================

CVD_WINDOW       = 96   # Candles para calcular CVD (= 24h en 15m)
CVD_SLOPE_WIN    = 48   # Ventana para la pendiente del CVD
ABSORB_WINDOW    = 48   # Ventana para absorción
SWEEP_WINDOW     = 48   # Ventana para liquidity sweeps
VOL_ZSCORE_WIN   = 96   # Ventana para z-score de volumen
VOLUME_CLOCK_WIN = 32   # Ventana para análisis de volume clock

# Umbrales ajustados para el nuevo score máximo (~590 pts)
# SCORE >= 420 → ULTRA  |  >= 280 → HIGH  |  >= 150 → MEDIUM  |  >= 60 → LOW
CONFIDENCE_ULTRA   = 420
CONFIDENCE_HIGH    = 280
CONFIDENCE_MEDIUM  = 150
CONFIDENCE_LOW     =  60


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
# 2. CVD DIVERGENCE (Fase 3)
# ===================================================================

def cvd_divergence(df: pd.DataFrame, window: int = 96) -> Tuple[Optional[str], float]:
    """
    Detecta divergencia entre CVD y precio.

    Divergencia alcista: precio baja pero CVD sube → acumulación oculta.
    Divergencia bajista: precio sube pero CVD baja → distribución oculta.

    Returns: (divergence_type, strength)
    - divergence_type: 'BULLISH', 'BEARISH', or None
    - strength: magnitud (0.0 a 1.0+)
    """
    try:
        data = df.iloc[-window:].copy()
        cvd_series = _cvd(df, window)

        mid = window // 2
        price_change = (
            (data['close'].iloc[-1] - data['close'].iloc[mid]) /
            (data['close'].iloc[mid] + 1e-10)
        )
        cvd_change = (
            (cvd_series.iloc[-1] - cvd_series.iloc[mid]) /
            (abs(cvd_series.iloc[mid]) + 1e-10)
        )

        if price_change < -0.005 and cvd_change > 0.1:
            strength = abs(cvd_change) + abs(price_change)
            return 'BULLISH', strength

        if price_change > 0.005 and cvd_change < -0.1:
            strength = abs(cvd_change) + abs(price_change)
            return 'BEARISH', strength

        return None, 0.0
    except Exception:
        return None, 0.0


# ===================================================================
# 3. ABSORCIÓN (alto volumen + rango pequeño)
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
# 4. LIQUIDITY SWEEPS (stop hunts)
# ===================================================================

def liquidity_sweep(df: pd.DataFrame,
                    window: int = SWEEP_WINDOW,
                    wick_body_ratio: float = 2.5) -> List[Dict]:
    """
    Detecta wick extremos que indican barridos de liquidez (stop hunts).
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

            if lower_wick > body * wick_body_ratio and cl > o:
                events.append({'type': 'BULLISH', 'idx': i, 'wick_ratio': lower_wick / body})
            if upper_wick > body * wick_body_ratio and cl < o:
                events.append({'type': 'BEARISH', 'idx': i, 'wick_ratio': upper_wick / body})
    except Exception:
        pass
    return events


# ===================================================================
# 5. VOLUMEN Z-SCORE (spike detector)
# ===================================================================

def volume_zscore(df: pd.DataFrame, window: int = VOL_ZSCORE_WIN) -> float:
    """Z-score del volumen de las últimas 24 velas vs el historial. > 2.0 = spike significativo."""
    try:
        vol = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        if len(vol) < window + 24:
            return 0.0
        hist    = vol.iloc[-(window + 24):-24]
        current = vol.iloc[-24:].sum()
        blocks  = [hist.iloc[i:i+24].sum() for i in range(0, len(hist)-24, 24)]
        if not blocks:
            return 0.0
        mean = np.mean(blocks)
        std  = np.std(blocks)
        return float((current - mean) / std) if std > 0 else 0.0
    except Exception:
        return 0.0


# ===================================================================
# 6. VOLUME CLOCK BIAS
# ===================================================================

def volume_clock_bias(df: pd.DataFrame, window: int = VOLUME_CLOCK_WIN) -> float:
    """Compara volumen en las últimas 8 velas vs las 8 anteriores. >0 = aceleración reciente."""
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
# 7. TAKER BUY WHALE (Fase 3)
# ===================================================================

def taker_buy_whale_score(df: pd.DataFrame) -> Tuple[int, Optional[str]]:
    """
    Detecta acumulación agresiva cuando el volumen comprador domina
    AND el volumen total está en spike z-score > 1.5.

    Returns: (score: int, reason: str | None)
    """
    try:
        if 'taker_buy_vol' not in df.columns:
            return 0, None
        recent_buy = df['taker_buy_vol'].iloc[-48:].sum()
        recent_vol = df['volume'].iloc[-48:].sum()
        if recent_vol <= 0:
            return 0, None
        buy_ratio = recent_buy / recent_vol
        vol_z     = volume_zscore(df, window=VOL_ZSCORE_WIN)

        if buy_ratio > 0.68 and vol_z > 1.5:
            return 40, f"W_TAKER_WHALE_BUY({buy_ratio:.0%},z={vol_z:.1f})"
        elif buy_ratio > 0.62 and vol_z > 1.0:
            return 20, f"W_TAKER_BIAS({buy_ratio:.0%})"
        elif buy_ratio < 0.38 and vol_z > 1.5:
            return 40, f"W_TAKER_WHALE_SELL({1-buy_ratio:.0%},z={vol_z:.1f})"
        elif buy_ratio < 0.42 and vol_z > 1.0:
            return 20, f"W_SELLER_BIAS({1-buy_ratio:.0%})"
    except Exception:
        pass
    return 0, None


# ===================================================================
# 8. ORDER BOOK SCORE (Fase 2)
# ===================================================================

def _orderbook_imbalance(bids: list, asks: list, levels: int = 10) -> float:
    """OBI en [-1, +1]. Positivo = bids dominan."""
    try:
        bids_vol = sum(float(q) for _, q in bids[:levels])
        asks_vol = sum(float(q) for _, q in asks[:levels])
        total = bids_vol + asks_vol
        if total == 0:
            return 0.0
        return (bids_vol - asks_vol) / total
    except Exception:
        return 0.0


def _detect_large_walls(bids: list, asks: list,
                        levels: int = 20,
                        multiplier: float = 5.0) -> Dict:
    """Detecta paredes de órdenes institucionales (órdenes >= multiplier × promedio)."""
    try:
        def find_walls(side, n):
            qtys = [float(q) for _, q in side[:n]]
            if not qtys:
                return []
            avg = np.mean(qtys)
            if avg == 0:
                return []
            return [
                {'price': float(p), 'qty': float(q), 'multiple': round(float(q) / avg, 1)}
                for p, q in side[:n]
                if float(q) / avg >= multiplier
            ]

        bid_walls = find_walls(bids, levels)
        ask_walls = find_walls(asks, levels)
        return {
            'bid_walls': bid_walls,
            'ask_walls': ask_walls,
            'has_bid_wall': len(bid_walls) > 0,
            'has_ask_wall': len(ask_walls) > 0,
        }
    except Exception:
        return {'bid_walls': [], 'ask_walls': [], 'has_bid_wall': False, 'has_ask_wall': False}


def orderbook_score(bids: list, asks: list) -> Tuple[int, List[str]]:
    """
    Puntúa el estado del libro de órdenes como señal de actividad whale.

    Returns: (score: int, reasons: list[str])
    """
    score   = 0
    reasons = []

    if not bids or not asks:
        return 0, []

    # OBI
    obi = _orderbook_imbalance(bids, asks, levels=10)
    if obi > 0.40:
        score += 25
        reasons.append(f"OB_BID_PRESSURE({obi:+.2f})")
    elif obi > 0.25:
        score += 12
        reasons.append(f"OB_BID_MILD({obi:+.2f})")
    elif obi < -0.40:
        score += 25
        reasons.append(f"OB_ASK_PRESSURE({obi:+.2f})")
    elif obi < -0.25:
        score += 12
        reasons.append(f"OB_ASK_MILD({obi:+.2f})")

    # Walls
    walls = _detect_large_walls(bids, asks, levels=20, multiplier=5.0)
    if walls['has_bid_wall']:
        top = max(walls['bid_walls'], key=lambda w: w['multiple'])
        score += 20
        reasons.append(f"OB_BID_WALL({top['multiple']:.1f}x@{top['price']:.4f})")
    if walls['has_ask_wall']:
        top = max(walls['ask_walls'], key=lambda w: w['multiple'])
        score += 20
        reasons.append(f"OB_ASK_WALL({top['multiple']:.1f}x@{top['price']:.4f})")

    return score, reasons


# ===================================================================
# 9. ON-CHAIN CONTEXT SCORES (Fase 1)
# ===================================================================

def oi_delta_score(oi_history: list) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Calcula el delta de Open Interest para detectar dinero nuevo entrando/saliendo.

    Returns: (score, reason, direction_hint)
    """
    try:
        if not oi_history or len(oi_history) < 2:
            return 0, None, None

        vals = [float(x['sumOpenInterest']) for x in oi_history if 'sumOpenInterest' in x]
        if len(vals) < 2:
            return 0, None, None

        oldest = vals[0]
        newest = vals[-1]
        if oldest <= 0:
            return 0, None, None

        delta_pct = (newest - oldest) / oldest * 100

        if delta_pct >= 10.0:
            return 30, f"OI_SURGE(+{delta_pct:.1f}%)", 'LONG'
        elif delta_pct >= 5.0:
            return 18, f"OI_GROWING(+{delta_pct:.1f}%)", 'LONG'
        elif delta_pct <= -10.0:
            return 30, f"OI_COLLAPSE({delta_pct:.1f}%)", 'SHORT'
        elif delta_pct <= -5.0:
            return 18, f"OI_FALLING({delta_pct:.1f}%)", 'SHORT'
    except Exception:
        pass
    return 0, None, None


def funding_rate_score(funding_list: list) -> Tuple[int, Optional[str]]:
    """
    Detecta funding rates extremos que indican posicionamiento institucional.
    """
    try:
        if not funding_list:
            return 0, None
        latest = funding_list[-1].get('fundingRate', 0.0)

        if latest > 0.0008:
            return 20, f"FUNDING_EXTREME_BULL({latest*100:.3f}%)"
        elif latest > 0.0005:
            return 10, f"FUNDING_HIGH({latest*100:.3f}%)"
        elif latest < -0.0005:
            return 20, f"FUNDING_EXTREME_BEAR({latest*100:.3f}%)"
        elif latest < -0.0003:
            return 10, f"FUNDING_LOW({latest*100:.3f}%)"
    except Exception:
        pass
    return 0, None


def long_short_ratio_score(ls_data: list) -> Tuple[int, Optional[str]]:
    """
    Detecta extremos en el ratio de cuentas long/short (retail sentiment).
    """
    try:
        if not ls_data:
            return 0, None
        latest = ls_data[-1]
        long_acc  = float(latest.get('longAccount',  0.5))
        short_acc = float(latest.get('shortAccount', 0.5))

        if long_acc > 0.65:
            return 15, f"LS_CROWD_LONG({long_acc:.0%})"
        elif long_acc > 0.60:
            return 8, f"LS_LONG_BIAS({long_acc:.0%})"
        elif short_acc > 0.65:
            return 15, f"LS_CROWD_SHORT({short_acc:.0%})"
        elif short_acc > 0.60:
            return 8, f"LS_SHORT_BIAS({short_acc:.0%})"
    except Exception:
        pass
    return 0, None


# ===================================================================
# 10. MEGA-TRADES (Fase 4)
# ===================================================================

def mega_trade_score(trades: list, min_usd: float = 250_000) -> Tuple[int, Optional[str]]:
    """
    Detecta trades individuales de tamaño institucional (>min_usd USDT).
    """
    try:
        if not trades:
            return 0, None

        mega = [t for t in trades if float(t.get('p', 0)) * float(t.get('q', 0)) >= min_usd]
        total = len(mega)

        if total == 0:
            return 0, None

        buy_mega  = sum(1 for t in mega if not t.get('m', True))
        sell_mega = total - buy_mega
        dir_str   = f"buy={buy_mega}/sell={sell_mega}"

        if total >= 5:
            return 50, f"MEGA_CLUSTER(n={total},{dir_str})"
        elif total >= 3:
            return 35, f"MEGA_GROUP(n={total},{dir_str})"
        elif total >= 1:
            biggest = max(float(t['p']) * float(t['q']) for t in mega)
            return 18, f"MEGA_TRADE(n={total},max=${biggest/1e6:.2f}M)"
    except Exception:
        pass
    return 0, None


# ===================================================================
# 11. WHALE SCORE — Función principal de entrada
# ===================================================================

def whale_score(df: pd.DataFrame, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Calcula el score de actividad de ballena sobre un DataFrame de klines.
    """
    score   = 0
    reasons = []
    ctx     = context or {}

    # ─── Validación básica ──────────────────────────────────────
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in required) or len(df) < 96:
        return {'score': 0, 'direction': 'NEUTRAL', 'reasons': [],
                'confidence': 'NONE', 'cvd_slope': 0.0, 'absorption': 0}

    # ─── Señal 1: CVD slope ────────────────────────────────────
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

    # ─── Señal 2: Absorción ────────────────────────────────────
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

    # ─── Señal 3: Liquidity sweeps ──────────────────────────────
    sweeps      = liquidity_sweep(df)
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

    # ─── Señal 4: Volume z-score ────────────────────────────────
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

    # ─── Señal 5: Volume clock bias ─────────────────────────────
    vc_bias = volume_clock_bias(df)
    if vc_bias > 0.50:
        score += 20
        reasons.append(f"W_VOL_CLOCK_ACCEL({vc_bias:+.0%})")
    elif vc_bias < -0.30:
        score += 10
        reasons.append(f"W_VOL_CLOCK_DECAY({vc_bias:+.0%})")

    # ─── Bonus de convergencia ───────────────────────────────────
    if abs_count >= 2 and cv_slope > 0.02 and len(bull_sweeps) >= 1:
        score += 40
        reasons.append("W_CONVERGENCE_BULL")
    elif abs_count >= 2 and cv_slope < -0.02 and len(bear_sweeps) >= 1:
        score += 40
        reasons.append("W_CONVERGENCE_BEAR")

    # ─── Señal 6: CVD Divergence ────────────────────────────────
    div_type, div_strength = cvd_divergence(df, window=96)
    if div_type == 'BULLISH' and div_strength > 0.5:
        score += 50
        reasons.append(f"W_CVD_DIV_BULL_STRONG(str={div_strength:.2f})")
    elif div_type == 'BULLISH' and div_strength > 0.15:
        score += 25
        reasons.append(f"W_CVD_DIV_BULL(str={div_strength:.2f})")
    elif div_type == 'BEARISH' and div_strength > 0.5:
        score += 30
        reasons.append(f"W_CVD_DIV_BEAR(str={div_strength:.2f})")

    # ─── Señal 7: Taker Buy Whale ────────────────────────────────
    taker_pts, taker_reason = taker_buy_whale_score(df)
    if taker_pts > 0 and taker_reason:
        score += taker_pts
        reasons.append(taker_reason)

    # ─── Fase 2: Order Book Score ────────────────────────────────
    ob_bids = ctx.get('ob_bids', [])
    ob_asks = ctx.get('ob_asks', [])
    if ob_bids and ob_asks:
        ob_pts, ob_reasons = orderbook_score(ob_bids, ob_asks)
        if ob_pts > 0:
            score += ob_pts
            reasons.extend(ob_reasons)

    # ─── Fase 1: On-Chain Context ────────────────────────────────
    oi_pts, oi_reason, oi_dir = oi_delta_score(ctx.get('oi_history', []))
    if oi_pts > 0 and oi_reason:
        score += oi_pts
        reasons.append(oi_reason)

    fr_pts, fr_reason = funding_rate_score(ctx.get('funding_list', []))
    if fr_pts > 0 and fr_reason:
        score += fr_pts
        reasons.append(fr_reason)

    ls_pts, ls_reason = long_short_ratio_score(ctx.get('ls_data', []))
    if ls_pts > 0 and ls_reason:
        score += ls_pts
        reasons.append(ls_reason)

    # ─── Fase 4: Mega-Trades ─────────────────────────────────────
    mega_pts, mega_reason = mega_trade_score(ctx.get('agg_trades', []))
    if mega_pts > 0 and mega_reason:
        score += mega_pts
        reasons.append(mega_reason)

    # ─── Dirección ──────────────────────────────────────────────
    bull_pts = (
        (50 if cv_slope > 0.05 else 30 if cv_slope > 0.02 else 0) +
        len(bull_sweeps) * 15 +
        (20 if vc_bias > 0.50 else 0) +
        (50 if div_type == 'BULLISH' and div_strength > 0.5 else
         25 if div_type == 'BULLISH' else 0) +
        (taker_pts if 'BUY' in (taker_reason or '') else 0) +
        (oi_pts if oi_dir == 'LONG' else 0)
    )
    bear_pts = (
        (50 if cv_slope < -0.05 else 30 if cv_slope < -0.02 else 0) +
        len(bear_sweeps) * 15 +
        (30 if div_type == 'BEARISH' and div_strength > 0.5 else 0) +
        (taker_pts if 'SELL' in (taker_reason or '') else 0) +
        (oi_pts if oi_dir == 'SHORT' else 0)
    )

    if bull_pts > bear_pts + 10:
        direction = 'LONG'
    elif bear_pts > bull_pts + 10:
        direction = 'SHORT'
    else:
        direction = 'NEUTRAL'

    # ─── Confianza ──────────────────────────────────────────────
    if score >= CONFIDENCE_ULTRA:
        confidence = 'ULTRA'
    elif score >= CONFIDENCE_HIGH:
        confidence = 'HIGH'
    elif score >= CONFIDENCE_MEDIUM:
        confidence = 'MEDIUM'
    elif score >= CONFIDENCE_LOW:
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
