"""
Layer 1: TREND — V9 (restored). Clean scoring.
"""
from . import scanner_math as math


def score(df):
    close = df['close']
    last = close.iloc[-1]

    ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]

    if last < ema_21:
        return {'score': 0, 'reasons': []}

    score_val = 0
    reasons = []

    # 1. Hurst
    h = math.hurst_exponent(close.values[-100:])
    if h > 1.2:
        score_val += 20
        reasons.append(f"T_PARABOLA({h:.2f})⚠️")
    elif h > 1.0:
        score_val += 70
        reasons.append(f"T_HURST_GOD({h:.2f})")
    elif h > 0.75:
        score_val += 50
        reasons.append(f"T_HURST_STRONG({h:.2f})")

    # 2. EMA Alignment
    if ema_21 > ema_50:
        score_val += 40
        reasons.append("T_EMA_ALIGN")

    # 3. OBV (3-tier)
    slope = math.obv_slope(df)
    if slope > 0.3:
        score_val += 70
        reasons.append(f"T_OBV_MASSIVE({slope:.2f})")
    elif slope > 0.1:
        score_val += 50
        reasons.append(f"T_OBV_STRONG({slope:.2f})")
    elif slope > 0.05:
        score_val += 30
        reasons.append(f"T_OBV_ACCUM({slope:.2f})")

    # 4. Price > EMA50
    if last > ema_50:
        score_val += 20
        reasons.append("T_ABOVE_EMA50")

    # 5. Trend Freshness
    ema_21_s = close.ewm(span=21, adjust=False).mean()
    ema_50_s = close.ewm(span=50, adjust=False).mean()
    if len(ema_21_s) > 96:
        recent = (ema_21_s.iloc[-48:] > ema_50_s.iloc[-48:])
        older = (ema_21_s.iloc[-96:-48] > ema_50_s.iloc[-96:-48])
        if recent.iloc[-1] and not older.iloc[0]:
            score_val += 30
            reasons.append("T_FRESH_CROSS")

    # 6. Higher Lows
    is_hl, strength = math.higher_lows(df, window=48)
    if is_hl and strength > 0.5:
        score_val += 35
        reasons.append(f"T_HIGHER_LOWS({strength:.1f}%)")

    # 7. Mann-Kendall
    s_stat, p_val = math.mann_kendall(close, window=96)
    if s_stat > 0:
        if p_val < 0.01:
            score_val += 40
            reasons.append(f"T_MK_SIGNIF(p={p_val:.3f})")
        elif p_val < 0.05:
            score_val += 20
            reasons.append(f"T_MK_TREND(p={p_val:.3f})")

    # 8. Autocorrelation
    ac = math.autocorrelation(close, lag=1, window=48)
    if ac > 0.15:
        score_val += 30
        reasons.append(f"T_AUTOCORR({ac:.2f})")

    return {'score': score_val, 'reasons': reasons}
