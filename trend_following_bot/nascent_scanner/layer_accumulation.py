"""
Layer 4: ACCUMULATION — V6: Capped OBV, requires 2+ positive signals.
"""
from . import scanner_math as math
import numpy as np
import pandas as pd


def score(df):
    close = df['close']
    last = close.iloc[-1]
    score_val = 0
    reasons = []
    signal_count = 0  # V6: Track number of positive signals

    # 1. OBV DIVERGENCE (V6: CAPPED at 80, was 150)
    slope = math.obv_slope(df, window=48)
    price_change = (last - close.iloc[-48]) / close.iloc[-48] * 100

    if slope > 0.03 and price_change < 5.0:
        obv_score = min(80, int(slope * 300))
        if price_change < 0:
            obv_score = int(obv_score * 1.2)
        score_val += obv_score
        signal_count += 1
        reasons.append(f"A_OBV_DIV(obv:{slope:.2f}, px:{price_change:.1f}%)")

    # 2. TAKER BUY RATIO
    if 'taker_buy_vol' in df.columns:
        recent_buy = df['taker_buy_vol'].iloc[-24:].sum()
        recent_total = df['volume'].iloc[-24:].sum()
        if recent_total > 0:
            buy_ratio = recent_buy / recent_total
            if buy_ratio > 0.60:
                score_val += 60
                signal_count += 1
                reasons.append(f"A_BUYER_DOM({buy_ratio:.0%})")
            elif buy_ratio > 0.55:
                score_val += 30
                signal_count += 1
                reasons.append(f"A_BUYER_EDGE({buy_ratio:.0%})")

    # 3. VOLUME STRUCTURE
    vol_recent = df['volume'].iloc[-96:].mean()
    vol_7d = df['volume'].mean()
    vol_ratio = vol_recent / vol_7d if vol_7d > 0 else 1.0
    if vol_ratio > 1.5:
        score_val += 40
        signal_count += 1
        reasons.append(f"A_VOL_SURGE({vol_ratio:.1f}x)")
    elif vol_ratio > 1.2:
        score_val += 20
        signal_count += 1
        reasons.append(f"A_VOL_RISING({vol_ratio:.1f}x)")

    # 4. RANGE CONTRACTION
    if len(df) >= 192:
        range_recent = (df['high'].iloc[-96:].max() - df['low'].iloc[-96:].min()) / last * 100
        range_past = (df['high'].iloc[-192:-96].max() - df['low'].iloc[-192:-96].min()) / last * 100
        if range_past > 0:
            range_ratio = range_recent / range_past
            if range_ratio < 0.6 and vol_ratio >= 1.0:
                score_val += 30
                signal_count += 1
                reasons.append(f"A_RANGE_SQ({range_ratio:.2f})")

    # 5. HIGHER LOWS
    is_hl, strength = math.higher_lows(df, window=96, segments=4)
    if is_hl and strength > 0.3:
        score_val += 50
        signal_count += 1
        reasons.append(f"A_HIGHER_LOWS({strength:.1f}%)")

    # 6. KL Divergence
    kl = math.kl_divergence_volume(df, window_recent=48, window_hist=192)
    if kl > 0.5:
        score_val += 45
        signal_count += 1
        reasons.append(f"A_KL_UNUSUAL({kl:.2f})")
    elif kl > 0.2:
        score_val += 20
        signal_count += 1
        reasons.append(f"A_KL_SHIFT({kl:.2f})")

    # 7. Mann-Kendall on OBV
    change = df['close'].diff()
    direction = np.where(change > 0, 1, -1)
    direction[0] = 0
    obv_series = pd.Series((direction * df['volume'].values).cumsum())
    s_stat, p_val = math.mann_kendall(obv_series, window=96)
    if s_stat > 0 and p_val < 0.05:
        score_val += 35
        signal_count += 1
        reasons.append(f"A_MK_OBV(p={p_val:.3f})")

    # === V6 GATE: Must have at least 2 positive signals ===
    # Single-signal accumulation is too noisy
    if signal_count < 2:
        return {'score': 0, 'reasons': []}

    return {'score': score_val, 'reasons': reasons}
