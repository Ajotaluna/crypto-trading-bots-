"""
Layer 3: MOMENTUM — V4: Added Fisher Transform for precision RSI.
"""
from . import scanner_math as math


def score(df):
    close = df['close']
    last = close.iloc[-1]
    score_val = 0
    reasons = []

    # Pre-calc
    rsi_series = math.rsi(close)
    cur_rsi = rsi_series.iloc[-1]
    rsi_vel = cur_rsi - rsi_series.iloc[-4]
    vol_z = math.z_score(df['volume'], window=96).iloc[-1]
    slope = math.obv_slope(df)
    is_green = last > df['open'].iloc[-1]

    ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

    # V2 GATE: Must have minimum volume
    if vol_z < 1.0:
        return {'score': 0, 'reasons': []}

    # === V4: FISHER TRANSFORM RSI (replaces raw RSI thresholds) ===
    fisher = math.fisher_transform_rsi(close, period=14)

    # Fisher > 1.5 = strong bullish ignition (sharper than RSI > 65)
    # Fisher > 1.0 = moderate bullish
    if is_green:
        if fisher > 1.5 and rsi_vel > 10:
            score_val += 90
            reasons.append(f"M_FISHER_TURBO({fisher:.2f}, +{rsi_vel:.1f})")
        elif fisher > 1.0 and rsi_vel > 5:
            score_val += 50
            reasons.append(f"M_FISHER_IGNITION({fisher:.2f}, +{rsi_vel:.1f})")
        elif fisher > 0.5 and rsi_vel > 15:
            # Classic RSI turbo (Fisher less extreme but velocity high)
            score_val += 70
            reasons.append(f"M_RSI_TURBO(F={fisher:.2f}, +{rsi_vel:.1f})")

    # 2. Volume Surge
    if vol_z > 3.0:
        score_val += 60
        reasons.append(f"M_VOL_NOVA({vol_z:.1f}σ)")
    elif vol_z > 2.0:
        score_val += 30
        reasons.append(f"M_VOL_HIGH({vol_z:.1f}σ)")

    # 3. OBV Confirmation
    if score_val > 0 and slope > 0.03:
        score_val += 25
        reasons.append(f"M_OBV_CONFIRM({slope:.2f})")

    # 4. Green Candle Bonus
    if score_val > 0 and is_green:
        score_val += 15
        reasons.append("M_GREEN")

    # Penalty: Red candle
    if not is_green and score_val > 0:
        score_val = int(score_val * 0.3)
        reasons.append("M_RED_KILL")

    return {'score': score_val, 'reasons': reasons}
