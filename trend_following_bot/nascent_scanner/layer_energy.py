"""
Layer 2: ENERGY — V14: Birth signals + Funding Rate (nascent positioning).
Base = V9 (proven 6/20). No MULTI bonus applied (via engine).
"""
from . import scanner_math as math


def score(df, funding_df=None):
    close = df['close']
    score_val = 0
    reasons = []

    # Gate: Must be in squeeze
    is_squeezed = math.keltner_squeeze(df)
    if not is_squeezed:
        return {'score': 0, 'reasons': []}

    score_val += 100
    reasons.append("E_SQUEEZE_ON")

    # 2. Volume Calm
    vol_z = math.z_score(df['volume'], window=96).iloc[-1]
    if vol_z < 0.0:
        score_val += 60
        reasons.append(f"E_DEAD_CALM({vol_z:.2f}σ)")
    elif vol_z < 0.5:
        score_val += 30
        reasons.append(f"E_CALM({vol_z:.2f}σ)")

    # 3. ATR Contracting
    atr_ratio = math.atr_trend(df)
    if atr_ratio < 0.8:
        score_val += 30
        reasons.append(f"E_ATR_SHRINK({atr_ratio:.2f})")

    # 4. Structured
    h = math.hurst_exponent(close.values[-100:])
    if h > 0.6:
        score_val += 20
        reasons.append(f"E_STRUCTURED({h:.2f})")

    # 5. OBV during squeeze
    slope = math.obv_slope(df)
    if slope > 0.1:
        score_val += 25
        reasons.append(f"E_HIDDEN_LOAD({slope:.2f})")
    elif slope > 0.03:
        score_val += 12
        reasons.append(f"E_SOFT_LOAD({slope:.2f})")

    # 6. Volume drying
    vol_recent = df['volume'].iloc[-48:].mean()
    vol_past = df['volume'].iloc[-96:-48].mean()
    if vol_past > 0:
        vol_ratio = vol_recent / vol_past
        if vol_ratio < 0.7:
            score_val += 15
            reasons.append(f"E_VOL_DRYING({vol_ratio:.2f})")

    # 7. Price above EMA21
    ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    if close.iloc[-1] > ema_21:
        score_val += 10
        reasons.append("E_ABOVE_EMA21")

    # 8. Variance Ratio (bonus only)
    vr, deviation = math.variance_ratio(close, lag=5, window=96)
    if vr > 1.3:
        score_val += 35
        reasons.append(f"E_VR_MOMENTUM(VR={vr:.2f})")

    # 9. Higher Lows (bonus only)
    is_hl, strength = math.higher_lows(df, window=48)
    if is_hl and strength > 5.0:
        score_val = int(score_val * 1.4)
        reasons.append(f"E_HL_STRONG({strength:.1f}%)↑↑")
    elif is_hl and strength > 1.0:
        score_val = int(score_val * 1.15)
        reasons.append(f"E_HL_MILD({strength:.1f}%)↑")

    # === V13: BIRTH SIGNALS ===

    # 10. Taker Buy Trend
    buy_slope, buy_p = math.taker_buy_trend(df, window=96, segments=8)
    if buy_slope > 0.02 and buy_p < 0.1:
        score_val += 45
        reasons.append(f"E_BIRTH_BUYERS(+{buy_slope:.1%})")
    elif buy_slope > 0.01:
        score_val += 20
        reasons.append(f"E_BIRTH_BUY_SHIFT(+{buy_slope:.1%})")

    # 11. Squeeze Tightening Rate
    accel, tightness = math.squeeze_tightness_rate(df, window=48)
    if accel < -0.10 and tightness < 0.85:
        score_val += 40
        reasons.append(f"E_BIRTH_TIGHTENING({accel:.2f}, t={tightness:.2f})")
    elif accel < -0.05:
        score_val += 20
        reasons.append(f"E_BIRTH_COMPRESSING({accel:.2f})")

    # === V14: FUNDING RATE (nascent positioning) ===
    # 12. Funding rate transitioning neg→pos = longs building quietly
    if funding_df is not None:
        # Get the last timestamp of history to filter funding data
        history_end_ts = None
        if 'timestamp' in df.columns:
            try:
                history_end_ts = int(df['timestamp'].iloc[-1])
            except Exception:
                pass

        trend, avg_rate, is_transitioning = math.funding_rate_trend(
            funding_df, history_end_ts
        )

        if is_transitioning:
            score_val += 50
            reasons.append(f"E_BIRTH_FUNDING_FLIP(avg={avg_rate:.4%})")
        elif trend > 0.0001:
            score_val += 25
            reasons.append(f"E_FUNDING_RISING(Δ={trend:.4%})")

    return {'score': score_val, 'reasons': reasons}
