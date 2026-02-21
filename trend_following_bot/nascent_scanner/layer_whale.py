"""
Layer 4: WHALE — Detección de acumulación/distribución silenciosa institucional.

Filosofía: Las ballenas dejan huellas matemáticas en el order flow:
- El CVD diverge del precio (acumulan mientras el precio baja)
- El volumen es alto pero el rango de precio pequeño (absorción)
- El volumen crece en un precio lateralizado (iceberg)
- Los trades de tamaño institucional aparecen en clusters

Gate: Requiere ≥2 señales activas para evitar falsos positivos.
"""
from . import whale_math as wm
from . import scanner_math as math
import numpy as np


def score(df, funding_df=None):
    """
    Score una sola asset por actividad de ballenas.
    Sigue el mismo patrón que las otras capas: {'score': int, 'reasons': list}
    """
    if len(df) < 200:
        return {'score': 0, 'reasons': []}

    score_val = 0
    reasons = []
    signal_count = 0

    # ============================================================
    # SEÑAL 1: CVD DIVERGENCE — Acumulación oculta
    # ============================================================
    # CVD sube pero precio baja = ballena comprando mientras el retail vende
    div_type, div_strength = wm.cvd_divergence(df, window=96)

    if div_type == 'BULLISH' and div_strength > 0.5:
        score_val += 60
        signal_count += 1
        reasons.append(f"W_CVD_ACCUM(str={div_strength:.2f})")
    elif div_type == 'BULLISH' and div_strength > 0.15:
        score_val += 30
        signal_count += 1
        reasons.append(f"W_CVD_BIAS(str={div_strength:.2f})")
    elif div_type == 'BEARISH' and div_strength > 0.5:
        # Distribución silenciosa (informativo - señal para SHORT)
        score_val += 25
        signal_count += 1
        reasons.append(f"W_CVD_DISTRIB(str={div_strength:.2f})")

    # ============================================================
    # SEÑAL 2: ABSORCIÓN — Alto volumen, pequeño rango
    # ============================================================
    abs_score, abs_count = wm.absorption_score(df, window=48, vol_threshold=2.0, range_threshold=0.4)
    abs_dir, abs_bias = wm.net_absorption_direction(df, window=48)

    if abs_score > 0.5 and abs_dir == 'BULLISH':
        score_val += 50
        signal_count += 1
        reasons.append(f"W_ABSORPTION_BULL(n={abs_count},bias={abs_bias:.2f})")
    elif abs_score > 0.3 and abs_dir != 'NEUTRAL':
        score_val += 25
        signal_count += 1
        reasons.append(f"W_ABSORPTION({abs_dir},n={abs_count})")
    elif abs_score > 0.1:
        score_val += 10
        reasons.append(f"W_ABS_WEAK(n={abs_count})")

    # ============================================================
    # SEÑAL 3: ICEBERG ORDERS — Volumen creciente en precio plano
    # ============================================================
    is_iceberg, iceberg_conf = wm.iceberg_detection(df, window=48, price_stability=0.3, vol_growth=1.3)

    if is_iceberg and iceberg_conf >= 0.8:
        score_val += 45
        signal_count += 1
        reasons.append(f"W_ICEBERG_STRONG(conf={iceberg_conf:.2f})")
    elif is_iceberg and iceberg_conf >= 0.6:
        score_val += 25
        signal_count += 1
        reasons.append(f"W_ICEBERG(conf={iceberg_conf:.2f})")

    # ============================================================
    # SEÑAL 4: LARGE TRADE RATIO — Huella de manos institucionales
    # ============================================================
    lt_ratio, lt_count, lt_mult = wm.large_trade_ratio(df, window=96, threshold_mult=3.0)

    if lt_ratio > 0.08 and lt_mult > 4.0:
        score_val += 40
        signal_count += 1
        reasons.append(f"W_INST_PRINTS(n={lt_count},~{lt_mult:.1f}x)")
    elif lt_ratio > 0.05:
        score_val += 20
        signal_count += 1
        reasons.append(f"W_LARGE_TRADES(n={lt_count})")

    # ============================================================
    # SEÑAL 5: VOLUME CLOCK — Concentración en ventanas institucionales
    # ============================================================
    ny_ratio, asian_ratio, has_clock = wm.volume_clock(df, window=192)

    if has_clock:
        dominant = max(ny_ratio, asian_ratio)
        session = "NY" if ny_ratio > asian_ratio else "ASIA"
        if dominant > 3.0:
            score_val += 35
            signal_count += 1
            reasons.append(f"W_CLOCK_{session}({dominant:.1f}x)")
        elif dominant > 2.0:
            score_val += 18
            reasons.append(f"W_CLOCK_{session}_MILD({dominant:.1f}x)")

    # ============================================================
    # SEÑAL 6: TAKER BUY WHALE — Acumulación agresiva
    # ============================================================
    if 'taker_buy_vol' in df.columns:
        recent = df['taker_buy_vol'].iloc[-48:].sum()
        total = df['volume'].iloc[-48:].sum()
        if total > 0:
            buy_ratio = recent / total
            # Solo contar si es alto buy ratio + volumen extremo
            vol_z = math.z_score(df['volume'], window=96).iloc[-1]
            if buy_ratio > 0.68 and vol_z > 1.5:
                score_val += 40
                signal_count += 1
                reasons.append(f"W_WHALE_BUYER({buy_ratio:.0%},z={vol_z:.1f})")
            elif buy_ratio > 0.62 and vol_z > 1.0:
                score_val += 20
                signal_count += 1
                reasons.append(f"W_BUYER_BIAS({buy_ratio:.0%})")

    # ============================================================
    # GATE: Mínimo 2 señales activas
    # ============================================================
    if signal_count < 2:
        return {'score': 0, 'reasons': []}

    # ============================================================
    # BONUS DE CONVERGENCIA: Múltiples señales de ballena combinadas
    # ============================================================
    if signal_count >= 4:
        bonus = 60
        score_val += bonus
        reasons.append(f"W_CONVERGENCE_ULTRA(x{signal_count},+{bonus})")
    elif signal_count == 3:
        bonus = 30
        score_val += bonus
        reasons.append(f"W_CONVERGENCE_STRONG(x{signal_count},+{bonus})")

    return {'score': score_val, 'reasons': reasons}
