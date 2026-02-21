"""
Layer 5: MANIPULATION — Detección de manipulación de precio activa por ballenas.

Filosofía: Las ballenas manipulan el precio para:
1. Acumular barato → crean una caída artificial (fake breakdown / stop-hunt bajista)
2. Distribuir caro  → crean una subida artificial (pump) y luego venden
3. Atrapar al retail en una dirección para ir en la opuesta

Señales:
- Liquidity Sweep: barrida de stops con wick extremo y reversión
- Stop Hunt Pattern: nuevo mínimo/máximo falso que revierte agresivamente
- Fake Breakout: cierre fuera de rango que no confirma
- CVD Extreme Divergence: precio en máximos, CVD en caída (distribución)
- Pump & Dump: subida rápida seguida de distribución
- Smart Money Index divergence: insiders saliendo cuando precio está arriba
"""
from . import whale_math as wm
from . import scanner_math as math
import numpy as np


def score(df, funding_df=None):
    """
    Score una asset por manipulación institucional activa.
    Returns: {'score': int, 'reasons': list, 'bias': 'BULLISH'|'BEARISH'|'NEUTRAL'}
    """
    if len(df) < 200:
        return {'score': 0, 'reasons': [], 'bias': 'NEUTRAL'}

    score_val = 0
    reasons = []
    bull_signals = 0
    bear_signals = 0

    # ============================================================
    # SEÑAL 1: LIQUIDITY SWEEP — Barrida de stops + reversión
    # ============================================================
    sweeps = wm.liquidity_sweep(df, window=96, wick_body_ratio=2.5)

    if sweeps:
        bullish_sweeps = [s for s in sweeps if s['type'] == 'BULLISH']
        bearish_sweeps = [s for s in sweeps if s['type'] == 'BEARISH']
        
        recent_sweeps = [s for s in sweeps if s['idx'] >= len(sweeps) - 3]  # Últimas 3 velas

        if bullish_sweeps:
            max_wick = max(s['wick_ratio'] for s in bullish_sweeps)
            is_recent = any(s in recent_sweeps for s in bullish_sweeps)
            base_pts = 70 if is_recent else 45
            if max_wick > 4.0:
                score_val += base_pts + 10
                reasons.append(f"M_SWEEP_BULL_EXTREME(wick={max_wick:.1f}x{'*' if is_recent else ''})")
            else:
                score_val += base_pts
                reasons.append(f"M_SWEEP_BULL(n={len(bullish_sweeps)},wick={max_wick:.1f}x{'*' if is_recent else ''})")
            bull_signals += 1

        if bearish_sweeps:
            max_wick = max(s['wick_ratio'] for s in bearish_sweeps)
            is_recent = any(s in recent_sweeps for s in bearish_sweeps)
            base_pts = 70 if is_recent else 45
            if max_wick > 4.0:
                score_val += base_pts + 10
                reasons.append(f"M_SWEEP_BEAR_EXTREME(wick={max_wick:.1f}x{'*' if is_recent else ''})")
            else:
                score_val += base_pts
                reasons.append(f"M_SWEEP_BEAR(n={len(bearish_sweeps)},wick={max_wick:.1f}x{'*' if is_recent else ''})")
            bear_signals += 1

    # ============================================================
    # SEÑAL 2: STOP HUNT PATTERN — Nuevo extremo falso + reversión
    # ============================================================
    bull_hunts, bear_hunts, last_hunt = wm.stop_hunt_pattern(df, window=96, lookback=20)
    total_hunts = bull_hunts + bear_hunts

    if total_hunts >= 3:
        score_val += 65
        reasons.append(f"M_STOP_HUNT_CLUSTER(bull={bull_hunts},bear={bear_hunts},last={last_hunt})")
        if last_hunt == 'BULLISH':
            bull_signals += 1
        elif last_hunt == 'BEARISH':
            bear_signals += 1
    elif total_hunts >= 1:
        score_val += 30
        reasons.append(f"M_STOP_HUNT(n={total_hunts},last={last_hunt})")

    # ============================================================
    # SEÑAL 3: FAKE BREAKOUT — Trampa para retail
    # ============================================================
    bear_fakes, bull_fakes, latest_fake = wm.fake_breakout_score(df, window=96, lookback=24)
    total_fakes = bear_fakes + bull_fakes

    if total_fakes >= 2:
        score_val += 55
        reasons.append(f"M_FAKE_BO(bear={bear_fakes},bull={bull_fakes},latest={'YES' if latest_fake else 'NO'})")
        if bear_fakes > bull_fakes:
            bear_signals += 1
        else:
            bull_signals += 1
    elif total_fakes == 1:
        score_val += 22
        reasons.append(f"M_FAKE_BO_WEAK(n={total_fakes})")

    # ============================================================
    # SEÑAL 4: CVD EXTREME DIVERGENCE — Distribución en el techo
    # ============================================================
    div_type, div_strength = wm.cvd_divergence(df, window=96)

    if div_type == 'BEARISH' and div_strength > 0.4:
        score_val += 55
        reasons.append(f"M_CVD_DISTRIB(str={div_strength:.2f})")
        bear_signals += 1
    elif div_type == 'BULLISH' and div_strength > 0.4:
        # Manipulación bajista → rebote esperado (señal alcista de manipulación)
        score_val += 45
        reasons.append(f"M_CVD_ACCUM_HIDDEN(str={div_strength:.2f})")
        bull_signals += 1

    # ============================================================
    # SEÑAL 5: PUMP & DUMP — Distribución institucional
    # ============================================================
    pump_det, dump_phase, pump_str = wm.pump_dump_score(df, window=96)

    if pump_det and dump_phase:
        score_val += 60
        reasons.append(f"M_PUMP_DUMP(str={pump_str:.1f})")
        bear_signals += 1  # Dump phase = bearish
    elif pump_det and not dump_phase:
        score_val += 25
        reasons.append(f"M_PUMP_ONLY(str={pump_str:.1f})")

    # ============================================================
    # SEÑAL 6: SMART MONEY INDEX — Divergencia insiders
    # ============================================================
    smi_slope, smi_div = wm.smart_money_index(df, window=192)

    if smi_div:
        if smi_slope > 0:
            # Smart money comprando cuando precio baja = bullish
            score_val += 45
            reasons.append(f"M_SMI_ACCUM(slope={smi_slope:.2f})")
            bull_signals += 1
        elif smi_slope < 0:
            # Smart money vendiendo cuando precio sube = distribución
            score_val += 45
            reasons.append(f"M_SMI_DISTRIB(slope={smi_slope:.2f})")
            bear_signals += 1

    # ============================================================
    # BONUS: Coherencia de señales (todas apuntan al mismo lado)
    # ============================================================
    total_directional = bull_signals + bear_signals
    if total_directional >= 3:
        if bull_signals >= 3:
            bonus = 50
            score_val += bonus
            reasons.append(f"M_BULL_COHERENCE(x{bull_signals},+{bonus})")
        elif bear_signals >= 3:
            bonus = 50
            score_val += bonus
            reasons.append(f"M_BEAR_COHERENCE(x{bear_signals},+{bonus})")
    elif total_directional == 2:
        if bull_signals == 2:
            score_val += 20
            reasons.append(f"M_BULL_ALIGN(x{bull_signals})")
        elif bear_signals == 2:
            score_val += 20
            reasons.append(f"M_BEAR_ALIGN(x{bear_signals})")

    # Determinar bias dominante
    if bull_signals > bear_signals:
        bias = 'BULLISH'
    elif bear_signals > bull_signals:
        bias = 'BEARISH'
    else:
        bias = 'NEUTRAL'

    return {'score': score_val, 'reasons': reasons, 'bias': bias}
