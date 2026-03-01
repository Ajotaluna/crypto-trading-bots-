"""
Layer 0: BIRTH — Detector del momento exacto de nacimiento de tendencia.

Filosofía:
Una tendencia cripto ocurre en 1-3 velas de 15 minutos.
Este layer detecta ESE momento específico, no la tendencia que resulta de él.

Preguntas que responde:
1. ¿La vela ACTUAL rompe el rango promedio de las últimas 4h?
2. ¿El volumen de las últimas 3 velas explota vs las 10 anteriores?
3. ¿El cruce EMA21/EMA50 es reciente (≤4 velas)?
4. ¿El cuerpo de la vela muestra convicción (no wick)?
5. ¿El movimiento no es ya overextendido (RSI7 < 78)?
"""
from . import scanner_math as sm
import numpy as np


def score(df):
    """
    Puntúa el nacimiento de tendencia basado en la vela ACTUAL y señales
    de MUY corto plazo.

    Returns: {'score': int, 'reasons': list, 'is_birth': bool}
    """
    if len(df) < 60:
        return {'score': 0, 'reasons': [], 'is_birth': False}

    close = df['close']
    last_close = close.iloc[-1]
    last_open = df['open'].iloc[-1]
    is_green = last_close > last_open

    score_val = 0
    reasons = []

    # ============================================================
    # GATE: Si ya está muy extendido, NO es nascente — penalizar
    # ============================================================
    rsi7 = sm.rsi(close, period=7).iloc[-1]
    if rsi7 > 80:
        # Movimiento ya corrido, trampa potencial
        return {
            'score': 0,
            'reasons': [f'B_OVEREXTENDED(RSI7={rsi7:.1f})'],
            'is_birth': False
        }

    # ============================================================
    # SEÑAL 1: VELA DE NACIMIENTO — Expansión de rango + convicción
    # ============================================================
    body_ratio, range_exp, is_birth_candle = sm.candle_birth_score(df, range_window=16)

    if is_birth_candle and is_green:
        if range_exp > 3.5:
            score_val += 100
            reasons.append(f'B_BIRTH_EXPLOSION(range={range_exp:.1f}x,body={body_ratio:.2f})')
        elif range_exp > 2.5:
            score_val += 75
            reasons.append(f'B_BIRTH_STRONG(range={range_exp:.1f}x,body={body_ratio:.2f})')
        elif range_exp > 1.8:
            score_val += 50
            reasons.append(f'B_BIRTH_CANDLE(range={range_exp:.1f}x,body={body_ratio:.2f})')
    elif is_birth_candle and not is_green:
        # Vela de nacimiento bajista — válida pero menos prioritaria para long
        score_val += 20
        reasons.append(f'B_BIRTH_BEAR(range={range_exp:.1f}x)')
    elif range_exp > 1.5 and is_green:
        # Rango expandido aunque no cumpla todos los criterios
        score_val += 25
        reasons.append(f'B_RANGE_EXP(range={range_exp:.1f}x)')

    # ============================================================
    # SEÑAL 2: EXPLOSIÓN DE VOLUMEN — Ahora, no hace 6h
    # ============================================================
    vol_acc = sm.volume_acceleration(df, short=3, long=10)

    if vol_acc > 4.0:
        score_val += 90
        reasons.append(f'B_VOL_NOVA({vol_acc:.1f}x)')
    elif vol_acc > 3.0:
        score_val += 60
        reasons.append(f'B_VOL_EXPLOSION({vol_acc:.1f}x)')
    elif vol_acc > 2.0:
        score_val += 35
        reasons.append(f'B_VOL_SURGE({vol_acc:.1f}x)')
    elif vol_acc > 1.5:
        score_val += 15
        reasons.append(f'B_VOL_RISING({vol_acc:.1f}x)')

    # ============================================================
    # SEÑAL 3: CRUCE EMA FRESCO — El nacimiento del cruce, no el efecto
    # ============================================================
    cross_age = sm.ema_cross_age(df, short=21, long=50, max_lookback=20)

    if cross_age == 0:
        # Cruce en vela actual — perfecto
        score_val += 80
        reasons.append('B_EMA_CROSS_NOW')
    elif cross_age == 1:
        score_val += 60
        reasons.append('B_EMA_CROSS_1AGO')
    elif cross_age == 2:
        score_val += 40
        reasons.append('B_EMA_CROSS_2AGO')
    elif cross_age == 3:
        score_val += 20
        reasons.append('B_EMA_CROSS_3AGO')
    elif cross_age <= 5:
        score_val += 10
        reasons.append(f'B_EMA_CROSS_FRESH({cross_age}ago)')
    # >5 velas: no puntúa — ya es tendencia, no nascente

    # ============================================================
    # SEÑAL 4: RSI(7) en zona de aceleración (no overextendido)
    # ============================================================
    if 55 < rsi7 < 72:
        score_val += 30
        reasons.append(f'B_RSI7_SWEET({rsi7:.1f})')
    elif 50 < rsi7 <= 55:
        score_val += 10
        reasons.append(f'B_RSI7_WARMING({rsi7:.1f})')
    elif rsi7 >= 72:
        # Caliente pero no overextendido aún
        penalty = int(score_val * 0.2)
        score_val -= penalty
        reasons.append(f'B_RSI7_HOT({rsi7:.1f},-{penalty})')

    # ============================================================
    # SEÑAL 5: CAMBIO DE VOLUMEN EN VELA ACTUAL
    # La vela actual tiene >> volumen que la vela previa
    # ============================================================
    if len(df) >= 5:
        cur_vol = df['volume'].iloc[-1]
        avg_prev_4 = df['volume'].iloc[-5:-1].mean()
        if avg_prev_4 > 0:
            cur_vol_ratio = cur_vol / avg_prev_4
            if cur_vol_ratio > 5.0 and is_green:
                score_val += 50
                reasons.append(f'B_CANDLE_VOL_SPIKE({cur_vol_ratio:.1f}x)')
            elif cur_vol_ratio > 3.0 and is_green:
                score_val += 25
                reasons.append(f'B_CANDLE_VOL_HIGH({cur_vol_ratio:.1f}x)')

    # ============================================================
    # PENALIZACIÓN: Ya ganó mucho en las últimas horas
    # Si el precio ya movió mucho, ya pasó la acción
    # ============================================================
    ret_6h = 0.0
    if len(close) >= 24:
        ret_6h = (last_close - close.iloc[-24]) / close.iloc[-24] * 100
        if ret_6h > 12.0:
            # Más de 12% en 6h = ya corrió
            score_val = int(score_val * 0.15)
            reasons.append(f'B_LATE_6H({ret_6h:+.1f}%,kill)')
        elif ret_6h > 8.0:
            penalty = int(score_val * 0.4)
            score_val -= penalty
            reasons.append(f'B_LATE_6H({ret_6h:+.1f}%,-{penalty})')
        elif ret_6h > 5.0:
            penalty = int(score_val * 0.2)
            score_val -= penalty
            reasons.append(f'B_LATE_6H({ret_6h:+.1f}%,-{penalty})')

    is_birth = score_val >= 50

    return {'score': max(0, score_val), 'reasons': reasons, 'is_birth': is_birth}
