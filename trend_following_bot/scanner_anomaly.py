"""
scanner_anomaly.py — Nascent Trend Scanner (Raw Macro Observer)

Filosofía: Encontrar el PUNTO 0 de una nueva tendencia, no seguir una que ya existe.

Dos modos de operación controlados por `boot_mode`:

  BOOT MODE (primer scan al iniciar el bot):
    - Ignora TODOS los pares con tendencia activa > 4 horas (16 velas de 15m)
    - Solo acepta pares en compresión (sin tendencia definida)
    - Objetivo: arrancar limpio, sin herencias del mercado previo

  CONTINUOUS MODE (scans posteriores):
    - Detecta el PUNTO EXACTO de inicio de una nueva tendencia
    - Acepta pares donde la tendencia lleva ≤ 3 velas (45 minutos)
    - Puntúa la frescura del movimiento junto con Kick, Fuel y Barrier Break
"""
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════

BOOT_MAX_TREND_AGE  = 16   # Velas en 15m = 4 horas → si la tendencia es más vieja, se rechaza en boot
LIVE_MAX_TREND_AGE  = 3    # Velas en 15m = 45 min → max edad permitida en modo continuo
MIN_HISTORY         = 96   # Mínimo de velas para calcular métricas (~24h en 15m)
TREND_MOVE_PCT      = 0.8  # % de movimiento mínimo por vela para contar como "tendencia activa"

# Categorías de señal (alineadas con whale scanner)
# ULTRA ≥ 70 pts : kick fuerte + rvol + barrier + trend reciente
# HIGH  ≥ 20 pts : algún kick + algo de rvol → señal mínima ejecutable
# por debajo de HIGH: descartado
ANOMALY_SCORE_ULTRA = 70
ANOMALY_SCORE_HIGH  = 20



# ═══════════════════════════════════════════════════════════
# LÓGICA DE EDAD DE TENDENCIA
# ═══════════════════════════════════════════════════════════

def _trend_age_candles(close: pd.Series, direction: str) -> int:
    """
    Calcula cuántas velas consecutivas lleva activa la tendencia en `direction`.

    Empezando desde la última vela y retrocediendo, cuenta cuántas velas
    consecutivas se mueven en la dirección dada más de TREND_MOVE_PCT%.

    Returns:
        int — número de velas consecutivas en tendencia (0 = sin tendencia activa)
    """
    try:
        prices = close.values
        if len(prices) < 4:
            return 0

        age = 0
        # Comparar vela[i] con vela[i-1], de atrás hacia adelante
        for i in range(len(prices) - 1, max(len(prices) - 49, 0), -1):
            prev = prices[i - 1]
            curr = prices[i]
            if prev <= 0:
                break

            move_pct = (curr - prev) / prev * 100

            if direction == 'LONG' and move_pct >= TREND_MOVE_PCT:
                age += 1
            elif direction == 'SHORT' and move_pct <= -TREND_MOVE_PCT:
                age += 1
            else:
                # La cadena se rompió — la tendencia no es continua
                break

        return age
    except Exception:
        return 0


def _is_in_compression(close: pd.Series, high: pd.Series, low: pd.Series) -> bool:
    """
    Detecta si el par está en compresión (rango estrecho sin tendencia clara).
    Se usa en BOOT MODE para validar que el par realmente no ha roto aún.
    
    Returns True si el rango de las últimas 16 velas (4h) es < 2x el ATR de 14 velas.
    """
    try:
        recent_high = high.iloc[-16:].max()
        recent_low  = low.iloc[-16:].min()
        range_4h    = recent_high - recent_low

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.iloc[-30:-16].mean()  # ATR del período anterior a la ventana

        if atr_14 <= 0:
            return False
        return (range_4h / atr_14) < 2.5
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
# SCANNER PRINCIPAL
# ═══════════════════════════════════════════════════════════

class AnomalyScanner:
    """
    Nascent Trend Scanner con dos modos:
      - boot_mode=True  → solo acepta pares sin tendencia (>4h rechazados)
      - boot_mode=False → detecta el inicio exacto de nueva tendencia (≤3 velas)
    """

    @staticmethod
    def score_universe(
        pair_data: dict,
        now_idx: int,
        top_n: int = 10,
        long_ratio=None,
        boot_mode: bool = False,
    ) -> list:
        """
        Escanea el universo y retorna los mejores candidatos de nacientes.

        Args:
            pair_data:   {symbol: DataFrame con OHLCV}
            now_idx:     índice de la vela "actual" en el DataFrame
            top_n:       número máximo de picks a retornar
            long_ratio:  None = meritocracy | 0.6 = 60% longs | 1.0 = solo longs
            boot_mode:   True = primer inicio del bot (rechaza todo lo que ya movió)

        Returns:
            lista de dicts con keys: symbol, score, direction, layer, reasons
        """
        max_age = BOOT_MAX_TREND_AGE if boot_mode else LIVE_MAX_TREND_AGE
        metrics = {}

        for symbol, df in pair_data.items():
            if now_idx > len(df):
                continue

            history = df.iloc[:now_idx].copy()
            if len(history) < MIN_HISTORY:
                continue

            try:
                close  = pd.to_numeric(history['close'],  errors='coerce')
                open_p = pd.to_numeric(history['open'],   errors='coerce')
                high   = pd.to_numeric(history['high'],   errors='coerce')
                low    = pd.to_numeric(history['low'],    errors='coerce')
                volume = pd.to_numeric(history['volume'], errors='coerce')

                if close.isna().sum() > 5:
                    continue

                # ── Calcular edad de tendencia en cada dirección ──────────
                long_age  = _trend_age_candles(close, 'LONG')
                short_age = _trend_age_candles(close, 'SHORT')

                # ── BOOT MODE: aceptar tendencias muy recientes (≤ 2 velas = 30 min) ──
                if boot_mode:
                    # En boot: la tendencia no puede tener más de 2 velas de historia
                    # Si ambas son demasiado viejas → el par ya lleva demasiado tiempo moviéndose
                    long_too_old  = long_age  > 2
                    short_too_old = short_age > 2
                    if long_too_old and short_too_old:
                        continue   # Par ya en movimiento fuerte, saltar

                # ── CONTINUOUS MODE: solo tendencias recién nacidas ────────
                else:
                    # LIVE: aceptar tendencias recién nacidas (1 ≤ age ≤ max_age)
                    # O pares planos (age=0) que estén en compresión o cerca de romper
                    long_nascent  = 0 < long_age  <= max_age
                    short_nascent = 0 < short_age <= max_age
                    long_coiled   = long_age  == 0  # plano: sin direccion confirmada aun
                    short_coiled  = short_age == 0

                    # Checar si hay señales previas de ruptura (rvol o kick) — SOLO para age=0
                    _vol_now  = pd.to_numeric(history['volume'], errors='coerce')
                    _vol_last = _vol_now.iloc[-4:].sum()
                    _avg_vol  = _vol_now.iloc[-96:].sum() / 24 if len(_vol_now) >= 96 else None
                    _rvol_pre = (_vol_last / _avg_vol) if (_avg_vol and _avg_vol > 0) else 0

                    # Par en compresión a punto de romper
                    _in_comp = _is_in_compression(close, high, low)

                    # Aceptar si: tiene tendencia naciente O está plano pero con rvol ≥ 1.5x
                    has_nascent = long_nascent or short_nascent
                    has_coiled  = (long_coiled and short_coiled) and (_rvol_pre >= 1.5 or _in_comp)

                    if not has_nascent and not has_coiled:
                        continue


                # ── MÉTRICAS COMUNES ──────────────────────────────────────

                # Kick: anomalía en el cuerpo de las últimas 2 velas
                bodies       = (close - open_p).abs()
                avg_body_24h = bodies.iloc[-96:-2].mean()
                max_kick     = max(bodies.iloc[-1], bodies.iloc[-2])
                kick_mult    = (max_kick / avg_body_24h) if avg_body_24h > 0 else 0

                # Fuel: RVol de la última hora vs media horaria
                vol_last_1h = volume.iloc[-4:].sum()
                avg_vol_1h  = volume.iloc[-96:].sum() / 24
                rvol        = (vol_last_1h / avg_vol_1h) if avg_vol_1h > 0 else 0

                # Barrier Break: posición relativa al rango de las últimas 12h
                high_12h   = high.iloc[-48:-2].max()
                low_12h    = low.iloc[-48:-2].min()
                range_12h  = high_12h - low_12h
                range_pct  = ((close.iloc[-1] - low_12h) / range_12h * 100) if range_12h > 0 else 50

                # Frescura del movimiento (solo modo continuo — en boot siempre es 0)
                long_age_score  = max(0, max_age - long_age)   # Más fresco = mayor score
                short_age_score = max(0, max_age - short_age)

                metrics[symbol] = {
                    'kick_mult':       kick_mult,
                    'rvol':            rvol,
                    'range_pct':       range_pct,
                    'long_age':        long_age,
                    'short_age':       short_age,
                    'long_age_score':  long_age_score,
                    'short_age_score': short_age_score,
                    'long_valid':      (not boot_mode and 0 < long_age  <= max_age) or (boot_mode and long_age == 0),
                    'short_valid':     (not boot_mode and 0 < short_age <= max_age) or (boot_mode and short_age == 0),
                }

            except Exception:
                continue

        if not metrics:
            return []

        # ═══════════════════════════════════════════════════
        # SCORING
        # ═══════════════════════════════════════════════════
        long_candidates  = []
        short_candidates = []

        for s, m in metrics.items():

            # ── LONG CANDIDATE ────────────────────────────────────────────
            if m['long_valid'] or boot_mode:
                score   = 0
                reasons = []

                if boot_mode:
                    # En boot: prioritizamos compresión + potencial
                    reasons.append("BOOT_CLEAN_SLATE")
                else:
                    # En continuo: puntuar frescura del inicio
                    if m['long_age'] == 1:
                        score += 35
                        reasons.append("TREND_START_1V")
                    elif m['long_age'] == 2:
                        score += 25
                        reasons.append("TREND_START_2V")
                    elif m['long_age'] == 3:
                        score += 10
                        reasons.append("TREND_START_3V")

                # Kick (0–40 pts)
                if m['kick_mult'] > 5.0:
                    score += 40
                    reasons.append(f"KICK_MASSIVE({m['kick_mult']:.1f}x)")
                elif m['kick_mult'] > 3.0:
                    score += 25
                    reasons.append(f"KICK_STRONG({m['kick_mult']:.1f}x)")
                elif m['kick_mult'] > 1.5:
                    score += 10
                    reasons.append(f"KICK_MILD({m['kick_mult']:.1f}x)")

                # Fuel / RVol (0–35 pts)
                if m['rvol'] > 4.0:
                    score += 35
                    reasons.append(f"RVOL_EXPLOSIVE({m['rvol']:.1f}x)")
                elif m['rvol'] > 2.0:
                    score += 20
                    reasons.append(f"RVOL_HIGH({m['rvol']:.1f}x)")

                # Barrier Break UP (0–25 pts)
                if m['range_pct'] >= 95:
                    score += 25
                    reasons.append("BREAKING_12H_HIGH")
                elif m['range_pct'] >= 85:
                    score += 10
                    reasons.append("TESTING_12H_HIGH")

                # Filtro mínimo: debe tener algún kick y algún combustible
                if score > 0 and m['kick_mult'] > 1.2:
                    long_candidates.append({
                        'symbol':    s,
                        'score':     score,
                        'reasons':   ", ".join(reasons),
                        'direction': 'LONG',
                        'layer':     'ANOMALY',
                        'trend_age': m['long_age'],
                    })

            # ── SHORT CANDIDATE ───────────────────────────────────────────
            if m['short_valid'] or boot_mode:
                score   = 0
                reasons = []

                if boot_mode:
                    reasons.append("BOOT_CLEAN_SLATE")
                else:
                    if m['short_age'] == 1:
                        score += 35
                        reasons.append("TREND_START_1V")
                    elif m['short_age'] == 2:
                        score += 25
                        reasons.append("TREND_START_2V")
                    elif m['short_age'] == 3:
                        score += 10
                        reasons.append("TREND_START_3V")

                # Kick (0–40 pts)
                if m['kick_mult'] > 5.0:
                    score += 40
                    reasons.append(f"KICK_MASSIVE({m['kick_mult']:.1f}x)")
                elif m['kick_mult'] > 3.0:
                    score += 25
                    reasons.append(f"KICK_STRONG({m['kick_mult']:.1f}x)")
                elif m['kick_mult'] > 1.5:
                    score += 10
                    reasons.append(f"KICK_MILD({m['kick_mult']:.1f}x)")

                # Fuel
                if m['rvol'] > 4.0:
                    score += 35
                    reasons.append(f"RVOL_EXPLOSIVE({m['rvol']:.1f}x)")
                elif m['rvol'] > 2.0:
                    score += 20
                    reasons.append(f"RVOL_HIGH({m['rvol']:.1f}x)")

                # Barrier Break DOWN (0–25 pts)
                if m['range_pct'] <= 5:
                    score += 25
                    reasons.append("BREAKING_12H_LOW")
                elif m['range_pct'] <= 15:
                    score += 10
                    reasons.append("TESTING_12H_LOW")

                if score > 0 and m['kick_mult'] > 1.2:
                    short_candidates.append({
                        'symbol':    s,
                        'score':     score,
                        'reasons':   ", ".join(reasons),
                        'direction': 'SHORT',
                        'layer':     'ANOMALY',
                        'trend_age': m['short_age'],
                    })

        # ═══════════════════════════════════════════════════
        # ROSTER: meritocracy o ratio fijo
        # ═══════════════════════════════════════════════════
        if long_ratio is None:
            all_candidates = long_candidates + short_candidates
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            roster, seen = [], set()
            for pick in all_candidates:
                if len(roster) >= top_n:
                    break
                if pick['symbol'] not in seen:
                    # Asignar categoria HIGH/ULTRA — debajo de HIGH se descarta
                    score = pick['score']
                    if score >= ANOMALY_SCORE_ULTRA:
                        pick['confidence'] = 'ULTRA'
                    elif score >= ANOMALY_SCORE_HIGH:
                        pick['confidence'] = 'HIGH'
                    else:
                        continue   # Por debajo de HIGH: no entra al watchlist
                    roster.append(pick)
                    seen.add(pick['symbol'])
            return roster

        # Ratio fijo
        long_slots  = top_n if long_ratio >= 1.0 else max(0, int(top_n * long_ratio))
        long_candidates.sort( key=lambda x: x['score'], reverse=True)
        short_candidates.sort(key=lambda x: x['score'], reverse=True)

        roster, seen = [], set()
        for pick in long_candidates[:long_slots]:
            roster.append(pick)
            seen.add(pick['symbol'])
        for pick in short_candidates:
            if len(roster) >= top_n:
                break
            if pick['symbol'] not in seen:
                roster.append(pick)
                seen.add(pick['symbol'])
        return roster
