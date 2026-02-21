"""
WhaleScanner — Scanner standalone de ballenas institucionales.

Orquesta Layer 4 (WHALE) y Layer 5 (MANIPULATION) para analizar
el universo de pares y detectar donde están las ballenas.

Puede usarse de forma independiente al scanner_engine.py:
  from nascent_scanner.whale_scanner import WhaleScanner
  ws = WhaleScanner()
  results = ws.score_universe(pair_data, top_n=10)

O ejecutarse directamente con datos reales desde archivos CSV:
  python whale_scanner.py
"""
import pandas as pd
import numpy as np
import os
import sys

# Support both package and direct execution
try:
    from . import layer_whale
    from . import layer_manipulation
    from . import whale_math as wm
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import layer_whale
    import layer_manipulation
    import whale_math as wm


class WhaleScanner:
    """
    Scanner que detecta actividad de ballenas institucionales.

    Combina señales de acumulación silenciosa (L4: WHALE) con
    patrones de manipulación activa (L5: MANIPULATION) para identificar
    dónde están posicionadas las manos fuertes y hacia dónde van.
    """

    CONFIDENCE_LEVELS = {
        'ULTRA': 250,
        'HIGH': 160,
        'MEDIUM': 90,
        'LOW': 40,
    }

    def analyze(self, df, funding_df=None):
        """
        Analiza un solo DataFrame y retorna el resultado completo de ballenas.

        Returns dict:
        {
          'rejected': bool,
          'l4': {'score': int, 'reasons': list},  # WHALE layer
          'l5': {'score': int, 'reasons': list, 'bias': str},  # MANIPULATION layer
          'total_score': int,
          'confidence': str,
          'direction': str,  # Dirección inferida ('LONG', 'SHORT', 'NEUTRAL')
          'cvd_slope': float,
          'absorption_count': int,
        }
        """
        if len(df) < 96:
            return {'rejected': True, 'reason': 'Insufficient data'}

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            return {'rejected': True, 'reason': 'Missing columns'}

        # Filtros de liquidez básicos
        last = df['close'].iloc[-1]
        quote_vol = df['volume'].iloc[-96:].sum() * last
        if quote_vol < 50_000:
            return {'rejected': True, 'reason': 'Illiquid'}

        # Calcular capas
        l4 = layer_whale.score(df, funding_df=funding_df)
        l5 = layer_manipulation.score(df, funding_df=funding_df)

        total_score = l4['score'] + l5['score']

        # CVD slope para diagnóstico
        cvd_sl = wm.cvd_slope(df, window=96, slope_window=48)

        # Absorción count
        _, abs_count = wm.absorption_score(df, window=48)

        # Inferir dirección dominante
        direction = _infer_direction(l4, l5, df)

        # Bonus de convergencia: L4 y L5 activos juntos + coherentes
        if l4['score'] > 0 and l5['score'] > 0:
            if direction != 'NEUTRAL':
                convergence_bonus = 40
                total_score += convergence_bonus

        # Nivel de confianza
        confidence = 'NONE'
        for level, threshold in sorted(self.CONFIDENCE_LEVELS.items(), key=lambda x: x[1], reverse=True):
            if total_score >= threshold:
                confidence = level
                break

        return {
            'rejected': False,
            'l4': l4,
            'l5': l5,
            'total_score': total_score,
            'confidence': confidence,
            'direction': direction,
            'cvd_slope': cvd_sl,
            'absorption_count': abs_count,
        }

    def score_universe(self, pair_data, top_n=10, min_score=40, funding_data=None):
        """
        Analizar todo el universo de pares y retornar los mejores por actividad de ballenas.

        Args:
            pair_data: dict {symbol: DataFrame}
            top_n: número máximo de picks
            min_score: score mínimo para aparecer en el roster
            funding_data: dict opcional {symbol: funding_df}

        Returns: list de dicts ordenados por total_score descendente.
        """
        results = []

        for symbol, df in pair_data.items():
            try:
                funding_df = funding_data.get(symbol) if funding_data else None
                result = self.analyze(df, funding_df=funding_df)

                if result.get('rejected'):
                    continue
                if result['total_score'] < min_score:
                    continue

                results.append({
                    'symbol': symbol,
                    'total_score': result['total_score'],
                    'whale_score': result['l4']['score'],
                    'manip_score': result['l5']['score'],
                    'whale_reasons': result['l4']['reasons'],
                    'manip_reasons': result['l5']['reasons'],
                    'direction': result['direction'],
                    'confidence': result['confidence'],
                    'cvd_slope': result['cvd_slope'],
                    'absorption_count': result['absorption_count'],
                    'bias': result['l5'].get('bias', 'NEUTRAL'),
                })

            except Exception as e:
                continue

        results.sort(key=lambda x: x['total_score'], reverse=True)
        return results[:top_n]

    def format_alert(self, result):
        """
        Formatea un resultado de ballena en un string de alerta legible.
        """
        emoji_map = {'ULTRA': '🦈🦈', 'HIGH': '🐋', 'MEDIUM': '🐬', 'LOW': '🐟', 'NONE': '─'}
        dir_emoji = {'LONG': '🟢', 'SHORT': '🔴', 'NEUTRAL': '⚪'}

        emoji = emoji_map.get(result['confidence'], '─')
        d_emoji = dir_emoji.get(result['direction'], '⚪')

        lines = [
            f"{emoji} [{result['confidence']}] {result['symbol']} | Score: {result['total_score']} {d_emoji}{result['direction']}",
            f"   🐋 WHALE({result['whale_score']}): {', '.join(result['whale_reasons'])}",
            f"   ⚠️  MANIP({result['manip_score']}): {', '.join(result['manip_reasons'])}",
            f"   CVD slope: {result['cvd_slope']:+.3f} | Absorb candles: {result['absorption_count']}",
        ]
        return "\n".join(lines)


# ============================================================
# HELPERS
# ============================================================

def _infer_direction(l4, l5, df):
    """
    Infiere la dirección de la manipulación / seguimiento de ballena.

    Lógica:
    - Si hay manipulación bajista (stop-hunt de lows, sweep bullish, CVD acumulación)
      = ballena acumuló a precios bajos → señal LONG
    - Si hay distribución (CVD cayendo con precio alto, pump detectado)
      = ballena distribuyendo → señal SHORT
    """
    bull_evidence = 0
    bear_evidence = 0

    # L4 reasons
    for r in l4.get('reasons', []):
        if any(k in r for k in ['BULL', 'ACCUM', 'BUYER', 'RISING']):
            bull_evidence += 1
        if any(k in r for k in ['DISTRIB', 'BEAR']):
            bear_evidence += 1

    # L5 bias
    bias = l5.get('bias', 'NEUTRAL')
    if bias == 'BULLISH':
        bull_evidence += 2
    elif bias == 'BEARISH':
        bear_evidence += 2

    # CVD direction
    cvd_sl = wm.cvd_slope(df, window=96)
    if cvd_sl > 0.02:
        bull_evidence += 1
    elif cvd_sl < -0.02:
        bear_evidence += 1

    if bull_evidence > bear_evidence:
        return 'LONG'
    elif bear_evidence > bull_evidence:
        return 'SHORT'
    return 'NEUTRAL'


# ============================================================
# STANDALONE — Ejecutar directamente con archivos CSV locales
# ============================================================

def _run_standalone():
    """Corre el whale scanner sobre los datos locales del nascent_scanner."""
    print("🦈 WHALE SCANNER — Rastreo Institucional")
    print("=" * 65)

    # Buscar datos
    scanner_dir = os.path.dirname(os.path.abspath(__file__))
    data_dirs = [
        os.path.join(scanner_dir, "data"),
        os.path.join(scanner_dir, "data_monthly"),
    ]

    data_dir = None
    for d in data_dirs:
        if os.path.isdir(d):
            data_dir = d
            break

    if not data_dir:
        print("❌ No data directory found (expected 'data' or 'data_monthly')")
        return

    # Cargar pares
    pair_data = {}
    files = [f for f in os.listdir(data_dir) if f.endswith("_15m.csv")]
    print(f"📂 Cargando {len(files)} pares desde {os.path.basename(data_dir)}...")

    for f in files:
        symbol = f.replace("_15m.csv", "")
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            if len(df) >= 200:
                pair_data[symbol] = df
        except Exception:
            continue

    print(f"✅ {len(pair_data)} pares cargados\n")

    if not pair_data:
        print("❌ No hay datos suficientes para escanear.")
        return

    # Ejecutar scanner
    ws = WhaleScanner()
    results = ws.score_universe(pair_data, top_n=15, min_score=40)

    if not results:
        print("⚠️  No se detectó actividad significativa de ballenas.")
        return

    print(f"🎯 TOP {len(results)} ACTIVOS CON ACTIVIDAD DE BALLENA:\n")
    for i, r in enumerate(results, 1):
        print(f"#{i}  {ws.format_alert(r)}")
        print()

    # Estadísticas
    long_count = sum(1 for r in results if r['direction'] == 'LONG')
    short_count = sum(1 for r in results if r['direction'] == 'SHORT')
    ultra_count = sum(1 for r in results if r['confidence'] == 'ULTRA')
    high_count = sum(1 for r in results if r['confidence'] == 'HIGH')

    print("=" * 65)
    print(f"📊 RESUMEN:")
    print(f"   LONG: {long_count} | SHORT: {short_count} | NEUTRAL: {len(results) - long_count - short_count}")
    print(f"   🦈🦈 ULTRA: {ultra_count} | 🐋 HIGH: {high_count}")
    avg_score = np.mean([r['total_score'] for r in results])
    print(f"   Score promedio: {avg_score:.1f}")


if __name__ == "__main__":
    _run_standalone()
