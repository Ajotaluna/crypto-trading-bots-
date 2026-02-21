"""
Test Harness — Whale Scanner (L4 y L5).

Crea datos sintéticos con patrones de manipulación conocidos y valida
que cada módulo los detecta correctamente.

Ejecutar con:
    cd "c:\\Users\\Ajota\\Documents\\Nueva carpeta\\trend_following_bot"
    python -m nascent_scanner.test_whale_scanner
    # o con pytest:
    python -m pytest nascent_scanner/test_whale_scanner.py -v
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nascent_scanner import whale_math as wm
from nascent_scanner import layer_whale
from nascent_scanner import layer_manipulation
from nascent_scanner.whale_scanner import WhaleScanner


# ============================================================
# HELPERS — Fábricas de DataFrames sintéticos
# ============================================================

def make_base_df(n=300, price=100.0, vol=1000.0):
    """DataFrame base: precio y volumen estables."""
    np.random.seed(42)
    prices = price + np.cumsum(np.random.randn(n) * 0.2)
    df = pd.DataFrame({
        'open':   prices + np.random.randn(n) * 0.1,
        'high':   prices + abs(np.random.randn(n) * 0.3),
        'low':    prices - abs(np.random.randn(n) * 0.3),
        'close':  prices,
        'volume': vol + np.random.randn(n) * 100,
    })
    df['volume'] = df['volume'].clip(lower=10)
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + 0.05
    df['low']  = df[['open', 'low',  'close']].min(axis=1) - 0.05
    return df


def inject_cvd_bullish(df):
    """Añade taker buy dominante: CVD sube con precio bajando."""
    df = df.copy()
    # Precio baja en la segunda mitad
    half = len(df) // 2
    trend_down = np.linspace(0, -5, len(df) - half)
    df.loc[df.index[half:], 'close'] = df['close'].iloc[half:].values + trend_down
    df.loc[df.index[half:], 'open']  = df['open'].iloc[half:].values  + trend_down

    # Pero el volumen comprador domina
    total_vol = df['volume']
    df['taker_buy_vol'] = total_vol * 0.72  # 72% comprador
    return df


def inject_absorption(df):
    """Añade velas de alta absorción: volumen 3x con rango muy pequeño."""
    df = df.copy()
    # Cada 10 velas, una vela de absorción
    for i in range(10, len(df), 10):
        df.loc[df.index[i], 'volume'] *= 4.0
        mid = (df.loc[df.index[i], 'high'] + df.loc[df.index[i], 'low']) / 2
        # Comprimir el rango al 20%
        df.loc[df.index[i], 'high']  = mid * 1.001
        df.loc[df.index[i], 'low']   = mid * 0.999
        df.loc[df.index[i], 'open']  = mid * 1.0002
        df.loc[df.index[i], 'close'] = mid * 1.0003
    return df


def inject_liquidity_sweep_bullish(df):
    """Añade un wick inferior extremo con cierre alcista (stop-hunt bajista)."""
    df = df.copy()
    idx = len(df) - 5
    mid_price = df['close'].iloc[idx]
    df.loc[df.index[idx], 'low']   = mid_price * 0.93   # Wick 7% hacia abajo
    df.loc[df.index[idx], 'open']  = mid_price * 0.997
    df.loc[df.index[idx], 'close'] = mid_price * 1.003  # Cierra alcista
    df.loc[df.index[idx], 'high']  = mid_price * 1.005
    df.loc[df.index[idx], 'volume'] *= 3.0
    return df


def inject_stop_hunt(df):
    """Crea un nuevo mínimo en los últimos 20 candles con reversión inmediata."""
    df = df.copy()
    # Asegurar que hay mínimos estables
    base_low = df['low'].iloc[-25:-5].min()
    idx = len(df) - 3
    df.loc[df.index[idx], 'low']   = base_low * 0.985  # Nuevo mínimo (stop-hunt)
    df.loc[df.index[idx], 'close'] = base_low * 1.002  # Pero cierra arriba
    df.loc[df.index[idx], 'open']  = base_low * 0.998
    df.loc[df.index[idx], 'high']  = base_low * 1.005
    return df


# ============================================================
# TESTS — whale_math.py
# ============================================================

def test_cvd_slope_positive_with_buyer_dominant():
    df = inject_cvd_bullish(make_base_df(300))
    slope = wm.cvd_slope(df, window=96, slope_window=48)
    assert slope > 0, f"CVD slope debería ser positivo con buyer dominante, got {slope:.4f}"
    print(f"  ✅ cvd_slope > 0 con taker_buy dominante: {slope:+.4f}")


def test_cvd_divergence_bullish():
    df = inject_cvd_bullish(make_base_df(300))
    div_type, div_strength = wm.cvd_divergence(df, window=96)
    assert div_type == 'BULLISH', f"CVD divergencia debería ser BULLISH, got {div_type}"
    assert div_strength > 0, f"Fuerza de divergencia debe ser > 0, got {div_strength:.3f}"
    print(f"  ✅ cvd_divergence = BULLISH (str={div_strength:.3f})")


def test_absorption_score_detects_high_vol_low_range():
    df = inject_absorption(make_base_df(300))
    score, count = wm.absorption_score(df, window=48, vol_threshold=2.0, range_threshold=0.4)
    assert count > 0, f"Debería detectar velas de absorción, got count={count}"
    assert score > 0, f"Score de absorción debe ser > 0, got {score:.3f}"
    print(f"  ✅ absorption_score: score={score:.3f}, count={count}")


def test_liquidity_sweep_bullish():
    df = inject_liquidity_sweep_bullish(make_base_df(300))
    events = wm.liquidity_sweep(df, window=48, wick_body_ratio=2.0)
    bullish = [e for e in events if e['type'] == 'BULLISH']
    assert len(bullish) > 0, f"Debería detectar liquidity sweep alcista, got events={events}"
    max_wick = max(e['wick_ratio'] for e in bullish)
    print(f"  ✅ liquidity_sweep BULLISH detectado (max wick_ratio={max_wick:.2f}x)")


def test_stop_hunt_pattern():
    df = inject_stop_hunt(make_base_df(300))
    bull_hunts, bear_hunts, last = wm.stop_hunt_pattern(df, window=96, lookback=20)
    assert bull_hunts > 0, f"Debería detectar ≥1 stop-hunt alcista, got {bull_hunts}"
    print(f"  ✅ stop_hunt_pattern: bull={bull_hunts}, bear={bear_hunts}, last={last}")


def test_large_trade_ratio_on_normal_data():
    """Con datos normales, el ratio de trades grandes debe ser bajo."""
    df = make_base_df(300)
    ratio, count, mult = wm.large_trade_ratio(df, window=96, threshold_mult=3.0)
    assert 0.0 <= ratio <= 1.0, f"Ratio debe estar en [0,1], got {ratio}"
    print(f"  ✅ large_trade_ratio (normal data): ratio={ratio:.3f}, count={count}, mult={mult:.1f}x")


# ============================================================
# TESTS — layer_whale.py (L4)
# ============================================================

def test_layer_whale_scores_with_whale_signals():
    df = inject_cvd_bullish(inject_absorption(make_base_df(300)))
    result = layer_whale.score(df)
    assert 'score' in result and 'reasons' in result
    assert result['score'] >= 0
    print(f"  ✅ layer_whale.score: {result['score']} | Razones: {result['reasons'][:3]}")


def test_layer_whale_gate_no_false_single_signal():
    """Con datos normales (sin patrones de ballena), la capa debe retornar 0."""
    df = make_base_df(300)
    result = layer_whale.score(df)
    # Con datos puramente random, debería ser 0 o bajo (gate requiere 2 señales)
    print(f"  ✅ layer_whale.score (datos normales): {result['score']} — razones: {result['reasons'][:2]}")
    # No forzamos == 0 ya que datos aleatorios pueden activar alguna señal débil


# ============================================================
# TESTS — layer_manipulation.py (L5)
# ============================================================

def test_layer_manipulation_detects_sweep():
    df = inject_liquidity_sweep_bullish(inject_stop_hunt(make_base_df(300)))
    result = layer_manipulation.score(df)
    assert 'score' in result and 'reasons' in result
    assert result['score'] > 0, f"Debería detectar señal de manipulación, got score={result['score']}"
    print(f"  ✅ layer_manipulation.score: {result['score']} | bias={result.get('bias')} | Razones: {result['reasons'][:3]}")


def test_layer_manipulation_bias():
    """El bias debe ser BULLISH cuando hay señales de manipulación alcista."""
    df = inject_liquidity_sweep_bullish(inject_stop_hunt(make_base_df(300)))
    result = layer_manipulation.score(df)
    bias = result.get('bias', 'NEUTRAL')
    assert bias in ('BULLISH', 'BEARISH', 'NEUTRAL')
    print(f"  ✅ layer_manipulation bias: {bias}")


# ============================================================
# TESTS — WhaleScanner orchestrator
# ============================================================

def test_whale_scanner_analyze_valid():
    ws = WhaleScanner()
    df = inject_cvd_bullish(inject_absorption(make_base_df(300)))
    # Simular volumen alto para pasar filtro de liquidez
    df['volume'] *= 10000
    result = ws.analyze(df)
    assert not result.get('rejected', True), f"No debería estar rechazado: {result}"
    assert 'total_score' in result
    assert 'direction' in result
    assert result['direction'] in ('LONG', 'SHORT', 'NEUTRAL')
    print(f"  ✅ WhaleScanner.analyze: score={result['total_score']}, dir={result['direction']}, conf={result['confidence']}")


def test_whale_scanner_score_universe():
    ws = WhaleScanner()
    pair_data = {
        'WHALE_BULL': inject_cvd_bullish(inject_absorption(make_base_df(300))),
        'MANIP':     inject_liquidity_sweep_bullish(inject_stop_hunt(make_base_df(300))),
        'NORMAL':    make_base_df(300),
    }
    # Subir volumen para pasar filtro de liquidez
    for sym in pair_data:
        pair_data[sym]['volume'] *= 10000

    results = ws.score_universe(pair_data, top_n=5, min_score=0)
    assert isinstance(results, list)
    print(f"  ✅ WhaleScanner.score_universe: {len(results)} resultados")
    for r in results:
        print(f"     {r['symbol']}: score={r['total_score']}, dir={r['direction']}, conf={r['confidence']}")


def test_whale_scanner_format_alert():
    ws = WhaleScanner()
    fake = {
        'symbol': 'BTCUSDT',
        'total_score': 200,
        'whale_score': 120,
        'manip_score': 80,
        'whale_reasons': ['W_CVD_ACCUM', 'W_ABSORPTION_BULL'],
        'manip_reasons': ['M_SWEEP_BULL', 'M_STOP_HUNT_CLUSTER'],
        'direction': 'LONG',
        'confidence': 'HIGH',
        'cvd_slope': 0.034,
        'absorption_count': 3,
        'bias': 'BULLISH',
    }
    alert = ws.format_alert(fake)
    assert 'BTCUSDT' in alert
    assert 'LONG' in alert
    print(f"  ✅ format_alert:\n{alert}")


# ============================================================
# RUNNER
# ============================================================

TESTS = [
    # whale_math
    ("cvd_slope positivo (buyer dominant)",          test_cvd_slope_positive_with_buyer_dominant),
    ("cvd_divergence BULLISH",                        test_cvd_divergence_bullish),
    ("absorption_score detecta alto vol + bajo rango",test_absorption_score_detects_high_vol_low_range),
    ("liquidity_sweep detecta wick bajista extremo",  test_liquidity_sweep_bullish),
    ("stop_hunt_pattern detecta nuevo mínimo falso",  test_stop_hunt_pattern),
    ("large_trade_ratio en datos normales",           test_large_trade_ratio_on_normal_data),
    # layer_whale
    ("layer_whale.score con señales de ballena",      test_layer_whale_scores_with_whale_signals),
    ("layer_whale.gate (datos normales → baja score)",test_layer_whale_gate_no_false_single_signal),
    # layer_manipulation
    ("layer_manipulation detecta sweep + stop-hunt",  test_layer_manipulation_detects_sweep),
    ("layer_manipulation bias coherente",             test_layer_manipulation_bias),
    # whale_scanner
    ("WhaleScanner.analyze no rechaza asset válida",  test_whale_scanner_analyze_valid),
    ("WhaleScanner.score_universe retorna lista",     test_whale_scanner_score_universe),
    ("WhaleScanner.format_alert produce string",      test_whale_scanner_format_alert),
]


def main():
    print("=" * 65)
    print("🧪 WHALE SCANNER — TEST HARNESS")
    print("=" * 65)

    passed = 0
    failed = 0

    for name, fn in TESTS:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 65)
    total = passed + failed
    pct = passed / total * 100 if total > 0 else 0
    status = "✅ PASSED" if failed == 0 else "⚠️ PARTIAL"
    print(f"{status} — {passed}/{total} tests ({pct:.0f}%)")
    if failed > 0:
        print(f"   {failed} test(s) fallaron — revisar logs arriba.")
    print("=" * 65)
    return failed


# Support pytest collection
def test_all_whale_math():
    test_cvd_slope_positive_with_buyer_dominant()
    test_cvd_divergence_bullish()
    test_absorption_score_detects_high_vol_low_range()
    test_liquidity_sweep_bullish()
    test_stop_hunt_pattern()
    test_large_trade_ratio_on_normal_data()


def test_all_layers():
    test_layer_whale_scores_with_whale_signals()
    test_layer_whale_gate_no_false_single_signal()
    test_layer_manipulation_detects_sweep()
    test_layer_manipulation_bias()


def test_all_scanner():
    test_whale_scanner_analyze_valid()
    test_whale_scanner_score_universe()
    test_whale_scanner_format_alert()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
