"""
TREND X-RAY — Radiografia de tendencias detectadas por el scanner.

Para cada señal del scanner, rastrea qué pasó en las siguientes 48h:
- Max Favorable Excursion (MFE): Hasta dónde llegó a nuestro favor
- Max Adverse Excursion (MAE): Cuánto fue en contra antes de moverse
- Peak Time: Cuánto tardó en llegar al máximo
- Pullback Profile: Qué tan profundo retrocede en el camino
- Trend Duration: Cuánto tiempo dura la tendencia

Esto nos dice EXACTAMENTE cómo se comportan las tendencias nacientes
para diseñar estrategias basadas en datos reales.
"""
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nascent_scanner.scanner_anomaly import AnomalyScanner
from nascent_scanner.backtest_realistic import (
    calculate_indicators, HISTORY_NEEDED, WARMUP_DAYS, CANDLES_PER_DAY, TOP_N
)

# How many candles to track after signal (48h = 192 candles at 15m)
TRACK_CANDLES = 192
# Track at these time intervals (in candles)
CHECKPOINTS = [4, 8, 16, 32, 48, 96, 192]  # 1h, 2h, 4h, 8h, 12h, 24h, 48h
CHECKPOINT_LABELS = ['1h', '2h', '4h', '8h', '12h', '24h', '48h']


def analyze_trend(df, start_idx, direction, track_candles=TRACK_CANDLES):
    """
    Analyze what happens to a pair from start_idx over the next track_candles.
    Returns a dict with detailed trend profile.
    """
    end_idx = min(start_idx + track_candles, len(df))
    if end_idx - start_idx < 8:
        return None

    entry_price = float(df.iloc[start_idx]['close'])
    atr_at_entry = float(df.iloc[start_idx]['atr'])
    if atr_at_entry <= 0:
        atr_at_entry = entry_price * 0.02
    
    # Track price movement candle by candle
    max_favorable = 0      # Best move in our direction (ATR)
    max_adverse = 0        # Worst move against us (ATR)
    peak_candle = 0        # When max favorable occurred
    trough_candle = 0      # When max adverse occurred
    
    # Pullback tracking
    pullbacks = []         # List of (depth_atr, recovery_candles)
    current_best_favorable = 0
    in_pullback = False
    pullback_start_favorable = 0
    
    # Checkpoint data
    checkpoint_data = {}
    
    # Time above breakeven
    candles_in_profit = 0
    
    for i in range(start_idx, end_idx):
        candle = df.iloc[i]
        c_high = float(candle['high'])
        c_low = float(candle['low'])
        c_close = float(candle['close'])
        offset = i - start_idx
        
        if direction == 'LONG':
            favorable = (c_high - entry_price) / atr_at_entry
            adverse = (entry_price - c_low) / atr_at_entry
            current_pnl = (c_close - entry_price) / atr_at_entry
        else:
            favorable = (entry_price - c_low) / atr_at_entry
            adverse = (c_high - entry_price) / atr_at_entry
            current_pnl = (entry_price - c_close) / atr_at_entry
        
        # Update max favorable/adverse
        if favorable > max_favorable:
            max_favorable = favorable
            peak_candle = offset
        if adverse > max_adverse:
            max_adverse = adverse
            trough_candle = offset
        
        # Count time in profit
        if current_pnl > 0:
            candles_in_profit += 1
        
        # Pullback detection
        if favorable > current_best_favorable:
            current_best_favorable = favorable
            if in_pullback:
                # Recovered from pullback
                pullback_depth = pullback_start_favorable - favorable
                pullbacks.append(pullback_depth)
                in_pullback = False
        elif current_best_favorable > 0.5 and favorable < current_best_favorable - 0.3:
            if not in_pullback:
                in_pullback = True
                pullback_start_favorable = current_best_favorable
        
        # Checkpoint recording
        if offset + 1 in CHECKPOINTS:
            idx = CHECKPOINTS.index(offset + 1)
            checkpoint_data[CHECKPOINT_LABELS[idx]] = {
                'pnl_atr': current_pnl,
                'mfe_atr': max_favorable,
                'mae_atr': max_adverse,
                'pnl_pct': ((c_close - entry_price) / entry_price * 100) if direction == 'LONG' 
                           else ((entry_price - c_close) / entry_price * 100),
            }
    
    total_candles = end_idx - start_idx
    
    return {
        'mfe_atr': max_favorable,
        'mae_atr': max_adverse,
        'peak_candle': peak_candle,
        'peak_time_h': peak_candle * 0.25,  # Each candle = 15min
        'trough_candle': trough_candle,
        'trough_time_h': trough_candle * 0.25,
        'pnl_at_end_atr': checkpoint_data.get(CHECKPOINT_LABELS[-1], {}).get('pnl_atr', 0),
        'time_in_profit_pct': (candles_in_profit / total_candles * 100) if total_candles > 0 else 0,
        'num_pullbacks': len(pullbacks),
        'avg_pullback_atr': np.mean(pullbacks) if pullbacks else 0,
        'checkpoints': checkpoint_data,
        'direction': direction,
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data_monthly")
    scanner = AnomalyScanner()
    
    # Load data
    pair_data = {}
    files = [f for f in os.listdir(data_dir) if f.endswith("_15m.csv")]
    for f in files:
        symbol = f.replace("_15m.csv", "")
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            if len(df) >= 1488:
                pair_data[symbol] = df
        except:
            continue
    print(f"Loaded {len(pair_data)} pairs\n")
    
    # Pre-calc indicators
    pair_data_ind = {}
    for symbol, df in pair_data.items():
        try:
            pair_data_ind[symbol] = calculate_indicators(df)
        except:
            continue
    
    # Test period setup
    lengths = [len(df) for df in pair_data.values()]
    median_len = int(np.median(lengths))
    warmup_candles = WARMUP_DAYS * CANDLES_PER_DAY + HISTORY_NEEDED
    test_start_idx = warmup_candles
    test_days = (median_len - test_start_idx - TRACK_CANDLES) // CANDLES_PER_DAY
    
    print(f"Analyzing {test_days} days of scanner picks...\n")
    
    all_profiles = []
    long_profiles = []
    short_profiles = []
    
    for day_num in range(test_days):
        day_start_idx = test_start_idx + day_num * CANDLES_PER_DAY
        
        picks = scanner.score_universe(pair_data, day_start_idx, top_n=TOP_N)
        
        for pick in picks:
            symbol = pick['symbol']
            direction = pick['direction']
            score = pick.get('score', 0)
            
            if symbol not in pair_data_ind:
                continue
            
            df = pair_data_ind[symbol]
            if day_start_idx + TRACK_CANDLES >= len(df):
                continue
            
            profile = analyze_trend(df, day_start_idx, direction, TRACK_CANDLES)
            if profile is None:
                continue
            
            profile['symbol'] = symbol
            profile['score'] = score
            profile['day'] = day_num + 1
            
            all_profiles.append(profile)
            if direction == 'LONG':
                long_profiles.append(profile)
            else:
                short_profiles.append(profile)
    
    print(f"Total signals analyzed: {len(all_profiles)} ({len(long_profiles)}L / {len(short_profiles)}S)")
    print()
    
    # ================================================================
    # REPORT 1: Overall Trend Behavior
    # ================================================================
    print("=" * 70)
    print("RADIOGRAFIA DE TENDENCIAS NACIENTES")
    print("=" * 70)
    
    for label, profiles in [("ALL", all_profiles), ("LONG", long_profiles), ("SHORT", short_profiles)]:
        if not profiles:
            continue
        mfes = [p['mfe_atr'] for p in profiles]
        maes = [p['mae_atr'] for p in profiles]
        peaks = [p['peak_time_h'] for p in profiles]
        troughs = [p['trough_time_h'] for p in profiles]
        profit_pcts = [p['time_in_profit_pct'] for p in profiles]
        
        print(f"\n--- {label} ({len(profiles)} signals) ---")
        print(f"{'Metric':<30} | {'Mean':>8} | {'Median':>8} | {'P25':>8} | {'P75':>8} | {'P90':>8}")
        print(f"{'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        
        for name, data in [
            ("MFE (ATR)", mfes),
            ("MAE (ATR)", maes),
            ("Peak Time (hours)", peaks),
            ("Trough Time (hours)", troughs),
            ("Time in Profit (%)", profit_pcts),
        ]:
            arr = np.array(data)
            print(f"{name:<30} | {np.mean(arr):8.2f} | {np.median(arr):8.2f} | "
                  f"{np.percentile(arr, 25):8.2f} | {np.percentile(arr, 75):8.2f} | "
                  f"{np.percentile(arr, 90):8.2f}")
    
    # ================================================================
    # REPORT 2: Time Profile — How trends evolve at each checkpoint
    # ================================================================
    print(f"\n{'=' * 70}")
    print("PERFIL TEMPORAL — Cómo evoluciona la tendencia promedio")
    print(f"{'=' * 70}")
    
    for label, profiles in [("ALL", all_profiles), ("LONG", long_profiles), ("SHORT", short_profiles)]:
        if not profiles:
            continue
        print(f"\n--- {label} ---")
        print(f"{'Time':>6} | {'Avg P&L':>10} | {'Med P&L':>10} | {'Avg MFE':>10} | {'Avg MAE':>10} | {'% Profitable':>12}")
        print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
        
        for cp_label in CHECKPOINT_LABELS:
            pnls = [p['checkpoints'].get(cp_label, {}).get('pnl_atr', 0) for p in profiles if cp_label in p['checkpoints']]
            mfes = [p['checkpoints'].get(cp_label, {}).get('mfe_atr', 0) for p in profiles if cp_label in p['checkpoints']]
            maes = [p['checkpoints'].get(cp_label, {}).get('mae_atr', 0) for p in profiles if cp_label in p['checkpoints']]
            
            if not pnls:
                continue
            
            profitable = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(f"{cp_label:>6} | {np.mean(pnls):>8.2f}R | {np.median(pnls):>8.2f}R | "
                  f"{np.mean(mfes):>8.2f}R | {np.mean(maes):>8.2f}R | {profitable:>10.1f}%")
    
    # ================================================================
    # REPORT 3: MFE Distribution — Where to set TP/Trail
    # ================================================================
    print(f"\n{'=' * 70}")
    print("DISTRIBUCION MFE — Cuantas tendencias alcanzan cada nivel")
    print(f"{'=' * 70}")
    
    for label, profiles in [("ALL", all_profiles), ("LONG", long_profiles), ("SHORT", short_profiles)]:
        if not profiles:
            continue
        mfes = [p['mfe_atr'] for p in profiles]
        arr = np.array(mfes)
        
        print(f"\n--- {label} ---")
        for level in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            pct = (arr >= level).sum() / len(arr) * 100
            bar = '#' * int(pct / 2)
            print(f"  MFE >= {level:>4.1f} ATR: {pct:5.1f}% {bar}")
    
    # ================================================================
    # REPORT 4: MAE Distribution — Where to set SL
    # ================================================================
    print(f"\n{'=' * 70}")
    print("DISTRIBUCION MAE — Cuanta adversidad soportan las tendencias")
    print(f"{'=' * 70}")
    
    for label, profiles in [("ALL", all_profiles), ("LONG", long_profiles), ("SHORT", short_profiles)]:
        if not profiles:
            continue
        
        # Only look at trends that eventually were profitable (MFE > 1 ATR)
        winning = [p for p in profiles if p['mfe_atr'] >= 1.0]
        losing = [p for p in profiles if p['mfe_atr'] < 1.0]
        
        print(f"\n--- {label} ---")
        print(f"  Winners (MFE>=1ATR): {len(winning)} signals ({len(winning)/len(profiles)*100:.0f}%)")
        print(f"  Losers  (MFE<1ATR):  {len(losing)} signals ({len(losing)/len(profiles)*100:.0f}%)")
        
        if winning:
            w_maes = np.array([p['mae_atr'] for p in winning])
            print(f"\n  MAE of WINNERS (how deep they dipped before recovering):")
            for level in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                pct = (w_maes <= level).sum() / len(w_maes) * 100
                bar = '#' * int(pct / 2)
                print(f"    MAE <= {level:>3.1f} ATR: {pct:5.1f}% {bar}")
            print(f"    Avg MAE: {np.mean(w_maes):.2f} ATR | Median: {np.median(w_maes):.2f} ATR")
        
        if losing:
            l_maes = np.array([p['mae_atr'] for p in losing])
            print(f"\n  MAE of LOSERS (confirmation bias — they go against us fast):")
            for level in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                pct = (l_maes >= level).sum() / len(l_maes) * 100
                print(f"    MAE >= {level:>3.1f} ATR: {pct:5.1f}%")
    
    # ================================================================
    # REPORT 5: Score vs Performance — Do higher scores = better trends?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("SCORE vs RENDIMIENTO — Las señales más fuertes son mejores?")
    print(f"{'=' * 70}")
    
    if all_profiles:
        scores = np.array([p['score'] for p in all_profiles])
        # Group by score buckets
        buckets = [(60, 80), (80, 100), (100, 120), (120, 150), (150, 200)]
        
        print(f"\n{'Score Range':>12} | {'Count':>5} | {'Avg MFE':>8} | {'Avg MAE':>8} | {'Avg P&L':>8} | {'Peak@':>6}")
        print(f"{'-'*12}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
        
        for lo, hi in buckets:
            bucket = [p for p in all_profiles if lo <= p['score'] < hi]
            if not bucket:
                continue
            avg_mfe = np.mean([p['mfe_atr'] for p in bucket])
            avg_mae = np.mean([p['mae_atr'] for p in bucket])
            avg_pnl = np.mean([p['pnl_at_end_atr'] for p in bucket])
            avg_peak = np.mean([p['peak_time_h'] for p in bucket])
            print(f"  {lo:3d}-{hi:3d}   | {len(bucket):5d} | {avg_mfe:6.2f}R | {avg_mae:6.2f}R | "
                  f"{avg_pnl:+6.2f}R | {avg_peak:5.1f}h")
    
    # ================================================================
    # REPORT 6: Optimal Strategy Insights
    # ================================================================
    print(f"\n{'=' * 70}")
    print("INSIGHTS PARA ESTRATEGIA OPTIMA")
    print(f"{'=' * 70}")
    
    if all_profiles:
        mfes = np.array([p['mfe_atr'] for p in all_profiles])
        maes = np.array([p['mae_atr'] for p in all_profiles])
        peaks = np.array([p['peak_time_h'] for p in all_profiles])
        
        # Optimal SL: wide enough that most winners survive
        winners = [p for p in all_profiles if p['mfe_atr'] >= 1.5]
        if winners:
            w_maes = np.array([p['mae_atr'] for p in winners])
            optimal_sl = np.percentile(w_maes, 85)
            print(f"\n  SL Optimo: {optimal_sl:.2f} ATR")
            print(f"    (85% de las tendencias ganadoras sobreviven este SL)")
        
        # Optimal entry window
        profitable_1h = sum(1 for p in all_profiles 
                          if p['checkpoints'].get('1h', {}).get('pnl_atr', 0) > 0)
        profitable_2h = sum(1 for p in all_profiles 
                          if p['checkpoints'].get('2h', {}).get('pnl_atr', 0) > 0)
        print(f"\n  Profitable at 1h: {profitable_1h/len(all_profiles)*100:.0f}%")
        print(f"  Profitable at 2h: {profitable_2h/len(all_profiles)*100:.0f}%")
        
        # Trail insights
        print(f"\n  Peak Time (median): {np.median(peaks):.1f}h")
        print(f"  Peak Time (75th): {np.percentile(peaks, 75):.1f}h")
        print(f"  => Tendencias alcanzan su pico en {np.median(peaks):.0f}-{np.percentile(peaks,75):.0f}h")
        
        # MFE insights
        print(f"\n  MFE (median): {np.median(mfes):.2f} ATR")
        print(f"  MFE (75th): {np.percentile(mfes, 75):.2f} ATR")
        print(f"  => Tendencia tipica se mueve {np.median(mfes):.1f}-{np.percentile(mfes,75):.1f} ATR")
        
        # Best approach
        print(f"\n  RECOMENDACION:")
        print(f"    - SL: ~{optimal_sl:.1f} ATR (protege 85% de winners)")
        print(f"    - Trail: ~{np.median(mfes)*0.4:.1f} ATR (40% del MFE tipico)")
        print(f"    - Hold max: ~{np.percentile(peaks, 75):.0f}h (75% ya hicieron peak)")


if __name__ == "__main__":
    main()
