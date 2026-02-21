"""
Diagnostic: Show what the scanner is picking and how the strategy uses it.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nascent_scanner.scanner_anomaly import AnomalyScanner
from nascent_scanner.backtest_realistic import (
    calculate_indicators, determine_direction, confirm_entry,
    HISTORY_NEEDED, WARMUP_DAYS, CANDLES_PER_DAY, TOP_N
)

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
    
    # Test period
    lengths = [len(df) for df in pair_data.values()]
    median_len = int(np.median(lengths))
    warmup_candles = WARMUP_DAYS * CANDLES_PER_DAY + HISTORY_NEEDED
    test_start_idx = warmup_candles
    test_days = (median_len - test_start_idx) // CANDLES_PER_DAY
    
    # Show scanner picks for first 5 days
    for day_num in range(min(5, test_days)):
        day_start_idx = test_start_idx + day_num * CANDLES_PER_DAY
        
        try:
            sample_df = list(pair_data.values())[0]
            day_label = str(sample_df.iloc[day_start_idx]['timestamp'])[:10]
        except:
            day_label = f"Day {day_num+1}"
        
        picks = scanner.score_universe(pair_data, day_start_idx, top_n=TOP_N)
        
        print(f"{'='*70}")
        print(f"DAY {day_num+1} | {day_label} | Scanner picks: {len(picks)}")
        print(f"{'='*70}")
        print(f"{'#':>2} | {'Symbol':<15} | {'Scanner':>8} | {'Score':>5} | {'Strategy':>10} | {'Entry?':>6} | Reasons")
        print(f"{'-'*2}-+-{'-'*15}-+-{'-'*8}-+-{'-'*5}-+-{'-'*10}-+-{'-'*6}-+-{'-'*30}")
        
        for i, pick in enumerate(picks):
            symbol = pick['symbol']
            scanner_dir = pick.get('direction', '?')
            scanner_score = pick.get('score', 0)
            reasons = pick.get('reasons', [])
            
            # What does the strategy say?
            strategy_dir = '---'
            entry_ok = '---'
            if symbol in pair_data_ind:
                df = pair_data_ind[symbol]
                if day_start_idx < len(df):
                    df_slice = df.iloc[:day_start_idx + 1]
                    strategy_dir = determine_direction(df_slice) or 'NONE'
                    if strategy_dir != 'NONE':
                        entry_ok = 'YES' if confirm_entry(df_slice, strategy_dir) else 'WAIT'
            
            # Check alignment
            aligned = ''
            if scanner_dir == strategy_dir:
                aligned = ' [ALIGNED]'
            elif strategy_dir == 'NONE':
                aligned = ' [SKIP]'
            else:
                aligned = ' [CONFLICT]'
            
            reason_str = ', '.join(reasons[:3]) if reasons else '?'
            print(f"{i+1:2d} | {symbol:<15} | {scanner_dir:>8} | {scanner_score:5.0f} | {strategy_dir:>10} | {entry_ok:>6} | {reason_str}{aligned}")
        
        print()

if __name__ == "__main__":
    main()
