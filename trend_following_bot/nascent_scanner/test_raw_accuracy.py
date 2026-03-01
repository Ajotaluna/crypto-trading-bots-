"""
Diagnostic tool to measure the raw predictive power of the RawMarketScanner.
It runs the scanner historically and then looks ahead 24-48 hours to see 
the Maximum Favorable Excursion (MFE) of the chosen assets.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nascent_scanner.scanner_raw import RawMarketScanner

def evaluate_scanner_accuracy():
    print("=" * 60)
    print(" RAW SCANNER PREDICTIVE ACCURACY DIAGNOSTIC ")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(__file__), "data_monthly")
    if not os.path.exists(data_dir):
        print("Data directory 'data_monthly' not found.")
        return

    # 1. Load Data
    files = [f for f in os.listdir(data_dir) if f.endswith("_15m.csv")]
    raw_data = {}
    all_timestamps = set()
    
    print(f"Loading {len(files)} pairs from data_monthly...")
    for f in files:
        symbol = f.replace("_15m.csv", "")
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            if len(df) < 500: continue # Need enough history
            raw_data[symbol] = df
            all_timestamps.update(df['timestamp'])
        except:
            continue

    timeline = sorted(list(all_timestamps))
    full_index = pd.DatetimeIndex(timeline)
    
    aligned_data = {}
    for symbol, df in raw_data.items():
        df_reindexed = df.set_index('timestamp').reindex(full_index).reset_index()
        df_reindexed.rename(columns={'index': 'timestamp'}, inplace=True)
        aligned_data[symbol] = df_reindexed

    scanner = RawMarketScanner()
    
    # 2. Run Scan Once a Day and Check Forward Returns
    # Skip first 7 days for history
    start_idx = 96 * 7 
    
    total_picks = 0
    success_picks_2_pct = 0
    success_picks_5_pct = 0
    
    print("\nSimulating Daily Scans and Forward Looking 24h...")
    
    # We step every 1 day (96 candles)
    for t_idx in range(start_idx, len(timeline) - 96, 96):
        current_time = timeline[t_idx]
        
        # Prepare data slice up to current time
        eligible_pairs = {}
        for s, df in aligned_data.items():
            slice_df = df.iloc[:t_idx+1].dropna(subset=['close'])
            if len(slice_df) > 96:
                eligible_pairs[s] = slice_df
                
        # Run Scanner
        picks = scanner.score_universe(eligible_pairs, -1, top_n=5)
        
        if not picks:
            continue
            
        print(f"\n--- {str(current_time)[:10]} ---")
        
        for pick in picks:
            symbol = pick['symbol']
            direction = pick['direction']
            score = pick['score']
            
            # Forward look 24 hours (next 96 candles)
            future_df = aligned_data[symbol].iloc[t_idx+1 : t_idx+97]
            if future_df['close'].isna().all():
                continue
                
            entry_price = float(aligned_data[symbol].iloc[t_idx]['close'])
            if pd.isna(entry_price): continue
            
            # Calculate Max Favorable Excursion (MFE)
            if direction == 'LONG':
                max_price = future_df['high'].max()
                max_gain_pct = ((max_price - entry_price) / entry_price) * 100
            else:
                min_price = future_df['low'].min()
                max_gain_pct = ((entry_price - min_price) / entry_price) * 100
                
            total_picks += 1
            if max_gain_pct >= 2.0: success_picks_2_pct += 1
            if max_gain_pct >= 5.0: success_picks_5_pct += 1
                
            highlight = "🔥 DUMP/PUMP CAUGHT!" if max_gain_pct >= 5.0 else ""
            print(f"Picked {symbol} ({direction}) | Score: {score} | Max 24h Move: +{max_gain_pct:.2f}% {highlight}")

    if total_picks > 0:
        print("\n" + "="*40)
        print(" ACCURACY SUMMARY ")
        print("="*40)
        print(f"Total Top Picks Evaluated: {total_picks}")
        print(f"Picks that moved > 2% in intended direction: {success_picks_2_pct} ({(success_picks_2_pct/total_picks)*100:.1f}%)")
        print(f"Picks that moved > 5% in intended direction: {success_picks_5_pct} ({(success_picks_5_pct/total_picks)*100:.1f}%)")

if __name__ == "__main__":
    evaluate_scanner_accuracy()
