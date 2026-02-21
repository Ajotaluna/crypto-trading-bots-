import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PROD = os.path.join(BASE_DIR, "data", "BTCUSDT_15m.csv")
FILE_TEST = os.path.join(BASE_DIR, "data_monthly", "BTCUSDT_15m.csv")

def compare_file(f1, f2):
    print(f"Comparing...\n1: {f1}\n2: {f2}")
    
    if not os.path.exists(f1) or not os.path.exists(f2):
        print("Files not found.")
        return

    df1 = pd.read_csv(f1)
    df1['timestamp'] = pd.to_datetime(df1['timestamp'])
    df1.set_index('timestamp', inplace=True)
    
    df2 = pd.read_csv(f2)
    df2['timestamp'] = pd.to_datetime(df2['timestamp'])
    df2.set_index('timestamp', inplace=True)
    
    # Check overlap
    common_idx = df1.index.intersection(df2.index)
    if len(common_idx) == 0:
        print("No overlap found.")
        return

    print(f"Common Candles: {len(common_idx)}")
    
    # Check last 5 candles
    last_idx = common_idx[-5:]
    print("\n--- Last 5 Common Candles (Close Price) ---")
    for idx in last_idx:
        c1 = df1.loc[idx]['close']
        c2 = df2.loc[idx]['close']
        diff = c1 - c2
        print(f"{idx} | PROD: {c1:.2f} | TEST: {c2:.2f} | Diff: {diff:.4f}")

    # Check whole overlap differences
    diff_series = (df1.loc[common_idx]['close'] - df2.loc[common_idx]['close']).abs()
    max_diff = diff_series.max()
    print(f"\nMax Close Price Difference across {len(common_idx)} candles: {max_diff:.6f}")
    
    if max_diff > 0.0001:
        print("⚠️  DATA MISMATCH DETECTED!")
    else:
        print("✅ Data is Identical (Price-wise).")

if __name__ == "__main__":
    compare_file(FILE_PROD, FILE_TEST)
