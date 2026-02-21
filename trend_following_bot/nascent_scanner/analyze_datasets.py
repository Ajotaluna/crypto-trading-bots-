import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROD = os.path.join(BASE_DIR, "data")
DATA_TEST = os.path.join(BASE_DIR, "data_monthly")

def get_pairs(folder):
    files = [f for f in os.listdir(folder) if f.endswith("_15m.csv")]
    return sorted(files)

def analyze_file_metrics(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty or len(df) < 96: return None
        
        # Last 5 days (480 candles) for recent profile
        df = df.tail(480).copy()
        
        # Est. Daily Volume (USDT)
        # Volume column in these CSVs is usually base asset volume? 
        # But let's assume close * volume = notional
        if 'close' not in df.columns or 'volume' not in df.columns: return None
        
        df['notional'] = df['close'] * df['volume']
        avg_daily_vol = df['notional'].sum() / 5 # 5 days
        
        # Volatility (ATR % of Price)
        df['tr'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))))
        df['atr_pct'] = (df['tr'] / df['close']) * 100
        avg_volatility = df['atr_pct'].mean()
        
        return avg_daily_vol, avg_volatility
    except Exception as e:
        return None

def analyze_group(file_list, folder_name, label):
    print(f"\nScanning {label} ({len(file_list)} files)...")
    vols = []
    vola = []
    
    for f in file_list:
        path = os.path.join(folder_name, f)
        res = analyze_file_metrics(path)
        if res:
            v, a = res
            vols.append(v)
            vola.append(a)
    
    if not vols: return 0, 0
    
    avg_vol_group = np.mean(vols)
    avg_vola_group = np.mean(vola)
    
    print(f"--- {label} RESULTS ---")
    print(f"Avg Daily Volume: ${avg_vol_group:,.0f}")
    print(f"Avg Volatility (ATR%): {avg_vola_group:.2f}%")
    return avg_vol_group, avg_vola_group

def main():
    prod_files = set(get_pairs(DATA_PROD))
    test_files = set(get_pairs(DATA_TEST))
    
    common = prod_files.intersection(test_files)
    only_prod = prod_files - test_files
    
    print(f"Production Total: {len(prod_files)}")
    print(f"Test Total: {len(test_files)}")
    print(f"Common Pairs: {len(common)}")
    print(f"Only in Production (The 'Extra' Risk): {len(only_prod)}")
    
    # Analyze Test Group (The "Good" Results)
    analyze_group(list(test_files), DATA_TEST, "TEST DATASET (data_monthly)")
    
    # Analyze The Extra Pairs (The "Bad" Results?)
    # We must find them in PROD folder
    analyze_group(list(only_prod), DATA_PROD, "EXTRA PAIRS (Only in Production)")

if __name__ == "__main__":
    main()
