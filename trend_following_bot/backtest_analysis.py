import pandas as pd
import time
import os
import requests
from patterns import PatternDetector
from config import config

import csv
import os
import ccxt # Added missing import

# CONFIGURATION
# Set to TRUE to scan the entire market
SCAN_ALL_MARKET = True
Top_Limit = 50 # Set to None for ALL 500+ pairs (Warning: Takes time)

# Manual List (Used if SCAN_ALL_MARKET = False)
PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 
    'DOGEUSDT', 'ADAUSDT', 'PEPEUSDT', 'SHIBUSDT'
]

def get_all_usdt_pairs(limit=None):
    """ Fetches all USDT pairs from Binance, excluding leveraged tokens """
    try:
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        usdt_pairs = []
        
        for symbol in markets:
            if symbol.endswith('/USDT'):
                # Filter out UP/DOWN/BEAR/BULL leveraged tokens
                base = symbol.split('/')[0]
                if not any(x in base for x in ['UP', 'DOWN', 'BEAR', 'BULL']):
                    # Convert fit/USDT to fitUSDT for compatibility
                    clean_symbol = symbol.replace('/', '')
                    usdt_pairs.append(clean_symbol)
        
        print(f"✅ Found {len(usdt_pairs)} USDT Pairs on Binance.")
        if limit:
            print(f"⚠️ Limiting scan to Top {limit} for speed.")
            return usdt_pairs[:limit]
        return usdt_pairs
    except Exception as e:
        print(f"❌ Error fetching market list: {e}")
        return PAIRS # Fallback

if SCAN_ALL_MARKET:
    PAIRS = get_all_usdt_pairs(Top_Limit)

INTERVAL = '15m' # Align with Strategy Timeframe
START_DATE = '2025-10-01'
END_DATE = '2026-01-15'
BASE_URL = "https://api.binance.com/api/v3/klines"

def get_timestamp(date_str):
    return int(pd.Timestamp(date_str).timestamp() * 1000)

def fetch_klines(symbol, interval, start_ts, end_ts):
    DATA_DIR = "backtest_data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    cache_file = os.path.join(DATA_DIR, f"{symbol}_{interval}_{START_DATE}_{END_DATE}.csv")
    
    if os.path.exists(cache_file):
        print(f"Loading {symbol}...")
        df = pd.read_csv(cache_file)
        df['open_time'] = pd.to_datetime(df['open_time'])
        if interval == '2m': df.set_index('open_time', inplace=True)
        return df

    # Fetch from API logic (Simplified for robustness)
    fetch_interval = '1m'
    all_klines = []
    current_start = start_ts
    
    print(f"Fetching API {symbol}...")
    
    while current_start < end_ts:
        params = {'symbol': symbol, 'interval': fetch_interval, 'startTime': current_start, 'endTime': end_ts, 'limit': 1000}
        try:
            r = requests.get(BASE_URL, params=params).json()
            if not isinstance(r, list): break
            all_klines.extend(r)
            current_start = r[-1][6] + 1
            time.sleep(0.1)
        except: break
            
    df = pd.DataFrame(all_klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qav', 'nt', 'tbba', 'tbqa', 'ig'])
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float, 'tbba': float})
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['taker_buy_vol'] = df['tbba']
    
    if interval == '15m':
        df.set_index('open_time', inplace=True)
        # Ensure no duplicates
        df = df[~df.index.duplicated(keep='first')]
        # Optional: Resample to strictly regular 15m grid
        df = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'taker_buy_vol': 'sum'}).dropna()
        
    df.to_csv(cache_file)
    return df

def run_backtest():
    start_ts = get_timestamp(START_DATE)
    end_ts = get_timestamp(END_DATE)
    detector = PatternDetector()
    
    print(f"\n{'PAIR':<12} | {'WINS':<6} | {'LOSS':<6} | {'WR%':<6}")
    print("-" * 40)
    
    total_wins = 0
    total_losses = 0
    results = {} # Initialize results dictionary
    
    for symbol in PAIRS:
        df = fetch_klines(symbol, INTERVAL, start_ts, end_ts)
        if len(df) < 500: continue
        
        wins = 0
        losses = 0
        
        # Window size for indicators
        window = 1500
        
        for i in range(window, len(df)):
            # Slice concept: strict past data only
            current_slice = df.iloc[i-window : i+1].copy()
            
            # THE TEST: Consensus Engine V4
            try:
                signal = detector.analyze(current_slice, symbol=symbol)
            except Exception as e:
                # print(f"SKIP {symbol} {i}: {e}") # Silence error to speed up
                continue
            
            if signal:
                # Signal details
                direction = signal['direction']
                tp = signal.get('tp_price')
                sl = signal.get('sl_price')
                
                if not tp or not sl: continue
                
                # Verify Logic
                outcome = 'PENDING'
                # Look ahead next 720 candles (24h)
                future = df.iloc[i+1 : i+1+720]
                
                for _, row in future.iterrows():
                    if direction == 'LONG':
                        if row['low'] <= sl: 
                            outcome = 'LOSS'
                            break
                        if row['high'] >= tp: 
                            outcome = 'WIN'
                            break
                    elif direction == 'SHORT':
                        if row['high'] >= sl: 
                            outcome = 'LOSS'
                            break
                        if row['low'] <= tp: 
                            outcome = 'WIN'
                            break
                
                if outcome == 'WIN': wins += 1
                elif outcome == 'LOSS': losses += 1
        
        # Result for Symbol
        total = wins + losses
        wr = (wins/total*100) if total > 0 else 0
        print(f"{symbol:<12} | {wins:<6} | {losses:<6} | {wr:.1f}%")
        
        total_wins += wins
        total_losses += losses
        
        # Store results for CSV report
        results[symbol] = {'wins': wins, 'losses': losses, 'total': wins + losses}
        
    print("-" * 40)
    grand_total = total_wins + total_losses
    grand_wr = (total_wins/grand_total*100) if grand_total > 0 else 0
    print(f"TOTAL: {grand_total} Trades")
    print(f"GLOBAL WIN RATE: {grand_wr:.1f}%")
    print("-" * 40)

    # Save to CSV
    csv_file = "market_scan_results.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pair", "Wins", "Losses", "WinRate%", "Status"])
        
        passed = 0
        failed = 0
        
        for pair, stats in results.items():
            wins = stats['wins']
            losses = stats['losses']
            total = wins + losses
            wr = (wins / total * 100) if total > 0 else 0
            
            status = "PASSED" if wr >= 40 else "FAILED"
            if status == "PASSED": passed += 1
            else: failed += 1
            
            writer.writerow([pair, wins, losses, f"{wr:.1f}", status])
            
    print(f"\n📄 Report saved to: {csv_file}")
    print(f"✅ PASSED (Fits Rule): {passed}")
    print(f"❌ FAILED (Need New Rules): {failed}")

if __name__ == "__main__":
    run_backtest()
