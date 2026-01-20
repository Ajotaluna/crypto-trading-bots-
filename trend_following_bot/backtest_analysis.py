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
# CONFIGURATION
# Set to TRUE to scan the entire market
SCAN_ALL_MARKET = True
Top_Limit = None 
MIN_DAILY_VOLUME = 10_000_000 # Filter: Only pairs with > $10M Volume

# 20 Selected Pairs (Fallback/Reference)
PAIRS = [
    'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'LINKUSDT', 'TRXUSDT', 'ETCUSDT', 'VETUSDT', 
    'ONTUSDT', 'HOTUSDT', 'ZILUSDT', 'ZRXUSDT', 'FETUSDT', 
    'BATUSDT', 'ZECUSDT', 'THETAUSDT', 'ATOMUSDT', 'ALGOUSDT'
]

def get_all_usdt_pairs(limit=None):
    """ Fetches all USDT pairs with volume > MIN_DAILY_VOLUME """
    try:
        exchange = ccxt.binance()
        # Fetch detailed ticker data (includes volume)
        print("⏳ Fetching 24h Tickers to filter by Volume...")
        tickers = exchange.fetch_tickers()
        
        usdt_pairs = []
        
        for symbol, data in tickers.items():
            if symbol.endswith('/USDT'):
                # Filter out Leveraged Tokens
                base = symbol.split('/')[0]
                if any(x in base for x in ['UP', 'DOWN', 'BEAR', 'BULL']):
                    continue
                    
                # VOLUME FILTER
                vol_usdt = data.get('quoteVolume', 0)
                if vol_usdt < MIN_DAILY_VOLUME:
                    continue
                    
                clean_symbol = symbol.replace('/', '')
                usdt_pairs.append((clean_symbol, vol_usdt))
        
        # Sort by Volume (Descending)
        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the symbols
        final_list = [x[0] for x in usdt_pairs]
        
        print(f"✅ Found {len(final_list)} High-Volume Pairs (> ${MIN_DAILY_VOLUME/1_000_000:.0f}M).")
        
        if limit:
            print(f"⚠️ Limiting scan to Top {limit} for speed.")
            return final_list[:limit]
            
        return final_list
        
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
    
    print(f"\n{'PAIR':<12} | {'WINS':<6} | {'LOSS':<6} | {'WR%':<6} | {'PnL (R)':<6} | {'PnL ($)':<8}")
    print("-" * 65)
    
    total_wins = 0
    total_losses = 0
    results = {} # Initialize results dictionary
    
    for symbol in PAIRS:
        df = fetch_klines(symbol, INTERVAL, start_ts, end_ts)
        if len(df) < 500: continue
        
        wins = 0
        losses = 0
        total_pnl_r = 0.0 # Track Net Profit in Risk Units (R)
        
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
                
                # Calculate Risk/Reward Ratio for this specific trade
                entry_price = current_slice['close'].iloc[-1]
                risk = abs(entry_price - sl)
                reward = abs(tp - entry_price)
                if risk == 0: continue
                rr_ratio = reward / risk
                
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
                
                if outcome == 'WIN': 
                    wins += 1
                    total_pnl_r += rr_ratio # Add the specific R multiple (e.g., +1.5 or +1.0)
                elif outcome == 'LOSS': 
                    losses += 1
                    total_pnl_r -= 1.0 # Standard Loss is -1R
        
        # Result for Symbol
        total = wins + losses
        wr = (wins/total*100) if total > 0 else 0
        
        # Calculate Simulated Profit
        pnl_usdt = total_pnl_r * RISK_AMOUNT
        
        print(f"{symbol:<12} | {wins:<6} | {losses:<6} | {wr:<6.1f}% | {total_pnl_r:<6.1f}R | ${pnl_usdt:+.2f}")
        
        total_wins += wins
        total_losses += losses
        # Store results for CSV report
        results[symbol] = {'wins': wins, 'losses': losses, 'total': wins + losses, 'pnl_r': total_pnl_r, 'pnl_usdt': pnl_usdt}
        
    print("-" * 65)
    grand_total = total_wins + total_losses
    grand_wr = (total_wins/grand_total*100) if grand_total > 0 else 0
    
    # Portfolio Stats
    total_portfolio_r = sum(r['pnl_r'] for r in results.values())
    total_profit_usdt = total_portfolio_r * RISK_AMOUNT
    final_balance = INITIAL_BALANCE + total_profit_usdt
    roi_pct = (total_profit_usdt / INITIAL_BALANCE) * 100
    
    print(f"TOTAL TRADES: {grand_total}")
    print(f"GLOBAL WIN RATE: {grand_wr:.1f}%")
    print("-" * 65)
    print(f"INITIAL BALANCE: ${INITIAL_BALANCE:.2f}")
    print(f"RISK PER TRADE:  ${RISK_AMOUNT:.2f} ({RISK_PER_TRADE_PCT}%)")
    print(f"NET PROFIT:      ${total_profit_usdt:+.2f}")
    print(f"FINAL BALANCE:   ${final_balance:.2f}")
    print(f"ROI:             {roi_pct:+.2f}%")
    print("-" * 65)

    # Save to CSV
    csv_file = "market_scan_results.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pair", "Wins", "Losses", "WinRate%", "PnL_R", "Status"])
        
        passed = 0
        failed = 0
        
        for pair, stats in results.items():
            wins = stats['wins']
            losses = stats['losses']
            pnl_r = stats['pnl_r']
            pnl_u = stats['pnl_usdt']
            total = wins + losses
            wr = (wins / total * 100) if total > 0 else 0
            
            # Status based on Profitability, not just WR
            # Pass if PnL is POSITIVE
            status = "PASSED" if pnl_r > 0 else "FAILED"
            if status == "PASSED": passed += 1
            else: failed += 1
            
            writer.writerow([pair, wins, losses, f"{wr:.1f}", f"{pnl_r:.2f}", f"{pnl_u:.2f}", status])
            
    print(f"\n📄 Report saved to: {csv_file}")
    print(f"✅ PASSED (Profitable): {passed}")
    print(f"❌ FAILED (Unprofitable): {failed}")

if __name__ == "__main__":
    run_backtest()
