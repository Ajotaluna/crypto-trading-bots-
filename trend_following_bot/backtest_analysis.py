import pandas as pd
import ccxt
import time
import os
from patterns_v2 import PatternDetector
from technical_analysis import TechnicalAnalysis

# CONFIGURATION
# TEST MODE: Major Strategy Refinement (Strict Isolation)
# CONFIGURATION
# STRATEGY LISTS (Segregated by Logic)

# STRATEGY 1: MAJOR REVERSION
# STRATEGY 1: MAJOR REVERSION
# Target: High Cap, Mean Reversion.
LIST_MAJORS = [
    'BTCUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT'
]

# STRATEGY 2: GRINDER
# Target: Trend Following (High Vol or Recovering Alts)
# VIP GRINDERS: Proven history of clean trends
LIST_GRINDERS = [
    'TRXUSDT', 'SUIUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'APTUSDT', 'ETHUSDT'
]

# STRATEGY 3: SCALPER
# Target: High Volatility, Panic Reversals (Pinbar)
LIST_SCALPERS = [
    # GROUP A: PROVEN WINNERS
    'WIFUSDT', 'FLOKIUSDT', 'PEPEUSDT', 
    # GROUP B: NEW PROVEN (Strict RSI)
    'BONKUSDT', 'BOMEUSDT', 'ORDIUSDT', '1000SATSUSDT', 'MEMEUSDT',
    # GROUP C: MACD RECOVERY
    'DOGEUSDT', 'SHIBUSDT'
]

INTERVAL = '15m'
LIMIT = 6000 # ~62 Days (15m)

def fetch_data(symbol, interval='15m'):
    """ Fetches historical data from Binance via CCXT with Local Caching """
    
    # Adjust Limit for 1m data to cover same period
    # 6000 candles * 15m = 90,000 minutes
    current_limit = LIMIT if interval == '15m' else (LIMIT * 16) 
    
    DATA_DIR = "backtest_data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    cache_file = os.path.join(DATA_DIR, f"{symbol}_{interval}_{current_limit}.csv")
    
    # 1. OPTIMIZATION: Check Cache
    if os.path.exists(cache_file):
        # print(f"DEBUG: Loading {symbol} {interval} from cache...")
        df = pd.read_csv(cache_file)
        df['time'] = pd.to_datetime(df['time'])
        return df
        
    exchange = ccxt.binance()
    all_ohlcv = []
    
    # Calculate start time
    duration_ms = current_limit * (15 if interval == '15m' else 1) * 60 * 1000
    since = exchange.milliseconds() - duration_ms
    
    print(f"Downloading {interval} data for {symbol} (High-Res)...")
    
    try:
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=1000, since=since)
            if not ohlcv: break
            
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            since = last_ts + 1
            
            if len(all_ohlcv) >= current_limit: break
            if len(ohlcv) < 1000: break
            time.sleep(0.5)
            
        if len(all_ohlcv) > current_limit: all_ohlcv = all_ohlcv[-current_limit:]
        
        df = pd.DataFrame(all_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['symbol'] = symbol
        
        # 2. SAVE TO CACHE
        df.to_csv(cache_file, index=False)
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def run_backtest(symbol):
    """
    PHASE 2: TRADING SIMULATION
    Executes the strategies based on the regime detected candle-by-candle.
    """
    # DATA LOADING
    df_15m = fetch_data(symbol, interval='15m')
    if df_15m is None: return None
    
    # OPTION: Load 1m data for High-Res Check?
    # For speed, we will load it but index it efficiently.
    df_1m = fetch_data(symbol, interval='1m')
    if df_1m is None: return None
    
    # Index 1m data by time for O(1) lookup or fast slicing
    df_1m.set_index('time', inplace=True)
    df_1m.sort_index(inplace=True)
    
    detector = PatternDetector()
    
    # SIMULATION VARIABLES
    initial_balance = 1000.0
    current_balance = initial_balance
    risk_per_trade = 0.02
    
    trades = [] 
    trades_log = [] 
    log_details = [] 
    active_trade = None 
    
    # DATA SIZING: DYNAMIC CONFIGURATION
    required_limit = PatternDetector.get_candle_limit(symbol)
    warmup = required_limit
    window = required_limit
    
    if len(df_15m) < warmup: return None
    
    # SIMULATION LOOP (Iterate over 15m candles)
    for i in range(warmup, len(df_15m)):
        # Current 15m Context
        start_idx = max(0, i - window) 
        subset = df_15m.iloc[start_idx:i+1].copy() 
        curr_price = subset['close'].iloc[-1]
        curr_time = subset['time'].iloc[-1]
        
        # 1. MANAGE ACTIVE POSITIONS (HIGH RES CHECK)
        if active_trade:
            entry = active_trade['entry']
            tp = active_trade['tp']
            sl = active_trade['sl']
            size = active_trade['size']
            
            end_time = curr_time + pd.Timedelta(minutes=15)
            
            # Safe slice using searchsorted logic or simple loc if Index is datetime
            try:
                # Get minute candles for this period
                minute_candles = df_1m.loc[curr_time:end_time]
            except KeyError:
                minute_candles = pd.DataFrame() # No data found for this period
            
            if not minute_candles.empty:
                # Minute-by-Minute Replay
                trade_closed = False
                for t_idx, row in minute_candles.iterrows():
                    # Check Low vs SL
                    if row['low'] <= sl:
                        loss_usdt = (sl - entry) * size
                        current_balance += loss_usdt
                        trades.append(loss_usdt)
                        trades_log.append('LOSS')
                        log_details.append(active_trade)
                        active_trade = None
                        trade_closed = True
                        break # Stop checking minutes
                    
                    # Check High vs TP
                    if row['high'] >= tp:
                        profit_usdt = (tp - entry) * size
                        current_balance += profit_usdt
                        trades.append(profit_usdt)
                        trades_log.append('WIN')
                        log_details.append(active_trade)
                        active_trade = None
                        trade_closed = True
                        break # Stop checking minutes
                
                if trade_closed: continue # Trade done, move to next 15m candle logic
        
        # 2. LOOK FOR NEW ENTRIES
        # (Only if no active trade)
        if active_trade is None:
            signal = detector.analyze(subset, symbol=symbol)
            
            if signal:
                atr = subset['atr'].iloc[-1] if 'atr' in subset.columns else (curr_price * 0.01)
                
                risk_amount = current_balance * risk_per_trade
                stop_distance = atr * 2.0
                if stop_distance == 0: continue
                
                position_size = risk_amount / stop_distance
                # Major Strat: 1.5 R/R? Or just Trail? 
                take_profit = curr_price + (atr * 3.0) 
                stop_loss = curr_price - stop_distance
                
                active_trade = {
                    'entry': curr_price,
                    'tp': take_profit,
                    'sl': stop_loss,
                    'size': position_size,
                    'time': curr_time,
                    'type': signal['type'],
                    'strategy': signal.get('strategy', 'UNKNOWN')
                }
            
    # RESULTS REPORT
    wins = trades_log.count('WIN')
    losses = trades_log.count('LOSS')
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    net_profit = current_balance - initial_balance
    
    # Strategy Mix Claculation
    strat_list = [t.get('strategy', 'UNK') for t in log_details]
    if not strat_list:
        main_strat = "NONE"
    else:
        # returns mostly used strategy
        from collections import Counter
        c = Counter(strat_list)
        main_strat = c.most_common(1)[0][0]
    
    # Volume Calculation
    total_vol_usdt = (df_15m['volume'] * df_15m['close']).sum()
    days = (df_15m['time'].iloc[-1] - df_15m['time'].iloc[0]).total_seconds() / 86400
    daily_vol_m = (total_vol_usdt / days) / 1_000_000 if days > 0 else 0
    
    return {
        'symbol': symbol,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'wr': win_rate,
        'pnl_usd': net_profit,
        'final_bal': current_balance,
        'vol_m': daily_vol_m,
        'strategy': main_strat
    }

if __name__ == "__main__":
    
    grand_total_profit = 0
    
    # Helper to run a batch
    def run_batch(name, pair_list):
        print(f"\n=== STRATEGY GROUP: {name} ===")
        print(f"{'SYMBOL':<10} | {'VOL (M)':<10} | {'STRATEGY':<15} | {'TRADES':<8} | {'W/L':<8} | {'WR%':<8} | {'PROFIT $':<12} | {'FINAL BAL':<12}")
        print("-" * 125)
        
        batch_profit = 0
        batch_trades = 0
        batch_wins = 0
        
        for pair in pair_list:
            stats = run_backtest(pair)
            if stats:
                batch_profit += stats['pnl_usd']
                batch_trades += stats['trades']
                batch_wins += stats['wins']
                
                wl_str = f"{stats['wins']}/{stats['losses']}"
                print(f"{stats['symbol']:<10} | ${stats['vol_m']:>8.1f}M | {stats['strategy']:<15} | {stats['trades']:>8} | {wl_str:>8} | {stats['wr']:>7.1f}% | ${stats['pnl_usd']:>10.2f} | ${stats['final_bal']:>10.2f}")
        
        avg_wr = (batch_wins / batch_trades * 100) if batch_trades > 0 else 0
        print("-" * 125)
        print(f"{name} NET PROFIT: ${batch_profit:.2f} | AVG WR: {avg_wr:.1f}%")
        return batch_profit

    # EXECUTE BATCHES
    # FULL PORTFOLIO TEST (Production Validation)
    grand_total_profit += run_batch("MAJORS", LIST_MAJORS)
    grand_total_profit += run_batch("GRINDERS", LIST_GRINDERS)
    grand_total_profit += run_batch("SCALPERS", LIST_SCALPERS)

    print("\n" + "="*50)
    print(f"SCALPER STRATEGY NET PROFIT: ${grand_total_profit:.2f}")
    print("="*50 + "\n")
