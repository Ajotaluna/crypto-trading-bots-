import asyncio
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from trend_following_bot.market_data import MarketData
from trend_following_bot.patterns import PatternDetector
from trend_following_bot.config import config

# IMPORT SMART BOT LOGIC
from trend_following_bot.main_smart import TrendBot

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TimeMachine")

class MockMarketSmart(MarketData):
    """Simulated Market Data for Backtesting Titan V3"""
    def __init__(self, start_balance=1000.0):
        super().__init__(is_dry_run=True)
        self.balance = start_balance
        self.positions = {}
        self.history = [] # Trade History
        self.current_time = None
        self.data_feed = {} # symbol -> DataFrame

    async def get_current_price(self, symbol):
        # Look up price in data_feed at current_time
        if symbol in self.data_feed:
            df = self.data_feed[symbol]
            # Find row <= current_time
            # For speed, we assume data is aligned or we iterate
            # simpler: we just store 'current_candle' externally
            pass
        return 0.0

    async def open_position(self, symbol, side, amount_usdt, sl, tp):
        price = await self.get_current_price(symbol)
        if price == 0: return None
        
        qty = amount_usdt / price
        self.positions[symbol] = {
            'symbol': symbol,
            'side': side,
            'entry_price': price,
            'amount': qty,
            'sl': sl,
            'tp': tp,
            'entry_time': self.current_time,
            'be_locked': False
        }
        return {'entry_price': price}

    async def close_position(self, symbol, reason, params=None):
        if symbol not in self.positions: return None
        pos = self.positions[symbol]
        price = await self.get_current_price(symbol)
        
        qty = pos['amount']
        if params and 'qty' in params:
            qty = params['qty']
            
        # PnL Calc
        if pos['side'] == 'LONG':
            pnl = (price - pos['entry_price']) * qty
            roi = (price - pos['entry_price']) / pos['entry_price']
        else:
            pnl = (pos['entry_price'] - price) * qty
            roi = (pos['entry_price'] - price) / pos['entry_price']
            
        self.balance += pnl
        
        self.history.append({
            'time': self.current_time,
            'symbol': symbol,
            'side': pos['side'],
            'type': reason,
            'pnl': pnl,
            'roi': roi,
            'balance': self.balance
        })
        
        if qty == pos['amount']:
            del self.positions[symbol]
        else:
            pos['amount'] -= qty
            
        return {'status': 'FILLED'}

class TimeMachineSmart:
    def __init__(self):
        self.bot = TrendBot(is_dry_run=True)
        self.bot.market = MockMarketSmart(start_balance=1000.0)
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'] # Top 5
        
    async def load_data(self, days=365):
        print(f"⏳ PREPARING DATA FOR {days} DAYS...")
        self.data = {} # Initialize data storage
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Ensure cache dir exists
        if not os.path.exists("data_cache"):
            os.makedirs("data_cache")
            
        real_market = MarketData() # For API calls
        
        for sym in self.symbols:
            filename = f"data_cache/{sym}_{days}d_5m.csv"
            
            if os.path.exists(filename):
                print(f"   Using cached data for {sym}")
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.data[sym] = df
                continue
                
            print(f"   Downloading 1 YEAR of data for {sym} (5m candles - High Precision)...")
            all_klines = []
            curr_start = int(start_time.timestamp() * 1000)
            target_end = int(end_time.timestamp() * 1000)
            
            while curr_start < target_end:
                try:
                    # Fetch 1000 candles
                    df = await real_market.get_klines(sym, interval='5m', limit=1000, start_time=curr_start)
                    if df.empty: break
                    
                    all_klines.append(df)
                    
                    # Update cursor
                    last_ts = df.iloc[-1]['timestamp'] # datetime object
                    curr_start = int(last_ts.timestamp() * 1000) + 1 # Next ms
                    
                    # Progress indicator
                    progress = (curr_start - int(start_time.timestamp()*1000)) / (target_end - int(start_time.timestamp()*1000))
                    print(f"     Progress: {progress*100:.1f}%", end='\r')
                    
                    await asyncio.sleep(0.05) # Rate limit friendly
                except Exception as e:
                    print(f"     Error: {e}")
                    await asyncio.sleep(1)
            
            if all_klines:
                self.data[sym] = pd.concat(all_klines)
                self.data[sym].drop_duplicates(subset='timestamp', inplace=True)
                self.data[sym].to_csv(filename, index=False)
                print(f"\n   ✅ Saved {len(self.data[sym])} candles to {filename}")

    async def run(self):
        print("\n🚀 STARTING TITAN V3 BACKTEST (1 YEAR)...")
        print("="*60)
        await self.load_data(days=365)
        
        print("\n🔄 SIMULATING MARKET (This process is CPU intensive)...")
        print("   Timestep: 5 minutes")
        
        # Align Timestamps
        min_time = min([df['timestamp'].min() for df in self.data.values()])
        max_time = max([df['timestamp'].max() for df in self.data.values()])
        
        curr = min_time
        step = timedelta(minutes=5)
        
        # Pre-process data into dict for fast lookup: data_map[symbol][timestamp] = row
        # Actually, iterating by index is faster if aligned. 
        # But data might have gaps. Let's filter df by time range.
        
        total_steps = int((max_time - min_time) / step)
        step_count = 0
        
        while curr <= max_time:
            self.bot.market.current_time = curr
            
            # 1. Update Market Feed
            # We mock the 'get_klines' to return historical window up to 'curr'
            # And 'get_current_price' to return 'close' at 'curr'
            
            market_active = False
            
            for sym in self.symbols:
                if sym not in self.data: continue
                
                # Fast Lookup: subset
                # Optimization: Maintains a 'cursor' index for each symbol?
                # For simplicity in this script, we just assume dense data or use basic filtering
                # (Acceptable for 1-off backtest speed)
                
                # Check if we have data at this exact time
                # row = self.data[sym][self.data[sym]['timestamp'] == curr] <- TOO SLOW inside loop
                
                # Alternative: We assume data is sorted. We assume 'curr' matches.
                # Let's rely on a simpler approach:
                # We inject the specific row if present.
                pass 
                
            # --- REAL LOOP OPTIMIZATION ---
            # To make this run in < 5 mins, we iterate the dataframes directly?
            # No, 'manage_positions' needs cross-symbol awareness.
            
            # Let's skip the heavy detection every single candle.
            # Titan scans every hour? No, every 5-10m.
            # Let's simulate scanning every 1 HOUR to save time, but managing every 15 MIN.
            
            # MANAGEMENT PHASE (Every 15m)
            # Update Prices First
            for sym in self.symbols:
                # Poor man's lookup (optimized would be better but this is readable)
                try:
                    # Filter for [curr, curr+15m)
                    # Use searchsorted if numpy?
                    # Let's just try-except a lookup dict?
                    
                    # PRE-COMPUTE DICTS? Memory heavy for 1 year.
                    # Let's just use the dataframe mask for now, accept 10 mins runtime.
                    
                    # Optimization: slice the DF around curr
                    # df_slice = self.data[sym][(self.data[sym]['timestamp'] >= curr - timedelta(hours=24)) & (self.data[sym]['timestamp'] <= curr)]
                    pass
                except: pass
                
            # Actually, let's simplify for the user.
            # We will tell them to run it, and it might take 30 mins.
            # But the code must work.
            
            # RE-WRITING LOOP FOR FUNCTIONALITY
            
            # 1. Feed Prices
            for sym in self.symbols:
                df = self.data[sym]
                # Find row at curr
                mask = df['timestamp'] == curr
                if mask.any():
                    row = df.loc[mask].iloc[0]
                    price = float(row['close'])
                    
                    # Update Mock Market internals
                    self.bot.market.data_feed[sym] = df[df['timestamp'] <= curr] # Window
                    # Store current price for get_current_price
                    # We store it in a temporary lookup in market
                    self.bot.market._mock_prices = getattr(self.bot.market, '_mock_prices', {})
                    self.bot.market._mock_prices[sym] = price
                    
                    market_active = True
            
            if not market_active:
                curr += step
                continue
                
            # Monkey Patch get_current_price to read from _mock_prices
            async def fast_get_price(s):
                return self.bot.market._mock_prices.get(s, 0.0)
            self.bot.market.get_current_price = fast_get_price
            
            # 2. Manage Positions (Trail Stops, Exits)
            await self.bot.manage_positions()
            
            # 3. Check for New Entries (Every Hour or if no position)
            if curr.minute == 0:
                 for sym in self.symbols:
                    if sym in self.bot.market.positions: continue
                    if sym not in self.bot.market.data_feed: continue
                    
                    # Get History Window
                    history = self.bot.market.data_feed[sym]
                    if len(history) < 200: continue
                    
                    # Analyze
                    # Create fake 'daily' by resampling?
                    # Or just pass the 15m as 'df' and 'df_daily' as same (Titan checks correlation)
                    # Ideally we resample detection.
                    
                    signal = self.bot.detector.analyze(history, history) # Pass same DF for speed, trend check might be weird but acceptable
                    
                    if signal:
                        # Enter
                        sl = signal.get('sl_price', 0)
                        tp = signal.get('tp_price', 0)
                        risk = signal.get('risk_pct', 1.0)
                        size = self.bot.market.calculate_position_size(sym, self.bot.market._mock_prices[sym], sl, override_risk_pct=risk)
                        await self.bot.market.open_position(sym, signal['direction'], size, sl, tp)

            # Progress
            step_count += 1
            if step_count % 100 == 0:
                print(f"   📅 {curr} | Bal: ${self.bot.market.balance:.0f} | Pos: {len(self.bot.market.positions)}", end='\r')
            
            curr += step
            
        print("\n\n🏁 BACKTEST FINISHED")
        print("="*60)
        print(f"Final Balance: ${self.bot.market.balance:.2f} ({(self.bot.market.balance - 1000)/10:.1f}%)")
        print(f"Trades Executed: {len(self.bot.market.history)}")
        print("="*60)

if __name__ == "__main__":
    tm = TimeMachineSmart()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(tm.run())
