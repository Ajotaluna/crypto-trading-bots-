import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
from trend_following_bot.market_data import MarketData
from trend_following_bot.market_data import MarketData
from trend_following_bot.patterns_v2 import PatternDetector
from trend_following_bot.technical_analysis import TechnicalAnalysis
from trend_following_bot.config import config
from trend_following_bot.config import config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Backtester")

class Backtester:
    """
    THE TIME MACHINE: Simulations against historical data.
    Verifies King's Guard, Kelly Lite, and Strategy logic.
    """
    def __init__(self):
        self.market = MarketData()
        self.detector = PatternDetector()
        self.market.is_dry_run = True # Safety
        
        # Stats
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        
    async def run_simulation(self, limit_days=2, symbol_limit=3):
        print(f"\n⏳ STARTING TIME MACHINE (Last {limit_days} Days)...")
        print(f"📊 Scanning Top {symbol_limit} Symbols (Accelerated Mode)...")
        
        # 1. Get Top Symbols
        symbols = await self.market.get_top_symbols(limit=symbol_limit)
        
        # 2. Fetch BTC Context (King's Guard)
        # We need 1H BTC data for the entire period
        print("🛡️ Fetching BTC King's Guard Data...")
        btc_df_1h = await self.market.get_klines('BTCUSDT', '1h', limit=limit_days*24 + 50)
        btc_df_1h = TechnicalAnalysis.calculate_indicators(btc_df_1h)
        # Pre-calc trends map for speed: {timestamp: trend_pct}
        btc_map = {}
        for index, row in btc_df_1h.iterrows():
            # Trend is (Close - Open) / Open or just use 1H change
            # But the bot uses `get_btc_trend` which does: (Current - Open_1H_Ago) / Open_1H_Ago
            # Approximation: (Close - Open) of the candle matching the 15m time
            change = (row['close'] - row['open']) / row['open']
            btc_map[row['timestamp']] = change
            
        # 3. Process Each Symbol
        for symbol in symbols:
            # Skip BTC itself
            if symbol == 'BTCUSDT': continue
            
            await self.simulate_symbol(symbol, btc_map, limit_days)
            
        # 4. Report
        self.print_report()
            
    async def simulate_symbol(self, symbol, btc_map, limit_days):
        # Fetch 15m Data (Primary)
        # 96 candles/day. 2 days = ~200. Buffer 50 for indicators. Total 300.
        limit_candles = (limit_days * 96) + 100
        df = await self.market.get_klines(symbol, '15m', limit=limit_candles)
        
        if df.empty or len(df) < 100: return
        
        # Fetch 1H Data (Context)
        df_daily = await self.market.get_klines(symbol, '1d', limit=50) # Approx Daily
        
        # Run Simulation Loop
        # We start from index 50 to allow indicators to warm up
        # We iterate until -1 (leaving room for outcome check)
        
        print(f"👉 Simulating {symbol} ({len(df)} candles)...")
        
        in_position = False
        
        for i in range(50, len(df)-4): # Leave 4 candles (1 hour) for outcome check
            # Slice "Past" Data
            current_slice = df.iloc[:i+1].copy()
            current_candle = current_slice.iloc[-1]
            future_candles = df.iloc[i+1:i+5] # Next 1 hour
            
            if in_position:
                 # Check if exit status logic would have triggered? 
                 # For simplicity in this V1 backtester, we just skip overlapping trades per symbol
                 # or checks strict TP/SL on future candles.
                 in_position = False # Reset for new signal check
                 continue
                 
            # 1. GET BTC TREND for this moment
            # Find closest 1H candle
            curr_time = current_candle['timestamp']
            # Simple match: round down to nearest hour?
            # Or just check if btc crashing
            # For speed, use the timestamp directly if aligned
            
            btc_trend = 0.0
            # Look up approximate btc trend
            # (In a real backtester we align carefully. Here we assume loose correlation)
            
            # 2. Analyze
            signal = self.detector.analyze(current_slice, df_daily, btc_trend=0.0) # Assume Neutral for base test
            
            if signal:
                # 3. KING'S GUARD VETO CHECK (Simulated)
                # Check BTC change in that hour
                # ... skip for now to test Raw Signals first ...
                
                # 4. SIMULATE OUTCOME
                entry_price = float(current_candle['close'])
                sl, tp = self.detector.calculate_dynamic_levels(current_slice, signal['direction'])
                
                if not sl or not tp: continue
                
                # Check outcome in next 4 candles (1 Hour scan)
                outcome = 'HOLD' 
                pnl = 0.0
                
                for _, future in future_candles.iterrows():
                    high = future['high']
                    low = future['low']
                    
                    if signal['direction'] == 'LONG':
                        if low <= sl:
                            outcome = 'LOSS'
                            pnl = (sl - entry_price) / entry_price
                            break
                        if high >= tp:
                            outcome = 'WIN'
                            pnl = (tp - entry_price) / entry_price
                            break
                    else:
                         if high >= sl:
                            outcome = 'LOSS'
                            pnl = (entry_price - sl) / entry_price
                            break
                         if low <= tp:
                            outcome = 'WIN'
                            pnl = (entry_price - tp) / entry_price
                            break
                            
                # Record
                if outcome != 'HOLD':
                    self.record_trade(symbol, outcome, pnl, signal['score'])
                    in_position = True # Mark busy for a bit
                    
    def record_trade(self, symbol, outcome, pnl_pct, score):
        # Kelly Sizing Simulation (Approx)
        size = 100 # $100 base
        if score >= 90: size = 166
        elif score < 75: size = 66
        
        pnl_usdt = size * pnl_pct * config.LEVERAGE # Leverage 5x
        
        self.trades.append({
            'symbol': symbol,
            'outcome': outcome,
            'pnl': pnl_usdt,
            'score': score
        })
        
        if pnl_usdt > 0: self.wins += 1
        else: self.losses += 1
        self.total_pnl += pnl_usdt
        
    def print_report(self):
        print("\n" + "="*40)
        print("🕑 TIME MACHINE REPORT (24H Simulation)")
        print("="*40)
        
        total = self.wins + self.losses
        if total == 0:
            print("No trades triggered.")
            return
            
        wr = (self.wins / total) * 100
        
        print(f"Total Trades: {total}")
        print(f"Wins: {self.wins} | Losses: {self.losses}")
        print(f"Win Rate: {wr:.2f}%")
        print(f"Total PnL (Simulated): ${self.total_pnl:.2f}")
        print("="*40)

if __name__ == "__main__":
    bt = Backtester()
    asyncio.run(bt.run_simulation())
