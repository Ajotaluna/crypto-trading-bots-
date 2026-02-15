"""
Main Bot Logic - REFACTORED FOR DOUBLE FUNNEL STRATEGY
Integrates AnomalyScanner (Macro) and TradingStrategy (Micro).
"""
import asyncio
import logging
import sys
import os
import time
from datetime import datetime, timedelta

# Local imports
from config import config
from market_data import MarketData
from win_rate_tracker import WinRateTracker
import pandas as pd
from blacklist_manager import BlacklistManager

# NEW IMPORTS (The "New Brain")
from scanner_anomaly import AnomalyScanner
from trading_strategy import (
    confirm_entry, 
    calculate_indicators, 
    PositionManager,
    RiskManager, # NEW
    MAX_SCORE,
    TOP_N,
    # Strategy Configs
    INITIAL_CAPITAL,
    LEVERAGE,
    COMMISSION,
    RISK_PER_ENTRY,
    MAX_CAPITAL_PER_TRADE,
    MAX_SIGNALS,
    MAX_HOLD_CANDLES,
    DAILY_LOSS_CAP,
    INITIAL_SL_ATR,
    BE_LOCK_ATR,
    TRAIL_DISTANCE_ATR,
    SCALE_LEVEL_2,
    SCALE_LEVEL_3,
    MAX_SCALE
)

# ... (rest of imports)

async def manage_positions(self):
        """
        Manages active positions using RiskManager from Strategy File.
        """
        for symbol, pos in list(self.market.positions.items()):
            current_price = await self.market.get_current_price(symbol)
            if current_price == 0: continue
            
            # 1. Get Trend Data (ATR)
            try:
                df = await self.market.get_klines(symbol, interval='15m', limit=20)
                if df.empty: continue
                
                # Simple ATR calc if not present
                high = df['high'].astype(float)
                low = df['low'].astype(float)
                close = df['close'].astype(float)
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
            except Exception:
                atr = 0

            # 2. DELEGATE TO STRATEGY FILE (Risk Manager)
            action, new_value, reason = RiskManager.check_exit_conditions(pos, current_price, atr)
            
            if action == 'CLOSE':
                logger.warning(f"🛑 EXIT TRIGGER: {symbol} | Reason: {reason}")
                await self.market.close_position(symbol, reason)
                continue # Moved to next position
                
            elif action == 'UPDATE_SL':
                if new_value != pos['sl']:
                    pos['sl'] = new_value
                    if reason == 'BE_LOCK':
                        pos['be_locked'] = True
                        logger.info(f"🛡️ BE LOCKED: {symbol} (Strategy Logic)")
            
            # 3. MAX HOLD TIME CHECK (Strategy Replication)
            # Strategy: Exit after MAX_HOLD_CANDLES (96 * 15m = 24h)
            entry_time = pos.get('entry_time')
            if entry_time:
                # Calculate elapsed time
                elapsed = datetime.now() - entry_time
                max_hold_time = timedelta(minutes=MAX_HOLD_CANDLES * 15)
                
                if elapsed > max_hold_time:
                    logger.warning(f"⏳ MAX HOLD EXPIRED: {symbol} held for {elapsed}. Closing.")
                    await self.market.close_position(symbol, "MAX_HOLD_TIME")

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trend_bot.log")
    ]
)
logger = logging.getLogger("TrendBot")

import concurrent.futures

class TrendBot:
    def __init__(self, is_dry_run=True, api_key=None, api_secret=None):
        self.market = MarketData(is_dry_run, api_key, api_secret)
        self.blacklist = BlacklistManager()
        self.running = True
        self.start_balance = 0.0
        
        # Strategies
        self.scanner = AnomalyScanner()
        self.position_mgr = PositionManager() # Replacing internal logic? 
        # Actually PositionManager in trading_strategy is for backtest state.
        # Producton usually manages state via exchange/MarketData.
        # But we need the PID logic.
        # For now, let's keep the production position management (Safety Loop)
        # but use confirm_entry for signals.
        
        self.daily_watchlist = [] # The Top 10 for the day
        
        # SCOREBOARD
        self.tracker = WinRateTracker()
        self.tracker.log_summary()

        # ThreadPool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def start(self):
        """Main Entry Point & Orchestrator"""
        mode_str = "PRODUCTION (REAL MONEY)" if not self.market.is_dry_run else "DRY RUN (PAPER TRADING)"
        logger.info(f"\n{'='*50}\n>>> STARTING TREND BOT V5 (DOUBLE FUNNEL): {mode_str} <<<\n{'='*50}")
        
        # 1. Initialize Balance
        if not self.market.is_dry_run:
             status = await self.market.get_real_account_status()
             if status:
                 self.start_balance = status['equity']
                 logger.info(f"Initial Equity: ${self.start_balance:.2f}")
        else:
             self.start_balance = self.market.balance
        
        self.current_day_str = datetime.utcnow().strftime('%Y-%m-%d')
             
        # LOGGING CAPITAL
        daily_target = self.start_balance * 0.03
        logger.info(f"💰 BALANCE: ${self.start_balance:.2f} | 🎯 DAILY TARGET (3%): ${daily_target:.2f}")

        # 2. Launch Background Loops
        asyncio.create_task(self.reporting_loop())
        asyncio.create_task(self.safety_loop()) 
        
        # 3. Start The Core Brain
        await self.execution_loop()

    async def execution_loop(self):
        """
        The Double Funnel Engine:
        1. Macro Scan (Once per day 00:00 UTC) -> Selects Top 10.
        2. Micro Scan (Continuous/15m) -> Checks confirmation on Top 10.
        """
        logger.info("🧠 Started Execution Loop (Double Funnel)...")
        
        # Initial Macro Scan
        await self.run_macro_scan()
        
        last_scan_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        while self.running:
            try:
                now_date = datetime.utcnow().strftime('%Y-%m-%d')
                
                # --- A. DAILY MACRO RE-SCAN (UTC Midnight) ---
                if now_date != last_scan_date:
                    logger.info(f"📅 NEW DAY {now_date}: Running Macro Scan...")
                    await self.run_macro_scan()
                    last_scan_date = now_date
                    
                    # Reset Daily PnL
                    self.start_balance = await self.get_equity()
                    self.tracker.reset_daily_pnl_if_new_day()

                # --- B. MICRO SCAN (Intraday) ---
                # Iterate through the Watchlist
                if not self.daily_watchlist:
                    logger.warning("⚠️ Watchlist empty. Retrying Macro Scan...")
                    await self.run_macro_scan()
                    await asyncio.sleep(60)
                    continue

                for pick in self.daily_watchlist:
                    symbol = pick['symbol']
                    direction = pick['direction']
                    
                    # Capacity Check
                    if len(self.market.positions) >= MAX_SIGNALS:
                        break
                    
                    # Skip if already open
                    if symbol in self.market.positions:
                        continue
                        
                    # MICRO CHECK
                    await self.check_micro_entry(symbol, direction, pick)
                    
                    await asyncio.sleep(1) # Pace requests

                # Wait for next cycle
                # 15m candles update every minute, but we can check more often?
                # Let's check every minute to be safe? Or match candle close?
                # Real-time can check every minute.
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Execution Loop Error: {e}")
                await asyncio.sleep(60)

    async def run_macro_scan(self):
        """
        Runs AnomalyScanner on the entire market to pick the daily Top N.
        """
        try:
            logger.info("🔭 MACRO SCAN: Fetching market data...")
            
            # 1. Get Tickers/Candidates (Whitelist Logic)
            tickers = await self.market.get_trading_universe() 
            if not tickers:
                logger.error("No tickers found in Universe.")
                return

            # 2. Add 'whitelist' logic here if implemented later
            # ...
            
            # 3. Build Data Dictionary for Scanner
            pair_data = {}
            logger.info(f"📥 Downloading history for {len(tickers)} pairs...")
            
            for symbol in tickers:
                # We need ~480 candles for history
                df = await self.market.get_klines(symbol, interval='15m', limit=490)
                if not df.empty:
                    # Rename likely required columns if get_klines doesn't match
                    # Assuming market_data returns DataFrame with lower case cols
                    pair_data[symbol] = df
                await asyncio.sleep(0.05) # Rate limit
                
            logger.info(f"📊 Analyzing {len(pair_data)} pairs...")
            
            # 4. Run AnomalyScanner
            # It expects synchronous dict of DF.
            # score_universe(pair_data, now_idx, top_n)
            # now_idx = -1 (latest)
            
            picks = self.scanner.score_universe(pair_data, -1, top_n=TOP_N)
            
            # Filter by MAX_SCORE
            self.daily_watchlist = [p for p in picks if p['score'] < MAX_SCORE]
            
            logger.info(f"✅ MACRO RESULTS: {len(self.daily_watchlist)} Candidates Selected.")
            for p in self.daily_watchlist:
                logger.info(f"  > {p['symbol']} ({p['direction']}) | Score: {p['score']} | Reasons: {p['reasons']}")
                
        except Exception as e:
            logger.error(f"Macro Scan Error: {e}")

    async def check_micro_entry(self, symbol, direction, pick_info):
        """
        Runs confirm_entry on the specific candidate.
        """
        try:
            # 1. Get Fresh Data
            # We need enough validation data (~50 candles minimum for indicators/Kalman)
            limit = 100 
            df = await self.market.get_klines(symbol, interval='15m', limit=limit)
            if df.empty: return

            # 2. Calculate Indicators (Using the NEW optimized function)
            # Run in thread to avoid blocking loop
            loop = asyncio.get_running_loop()
            df_indicators = await loop.run_in_executor(
                self.executor,
                calculate_indicators,
                df
            )
            
            # 3. Micro Confirmation (Kalman, Entropy, RSI)
            # confirm_entry(df, direction)
            is_valid = await loop.run_in_executor(
                self.executor,
                confirm_entry,
                df_indicators,
                direction
            )
            
            if is_valid:
                logger.info(f"✨ MICRO TRIGGER: {symbol} ({direction}) Confirmed!")
                
                # 4. Execute
                # Signal details
                signal = {
                    'direction': direction,
                    'score': pick_info['score'],
                    'strategy_mode': 'TREND',
                    'reasons': pick_info['reasons']
                }
                
                await self.execute_trade(symbol, signal, df_indicators)
                
        except Exception as e:
            logger.error(f"Micro Scan Error {symbol}: {e}")

    async def execute_trade(self, symbol, signal, df):
        # ... (Reusing existing execution logic, stripped of shadow/legacy guards) ...
        
        curr_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # SIZING & SL/TP based on STRATEGY CONSTANTS
        # INITIAL_SL_ATR = 5.0 (Default)
        sl_dist = atr * INITIAL_SL_ATR 
        
        if signal['direction'] == 'LONG':
            sl = curr_price - sl_dist
            tp = curr_price + (atr * 10) # Open ended really
        else:
            sl = curr_price + sl_dist
            tp = curr_price - (atr * 10)

        # RISK SIZING: RISK_PER_ENTRY (1%)
        # Logic: Position Size = (Risk Amount) / (Distance to SL %) / Leverage?
        # MarketData.calculate_position_size usually handles 'Risk Amount / Distance'.
        # We pass override_risk_pct = RISK_PER_ENTRY.
        
        amount = self.market.calculate_position_size(
            symbol, 
            curr_price, 
            sl, 
            override_risk_pct=RISK_PER_ENTRY,
            override_leverage=LEVERAGE
        )
        
        # LEVERAGE CHECK (Strategy Requirement)
        # We should ensure MarketData uses LEVERAGE constant or we enforce it here.
        # MarketData usually takes leverage from config. 
        # But 'amount' is Notional USDT size.
        
        if amount < 6.0: return
        
        logger.info(f"⚡ EXECUTING {symbol} | SL: {sl:.4f} | Size: ${amount:.0f} (Risk {RISK_PER_ENTRY*100}%)")
        
        # Execute with Strategy Leverage if possible, or just open
        result = await self.market.open_position(symbol, signal['direction'], amount, sl, tp)
        if result:
             logger.info(f"🔫 OPEN SUCCESS {symbol}")

    # ... (Keeping Safety Loop, Reporting Loop, but stripped of Shadow Logic if user wants clean slate) ...
    # User said: "Borra toda la logica que tenga que ver con señales"
    # But Safety Loop is "Management", not "Signals".
    # I will adapt Safety Loop to use PositionManager logic? 
    # Or just keep the existing safety loop for now?
    # User said "borra... señales". Safety is management.
    # However, existing safety loop uses old logic.
    # Let's keep it simple: Use MarketData's basic close and maybe rudimentary management.
    
    async def safety_loop(self):
        """Active Position Manager (Simple)"""
        logger.info("Started Safety Monitor...")
        while self.running:
            try:
                await self.manage_positions()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Safety Loop Error: {e}")
                await asyncio.sleep(5)

    async def manage_positions(self):
        """
        Manages active positions with High Win Rate Logic:
        - Hard SL/TP checks.
        - BE Lock at 1.5 ATR.
        - Trailing Stop at 2.0 ATR (Dynamic).
        """
        for symbol, pos in list(self.market.positions.items()):
            current_price = await self.market.get_current_price(symbol)
            if current_price == 0: continue
            
            # 1. Get Trend Data (ATR) for Management
            try:
                # We need fresh ATR for dynamic trailing
                df = await self.market.get_klines(symbol, interval='15m', limit=20)
                if df.empty: continue
                
                # Simple ATR calc if not present
                high = df['high'].astype(float)
                low = df['low'].astype(float)
                close = df['close'].astype(float)
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
            except Exception:
                atr = 0

            # 2. Hard Stop Checks (Safety Net)
            is_sl = False
            if pos['side'] == 'LONG':
                if current_price <= pos['sl']: is_sl = True
            else:
                if current_price >= pos['sl']: is_sl = True
                
            if is_sl:
                logger.warning(f"🛑 STOP LOSS HIT: {symbol}")
                await self.market.close_position(symbol, "STOP LOSS")
                continue

            # 3. Dynamic Trailing Logic (if ATR available)
            if atr > 0:
                pnl_atr = ((current_price - pos['entry_price']) if pos['side'] == 'LONG' else (pos['entry_price'] - current_price)) / atr
                
                # BE LOCK (1.5 ATR)
                if pnl_atr >= 1.5 and not pos.get('be_locked', False):
                    buffer = current_price * 0.001
                    new_sl = pos['entry_price'] + buffer if pos['side'] == 'LONG' else pos['entry_price'] - buffer
                    pos['sl'] = new_sl
                    pos['be_locked'] = True
                    logger.info(f"🛡️ BE LOCKED: {symbol} (+1.5 ATR)")
                    
                # TRAILING (2.0 ATR) - Only active after some profit or always? 
                # Strategy says "Trail 2.0 ATR".
                # If we trail immediately, we might choke the trade.
                # Usually trail activates after BE or when deep in profit.
                # Let's simple Trail if Locked.
                if pos.get('be_locked', False):
                    trail_dist = 2.0 * atr
                    if pos['side'] == 'LONG':
                        potential_sl = current_price - trail_dist
                        if potential_sl > pos['sl']:
                            pos['sl'] = potential_sl
                            # logger.info(f"🪜 TRAIL UPDATE: {symbol} to {pos['sl']:.4f}")
                    else:
                        potential_sl = current_price + trail_dist
                        if potential_sl < pos['sl']:
                            pos['sl'] = potential_sl
                            # logger.info(f"🪜 TRAIL UPDATE: {symbol} to {pos['sl']:.4f}")

    async def reporting_loop(self):
        """Periodically reports status."""
        while self.running:
             try:
                 total_equity = self.market.balance
                 if not self.market.is_dry_run:
                     st = await self.market.get_real_account_status()
                     if st: total_equity = st['equity']
                 
                 open_count = len(self.market.positions)
                 logger.info(f"--- 📊 STATUS: Equity ${total_equity:.2f} | Open Logic: {open_count} ---")
                 if open_count > 0:
                     for s, p in self.market.positions.items():
                         logger.info(f"   > {s} ({p['side']}) | Entry: {p['entry_price']} | SL: {p['sl']:.4f}")
                 
                 await asyncio.sleep(300) # 5 Minutes
             except Exception as e:
                 logger.error(f"Reporting Error: {e}")
                 await asyncio.sleep(60)
    
    async def get_equity(self):
        if not self.market.is_dry_run:
             s = await self.market.get_real_account_status()
             return s['equity'] if s else self.market.balance
        return self.market.balance
