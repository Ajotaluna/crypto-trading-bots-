"""
Main Bot Logic
Integrates Data, Patterns, and Execution.
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
from patterns_v2 import PatternDetector # V2 (Verified Strategy)
from win_rate_tracker import WinRateTracker
import pandas as pd
from blacklist_manager import BlacklistManager
from calibration import CalibrationManager # NEW: Production Calibration Engine
from technical_analysis import TechnicalAnalysis # For local indicator calc

# Local imports
from config import config
from market_data import MarketData
from patterns_v2 import PatternDetector # V2 (Verified Strategy)
from win_rate_tracker import WinRateTracker
import pandas as pd
from blacklist_manager import BlacklistManager
from calibration import CalibrationManager # NEW: Production Calibration Engine

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
        self.detector = PatternDetector()
        self.blacklist = BlacklistManager()
        self.calibrator = CalibrationManager(use_cache=False) 
        self.running = True
        self.start_balance = 0.0
        self.watchlist = {} 
        self.pending_entries = {}
        self.trap_blacklist = {} 
        
        # PRODUCTION TRADING LIST (Hot Start)
        saved_map = self.calibrator.load_strategy_map()
        if saved_map:
            self.active_trading_list = list(saved_map.keys())
            logger.info(f"🚀 HOT START: Restored {len(self.active_trading_list)} pairs from cache.")
        else:
            self.active_trading_list = list(self.calibrator.vip_majors)
            logger.info(f"🚀 BOOTSTRAP: Loaded {len(self.active_trading_list)} Major Pairs immediately.")
        
        # SHADOW MODE STATE MACHINE (Pre-Ban Logic)
        self.pair_states = {} 
        self.shadow_positions = {} 
        
        # PERSISTENCE TRACKING (Core vs Ephemeral)
        # Pairs in this set are "Protected" from strict pruning (Volume/Core tiers)
        self.persistent_pairs = set()
        if saved_map:
             self.persistent_pairs.update(saved_map.keys())
        
        # GRACE PERIOD TRACKER (Anti-Flicker)
        # {symbol: timestamp_when_trend_lost}
        self.trend_grace_timers = {}

        # SCOREBOARD
        self.tracker = WinRateTracker()
        self.tracker.log_summary()

        # ThreadPool used because CalibrationManager holds unpicklable objects (ccxt sockets)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def start(self):
        """Main Entry Point & Orchestrator"""
        mode_str = "PRODUCTION (REAL MONEY)" if not self.market.is_dry_run else "DRY RUN (PAPER TRADING)"
        logger.info(f"\n{'='*50}\n>>> STARTING TREND BOT: {mode_str} <<<\n{'='*50}")
        
        # 1. Initialize Balance
        if not self.market.is_dry_run:
             status = await self.market.get_real_account_status()
             if status:
                 self.start_balance = status['equity']
                 logger.info(f"Initial Equity: ${self.start_balance:.2f}")
        else:
             self.start_balance = self.market.balance
        
        self.current_day_str = datetime.utcnow().strftime('%Y-%m-%d')
             
        # LOGGING CAPITAL & TARGET
        daily_target = self.start_balance * 0.03
        logger.info(f"💰 BALANCE: ${self.start_balance:.2f} | 🎯 DAILY TARGET (3%): ${daily_target:.2f}")

        # 2. Launch Background Loops
        asyncio.create_task(self.reporting_loop())
        asyncio.create_task(self.safety_loop()) 
        asyncio.create_task(self.confirmation_loop())
        
        # 3. Start Continuous Scanner
        self.scanner_task = asyncio.create_task(self.continuous_scanner_loop())
        
        # 4. Maintenance & Calibration Loop
        last_maintenance = 0
        last_discovery = 0
        
        while self.running:
            try:
                now = time.time()
                
                # --- PRIORITY 1: SCHEDULED MAINTENANCE (4H / 7D) ---
                if now - last_maintenance > (4 * 3600):
                    
                    # Determine Scan Type (Weekly vs 4H)
                    current_hour = datetime.utcnow().hour
                    is_tokyo = 0 <= current_hour < 9
                    # Last FULL scan time needs to be tracked persistently or roughly estimated
                    # Using a rough check here. If we want strict 7 days + Tokyo, we need another variable.
                    # Re-using last_full_scan logic from self if accessible, or just checking condition.
                    # Let's simplify: Weekly logic inside here.
                    
                    is_full_scan = False
                    # Check if 7 days passed since last FULL reset (we need a var for this)
                    if not hasattr(self, 'last_weekly_reset'): self.last_weekly_reset = 0
                    
                    days_since_weekly = (now - self.last_weekly_reset) / (24 * 3600)
                    
                    if days_since_weekly >= 7 and is_tokyo:
                        logger.info("🕒 WEEKLY MAINTENANCE (TOKYO): Full Market Reset (Volume 10M+ + Trending)")
                        is_full_scan = True
                        self.last_weekly_reset = now
                        
                        # WIPE SLATE CLEAN
                        logger.warning("🧹 FLUSHING LISTS: Starting fresh calibration cycle.")
                        self.active_trading_list = [] 
                    else:
                        logger.info("🕒 4H UPDATE: Quick Scan (Trending Only)")
                        
                    # Scan
                    candidates = await self.scan_market_candidates(full_scan=is_full_scan)
                    if candidates:
                        logger.info(f"⏳ Starting Maintenance Calibration (Reset=True) for {len(candidates)} pairs...")
                        loop = asyncio.get_running_loop()
                        approved_map = await loop.run_in_executor(
                            self.executor, 
                            self.calibrator.calibrate, 
                            candidates,
                            True # RESET LISTS = TRUE
                        )
                        # Sync List
                        self.active_trading_list = list(approved_map.keys())
                        
                        # UPDATE PERSISTENCE (Core Pairs)
                        self.persistent_pairs = set(approved_map.keys())
                        logger.info(f"🔒 PERSISTENCE: {len(self.persistent_pairs)} Core Pairs protected from strict pruning.")
                        
                        logger.info(f"✅ Active Trading List Updated ({len(self.active_trading_list)} pairs)")
                        
                    last_maintenance = now
                    last_discovery = now # Reset discovery timer
                    
                # --- PRIORITY 2: REACTIVE DISCOVERY (Every 10 Mins) ---
                elif now - last_discovery > 600:
                    logger.info("🔭 REACTIVE DISCOVERY: Scanning for New Trends...")
                    
                    # 1. Get Top Trending Gainers
                    gainers = await self.market.get_top_gainers(limit=20)
                    
                    # 2. Filter: Only NEW pairs not in current list
                    unknowns = [s for s in gainers if s not in self.active_trading_list]
                    
                    if unknowns:
                        logger.info(f"⚡ NEW TRENDS DETECTED: {unknowns} -> Calibrating Incrementally...")
                        loop = asyncio.get_running_loop()
                        
                        # Run Calibration with RESET=FALSE (Append Mode)
                        # This adds winners to the existing list in memory/disk
                        await loop.run_in_executor(
                             self.executor,
                             self.calibrator.calibrate,
                             unknowns,
                             False # RESET LISTS = FALSE (Incremental)
                        )
                        # Live Sync in Scanner Loop will pick these up automatically!
                    else:
                        logger.info("🔭 No new trends found.")
                        
                    last_discovery = now

                await asyncio.sleep(60) # Check timers every minute
                
            except Exception as e:
                logger.error(f"Main Loop Error: {e}")
                await asyncio.sleep(60)


    async def scan_market_candidates(self, full_scan=True):
        """
        Sourcing Candidates.
        Full Scan: Top Volume (50) + Top Gainers (20)
        Quick Scan: Top Gainers (20) only
        """
        candidates = set()
        
        try:
            # 1. Trending (Gainers) - Always Run
            gainers = await self.market.get_top_gainers(limit=20)
            candidates.update(gainers)
            
            # 2. Volume - Only on Full Scan (Weekly Tokyo Reset)
            # Requirement: Volume > 10M
            if full_scan:
                vol_pairs = await self.market.get_top_volume_pairs(limit=50, min_vol=10000000)
                candidates.update(vol_pairs)
                
            logger.info(f"🔎 Scanned Market. Found {len(candidates)} unique candidates.")
            return list(candidates)
            
        except Exception as e:
            logger.error(f"Market Scan Failed: {e}")
            return []

    async def safety_loop(self):
        """Active Position Manager Wrapper"""
        logger.info("Started Safety Monitor...")
        while self.running:
            try:
                # 1. Manage Active Positions
                await self.manage_positions()
                
                # 2. Manage Shadow Positions
                await self.manage_shadow_positions()
                
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Safety Loop Error: {e}")
                await asyncio.sleep(5)

    async def manage_shadow_positions(self):
        """Virtual Trade Management"""
        if not self.shadow_positions: return
        
        current_time = datetime.now()
        
        for symbol in list(self.shadow_positions.keys()):
            pos = self.shadow_positions[symbol]
            curr_price = await self.market.get_current_price(symbol)
            
            if curr_price == 0: continue
            
            outcome = None
            
            if pos['side'] == 'LONG':
                if curr_price >= pos['tp']: outcome = 'WIN'
                elif curr_price <= pos['sl']: outcome = 'LOSS'
            else:
                if curr_price <= pos['tp']: outcome = 'WIN'
                elif curr_price >= pos['sl']: outcome = 'LOSS'
                
            # Timeout (24h)
            if (current_time - pos['entry_time']).total_seconds() > 86400:
                outcome = 'LOSS' # Time limit treated as failure for shadow?
            
            if outcome:
                await self.handle_shadow_outcome(symbol, outcome)

    async def handle_shadow_outcome(self, symbol, outcome):
        """ Resolves a Virtual Trade for Pre-Ban Recovery """
        # Close Virtual
        if symbol in self.shadow_positions:
            del self.shadow_positions[symbol]
            
        state = self.pair_states.get(symbol)
        if not state: return
        
        state['shadow_total'] = state.get('shadow_total', 0) + 1
        
        if outcome == 'WIN':
            state['shadow_wins'] = state.get('shadow_wins', 0) + 1
            logger.info(f"👻 SHADOW WIN: {symbol} (Wins: {state['shadow_wins']}/5 to Recover)")
            
            if state['shadow_wins'] >= 5:
                logger.info(f"🌟 REDEMPTION: {symbol} has proven itself! Returning to ACTIVE status.")
                state['status'] = 'ACTIVE'
                state['real_losses'] = 0
                state['shadow_wins'] = 0
                state['shadow_total'] = 0
                
        else: # LOSS
            logger.info(f"💀 SHADOW LOSS: {symbol} failed virtual test.")
            # Ban if fails 3 times with 0 wins
            if state['shadow_total'] >= 3 and state['shadow_wins'] == 0:
                 logger.warning(f"⛔ BANHAMMER: {symbol} failed Shadow Mode (0/3). Blacklisting.")
                 self.blacklist.add_to_blacklist(symbol, "Failed Shadow Mode")
                 if symbol in self.active_trading_list:
                    self.active_trading_list.remove(symbol)
                 del self.pair_states[symbol]

    async def register_real_loss(self, symbol):
        """ Called when a REAL trade hits Stop Loss """
        state = self.pair_states.setdefault(symbol, {'status': 'ACTIVE', 'real_losses': 0})
        state['real_losses'] += 1
        
        logger.warning(f"⚠️ REAL LOSS: {symbol} (Count: {state['real_losses']}/2)")
        
        if state['real_losses'] >= 2:
            logger.warning(f"🛡️ PRE-BAN TRIGGERED: {symbol} entered SHADOW MODE. Simulation only.")
            state['status'] = 'SHADOW'
            state['shadow_wins'] = 0
            state['shadow_total'] = 0

    async def continuous_scanner_loop(self):
        """WATERFALL FINDER: Prioritized Continuous Scanning"""
        logger.info("🌊 Started Waterfall Scanner (Continuous Flow)...")
        
        # STARTUP COOLDOWN: Give indicators time to stabilize
        logger.info("🧊 STARTUP COOLDOWN: Waiting 60s for market data stabilization...")
        await asyncio.sleep(60)
        
        while self.running:
            try:
                # 1. Capacity Check
                if len(self.market.positions) >= config.MAX_OPEN_POSITIONS:
                    await asyncio.sleep(30)
                    continue
                    
                # 1.5. LIVE SYNC (Calibrator Integration)
                # "Conforme se va llenando la lista..." - User Request
                # If calibration is running, this picks up new winners immediately.
                if self.calibrator and self.calibrator.approved_pairs:
                    live_winners = list(self.calibrator.approved_pairs.keys())
                    
                    # INTELLIGENT SYNC (Append Only)
                    # Prevents overwriting pruning work. Only adds TRULY NEW discoveries.
                    current_set = set(self.active_trading_list)
                    new_items = [x for x in live_winners if x not in current_set]
                    
                    if new_items:
                        # Only add if they are not in our "Kill List" (Grace Timer Expiry)
                        # We use 'trend_grace_timers' keys? No, keys exist during grace.
                        # We need to trust that if we removed it from calibrator, it won't be here.
                        # But if it is here, it's new.
                        self.active_trading_list.extend(new_items)
                        # logger.info(f"⚡ LIVE SYNC: Added {len(new_items)} new pairs.")
                    
                # 2. TREND AUDIT & PRIORITY SORT (Dynamic)
                # User Requirement: "Priority to Trend", "Remove if Trend Ends"
                # To save API, we assume calibration set the baseline, but we check here too.
                
                scored_symbols = []
                # Use a copy to avoid modification issues
                for symbol in list(self.active_trading_list):
                    # Quick Trend Check (Cached or Fresh)
                    trend_score = await self.market.get_adx_now(symbol) # Ensure this method exists or use fallback
                    
                    if trend_score < 20: 
                        # PRUNE: Trend Ended (Strict Mode)
                        # EXCEPTION: Do not strictly prune "Core Pairs" (Volume/Maintenance List)
                        # User: "No toda la lista en la que se incluyen los 50 pares..."
                        
                        if symbol in self.persistent_pairs:
                            continue
                            
                        # GRACE PERIOD LOGIC (2 HOURS)
                        # "Creemos ciclos... solo se elimina 2 horas despues"
                        
                        grace_start = self.trend_grace_timers.get(symbol)
                        
                        if not grace_start:
                             # Start Timer (First Failure)
                             self.trend_grace_timers[symbol] = time.time()
                             logger.warning(f"📉 TREND LOST: {symbol} (ADX {trend_score:.1f}). Entering 2H Grace Period (Deprioritized).")
                             # Skip adding to 'scored_symbols' -> Deprioritized immediately
                             continue
                             
                        else:
                             # Timer Running
                             elapsed_hours = (time.time() - grace_start) / 3600
                             if elapsed_hours >= 2.0:
                                  # EXPIRED -> KILL
                                  logger.warning(f"💀 GRACE EXPIRED: {symbol} failed to recover in 2h. REMOVING permanently.")
                                  if symbol in self.active_trading_list:
                                       self.active_trading_list.remove(symbol)
                                  if self.calibrator and symbol in self.calibrator.approved_pairs:
                                       del self.calibrator.approved_pairs[symbol] # Ensure Sync doesn't revive it
                                  del self.trend_grace_timers[symbol]
                                  continue
                             else:
                                  # Waiting... (Silent Deprioritization)
                                  continue
                    
                    # Trend OK (>= 20)
                    if symbol in self.trend_grace_timers:
                        logger.info(f"♻️ TREND RECOVERED: {symbol} (ADX {trend_score:.1f}) is back!")
                        del self.trend_grace_timers[symbol] 
                        
                    scored_symbols.append((symbol, trend_score))
                
                # Sort: Strongest Trend FIRST
                scored_symbols.sort(key=lambda x: x[1], reverse=True)
                symbols = [x[0] for x in scored_symbols]
                
                # 3. Global Conditions & FILTERS
                # (symbols is now sorted by ADX Descending from the block above)
                
                if not symbols:
                    if not self.active_trading_list:
                        logger.warning("No Active Trading List! Waiting for Calibration...")
                    await asyncio.sleep(10)
                    continue

                # --- PRE-CALCULATION: EQUITY ---
                current_equity = self.start_balance # Default (Safe Fallback)
                if not self.market.is_dry_run:
                     st = await self.market.get_real_account_status()
                     if st: current_equity = float(st['equity'])
                else:
                     # Dry Run: Use Market Balance (Simulation)
                     current_equity = self.market.balance

                # --- A. NEW DAY RECALCULATION ---
                current_date_str = datetime.utcnow().strftime('%Y-%m-%d')
                if current_date_str != self.current_day_str:
                     logger.info(f"📅 NEW DAY (UTC): Recalculating Daily Goals...")
                     self.current_day_str = current_date_str
                     self.start_balance = current_equity # Reset Baseline for Compounding
                     # Also ensure tracker resets
                     self.tracker.reset_daily_pnl_if_new_day()

                # --- B. DAILY TARGET CHECK (3% of SESSION/DAY START) ---
                target_equity = self.start_balance * 1.03
                
                if current_equity >= target_equity:
                    profit = current_equity - self.start_balance
                    logger.info(f"💰 💰 💰 DAILY TARGET HIT! Equity ${current_equity:.2f} >= ${target_equity:.2f} (+${profit:.2f})")
                    logger.info("🛑 CLOSING ALL POSITIONS & SLEEPING UNTIL AFTER TOKYO.")
                    
                    # 1. Close All Trades
                    await self.market.emergency_close_all(reason="DAILY TARGET SECURED")
                    
                    # 2. Calculate Sleep Time until Next 09:00 UTC
                    utc_now = datetime.utcnow()
                    
                    # Target: Today 09:00 or Tomorrow 09:00?
                    # If we hit it BEFORE 09:00, we sleep until Today 09:00.
                    # If we hit it AFTER 09:00, we sleep until Tomorrow 09:00.
                    
                    target_time = utc_now.replace(hour=9, minute=0, second=0, microsecond=0)
                    if utc_now >= target_time:
                        target_time += timedelta(days=1) # Tomorrow
                        
                    sleep_seconds = (target_time - utc_now).total_seconds()
                    hours = sleep_seconds / 3600
                    
                    logger.info(f"😴 GOODNIGHT: Sleeping {hours:.1f} hours until {target_time} UTC...")
                    await asyncio.sleep(sleep_seconds) 
                    
                    # Upon waking, loop continues. New Day check will eventually trigger if we crossed midnight.
                    continue

                # --- C. TOKYO SESSION FILTER (00:00 - 09:00 UTC) ---
                # Only strictly pause if we haven't hit target yet (implied by execution flow)
                current_hour_utc = datetime.utcnow().hour
                if 0 <= current_hour_utc < 9:
                    logger.info("⏸️ TOKYO SESSION (00-09 UTC): Pausing Entries (Fluidez Perfecta).")
                    await asyncio.sleep(600) 
                    continue
                
                # --- D. KING'S GUARD (Restored) ---
                btc_trend = await self.market.get_btc_trend()
                if btc_trend < -0.01:
                    logger.warning(f"🛡️ KING'S GUARD: BTC Crash ({btc_trend*100:.2f}%). Pausing.")
                    await asyncio.sleep(60)
                    continue
                
                # --- D. KING'S GUARD ---
                    
                # 4. WATERFALL EXECUTION
                for symbol in symbols:
                    if len(self.market.positions) >= config.MAX_OPEN_POSITIONS: break
                    if symbol in self.market.positions or symbol in self.pending_entries: continue
                        
                    await asyncio.sleep(0.2) 
                    
                    candidate = await self.analyze_single_symbol(symbol, btc_trend)
                    
                    if candidate:
                        logger.info(f"✨ OPPORTUNITY FOUND: {symbol} (Score {candidate['score']}) -> Dispatching!")
                        if config.SMART_ENTRY_ENABLED:
                            await self.add_to_pending(candidate['symbol'], candidate['signal'], candidate['df'])
                        else:
                            await self.execute_trade(candidate['symbol'], candidate['signal'], candidate['df'])
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Scanner Loop Error: {e}")
                await asyncio.sleep(5)

    async def analyze_single_symbol(self, symbol, btc_trend):
        """Analyze ONE symbol fully"""
        try:
            if not self.blacklist.is_allowed(symbol): return None
            
            # Dynamic Sizing
            required_limit = self.detector.get_candle_limit(symbol)
            df = await self.market.get_klines(symbol, interval=config.TIMEFRAME, limit=required_limit)
            df_daily = await self.market.get_klines(symbol, interval='1d', limit=90)
            if df.empty or df_daily.empty: return None
            
            # CALCULATE INDICATORS LOCALLY (To fix ATR missing error)
            df = TechnicalAnalysis.calculate_indicators(df)
            
            loop = asyncio.get_running_loop()
            tech_signal = await loop.run_in_executor(
                self.executor, 
                self.detector.analyze, 
                df, df_daily, btc_trend
            )
            
            if not tech_signal: return None
            
            # Validation
            min_score = config.MIN_SIGNAL_SCORE
            
            is_override = False
            if config.ALLOW_MOMENTUM_OVERRIDE and tech_signal['score'] >= config.MOMENTUM_SCORE_THRESHOLD:
                is_override = True
                
            if not is_override:
                if tech_signal['score'] < min_score: return None
                
                # King's Guard
                if btc_trend < -0.01 and tech_signal['direction'] == 'LONG': return None
                if btc_trend > 0.01 and tech_signal['direction'] == 'SHORT': return None
            
            return {
                'symbol': symbol,
                'signal': tech_signal,
                'df': df,
                'score': tech_signal['score']
            }
        except Exception as e:
            return None

    async def confirmation_loop(self):
        """SMART ENTRY V4: Anti-Trap Logic"""
        logger.info("Started Smart Entry Monitor (V4)...")
        while self.running:
            try:
                current_time = datetime.now()
                for symbol, entry in list(self.pending_entries.items()):
                    if (current_time - entry['queued_time']).total_seconds() > (config.CONFIRMATION_TIMEOUT_MINS * 60):
                        del self.pending_entries[symbol]
                        continue
                        
                    df_5m = await self.market.get_klines(symbol, interval='5m', limit=25)
                    if df_5m.empty: continue
                    
                    try:
                        delta = df_5m['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / loss
                        df_5m['rsi_val'] = 100 - (100 / (1 + rs))
                        current_rsi = df_5m.iloc[-1]['rsi_val']
                    except:
                        current_rsi = 50.0

                    closed_candle = df_5m.iloc[-2] 
                    last_candle = df_5m.iloc[-1]   
                    trigger = entry['trigger_price']
                    state = entry.get('state', 'WAIT_BREAK')

                    if state == 'WAIT_BREAK':
                        breakout_confirmed = False
                        is_trap = False
                        
                        if entry['direction'] == 'LONG':
                            if closed_candle['close'] > trigger:
                                breakout_confirmed = True
                                if current_rsi > 75: is_trap = True
                        else:
                            if closed_candle['close'] < trigger:
                                breakout_confirmed = True
                                if current_rsi < 25: is_trap = True
                                
                        if breakout_confirmed:
                            if is_trap:
                                logger.warning(f"⚠️ TRAP DETECTED {symbol}: RSI {current_rsi:.1f}. Trap Blacklist.")
                                self.trap_blacklist[symbol] = datetime.now() + timedelta(minutes=60)
                                del self.pending_entries[symbol]
                                continue
                                
                            entry['state'] = 'WAIT_RETEST'
                            logger.info(f"💥 BREAKOUT CONFIRMED {symbol}. Waiting Retest...")

                    elif state == 'WAIT_RETEST':
                        retest_success = False
                        if entry['direction'] == 'LONG':
                            if float(last_candle['low']) <= trigger * 1.002: retest_success = True
                        else:
                            if float(last_candle['high']) >= trigger * 0.998: retest_success = True
                                
                        if retest_success:
                            logger.info(f"🎯 SNIPER ENTRY {symbol}")
                            await self.execute_trade(symbol, entry['signal'], entry['df'])
                            del self.pending_entries[symbol]

                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Confirmation Loop Error: {e}")
                await asyncio.sleep(5)

    async def add_to_pending(self, symbol, signal, df):
        if symbol in self.trap_blacklist:
            if datetime.now() < self.trap_blacklist[symbol]: return
            else: del self.trap_blacklist[symbol]

        if symbol in self.market.positions or symbol in self.pending_entries: return
        
        last_candle = df.iloc[-1]
        trigger_price = float(last_candle['high']) if signal['direction'] == 'LONG' else float(last_candle['low'])
            
        self.pending_entries[symbol] = {
            'symbol': symbol,
            'signal': signal,
            'df': df,
            'queued_time': datetime.now(),
            'trigger_price': trigger_price,
            'direction': signal['direction'],
            'state': 'WAIT_BREAK'
        }
        logger.info(f"⏳ QUEUED {symbol} {signal['direction']} | Trigger: {trigger_price:.4f}")

    async def execute_trade(self, symbol, signal, df):
        # 1. Check State
        state = self.pair_states.get(symbol, {'status': 'ACTIVE', 'real_losses': 0})
        is_shadow = state['status'] == 'SHADOW'
        
        curr_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        tp = signal.get('tp_price', 0)
        sl = signal.get('sl_price', 0)
        risk_pct = signal.get('risk_pct', config.RISK_PER_TRADE_PCT)
        
        if not tp or not sl:
             if signal['direction'] == 'LONG':
                sl = curr_price * (1 - config.STOP_LOSS_PCT/100)
                tp = curr_price * (1 + config.TAKE_PROFIT_PCT/100)
             else:
                sl = curr_price * (1 + config.STOP_LOSS_PCT/100)
                tp = curr_price * (1 - config.TAKE_PROFIT_PCT/100)

        # SHADOW EXECUTION
        if is_shadow:
            logger.info(f"👻 SHADOW TRADE: Simulating {symbol} {signal['direction']}...")
            self.shadow_positions[symbol] = {
                'symbol': symbol,
                'side': signal['direction'],
                'entry_price': curr_price,
                'amount': 0, 
                'tp': tp,
                'sl': sl,
                'entry_time': datetime.now(),
                'strategy': signal.get('strategy', 'UNKNOWN')
            }
            return

        # CAPACITY CHECK (Strict)
        if len(self.market.positions) >= config.MAX_OPEN_POSITIONS:
            logger.warning(f"⚠️ CAPACITY FULL: Cannot execute {symbol}. ({len(self.market.positions)}/{config.MAX_OPEN_POSITIONS})")
            return

        # REAL EXECUTION
        amount = self.market.calculate_position_size(symbol, curr_price, sl, override_risk_pct=risk_pct)
        if amount < 12.0: amount = 12.0 # Min Boost
        
        if amount < 6.0: return # Skip too small
        
        margin_needed = amount / config.LEVERAGE
        if margin_needed > self.market.balance: return
        
        logger.info(f"⚡ EXECUTING {symbol} | SL: {sl:.4f} | TP: {tp:.4f} | Size: ${amount:.0f}")
        
        result = await self.market.open_position(symbol, signal['direction'], amount, sl, tp)
        
        if result:
            mode = signal.get('strategy_mode', 'TREND')
            if symbol in self.market.positions:
                 self.market.positions[symbol]['strategy_mode'] = mode
            logger.info(f"🔫 OPEN SUCCESS {symbol} ({mode}) | Size: ${amount:.2f}")

    async def manage_positions(self):
        """TITAN V5: RATCHET + SCALP HYBRID"""
        for symbol, pos in list(self.market.positions.items()):
            current_price = await self.market.get_current_price(symbol)
            if current_price == 0: continue
            
            duration_sec = (datetime.now() - pos['entry_time']).total_seconds()
            
            # PnL Calculation
            pnl_pct = 0.0
            if pos['side'] == 'LONG':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
                
            # 1. HARD STOPS
            is_sl = False
            is_tp = False
            if pos['side'] == 'LONG':
                if current_price <= pos['sl']: is_sl = True
                if current_price >= pos['tp']: is_tp = True
            else:
                if current_price >= pos['sl']: is_sl = True
                if current_price <= pos['tp']: is_tp = True
                
            if is_sl:
                logger.warning(f"🛑 STOP LOSS HIT: {symbol}")
                res = await self.market.close_position(symbol, "STOP LOSS")
                if res:
                    self.blacklist.record_loss(symbol)
                    await self.register_real_loss(symbol) # PRE-BAN TRIGGER
                continue
                
            if is_tp:
                await self.market.close_position(symbol, "TAKE PROFIT")
                # Reset streak if profitable?
                if symbol in self.pair_states: self.pair_states[symbol]['real_losses'] = 0
                continue
                
            # 2. STRATEGY MANAGEMENT
            mode = pos.get('strategy_mode', 'TREND')
            
            if mode == 'SCALP':
                # Fast Break Even
                if pnl_pct >= 0.015 and not pos.get('be_locked', False):
                    buffer = 0.001
                    new_sl = pos['entry_price'] * (1 + buffer if pos['side'] == 'LONG' else 1 - buffer)
                    pos['sl'] = new_sl
                    pos['be_locked'] = True
                    logger.info(f"🛡️ SCALP: Fast BE {symbol}")
                
            else: # TREND (Ratchet)
                # Get ATR
                try:
                    df = await self.market.get_klines(symbol, interval=config.TIMEFRAME, limit=20)
                    atr = self.detector.calculate_atr(df).iloc[-1]
                except: atr = 0
                
                if atr > 0:
                     pnl_atr = ((current_price - pos['entry_price']) if pos['side'] == 'LONG' else (pos['entry_price'] - current_price)) / atr
                     
                     # Immortal Lock (1 ATR)
                     if pnl_atr >= 1.0 and not pos.get('be_locked', False):
                          buffer = current_price * 0.001
                          new_sl = pos['entry_price'] + buffer if pos['side'] == 'LONG' else pos['entry_price'] - buffer
                          pos['sl'] = new_sl
                          pos['be_locked'] = True
                          logger.info(f"🔒 RATCHET: Immortal Lock {symbol}")
                          
                     # Step Ladder (Every 0.5 ATR)
                     if pos.get('be_locked', False):
                         current_step = int(pnl_atr * 2) / 2
                         if current_step >= 1.5:
                             target_sl_dist = (current_step - 0.5) * atr
                             new_sl = pos['entry_price'] + target_sl_dist if pos['side'] == 'LONG' else pos['entry_price'] - target_sl_dist
                             
                             update = False
                             if pos['side'] == 'LONG' and new_sl > pos['sl']: update = True
                             if pos['side'] == 'SHORT' and new_sl < pos['sl']: update = True
                             
                             if update:
                                 pos['sl'] = new_sl
                                 logger.info(f"🪜 RATCHET: Step {current_step} {symbol}")

            # 3. Timeout
            if duration_sec > config.MAX_POSITION_TIME_SEC:
                await self.market.close_position(symbol, "MAX TIME LIMIT")

    async def reporting_loop(self):
        logger.info("Started Reporter...")
        while self.running:
            try:
                total_equity = self.market.balance # fallback
                if not self.market.is_dry_run:
                    # 1. FORCE SYNC (The Fix)
                    await self.market.sync_positions()
                    
                    st = await self.market.get_real_account_status()
                    if st: total_equity = st['equity']
                
                logger.info(f"--- 📊 STATUS: Equity ${total_equity:.2f} | Open: {len(self.market.positions)} ---")
                
                # SHOW OPEN POSITIONS
                if self.market.positions:
                   for sym, pos in self.market.positions.items():
                       # Estimate PnL
                       try:
                           cp = await self.market.get_current_price(sym)
                           if pos['side'] == 'LONG':
                               pnl_pct = (cp - pos['entry_price']) / pos['entry_price'] * 100
                           else:
                            pnl_pct = (pos['entry_price'] - cp) / pos['entry_price'] * 100
                               
                           duration_m = (datetime.now() - pos['entry_time']).total_seconds() / 60
                           
                           # Calc Percentages
                           sl_dist = abs(pos['sl'] - pos['entry_price'])
                           tp_dist = abs(pos['tp'] - pos['entry_price'])
                           
                           sl_pct = (sl_dist / pos['entry_price']) * 100
                           tp_pct = (tp_dist / pos['entry_price']) * 100
                           
                           # Format Logging (Percent)
                           logger.info(f"   👉 {sym}: {pos['side']} | PnL: {pnl_pct:+.2f}% | Entry: {pos['entry_price']:.4f} | SL: {sl_pct:.2f}% | TP: {tp_pct:.2f}%")
                       except: pass
                       
                await asyncio.sleep(config.MONITOR_INTERVAL)
            except Exception as e:
                logger.error(f"Reporter Error: {e}")
                await asyncio.sleep(60)

    async def check_watchlist(self): pass

if __name__ == "__main__":
    import os
    
    # 0. Load .env manually to avoid extra dependencies
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        print(f">>> Loading config from {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    os.environ[key] = val.strip()

    dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    
    bot = TrendBot(is_dry_run=dry_run, api_key=api_key, api_secret=api_secret)
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("Bot stopped.")
