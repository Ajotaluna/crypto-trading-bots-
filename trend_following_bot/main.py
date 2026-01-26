"""
Main Bot Logic
Integrates Data, Patterns, and Execution.
"""
import asyncio
import logging
import sys
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trend_bot.log")
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
        
        # PRODUCTION TRADING LIST (Pre-load Majors)
        self.active_trading_list = list(self.calibrator.vip_majors)
        if self.active_trading_list:
            logger.info(f"🚀 BOOTSTRAP: Loaded {len(self.active_trading_list)} Major Pairs immediately.")
        
        # SHADOW MODE STATE MACHINE (Pre-Ban Logic)
        self.pair_states = {} 
        self.shadow_positions = {} 
        
        # SCOREBOARD
        self.tracker = WinRateTracker()
        self.tracker.log_summary()
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

    async def start(self):
        """Main Entry Point & Orchestrator"""
        logger.info(">>> STARTING TREND BOT (PRODUCTION) <<<")
        
        # 1. Initialize Balance
        if not self.market.is_dry_run:
             status = await self.market.get_real_account_status()
             if status:
                 self.start_balance = status['equity']
                 logger.info(f"Initial Equity: ${self.start_balance:.2f}")
        else:
             self.start_balance = self.market.balance

        # 2. Launch Background Loops
        asyncio.create_task(self.reporting_loop())
        asyncio.create_task(self.safety_loop()) 
        asyncio.create_task(self.confirmation_loop())
        
        # 3. Start Continuous Scanner
        self.scanner_task = asyncio.create_task(self.continuous_scanner_loop())
        
        # 4. Maintenance & Calibration Loop
        last_full_scan = 0
        
        while self.running:
            try:
                now = time.time()
                is_full_scan = False
                
                # Determine Scan Type
                if now - last_full_scan > (24 * 3600):
                    logger.info("🕒 DAILY MAINTENANCE: Full Market Scan (Volume + Trending)")
                    is_full_scan = True
                    last_full_scan = now
                else:
                    logger.info("🕒 4H UPDATE: Quick Scan (Trending Only)")
                
                # A. Scan Candidates
                candidates = await self.scan_market_candidates(full_scan=is_full_scan)
                if not candidates:
                    logger.warning("No candidates found. Retrying in 10m...")
                    await asyncio.sleep(600)
                    continue
                    
                # B. Calibrate (Tournament)
                # This updates the strategy map and returns the winners
                approved_map = self.calibrator.calibrate(candidates)
                
                # C. Update Active List
                new_list = list(approved_map.keys())
                if new_list:
                    self.active_trading_list = new_list
                    logger.info(f"✅ Active Trading List Updated ({len(new_list)} pairs): {new_list}")
                
                # Wait 4 Hours before next check
                await asyncio.sleep(4 * 3600)
                
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
            
            # 2. Volume - Only on Full Scan
            if full_scan:
                vol_pairs = await self.market.get_top_volume_pairs(limit=50)
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
        
        while self.running:
            try:
                # 1. Capacity Check
                if len(self.market.positions) >= config.MAX_OPEN_POSITIONS:
                    await asyncio.sleep(30)
                    continue

                # 2. Refresh Priority List
                if not self.active_trading_list:
                    logger.warning("No Active Trading List! Waiting for Calibration...")
                    await asyncio.sleep(10)
                    continue
                    
                symbols = self.active_trading_list 
                
                # 3. Global Conditions
                btc_trend = await self.market.get_btc_trend()
                if btc_trend < -0.01:
                    logger.warning(f"🛡️ KING'S GUARD: BTC Crash ({btc_trend*100:.2f}%). Pausing.")
                    await asyncio.sleep(60)
                    continue
                    
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

        # REAL EXECUTION
        amount = self.market.calculate_position_size(symbol, curr_price, sl, override_risk_pct=risk_pct)
        if amount < 12.0: amount = 12.0 # Min Boost
        
        if amount < 6.0: return # Skip too small
        
        margin_needed = amount / config.LEVERAGE
        if margin_needed > self.market.balance: return
        
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
                    st = await self.market.get_real_account_status()
                    if st: total_equity = st['equity']
                
                logger.info(f"--- STATUS: Equity ${total_equity:.2f} | Open: {len(self.market.positions)} ---")
                await asyncio.sleep(config.MONITOR_INTERVAL)
            except Exception:
                await asyncio.sleep(60)

    async def check_watchlist(self): pass

if __name__ == "__main__":
    import os
    dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    
    bot = TrendBot(is_dry_run=dry_run, api_key=api_key, api_secret=api_secret)
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("Bot stopped.")
