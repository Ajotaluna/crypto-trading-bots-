"""
Main Bot Logic
Integrates Data, Patterns, and Execution.
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta

# Local imports
from config import config
from market_data import MarketData
from patterns import PatternDetector, SentimentAnalyzer
from win_rate_tracker import WinRateTracker
import pandas as pd
from blacklist_manager import BlacklistManager

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
        self.running = True
        self.start_balance = 0.0
        self.watchlist = {} # Symbol -> Score
        self.pending_entries = {}
        self.trap_blacklist = {} # {symbol: expiry_time} # {symbol: {signal, time, trigger_price}}
        
        # SCOREBOARD (Persistent Stats)
        self.tracker = WinRateTracker()
        self.tracker.log_summary()
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

    def __del__(self):
        self.executor.shutdown(wait=False)
        
    async def start(self):
        await self.market.initialize_balance()
        await self.market.sync_positions() # Recover Phantom Positions
        self.start_balance = self.market.balance
        
        mode = "TEST (DRY RUN)" if self.market.is_dry_run else "PRODUCTION (REAL MONEY)"
        logger.info(f">>> TREND FOLLOWING BOT V2 (REAL-TIME) STARTED [{mode}] <<<")
        logger.info(f"Config: Score {config.MIN_SIGNAL_SCORE}+ | Hold {config.MIN_POSITION_TIME_SEC/3600}h-{config.MAX_POSITION_TIME_SEC/3600}h")
        logger.info(f"Daily Target: {config.DAILY_PROFIT_TARGET_PCT}% (Stop at ${self.start_balance * (1 + config.DAILY_PROFIT_TARGET_PCT/100):.2f})")
        
        # Run loops concurrently
        await asyncio.gather(
            self.safety_loop(),
            self.reporting_loop(),
            self.confirmation_loop(),
            self.slow_scan()
        )
        
    async def safety_loop(self):
        """REAL-TIME SAFETY: Monitors SL/TP & Exits (Every 5s)"""
        logger.info("Started Safety Monitor...")
        while self.running:
            try:
                # 1. Manage Open Positions (CRITICAL)
                await self.manage_positions()
                
                await asyncio.sleep(config.SAFETY_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Safety Loop Error: {e}")
                await asyncio.sleep(1)

    async def reporting_loop(self):
        """REPORTING LOOP: Logs Status Loop (Every 5m)"""
        logger.info("Started Status Reporter...")
        while self.running:
            try:
                # 0. DUAL REPORTING LOGIC (REAL vs DRY RUN)
                margin_used = 0.0
                unrealized_pnl_usdt = 0.0
                total_equity = 0.0
                
                if not self.market.is_dry_run:
                    # --- PRODUCTION: USE BINANCE "SOURCE OF TRUTH" ---
                    real_status = await self.market.get_real_account_status()
                    if real_status:
                        total_equity = real_status['equity']
                        unrealized_pnl_usdt = real_status['unrealized_pnl']
                        # Calculate Margin from Balance - Available (approx) or iterate positions if needed
                        # Ideally get 'totalInitialMargin' from API if available, but for now we trust Equity.
                        # We don't strictly need 'margin_used' for the trigger, just Equity.
                        
                        logger.debug(f"[REAL] API Status: Eq={total_equity}, PnL_Unr={unrealized_pnl_usdt}")
                    else:
                        logger.warning("Failed to fetch Real Account Status. Skipping report.")
                        await asyncio.sleep(5)
                        continue
                else:
                    # --- DRY RUN: USE MANUAL CALCULATOR ---
                    if self.market.positions:
                        for sym, pos in self.market.positions.items():
                            # Calculate Margin Used (USDT Value / Leverage)
                            notional_value = pos['amount'] * pos['entry_price']
                            margin_used += notional_value / config.LEVERAGE
                            
                            # Calculate Unrealized PnL (USDT)
                            curr_price = await self.market.get_current_price(sym)
                            if curr_price > 0:
                                if pos['side'] == 'LONG':
                                    pnl_val = pos['amount'] * (curr_price - pos['entry_price'])
                                else:
                                    pnl_val = pos['amount'] * (pos['entry_price'] - curr_price)
                                
                                unrealized_pnl_usdt += pnl_val
                    
                    # Formula: Equity = Balance (Available) + Margin + Unrealized PnL
                    # WAIT! In "Tracker Mode" (Dry Run), we typically DO NOT deduct margin from self.balance anymore.
                    # So self.balance is already the Full Principal ($1000).
                    # If we add Margin Used ($66) to it, we create Fake Money ($1066).
                    
                    # CORRECTION:
                    # If Dry Run (Tracker Mode): Equity = Gross Balance + PnL
                    # If Real Mode: Handled above by API.
                    
                    total_equity = self.market.balance + unrealized_pnl_usdt
                
                # Formula: Total PnL % = (Current Equity - Initial Daily Balance) / Initial Daily Balance
                if self.start_balance > 0:
                    total_pnl_pct = (total_equity - self.start_balance) / self.start_balance
                else:
                    total_pnl_pct = 0.0
                
                # Log Status
                if self.market.positions:
                    logger.info(f"--- STATUS REPORT (Equity: {total_equity:.2f} | PnL: {total_pnl_pct*100:.2f}%) ---")
                    for sym, pos in self.market.positions.items():
                        curr_price = await self.market.get_current_price(sym)
                        pnl_roi = 0.0
                        if curr_price > 0:
                            if pos['side'] == 'LONG':
                                pnl_roi = (curr_price - pos['entry_price']) / pos['entry_price'] * 100
                            else:
                                pnl_roi = (pos['entry_price'] - curr_price) / pos['entry_price'] * 100
                        
                        duration_min = (datetime.now() - pos['entry_time']).total_seconds() / 60
                        logger.info(f"{sym} {pos['side']} | ROI: {pnl_roi:.2f}% | Time: {duration_min:.1f}m")
                else:
                    logger.info(f"--- STATUS REPORT: No Open Positions (PnL: {total_pnl_pct*100:.2f}%) ---")
                    
                # TARGET HIT LOGIC (Simple Reset)
                if total_pnl_pct >= (config.DAILY_PROFIT_TARGET_PCT/100):
                    logger.info(f"🏆 DAILY TARGET HIT (+{total_pnl_pct*100:.2f}%)! Stopping to Secure Profits.")
                    
                    # 1. Force Close All (Bank it)
                    for symbol in list(self.market.positions.keys()):
                        await self.market.close_position(symbol, "TARGET REACHED")
                    
                    # 2. Reset Monthly/Daily Tracker (Start New Loop)
                    # We update the baseline to the current equity so PnL goes back to 0.0%
                    self.start_balance = total_equity
                    self.market.cumulative_pnl_daily = 0.0
                    
                    logger.info(f"🔄 Loop Reset. New Baseline: ${self.start_balance:.2f}. Searching for next +{config.DAILY_PROFIT_TARGET_PCT}%...")
                    await asyncio.sleep(5) # Cooldown
                    continue
                
                # Check Watchlist just in case (optional, low priority)
                if len(self.market.positions) < config.MAX_OPEN_POSITIONS:
                    await self.check_watchlist()
                
                await asyncio.sleep(config.MONITOR_INTERVAL)
                
            except Exception as e:
                logger.error(f"Reporting Loop Error: {e}")
                await asyncio.sleep(60)

    async def slow_scan(self):
        """BACKGROUND LOOP: Batch Scan & Fill (Every 60s)"""
        logger.info("Started Batch Scanner...")
        while self.running:
            try:
                # Only scan if we have empty slots
                open_slots = config.MAX_OPEN_POSITIONS - len(self.market.positions)
                
                if open_slots > 0:
                    logger.info(f"Scanning to fill {open_slots} slots...")
                    await self.scan_and_fill_batch(open_slots)
                else:
                    logger.info("Slots full (10/10). Waiting for positions to close...")
                    # IMPORTANT: Wait to prevent tight loop if full
                    await asyncio.sleep(60) 
                    continue # Skip the rest of loop to re-check
                
                await asyncio.sleep(config.CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Batch Loop Error: {e}")
                await asyncio.sleep(5)

    async def scan_and_fill_batch(self, slots_needed):
        """Analyze market and pick TOP N candidates (Parallel Engine)"""
        # 0. THE KING'S GUARD (BTC Trend Filter)
        btc_trend = await self.market.get_btc_trend()
        
        if btc_trend < -0.01: # Check for CRASH (Net Drop > 1%)
            logger.warning(f"🛡️ KING'S GUARD: BTC IS CRASHING ({btc_trend*100:.2f}%)! Pausing Scans.")
            return

        # 0b. THE WATCHTOWER (Market Breadth)
        breadth = await self.market.get_market_breadth(limit=50)
        logger.info(f"🔭 WATCHTOWER: Market Sentiment: {breadth['bullish_pct']:.1f}% Bullish ({breadth['sentiment']})")

        # CALENDAR FILTER & PROFIT STOP (USER REQUEST)
        today = datetime.now()
        current_hour = today.utcnow().hour
        is_tokyo_blackout = 3 <= current_hour < 10
        
        # Check Daily Profit Target
        daily_target_hit = False
        try:
            current_pnl_pct = self.pnl_tracker.current_pnl_pct * 100 # Convert to %
            if current_pnl_pct >= config.DAILY_PROFIT_TARGET_PCT:
                daily_target_hit = True
        except:
            pass # PnL tracker might not be ready

        # ------------------------------------------------------------------
        # BLACKOUTS REMOVED - 24/7 OPERATION
        # ------------------------------------------------------------------
        pass

        # ------------------------------------------------------------------
        # ACTIVE TRADING MODE (09:00 - 23:59 UTC)
        # ------------------------------------------------------------------
        # Base Settings (Sniper Mode Defaults)
        min_score_required = config.MIN_SIGNAL_SCORE
        allow_override = config.ALLOW_MOMENTUM_OVERRIDE
        
        mode_name = "SNIPER MODE (DAYTIME)"
        
        # No more Tokyo Guard logic needed since we don't trade Tokyo.
        # logger.info(f"📅 CALENDAR MODE: {mode_name} | Min Score: {min_score_required} | Override: {allow_override}")
            
        symbols = await self.market.get_top_symbols(limit=None)
        final_candidates = []
        loop = asyncio.get_running_loop()

        # --- PARALLEL ANALYSIS ENGINE ---
        async def analyze_symbol(symbol):
            try:
                # Blacklist Check
                if not self.blacklist.is_allowed(symbol): return None
                if symbol in self.market.positions: return None
                
                # 1. Fetch Data
                df = await self.market.get_klines(symbol, interval=config.TIMEFRAME)
                df_daily = await self.market.get_klines(symbol, interval='1d', limit=90)
                if df.empty or df_daily.empty: return None
                
                # 2. Analyze Technicals
                tech_signal = await loop.run_in_executor(self.executor, self.detector.analyze, df, df_daily, btc_trend)
                if not tech_signal: return None
                
                # [REMOVED] WEEKDAY VOLUME FILTER
                # User requested full Weekend Logic always.
                
                # --- MOMENTUM OVERRIDE CHECK ---
                is_override = False
                if allow_override:
                    if tech_signal['score'] >= config.MOMENTUM_SCORE_THRESHOLD:
                        is_override = True
                
                # 3. TITAN DATA & CHECKS
                if not is_override:
                    if tech_signal['score'] < min_score_required: return None
                    
                    
                    # King's Guard (Already handled in analyze, but double check doesn't hurt)
                    if btc_trend < -0.01 and tech_signal['direction'] == 'LONG': return None
                    if btc_trend > 0.01 and tech_signal['direction'] == 'SHORT': return None

                    # Titan Sentiment (Simplified for speed in parallel)
                    try:
                        oi_data = await self.market.get_open_interest(symbol, period='1h')
                        top_ls_data = await self.market.get_top_long_short_ratio(symbol, period='1h')
                        global_ls_data = await self.market.get_global_long_short_ratio(symbol, period='1h')
                        sentiment = SentimentAnalyzer.analyze_sentiment(oi_data, top_ls_data, global_ls_data)
                        
                        valid_titan = False
                        if tech_signal['direction'] == 'LONG' and sentiment['signal'] == 'BULLISH':
                            valid_titan = True
                            tech_signal['score'] += 10
                        elif tech_signal['direction'] == 'SHORT' and sentiment['signal'] == 'BEARISH':
                            valid_titan = True
                            tech_signal['score'] += 10
                            
                        if not valid_titan: return None
                    except:
                        # If Titan API fails, be conservative and skip finding
                        return None
                
                return {
                    'symbol': symbol,
                    'df': df,
                    'signal': tech_signal,
                    'score': tech_signal['score'],
                    'is_override': is_override
                }
            except Exception as e:
                return None

        # EXECUTE PARALLEL ANALYSIS
        logger.info(f"⚡ Analyzing {len(symbols)} pairs in PARALLEL...")
        results = await asyncio.gather(*[analyze_symbol(sym) for sym in symbols])
        
        # Filter valid results
        final_candidates = [r for r in results if r is not None]

        # Sort by Score (Highest first)
        final_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Pick Top N
        top_picks = final_candidates[:slots_needed]
        
        if top_picks:
            logger.info(f"Found {len(top_picks)} candidates. Processing...")
            for pick in top_picks:
                
                # STABILITY FILTER V5: Trend Alignment
                # Weekday Rule: MANDATORY ALIGNMENT (No excuses)
                # Weekend Rule (Now 24/7): Skippable for Overrides
                if pick['is_override']: must_align = False
                
                if must_align:
                    try:
                        df_1h = await self.market.get_klines(pick['symbol'], interval=config.TREND_ALIGN_INTERVAL, limit=config.TREND_ALIGN_EMA + 10)
                        if not df_1h.empty and len(df_1h) > config.TREND_ALIGN_EMA: 
                            ema_200 = df_1h['close'].ewm(span=config.TREND_ALIGN_EMA, adjust=False).mean().iloc[-1]
                            price = df_1h.iloc[-1]['close']
                            # COUNTER TREND CHECK
                            if pick['signal']['direction'] == 'LONG' and price < ema_200: continue
                            if pick['signal']['direction'] == 'SHORT' and price > ema_200: continue
                    except: pass
                
                if config.SMART_ENTRY_ENABLED:
                    await self.add_to_pending(pick['symbol'], pick['signal'], pick['df'])
                else:
                    await self.execute_trade(pick['symbol'], pick['signal'], pick['df'])
        else:
            logger.info("No suitable candidates found.")

    async def check_watchlist(self):
        """Legacy method"""
        pass

    async def add_to_pending(self, symbol, signal, df):
        """Queue trade for Smart Entry (Confirmation)"""
        # 0. Check Trap Blacklist
        if symbol in self.trap_blacklist:
            if datetime.now() < self.trap_blacklist[symbol]:
                # logger.info(f"🚫 Ignoring {symbol} (Trap Blacklist Active)")
                return
            else:
                del self.trap_blacklist[symbol] # Expired

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
        logger.info(f"⏳ QUEUED {symbol} {signal['direction']} | Wait for break of {trigger_price:.4f}")

    async def confirmation_loop(self):
        """SMART ENTRY V4: Anti-Trap Logic (5m Confirmation)"""
        logger.info("Started Smart Entry Monitor (V4 - Anti-Trap)...")
        while self.running:
            try:
                current_time = datetime.now()
                for symbol, entry in list(self.pending_entries.items()):
                    # check timeout
                    if (current_time - entry['queued_time']).total_seconds() > (config.CONFIRMATION_TIMEOUT_MINS * 60):
                        logger.info(f"🗑️ EXPIRED {symbol}")
                        del self.pending_entries[symbol]
                        continue
                        
                    # 1. GET 5M CANDLES (Stricter Timeframe)
                    df_5m = await self.market.get_klines(symbol, interval='5m', limit=25)
                    if df_5m.empty: continue
                    
                    # Calculate RSI for Trap Detection
                    df_5m['rsi'] = self.detector.calculate_atr(df_5m) # Placeholder calc logic? No let's use Patterns
                    # Actually we need PatternDetector instance to calc RSI quickly or manual calc
                    # Let's do a quick manual RSI here to avoid overhead
                    try:
                        delta = df_5m['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / loss
                        df_5m['rsi_val'] = 100 - (100 / (1 + rs))
                        current_rsi = df_5m.iloc[-1]['rsi_val']
                    except:
                        current_rsi = 50.0

                    closed_candle = df_5m.iloc[-2] # Confirmed Candle
                    last_candle = df_5m.iloc[-1]   # Current Tick
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
                                logger.warning(f"⚠️ TRAP DETECTED {symbol}: Breakout but RSI Extended ({current_rsi:.1f}). Blacklisting for 60m.")
                                self.trap_blacklist[symbol] = datetime.now() + timedelta(minutes=60)
                                del self.pending_entries[symbol]
                                continue
                                
                            entry['state'] = 'WAIT_RETEST'
                            logger.info(f"💥 BREAKOUT CONFIRMED (5m) for {symbol}! Waiting for Pullback...")

                    elif state == 'WAIT_RETEST':
                        retest_success = False
                        if entry['direction'] == 'LONG':
                            limit_buy = trigger * 1.002
                            if float(last_candle['low']) <= limit_buy: retest_success = True
                        else:
                            limit_sell = trigger * 0.998
                            if float(last_candle['high']) >= limit_sell: retest_success = True
                                
                        if retest_success:
                            logger.info(f"🎯 SNIPER ENTRY V4 {symbol}")
                            await self.execute_trade(symbol, entry['signal'], entry['df'])
                            del self.pending_entries[symbol]

                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Confirmation Loop Error: {e}")
                await asyncio.sleep(5)

    async def execute_trade(self, symbol, signal, df):
        # 1. GET PRICE
        price = df.iloc[-1]['close']
        
        # 2. GET DYNAMIC SL/TP FROM SIGNAL (Titan V3)
        # These were calculated by PatternDetector based on ATR/Volatility
        sl = signal.get('sl_price', 0)
        tp = signal.get('tp_price', 0)
        risk_pct = signal.get('risk_pct', config.RISK_PER_TRADE_PCT)
        
        # Fallback (Safety)
        if not sl or not tp:
             logger.warning(f"⚠️ Dynamic Levels missing for {symbol}. Falling back to config.")
             if signal['direction'] == 'LONG':
                sl = price * (1 - config.STOP_LOSS_PCT/100)
                tp = price * (1 + config.TAKE_PROFIT_PCT/100)
             else:
                sl = price * (1 + config.STOP_LOSS_PCT/100)
                tp = price * (1 - config.TAKE_PROFIT_PCT/100)

        # 3. THE RISK VAULT: Calculate Position Size based on Risk
        # TITAN V3: Use the Dynamic 'risk_pct' from the signal (0.5% - 2.0%)
        # Note: We do not apply the "Casino Manager" scaler anymore because 
        # Risk Factor is already built into risk_pct.
        
        amount = self.market.calculate_position_size(symbol, price, sl, override_risk_pct=risk_pct)
        
        # LOW CAPITAL OVERRIDE: Force Minimum Size $12
        # User Strategy: Ensure size is enough for 50% Partial Close ($6) > Min Notional ($5)
        if amount < 12.0:
            logger.info(f"⚠️ LOW CAP SIZE ADJUST: Boosting {symbol} position from ${amount:.2f} to $12.00 to enable Partial TP.")
            amount = 12.0

        # Safety Check (should be covered by above, but keeping for logic integrity)
        if amount < 6.0:
            logger.warning(f"⚠️ Position Size {amount:.2f} too small for {symbol}. Skipping.")
            return

        # Check affordability (Margin)
        margin_needed = amount / config.LEVERAGE
        if margin_needed > self.market.balance:
            logger.warning(f"Insufficient funds for {symbol}. Need Margin ${margin_needed:.2f}, have ${self.market.balance:.2f}")
            # Optional: Resize to Max Balance? No, violates Risk Vault. Skip.
            return
        
        # logger.info(f"🔫 OPENING {symbol} ...") <- REMOVED to avoid confusion
        
        result = await self.market.open_position(symbol, signal['direction'], amount, sl, tp)
        
        if result:
            # INJECT STRATEGY MODE & DYNAMIC SCALP TARGET
            mode = signal.get('strategy_mode', 'TREND')
            partial_tp = signal.get('partial_tp_price', 0.0)
            
            if symbol in self.market.positions:
                 self.market.positions[symbol]['strategy_mode'] = mode
                 self.market.positions[symbol]['partial_tp'] = partial_tp
            
            # LOG THE REAL EXECUTION PRICE (Honesty Fix)
            real_entry = result['entry_price']
            logger.info(f"🔫 OPEN SUCCESS {symbol} ({mode} MODE) {signal['direction']} | Size: ${amount:.2f} | Entry: {real_entry:.4f} | SL: {sl:.4f} | TP: {tp:.4f}")
        else:
            logger.error(f"❌ EXECUTION FAILED for {symbol}. Check logs for details (Precision/Margin/API).")

    async def manage_positions(self):
        """Real-time position management"""
        for symbol, pos in list(self.market.positions.items()):
            current_price = await self.market.get_current_price(symbol)
            if current_price == 0: continue
            
            # Calculate duration
            duration_sec = (datetime.now() - pos['entry_time']).total_seconds()
            
            # 1. HEARTBEAT MONITOR (LIFECYCLE TRACKER)
            # Log the vital signs of the trade constantly
            # ENABLED FOR ALL MODES (User Request)
            
            # Calculate Distances
            dist_tp_pct = abs(pos['tp'] - current_price) / current_price * 100
            dist_sl_pct = abs(current_price - pos['sl']) / current_price * 100
            
            # Calculate Current PnL
            if pos['side'] == 'LONG':
                curr_pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
            else:
                curr_pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
                
            logger.info(f"💓 [TRACKER] {symbol}: Price {current_price:.5f} | ROI {curr_pnl_pct:+.2f}% | To TP: {dist_tp_pct:.2f}% | To SL: {dist_sl_pct:.2f}%")
            
            # 1. Check Hard SL/TP (Instant)
            # Calculate PnL (ROI)
            pnl_pct = 0.0
            if pos['side'] == 'LONG':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                if current_price <= pos['sl']:
                    logger.warning(f"🛑 STOP LOSS HIT: {symbol} at {current_price}")
                    
                    # 1. ATTEMPT CLOSE FIRST
                    result = await self.market.close_position(symbol, "STOP LOSS")
                    
                    if result:
                        # SUCCESS: Record Loss & Stats
                        self.blacklist.record_loss(symbol) 
                        roi = (current_price - pos['entry_price']) / pos['entry_price']
                        pnl_usdt = roi * pos['amount'] * config.LEVERAGE
                        self.tracker.record_trade(pnl_usdt, symbol)
                    else:
                        # SURVIVAL MODE (Simple)
                        # API Failed to close. We do NOT record loss. We wait for next loop.
                        # If price recovers > SL, we survive.
                        logger.error(f"❌ CLOSE FAILED for {symbol}. Keeping position open for potential recovery.")
                    
                    continue
                if current_price >= pos['tp']:
                    await self.market.close_position(symbol, "TAKE PROFIT")
                    continue
            else:
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
                if current_price >= pos['sl']:
                    logger.warning(f"🛑 STOP LOSS HIT: {symbol} at {current_price}")
                    
                    # 1. ATTEMPT CLOSE FIRST
                    result = await self.market.close_position(symbol, "STOP LOSS")
                    
                    if result:
                        # SUCCESS
                        self.blacklist.record_loss(symbol)
                        # SHORT ROI = (Entry - Exit) / Entry
                        roi = (pos['entry_price'] - current_price) / pos['entry_price']
                        pnl_usdt = roi * pos['amount'] * config.LEVERAGE
                        self.tracker.record_trade(pnl_usdt, symbol)
                    else:
                        # SURVIVAL MODE (Simple)
                        logger.error(f"❌ CLOSE FAILED for {symbol}. Keeping position open for potential recovery.")
                        
                    continue
                if current_price <= pos['tp']:
                    # SCOREBOARD UPDATE
                    roi = (pos['entry_price'] - current_price) / pos['entry_price']
                    pnl_usdt = roi * pos['amount'] * config.LEVERAGE # Estimating
                    self.tracker.record_trade(pnl_usdt, symbol)
                    
                    await self.market.close_position(symbol, "TAKE PROFIT")
                    continue
            
            # 1b. THE HARVESTER: Secure Profits (Break Even & Partial TP)
            # We track R-Multiple (Profit / Risk) where Risk = |Entry - Initial SL|
            # Note: We need to store Initial SL in pos dict. If not present, estimate it.
            
            initial_sl = pos.get('initial_sl', pos['sl']) # Fallback to current SL if missing
            
            # 1b. THE HARVESTER V2: Simplified ROI-Based Management
            # We use strict ROI targets to secure bags early.
            
            # STATE TRACKING
            has_moved_to_be = pos.get('be_locked', False)
            has_taken_partial = pos.get('partial_taken', False)
            
            current_roi = pnl_pct
            
            # A. ADAPTIVE MANAGEMENT (Titan V4)
            mode = pos.get('strategy_mode', 'TREND') # Legacy positions treated as TREND
            
            # --- SCALP MODE LOGIC ---
            if mode == 'SCALP':
                # 1. Quick Break Even
                if current_roi >= 0.015 and not has_moved_to_be: # 1.5% Trigger
                     # Move SL to Entry
                     buffer = 0.001
                     new_sl = pos['entry_price'] * (1 + buffer if pos['side'] == 'LONG' else 1 - buffer)
                     pos['sl'] = new_sl
                     pos['be_locked'] = True
                     logger.info(f"🛡️ SCALP: Fast BE locked for {symbol} @ {new_sl:.4f}")

                # 2. Harvest (Partial Take Profit) - Dynamic ATR Target
                partial_tp = pos.get('partial_tp', 0.0)
                harvest_triggered = False
                
                if partial_tp > 0:
                    if pos['side'] == 'LONG' and current_price >= partial_tp: harvest_triggered = True
                    if pos['side'] == 'SHORT' and current_price <= partial_tp: harvest_triggered = True
                
                # Fallback to old 2.0% if missing (Legacy)
                if partial_tp == 0 and current_roi >= 0.020: harvest_triggered = True
                
                if harvest_triggered and not pos.get('harvested', False):
                     # Close 50%
                     await self.market.close_position(symbol, "SCALP_HARVEST", params={'qty': pos['amount'] * 0.5})
                     pos['harvested'] = True
                     logger.info(f"💰 SCALP: Harvested 50% of {symbol} @ {current_price:.4f} (Dynamic Target Hit)")
                     
            # --- TREND MODE LOGIC ---
            else: # TREND
                # 1. Patient Break Even
                if current_roi >= 0.030 and not has_moved_to_be: # 3.0% Trigger
                     buffer = 0.001
                     new_sl = pos['entry_price'] * (1 + buffer if pos['side'] == 'LONG' else 1 - buffer)
                     pos['sl'] = new_sl
                     pos['be_locked'] = True
                     logger.info(f"🛡️ TREND: Break Even locked for {symbol} @ {new_sl:.4f} (ROI > 3%)")

            # B. PARTIAL PROFIT - DISABLED FOR TITAN V3 (PURE TREND)
            # Logic Removed. We hold for the big move.
            pass

            # 1c. THE BLOODHOUND V3 (ATR Trailing Stop)
            # Only trail IF we have already locked Break Even (Don't choke early trade)
            if has_moved_to_be:
                try:
                    # Get ATR for current volatility
                    df_atr = await self.market.get_klines(symbol, interval=config.TIMEFRAME, limit=20)
                    if not df_atr.empty:
                        current_atr = self.detector.calculate_atr(df_atr).iloc[-1]
                        
                        # Determine Trailing Distance based on ROI
                        # < 1% Profit: Loose (3x ATR)
                        # > 1% Profit: Tightening (2x ATR)
                        # > 3% Profit: Sniper (1.5x ATR)
                        multiplier = 3.0
                        if pnl_pct > 0.03: multiplier = 1.5
                        elif pnl_pct > 0.01: multiplier = 2.0
                        
                        trailing_dist = current_atr * multiplier
                        
                        if pos['side'] == 'LONG':
                            new_sl = current_price - trailing_dist
                            # ONLY MOVE UP
                            if new_sl > pos['sl']:
                                pos['sl'] = new_sl
                                logger.info(f"🐕 BLOODHOUND: Trailed SL for {symbol} to {new_sl:.4f} (Price {current_price:.4f} | ATR {current_atr:.4f})")
                        else:
                            new_sl = current_price + trailing_dist
                            # ONLY MOVE DOWN
                            if new_sl < pos['sl']:
                                pos['sl'] = new_sl
                                logger.info(f"🐕 BLOODHOUND: Trailed SL for {symbol} to {new_sl:.4f} (Price {current_price:.4f} | ATR {current_atr:.4f})")
                except Exception as e:
                    logger.error(f"Trailing Stop Error: {e}")

            # 2. Check Major Resistance (REAL TIME PROTECTION)
            # We check this frequently to exit BEFORE SL if hitting a wall
            df = await self.market.get_klines(symbol, interval=config.TIMEFRAME, limit=100)
            if not df.empty:
                major_levels = self.detector.find_major_levels(df)
                for level in major_levels:
                    # DISTINCTION: Support vs Resistance
                    # LONG: We fear Resistance (Level > Price)
                    # SHORT: We fear Support (Level < Price)
                    
                    is_threat = False
                    if pos['side'] == 'LONG':
                        if level > current_price: # Resistance above
                            is_threat = True
                    else:
                        if level < current_price: # Support below
                            is_threat = True
                    
                    if not is_threat: continue

                    # CRITICAL FIX: DO NOT CLOSE LOSING TRADES DUE TO RESISTANCE
                    # We only alert/close if we are protecting profits.
                    # If we are underwater, we rely on the Stop Loss.
                    if pnl_pct <= 0:
                        continue

                    dist = abs(current_price - level) / current_price
                    if dist < 0.005: # Within 0.5% (Very close)
                        # Only exit if we are LOSING momentum or STUCK
                        if self.detector.check_exhaustion(df, pos['side']):
                            type_str = "RESISTANCE" if pos['side'] == 'LONG' else "SUPPORT"
                            await self.market.close_position(symbol, f"MAJOR {type_str} @ {level:.2f}")
                            break

            # 3. Time Constraints
            if duration_sec > config.MAX_POSITION_TIME_SEC:
                await self.market.close_position(symbol, "MAX TIME LIMIT")
                continue

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
