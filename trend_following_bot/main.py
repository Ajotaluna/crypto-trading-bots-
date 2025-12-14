"""
Main Bot Logic
Integrates Data, Patterns, and Execution.
"""
import asyncio
import logging
import sys
from datetime import datetime

# Local imports
from config import config
from market_data import MarketData
from patterns import PatternDetector

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
        self.running = True
        self.start_balance = 0.0
        self.watchlist = {} # Symbol -> Score
        self.pending_entries = {} # Symbol -> {signal, df, time, trigger_price}
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

    def __del__(self):
        self.executor.shutdown(wait=False)
        
    async def start(self):
        await self.market.initialize_balance()
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
                # 0. Check Total Equity (Balance + PnL)
                unrealized_pnl = 0.0
                if self.market.positions:
                    for sym, pos in self.market.positions.items():
                        # Estimate unrealized pnl roughly
                        curr_price = await self.market.get_current_price(sym)
                        if curr_price > 0:
                            entry = pos['entry_price']
                            amt = pos['amount']
                            if pos['side'] == 'LONG':
                                unrealized_pnl += (curr_price - entry) * amt
                            else:
                                unrealized_pnl += (entry - curr_price) * amt
                
                total_equity = self.market.balance + unrealized_pnl
                current_pnl_pct = ((total_equity - self.start_balance) / self.start_balance) * 100
                
                # Log Status
                if self.market.positions:
                    logger.info(f"--- STATUS REPORT (Total Equity: {total_equity:.2f} | PnL: {current_pnl_pct:.2f}%) ---")
                    for sym, pos in self.market.positions.items():
                        curr_price = await self.market.get_current_price(sym)
                        pnl = 0.0
                        if curr_price > 0:
                            entry = pos['entry_price']
                            if pos['side'] == 'LONG':
                                pnl = (curr_price - entry) / entry * 100
                            else:
                                pnl = (entry - curr_price) / entry * 100
                        
                        duration_min = (datetime.now() - pos['entry_time']).total_seconds() / 60
                        logger.info(f"{sym} {pos['side']} | PnL: {pnl:.2f}% | Time: {duration_min:.1f}m")
                else:
                    logger.info(f"--- STATUS REPORT: No Open Positions (PnL: {current_pnl_pct:.2f}%) ---")

                if current_pnl_pct >= config.DAILY_PROFIT_TARGET_PCT:
                    logger.info(f"DAILY TARGET REACHED! PnL: +{current_pnl_pct:.2f}%")
                    # CONTINUOUS COMPOUNDING: Don't stop, just log.
                    # self.running = False 
                    # break
                
                # REFRESH BALANCE FOR COMPOUNDING
                # We update balance here so the NEXT batch scan uses the new capital (Initial + Profit)
                await self.market.initialize_balance()
                
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
                
                await asyncio.sleep(config.CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Batch Loop Error: {e}")
                await asyncio.sleep(5)

    async def scan_and_fill_batch(self, slots_needed):
        """Analyze market and pick TOP N candidates"""
        symbols = await self.market.get_top_symbols(limit=None)
        candidates = []
        
        # 1. Analyze ALL symbols first (Parallel CPU)
        loop = asyncio.get_running_loop()
        tasks = []
        
        for symbol in symbols:
            if not self.running: break
            if symbol in self.market.positions: continue
            
            # Get data (Non-blocking network)
            df = await self.market.get_klines(symbol, interval=config.TIMEFRAME)
            if df.empty: continue
            
            # Offload heavy analysis to Process Pool
            tasks.append(
                loop.run_in_executor(self.executor, self.detector.analyze, df)
            )
            candidates.append({'symbol': symbol, 'df': df}) # Keep ref to DF

        if not tasks: return

        # Wait for all analyses
        results = await asyncio.gather(*tasks)
        
        # Merge results
        final_candidates = []
        for i, signal in enumerate(results):
            if signal and signal['score'] >= config.MIN_SIGNAL_SCORE:
                cand = candidates[i]
                cand['signal'] = signal
                cand['score'] = signal['score']
                final_candidates.append(cand)

        candidates = final_candidates # Replace with filtered list
        
        # 2. Sort by Score (Highest first)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. Pick Top N
        top_picks = candidates[:slots_needed]
        
        if top_picks:
            logger.info(f"Found {len(top_picks)} candidates. Processing...")
            for pick in top_picks:
                # STABILITY FILTER V4: 1H Trend Alignment
                # Only take LONG if Price > EMA200(1H)
                # Only take SHORT if Price < EMA200(1H)
                try:
                    df_1h = await self.market.get_klines(pick['symbol'], interval=config.TREND_ALIGN_INTERVAL, limit=config.TREND_ALIGN_EMA + 10)
                    if not df_1h.empty and len(df_1h) > config.TREND_ALIGN_EMA:
                        # Calculate EMA200 manually or using pandas (ta-lib might not be in main scope)
                        ema_200 = df_1h['close'].ewm(span=config.TREND_ALIGN_EMA, adjust=False).mean().iloc[-1]
                        current_price_1h = df_1h['close'].iloc[-1]
                        
                        direction = pick['signal']['direction']
                        if direction == 'LONG' and current_price_1h < ema_200:
                            logger.info(f"🚫 SKIPPED {pick['symbol']} LONG: Against 1H Trend (Price {current_price_1h:.2f} < EMA200 {ema_200:.2f})")
                            continue
                        if direction == 'SHORT' and current_price_1h > ema_200:
                            logger.info(f"🚫 SKIPPED {pick['symbol']} SHORT: Against 1H Trend (Price {current_price_1h:.2f} > EMA200 {ema_200:.2f})")
                            continue
                except Exception as e:
                    logger.error(f"Trend Alignment Check Error: {e}")
                    continue # Skip if check fails (Safety first)

                if config.SMART_ENTRY_ENABLED:
                    await self.add_to_pending(pick['symbol'], pick['signal'], pick['df'])
                else:
                    await self.execute_trade(pick['symbol'], pick['signal'], pick['df'])
        else:
            logger.info("No suitable candidates found this round.")

    async def check_watchlist(self):
        """
        Legacy method kept for compatibility, but main focus is now Batch Scan.
        We can still use this to monitor specific high-potential pairs if needed.
        """
        pass # Disabled for now to focus on Batch Strategy

    async def add_to_pending(self, symbol, signal, df):
        """Queue trade for Smart Entry (Confirmation)"""
        if symbol in self.market.positions or symbol in self.pending_entries: return
        
        # Calculate Trigger Price (Breakout of last candle high/low)
        last_candle = df.iloc[-1]
        trigger_price = 0.0
        
        if signal['direction'] == 'LONG':
            trigger_price = last_candle['high']
        else:
            trigger_price = last_candle['low']
            
        self.pending_entries[symbol] = {
            'symbol': symbol,
            'signal': signal,
            'df': df,
            'queued_time': datetime.now(),
            'trigger_price': trigger_price,
            'direction': signal['direction']
        }
        logger.info(f"⏳ QUEUED {symbol} {signal['direction']} | Wait for break of {trigger_price:.4f}")

    async def confirmation_loop(self):
        """SMART ENTRY: Watches Pending Entries for Breakout"""
        logger.info("Started Smart Entry Monitor...")
        while self.running:
            try:
                # Iterate copy to allow modification
                current_time = datetime.now()
                for symbol, entry in list(self.pending_entries.items()):
                    # check timeout
                    if (current_time - entry['queued_time']).total_seconds() > (config.CONFIRMATION_TIMEOUT_MINS * 60):
                        logger.info(f"🗑️ EXPIRED {symbol} - No breakout in {config.CONFIRMATION_TIMEOUT_MINS}m")
                        del self.pending_entries[symbol]
                        continue
                        
                    # DYNAMIC TRIGGER & VALIDATION (Smart Entry V3 - Sniper)
                    # Fetch 50 candles for RSI calculation
                    df_latest = await self.market.get_klines(symbol, interval=config.TIMEFRAME, limit=50)
                    if not df_latest.empty:
                        # 0. Helper for RSI
                        def calc_rsi(series, period=14):
                            delta = series.diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                            rs = gain / loss
                            return 100 - (100 / (1 + rs))

                        # 1. Update Dynamic Trigger (Last 3 Candles High/Low)
                        recent_candles = df_latest.iloc[-4:-1] # Exclude current candle to define "range"
                        
                        if entry['direction'] == 'LONG':
                            new_trigger = recent_candles['high'].max()
                            entry['trigger_price'] = new_trigger
                        else:
                            new_trigger = recent_candles['low'].min()
                            entry['trigger_price'] = new_trigger

                        # 2. Check Breakout Conditions
                        current_candle = df_latest.iloc[-1]
                        current_price = current_candle['close']
                        current_vol = current_candle['volume']
                        avg_vol = df_latest['volume'].iloc[-21:-1].mean() # 20 period avg (excl current)
                        
                        rsi_val = 50 # Default safe
                        try:
                            rsi_series = calc_rsi(df_latest['close'])
                            rsi_val = rsi_series.iloc[-1]
                        except: pass

                        triggered = False
                        reason = ""

                        if entry['direction'] == 'LONG':
                            # Price > Trigger + Buffer
                            target = entry['trigger_price'] * (1 + config.CONFIRM_BUFFER_PCT/100)
                            if current_price > target:
                                # Validation
                                if current_vol > (avg_vol * config.CONFIRM_VOLUME_FACTOR):
                                    if rsi_val < config.CONFIRM_RSI_MAX:
                                        triggered = True
                                        reason = f"Price {current_price:.2f} > {target:.2f} | Vol {current_vol:.0f} > {avg_vol:.0f} | RSI {rsi_val:.1f}"
                        else:
                            # Price < Trigger - Buffer
                            target = entry['trigger_price'] * (1 - config.CONFIRM_BUFFER_PCT/100)
                            if current_price < target:
                                # Validation
                                if current_vol > (avg_vol * config.CONFIRM_VOLUME_FACTOR):
                                    if rsi_val > config.CONFIRM_RSI_MIN:
                                        triggered = True
                                        reason = f"Price {current_price:.2f} < {target:.2f} | Vol {current_vol:.0f} > {avg_vol:.0f} | RSI {rsi_val:.1f}"
                        
                        if triggered:
                            logger.info(f"🎯 SNIPER ENTRY {symbol} {entry['direction']} | {reason}")
                            # Execute
                            await self.execute_trade(symbol, entry['signal'], entry['df'])
                            del self.pending_entries[symbol]
                        
                await asyncio.sleep(2) # Check frequently (2s)
            except Exception as e:
                logger.error(f"Confirmation Loop Error: {e}")
                await asyncio.sleep(5)

    async def execute_trade(self, symbol, signal, df):
        # Enforce Minimum Position Size (Binance usually requires 5-6 USDT)
        calc_amount = self.market.balance * (config.CAPITAL_PER_TRADE_PCT / 100)
        amount = max(calc_amount, 6.0) # Ensure at least 6 USDT
        
        # Check affordability
        if amount > self.market.balance:
            logger.warning(f"Insufficient funds for {symbol}. Need {amount}, have {self.market.balance}")
            return
        price = df.iloc[-1]['close']
        
        # Calculate Dynamic TP/SL
        sl, tp = self.detector.calculate_dynamic_levels(df, signal['direction'])
        
        # Fallback if dynamic calc fails (e.g. not enough data)
        if not sl or not tp:
            logger.warning(f"Dynamic TP/SL failed for {symbol}. Using fixed fallback.")
            if signal['direction'] == 'LONG':
                sl = price * (1 - config.STOP_LOSS_PCT/100)
                tp = price * (1 + config.TAKE_PROFIT_PCT/100)
            else:
                sl = price * (1 + config.STOP_LOSS_PCT/100)
                tp = price * (1 - config.TAKE_PROFIT_PCT/100)
        
        logger.info(f"OPENING {symbol} {signal['direction']} | Entry: {price:.4f} | SL: {sl:.4f} | TP: {tp:.4f}")
        result = await self.market.open_position(symbol, signal['direction'], amount, sl, tp)
        if not result:
            logger.error(f"❌ EXECUTION FAILED for {symbol}. Check logs for details (Precision/Margin/API).")

    async def manage_positions(self):
        """Real-time position management"""
        for symbol, pos in list(self.market.positions.items()):
            current_price = await self.market.get_current_price(symbol)
            if current_price == 0: continue
            
            # Calculate duration
            duration_sec = (datetime.now() - pos['entry_time']).total_seconds()
            
            # 1. Check Hard SL/TP (Instant)
            # Calculate PnL (ROI)
            pnl_pct = 0.0
            if pos['side'] == 'LONG':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                if current_price <= pos['sl']:
                    await self.market.close_position(symbol, "STOP LOSS")
                    continue
                if current_price >= pos['tp']:
                    await self.market.close_position(symbol, "TAKE PROFIT")
                    continue
            else:
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
                if current_price >= pos['sl']:
                    await self.market.close_position(symbol, "STOP LOSS")
                    continue
                if current_price <= pos['tp']:
                    await self.market.close_position(symbol, "TAKE PROFIT")
                    continue
            
            # 1b. BREAKEVEN Logic (Stricter Risk)
            # If profit > 1.5x Risk (approx 7.5% ROI at 5x), move SL to Entry
            risk_pct = config.STOP_LOSS_PCT / 100
            if pnl_pct > (risk_pct * 1.5) and not pos.get('breakeven', False):
                pos['sl'] = pos['entry_price']
                pos['breakeven'] = True
                logger.info(f"MOVED SL TO BREAKEVEN for {symbol} (Profit > 1.5R)")

            # 2. Check Major Resistance (REAL TIME PROTECTION)
            # We check this frequently to exit BEFORE SL if hitting a wall
            df = await self.market.get_klines(symbol, interval=config.TIMEFRAME, limit=100)
            if not df.empty:
                major_levels = self.detector.find_major_levels(df)
                for level in major_levels:
                    dist = abs(current_price - level) / current_price
                    if dist < 0.005: # Within 0.5% (Very close)
                        # Only exit if we are LOSING momentum or STUCK
                        if self.detector.check_exhaustion(df, pos['side']):
                            await self.market.close_position(symbol, f"MAJOR RESISTANCE @ {level:.2f}")
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
