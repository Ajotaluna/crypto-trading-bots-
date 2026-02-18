"""
Main Bot Logic - REFACTORED FOR DOUBLE FUNNEL STRATEGY
Integrates AnomalyScanner (Macro) and TradingStrategy (Micro).
ALIGNED WITH backtest_progressive.py PositionManager logic.
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

# Strategy imports (shared with backtest)
from scanner_anomaly import AnomalyScanner
from trading_strategy import (
    confirm_entry, 
    calculate_indicators, 
    PositionManager,
    PIDController,
    RiskManager,
    MAX_SCORE,
    TOP_N,
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
        
        self.daily_watchlist = [] # The Top 10 for the day
        self.pos_state = {}  # Local state tracker (backtest PositionManager alignment)
        
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
                if not self.daily_watchlist:
                    logger.warning("⚠️ Watchlist empty. Retrying Macro Scan...")
                    await self.run_macro_scan()
                    await asyncio.sleep(60)
                    continue

                for pick in list(self.daily_watchlist):  # Copy for safe removal
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
                    
                    # Remove from watchlist after entry (backtest alignment)
                    if symbol in self.market.positions and pick in self.daily_watchlist:
                        self.daily_watchlist.remove(pick)
                    
                    await asyncio.sleep(1) # Pace requests

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Execution Loop Error: {e}")
                await asyncio.sleep(60)

    async def run_macro_scan(self):
        """Runs AnomalyScanner on the entire market to pick the daily Top N."""
        try:
            logger.info("🔭 MACRO SCAN: Fetching market data...")
            
            tickers = await self.market.get_trading_universe() 
            if not tickers:
                logger.error("No tickers found in Universe.")
                return

            pair_data = {}
            logger.info(f"📥 Downloading history for {len(tickers)} pairs...")
            
            for symbol in tickers:
                df = await self.market.get_klines(symbol, interval='15m', limit=490)
                if not df.empty:
                    pair_data[symbol] = df
                await asyncio.sleep(0.05)
                
            logger.info(f"📊 Analyzing {len(pair_data)} pairs...")
            
            picks = self.scanner.score_universe(pair_data, -1, top_n=TOP_N)
            
            # Filter by MAX_SCORE (matches backtest)
            self.daily_watchlist = [p for p in picks if p['score'] < MAX_SCORE]
            
            logger.info(f"✅ MACRO RESULTS: {len(self.daily_watchlist)} Candidates Selected.")
            for p in self.daily_watchlist:
                logger.info(f"  > {p['symbol']} ({p['direction']}) | Score: {p['score']} | Reasons: {p['reasons']}")
                
        except Exception as e:
            logger.error(f"Macro Scan Error: {e}")

    async def check_micro_entry(self, symbol, direction, pick_info):
        """Runs confirm_entry on the specific candidate."""
        try:
            limit = 100 
            df = await self.market.get_klines(symbol, interval='15m', limit=limit)
            if df.empty: return

            loop = asyncio.get_running_loop()
            df_indicators = await loop.run_in_executor(
                self.executor,
                calculate_indicators,
                df
            )
            
            is_valid = await loop.run_in_executor(
                self.executor,
                confirm_entry,
                df_indicators,
                direction
            )
            
            if is_valid:
                logger.info(f"✨ MICRO TRIGGER: {symbol} ({direction}) Confirmed!")
                
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
        """Execute trade with backtest-aligned sizing and state initialization."""
        curr_price = float(df['close'].iloc[-1])
        atr = float(df['atr'].iloc[-1])
        
        if atr <= 0: return
        
        # SL/TP Calculation (matches backtest)
        if signal['direction'] == 'LONG':
            sl = curr_price - (atr * INITIAL_SL_ATR)
            tp = curr_price + (atr * 10)
        else:
            sl = curr_price + (atr * INITIAL_SL_ATR)
            tp = curr_price - (atr * 10)
        
        # POSITION SIZING (exact backtest formula)
        equity = await self.get_equity()
        risk_amt = equity * RISK_PER_ENTRY
        risk_dist = abs(curr_price - sl)
        if risk_dist <= 0: return
        
        margin = (risk_amt / (risk_dist / curr_price)) / LEVERAGE
        margin = min(margin, equity * MAX_CAPITAL_PER_TRADE)
        amount = margin * LEVERAGE  # Convert to notional for MarketData
        
        if amount < 6.0: return
        
        logger.info(f"⚡ EXECUTING {symbol} | SL: {sl:.4f} | Size: ${amount:.0f} (Risk {RISK_PER_ENTRY*100}%)")
        
        result = await self.market.open_position(symbol, signal['direction'], amount, sl, tp)
        if result:
            logger.info(f"🔫 OPEN SUCCESS {symbol}")
            
            # Initialize local state tracker (matches backtest PositionManager)
            pid = PIDController(Kp=0.4, Ki=0.0, Kd=0.1, setpoint=0, output_limits=(-0.8, 0.5))
            self.pos_state[symbol] = {
                'best_price': curr_price,
                'be_locked': False,
                'pid': pid,
                'avg_price': curr_price,
                'atr_at_entry': atr,
                'scale_level': 1,
                'total_amount': margin,
                'entry_price': curr_price,
            }

    async def scale_into_position(self, symbol, direction, notional):
        """Add to an existing position (scaling entry)."""
        try:
            if self.market.is_dry_run:
                # Mock: Update position amount directly
                pos = self.market.positions[symbol]
                price = await self.market.get_current_price(symbol)
                if price <= 0: return False
                new_qty = notional / price
                old_amount = pos['amount']
                pos['amount'] = old_amount + new_qty
                return True
            else:
                # Real: Place additional market order (Binance adds to position)
                price = await self.market.get_current_price(symbol)
                if price <= 0: return False
                qty = notional / price
                side_param = 'BUY' if direction == 'LONG' else 'SELL'
                info = await self.market._get_symbol_precision(symbol)
                qty_val = self.market._round_step_size(qty, info['q'])
                if qty_val <= 0: return False
                params = {
                    'symbol': symbol,
                    'side': side_param,
                    'type': 'MARKET',
                    'quantity': f"{qty_val}",
                }
                result = await self.market._signed_request('POST', '/fapi/v1/order', params)
                return result is not None
        except Exception as e:
            logger.error(f"Scale Error {symbol}: {e}")
            return False

    async def safety_loop(self):
        """Active Position Manager"""
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
        Manages active positions using PID Controller Logic.
        Faithfully matches Backtest PositionManager.update_positions() behavior.
        """
        for symbol, pos in list(self.market.positions.items()):
            # Initialize state for orphaned positions (e.g., after bot restart)
            if symbol not in self.pos_state:
                logger.info(f"⚠️ Initializing state for orphaned position: {symbol}")
                pid = PIDController(Kp=0.4, Ki=0.0, Kd=0.1, setpoint=0, output_limits=(-0.8, 0.5))
                self.pos_state[symbol] = {
                    'best_price': pos['entry_price'],
                    'be_locked': False,
                    'pid': pid,
                    'avg_price': pos['entry_price'],
                    'atr_at_entry': 0,
                    'scale_level': 1,
                    'total_amount': pos.get('amount', 0) * pos['entry_price'] / LEVERAGE,
                    'entry_price': pos['entry_price'],
                }
            
            state = self.pos_state[symbol]
            
            # 1. Get Fresh Market Data (ATR, Kalman, Price)
            try:
                df = await self.market.get_klines(symbol, interval='15m', limit=50)
                if df.empty: continue
                
                loop = asyncio.get_running_loop()
                df = await loop.run_in_executor(self.executor, calculate_indicators, df)
                
                current_price = float(df['close'].iloc[-1])
                atr = float(df['atr'].iloc[-1])
                kf_price = float(df['kf_price'].iloc[-1])
                
                if current_price == 0: continue
            except Exception as e:
                logger.error(f"Data Error {symbol}: {e}")
                continue
            
            # ATR fallback (matches backtest)
            if atr <= 0:
                atr = state['atr_at_entry']
            if atr <= 0:
                atr = current_price * 0.02
            if state['atr_at_entry'] <= 0:
                state['atr_at_entry'] = atr
            
            # Track best price & calculate PnL (matches backtest exactly)
            if pos['side'] == 'LONG':
                if current_price > state['best_price']:
                    state['best_price'] = current_price
                pnl_atr = (current_price - state['avg_price']) / atr
                dist_kalman_atr = (current_price - kf_price) / atr
            else:
                if current_price < state['best_price']:
                    state['best_price'] = current_price
                pnl_atr = (state['avg_price'] - current_price) / atr
                dist_kalman_atr = (kf_price - current_price) / atr
            
            # --- 1. CHECK STOP LOSS ---
            is_sl = False
            if pos['side'] == 'LONG':
                if current_price <= pos['sl']: is_sl = True
            else:
                if current_price >= pos['sl']: is_sl = True
            
            if is_sl:
                reason = 'TRAILING_STOP' if state['be_locked'] else 'STOP_LOSS'
                logger.warning(f"🛑 {reason}: {symbol}")
                await self.market.close_position(symbol, reason)
                self.pos_state.pop(symbol, None)
                continue
            
            # --- 2. CHECK SCALING (before BE lock, matches backtest) ---
            if state['scale_level'] < MAX_SCALE and not state['be_locked']:
                ref_price = state['entry_price']
                if pos['side'] == 'LONG':
                    adverse_atr = (ref_price - current_price) / atr
                else:
                    adverse_atr = (current_price - ref_price) / atr
                
                should_scale = False
                if state['scale_level'] == 1 and adverse_atr >= SCALE_LEVEL_2:
                    should_scale = True
                elif state['scale_level'] == 2 and adverse_atr >= SCALE_LEVEL_3:
                    should_scale = True
                
                if should_scale:
                    equity = await self.get_equity()
                    risk_amt = equity * RISK_PER_ENTRY
                    risk_dist = abs(current_price - pos['sl'])
                    if risk_dist > 0:
                        new_margin = (risk_amt / (risk_dist / current_price)) / LEVERAGE
                        new_margin = min(new_margin, equity * MAX_CAPITAL_PER_TRADE)
                        notional = new_margin * LEVERAGE
                        if notional >= 5:
                            success = await self.scale_into_position(symbol, pos['side'], notional)
                            if success:
                                old_total = state['total_amount']
                                state['total_amount'] += new_margin
                                state['avg_price'] = (state['avg_price'] * old_total + current_price * new_margin) / state['total_amount']
                                state['scale_level'] += 1
                                logger.info(f"📈 SCALE {state['scale_level']}: {symbol} @ {current_price:.4f} | Avg: {state['avg_price']:.4f}")
            
            # --- 3. BREAKEVEN LOCK (matches backtest: BE_LOCK_ATR = 1.5) ---
            if not state['be_locked'] and pnl_atr >= BE_LOCK_ATR:
                buffer = state['avg_price'] * 0.002
                if pos['side'] == 'LONG':
                    new_sl = state['avg_price'] + buffer
                else:
                    new_sl = state['avg_price'] - buffer
                state['be_locked'] = True
                await self.market.update_sl(symbol, new_sl)
                logger.info(f"🔒 BE LOCK: {symbol} SL->{new_sl:.4f} (PnL: {pnl_atr:.1f} ATR)")
            
            # --- 4. PID DYNAMIC TRAILING (after BE lock ONLY, matches backtest) ---
            if state['be_locked']:
                pid_adjust = state['pid'].update(dist_kalman_atr)
                
                # Trail = TRAIL_DISTANCE_ATR + PID output (matches backtest formula)
                current_trail_atr = TRAIL_DISTANCE_ATR + pid_adjust
                current_trail_atr = max(0.5, min(4.0, current_trail_atr))
                
                if pos['side'] == 'LONG':
                    trail_sl = state['best_price'] - (atr * current_trail_atr)
                    if trail_sl > pos['sl']:
                        await self.market.update_sl(symbol, trail_sl)
                        logger.debug(f"🔄 PID TRAIL: {symbol} SL->{trail_sl:.4f} (Adj:{pid_adjust:.2f})")
                else:
                    trail_sl = state['best_price'] + (atr * current_trail_atr)
                    if trail_sl < pos['sl']:
                        await self.market.update_sl(symbol, trail_sl)
                        logger.debug(f"🔄 PID TRAIL: {symbol} SL->{trail_sl:.4f} (Adj:{pid_adjust:.2f})")
            
            # --- 5. MAX HOLD TIME (candle-equivalent via elapsed time) ---
            entry_time = pos.get('entry_time')
            if entry_time:
                elapsed_minutes = (datetime.now() - entry_time).total_seconds() / 60
                candles_elapsed = int(elapsed_minutes / 15)  # 15m candles
                if candles_elapsed >= MAX_HOLD_CANDLES:
                    logger.warning(f"⏰ MAX_HOLD_TIME: {symbol} ({candles_elapsed} candles)")
                    await self.market.close_position(symbol, "MAX_HOLD_TIME")
                    self.pos_state.pop(symbol, None)
        
        # Clean up pos_state for externally closed positions
        closed_symbols = [s for s in self.pos_state if s not in self.market.positions]
        for s in closed_symbols:
            self.pos_state.pop(s, None)

    async def reporting_loop(self):
        """Periodically reports status."""
        while self.running:
             try:
                 total_equity = self.market.balance
                 if not self.market.is_dry_run:
                     st = await self.market.get_real_account_status()
                     if st: total_equity = st['equity']
                 
                 open_count = len(self.market.positions)
                 logger.info(f"--- 📊 STATUS: Equity ${total_equity:.2f} | Open: {open_count} ---")
                 if open_count > 0:
                     for s, p in self.market.positions.items():
                         state_info = ""
                         if s in self.pos_state:
                             st = self.pos_state[s]
                             state_info = f" | BE:{'✅' if st['be_locked'] else '❌'} | Scale:{st['scale_level']}"
                         logger.info(f"   > {s} ({p['side']}) | Entry: {p['entry_price']} | SL: {p['sl']:.4f}{state_info}")
                 
                 await asyncio.sleep(300) # 5 Minutes
             except Exception as e:
                 logger.error(f"Reporting Error: {e}")
                 await asyncio.sleep(60)
    
    async def get_equity(self):
        if not self.market.is_dry_run:
             s = await self.market.get_real_account_status()
             return s['equity'] if s else self.market.balance
        return self.market.balance

# ================================================================
# EXECUTION ENTRY POINT (Docker / Cloud Support)
# ================================================================
if __name__ == "__main__":
    # Load Environment Variables (Standard for Cloud)
    api_key = os.getenv('API_KEY', '')
    api_secret = os.getenv('API_SECRET', '')
    is_dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'

    if not api_key or not api_secret:
        if not is_dry_run:
            logging.error("❌ MISSING API KEYS! Set API_KEY and API_SECRET env vars.")
            sys.exit(1)
            
    try:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        bot = TrendBot(is_dry_run=is_dry_run, api_key=api_key, api_secret=api_secret)
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Fatal Error: {e}")
        sys.exit(1)
