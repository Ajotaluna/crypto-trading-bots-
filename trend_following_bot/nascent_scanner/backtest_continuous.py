import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nascent_scanner.scanner_anomaly import AnomalyScanner

# ================================================================
# CONFIGURATION (Reverted to "89% ROI" Baseline)
# ================================================================
INITIAL_CAPITAL = 1000
LEVERAGE = 5
COMMISSION = 0.04 / 100       # 0.04% per side

RISK_PER_ENTRY = 0.01         # 1% equity per entry
MAX_CAPITAL_PER_TRADE = 0.10  # 10% max position size
MAX_SIGNALS = 3               # Max concurrent trades
MAX_HOLD_CANDLES = 160        # 40h max hold
DAILY_LOSS_CAP = 0.08         # 8% daily loss cap

# Data Config
CANDLES_PER_DAY = 96
HISTORY_NEEDED = 480          # Warmup valid candles needed
WARMUP_DAYS = 3               # Not used in time loop directly, but implied by history
TOP_N = 10                    # Top N candidates to consider

# Strategy Config
INITIAL_SL_ATR = 5.0
BE_LOCK_ATR = 1.5
TRAIL_DISTANCE_ATR = 2.0

# Scaled Entry Config
SCALE_LEVEL_2 = 1.5
SCALE_LEVEL_3 = 3.0
MAX_SCALE = 3
MAX_SCORE = 120
ENTRY_WINDOW_CANDLES = 48


# ================================================================
# POSITION MANAGER (Copied & Adapted for Time-Loop)
# ================================================================
class PositionManager:
    """Manages scaled positions: up to 3 entries per signal."""
    def __init__(self):
        self.positions = {}

    def open_position(self, symbol, direction, entry_price, sl, amount, atr_at_entry, time_idx):
        self.positions[symbol] = {
            'symbol': symbol,
            'direction': direction,
            'entries': [{'price': entry_price, 'amount': amount, 'time': time_idx}],
            'entry_price': entry_price,
            'avg_price': entry_price,
            'total_amount': amount, # Margin amount
            'sl': sl,
            'entry_time': time_idx, # Timestamp
            'be_locked': False,
            'best_price': entry_price,
            'atr_at_entry': atr_at_entry,
            'scale_level': 1,
        }

    def add_to_position(self, symbol, entry_price, amount, time_idx):
        pos = self.positions[symbol]
        pos['entries'].append({'price': entry_price, 'amount': amount, 'time': time_idx})
        
        # Recalculate Average Price
        total_cost = sum(e['price'] * e['amount'] for e in pos['entries'])
        pos['total_amount'] = sum(e['amount'] for e in pos['entries'])
        pos['avg_price'] = total_cost / pos['total_amount'] if pos['total_amount'] > 0 else entry_price
        pos['scale_level'] += 1

    def update_positions(self, pair_data_dict, current_time):
        """
        Check SL, Trailing, Scaling for all open positions at 'current_time'.
        Returns list of closed trades.
        """
        closed = []
        
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            
            # 1. Get current candle for this symbol
            if symbol not in pair_data_dict:
                continue
                
            df = pair_data_dict[symbol]
            
            # Check if this timestamp exists for this symbol
            if current_time not in df.index:
                # Gaps happen. If we haven't seen data for X time, maybe close?
                # For now, just hold through gaps.
                # BUT, check if data ENDED (Zombie Fix adapted for Time Loop)
                if current_time > df.index[-1]:
                     last_close = float(df.iloc[-1]['close'])
                     closed.append(self._close_position(symbol, last_close, 'DATA_ENDED', current_time))
                continue

            # Get exact row for this time
            candle = df.loc[current_time]
            
            c_high = float(candle['high'])
            c_low = float(candle['low'])
            c_close = float(candle['close'])
            atr_val = float(candle['atr']) if candle['atr'] > 0 else pos['atr_at_entry']
            if atr_val <= 0: atr_val = c_close * 0.02

            # PnL & Best Price Tracking
            if pos['direction'] == 'LONG':
                if c_high > pos['best_price']: pos['best_price'] = c_high
                pnl_atr = (c_close - pos['avg_price']) / atr_val
            else:
                if c_low < pos['best_price']: pos['best_price'] = c_low
                pnl_atr = (pos['avg_price'] - c_close) / atr_val

            # 1. STOP LOSS
            sl_hit = False
            if pos['direction'] == 'LONG' and c_low <= pos['sl']: sl_hit = True
            elif pos['direction'] == 'SHORT' and c_high >= pos['sl']: sl_hit = True
            
            if sl_hit:
                reason = 'TRAILING_STOP' if pos['be_locked'] else 'STOP_LOSS'
                closed.append(self._close_position(symbol, pos['sl'], reason, current_time))
                continue

            # 2. BREAKEVEN LOCK
            if not pos['be_locked'] and pnl_atr >= BE_LOCK_ATR:
                buffer = pos['avg_price'] * 0.002
                if pos['direction'] == 'LONG': pos['sl'] = pos['avg_price'] + buffer
                else: pos['sl'] = pos['avg_price'] - buffer
                pos['be_locked'] = True

            # 3. TRAILING STOP
            if pos['be_locked']:
                trail_atr = 1.0 if pnl_atr > 6.0 else (1.5 if pnl_atr > 4.0 else TRAIL_DISTANCE_ATR)
                if pos['direction'] == 'LONG':
                    trail_sl = pos['best_price'] - (atr_val * trail_atr)
                    if trail_sl > pos['sl']: pos['sl'] = trail_sl
                else:
                    trail_sl = pos['best_price'] + (atr_val * trail_atr)
                    if trail_sl < pos['sl']: pos['sl'] = trail_sl
                
                # Check if trail hit immediately
                if (pos['direction'] == 'LONG' and c_close <= pos['sl']) or \
                   (pos['direction'] == 'SHORT' and c_close >= pos['sl']):
                       closed.append(self._close_position(symbol, c_close, 'TRAILING_STOP', current_time))
                       continue

            # 4. MAX HOLD TIME
            # Calculate duration
            if isinstance(pos['entry_time'], pd.Timestamp):
                duration = current_time - pos['entry_time']
                candles_held = duration.total_seconds() / 900 # 15m
            else:
                candles_held = 0
                
            if candles_held >= MAX_HOLD_CANDLES:
                 closed.append(self._close_position(symbol, c_close, 'MAX_TIME', current_time))

        return closed

    def check_scale(self, symbol, current_price, atr_val):
        if symbol not in self.positions: return False, 0
        pos = self.positions[symbol]
        if pos['scale_level'] >= MAX_SCALE: return False, 0
        if pos['be_locked']: return False, 0

        ref_price = pos['entry_price']
        if pos['direction'] == 'LONG':
             adverse = (ref_price - current_price) / atr_val
        else:
             adverse = (current_price - ref_price) / atr_val

        # "The Bug" Feature: Return multiplier 2/3, not amount
        if pos['scale_level'] == 1 and adverse >= SCALE_LEVEL_2:
            return True, 2
        
        if pos['scale_level'] == 2 and adverse >= SCALE_LEVEL_3:
            return True, 3

        return False, 0

    def _close_position(self, symbol, exit_price, reason, current_time):
        pos = self.positions.pop(symbol)
        
        if pos['direction'] == 'LONG':
             pnl_pct = (exit_price - pos['avg_price']) / pos['avg_price']
        else:
             pnl_pct = (pos['avg_price'] - exit_price) / pos['avg_price']

        notional = pos['total_amount'] * LEVERAGE
        fees = notional * COMMISSION * 2
        pnl_dollar = pnl_pct * notional - fees
        
        # Calculate hold duration properly for report
        start = pos['entry_time']
        end = current_time
        if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
            hold_str = str(end - start)
            hours = (end - start).total_seconds() / 3600
        else:
            hold_str = "N/A"
            hours = 0

        return {
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_price': pos['avg_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_dollar': pnl_dollar,
            'reason': reason,
            'hold_time': hold_str,
            'hold_hours': hours,
            'be_locked': pos['be_locked'],
            'scale_level': pos['scale_level']
        }

# ================================================================
# MAIN CONTINUOUS ENGINE
# ================================================================
class ContinuousBacktest:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data") # Default to 1-year folder
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.position_mgr = PositionManager()
        
    def load_and_prep_data(self):
        print("⏳ Loading CSVs...")
        loaded = {}
        files = [f for f in os.listdir(self.data_dir) if f.endswith("_15m.csv")]
        
        for f in files:
            symbol = f.replace("_15m.csv", "")
            try:
                df = pd.read_csv(os.path.join(self.data_dir, f))
                if len(df) < 480: continue # Basic check
                
                # Parse Dates
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Remove duplicates
                df = df[~df.index.duplicated(keep='first')]
                
                # Calc Indicators Here (Vectorized is faster than in-loop)
                loaded[symbol] = self._add_indicators(df)
            except:
                continue
                
        print(f"✅ Loaded {len(loaded)} pairs with indicators.")
        return loaded

    def _add_indicators(self, df):
        # reuse implementation from realistic py but without re-loading deps
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # EMA
        df['ema_20'] = close.ewm(span=20).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta>0, 0).rolling(14).mean()
        loss = (-delta.where(delta<0, 0)).rolling(14).mean()
        rs = gain/loss.replace(0, np.nan)
        df['rsi'] = 100 - (100/(1+rs))
        
        # ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # --- VECTORIZED STRATEGY METRICS (Replicating AnomalyScanner) ---
        
        # 1. Returns (6h=24, 12h=48, 24h=96)
        df['ret_6h'] = close.pct_change(24) * 100
        df['ret_12h'] = close.pct_change(48) * 100
        df['ret_24h'] = close.pct_change(96) * 100
        
        # 2. Volume Patterns
        # vol_24h = rolling 1-day sum
        vol_24h = volume.rolling(96).sum()
        df['vol_24h'] = vol_24h
        
        # Vol Z-Score: (Current 24h Vol - Avg 30d Vol) / Std 30d Vol
        # 30 days = 96 * 30 = 2880 candles
        # Note: We use shift(1) to avoid including current 24h in the baseline if we want strict lookback, 
        # but standard Z is usually vs trailing window.
        # CRITICAL FIX: Use min_periods=96 so young coins get a Z-score (vs 24h not 30d)
        vol_mean_30d = vol_24h.rolling(2880, min_periods=96).mean()
        vol_std_30d = vol_24h.rolling(2880, min_periods=96).std()
        
        # Avoid division by zero
        df['vol_z'] = (vol_24h - vol_mean_30d) / vol_std_30d.replace(0, 1)
        df['vol_z'] = df['vol_z'].fillna(0) # Default to 0 if not enough data
        
        # Volume Increasing? (Last 6h vs Prev 6h)
        vol_6h = volume.rolling(24).sum()
        vol_prev_6h = volume.shift(24).rolling(24).sum()
        df['vol_increasing'] = vol_6h > (vol_prev_6h * 1.2)
        df['vol_increasing'] = df['vol_increasing'].fillna(False)
        
        return df
        high_7d = high.rolling(672).max()
        low_7d = low.rolling(672).min()
        range_7d = high_7d - low_7d
        
        # Avoid division by zero
        df['range_pct'] = ((close - low_7d) / range_7d.replace(0, np.nan)) * 100
        df['range_pct'] = df['range_pct'].fillna(50) # Default mid-range if flat
        
        # 4. Acceleration
        # Up: 6h > 0, 12h > 0, 6h > 60% of 12h
        df['is_accel_up'] = (df['ret_6h'] > 0) & (df['ret_12h'] > 0) & (df['ret_6h'] > df['ret_12h'] * 0.6)
        
        # Down: 6h < 0, 12h < 0, 6h drop is > 60% of 12h drop
        df['is_accel_down'] = (df['ret_6h'] < 0) & (df['ret_12h'] < 0) & (df['ret_6h'] < df['ret_12h'] * 0.6)
        
        return df

    def run(self):
        pair_data = self.load_and_prep_data()
        if not pair_data: return
        
        # 1. Establish Timeline
        all_dates = sorted(list(set().union(*[df.index for df in pair_data.values()])))
        start_date = all_dates[0]
        end_date = all_dates[-1]
        
        print(f"📅 Simulation Window: {start_date} -> {end_date}")
        print(f"⏱ Total Candles: {len(all_dates)}")
        
        # Identify "Warmup" date (just for printing, logic handles per-coin)
        # We start trading immediately, but filtering logic handles warmup per coin
        
        equity = INITIAL_CAPITAL
        daily_equity_start = equity
        current_day = start_date.date()
        
        trades_log = []
        daily_log = []
        
        day_longs = 0
        day_shorts = 0
        day_trades_closed = []
        day_stopped = False
        
        # === MAIN LOOP ===
        for i, current_time in enumerate(all_dates):
            
            # Print debug for very first candle and daily noon to sample activity
            is_noon = (current_time.hour == 12 and current_time.minute == 0)
            show_debug = (i == 0) or is_noon
            
            # --- Daily Accounting ---
            if current_time.date() != current_day:
                # End of Day
                pnl_day = equity - daily_equity_start
                daily_log.append({
                    'date': current_day,
                    'equity': equity,
                    'pnl': pnl_day,
                    'trades': len(day_trades_closed),
                    'open_positions': len(self.position_mgr.positions)
                })
                print(f"Day {current_day} | PnL: ${pnl_day:+.2f} | Eq: ${equity:.2f} | Open: {len(self.position_mgr.positions)}")
                
                # Reset for new day
                current_day = current_time.date()
                daily_equity_start = equity
                day_longs, day_shorts = 0, 0
                day_trades_closed = []
                day_stopped = False
                
                # Check Daily Stop
                # Logic: We set 'day_stopped' flag if equity drops below limit
            
            if equity < daily_equity_start * (1 - DAILY_LOSS_CAP):
                day_stopped = True

            # --- 1. Manage Positions ---
            closed_now = self.position_mgr.update_positions(pair_data, current_time)
            for t in closed_now:
                equity += t['pnl_dollar']
                trades_log.append(t)
                day_trades_closed.append(t)
            
            # --- 2. Check Scaling ---
            for symbol in list(self.position_mgr.positions.keys()):
                if symbol in pair_data and current_time in pair_data[symbol].index:
                    df = pair_data[symbol]
                    row = df.loc[current_time]
                    should_scale, multiplier = self.position_mgr.check_scale(symbol, row['close'], row['atr'])
                    if should_scale:
                         self.position_mgr.add_to_position(symbol, row['close'], multiplier, current_time)

            # --- 3. Scan for New Trades ---
            if not day_stopped and len(self.position_mgr.positions) < MAX_SIGNALS:
                candidates = self.score_universe_at_time(pair_data, current_time, verbose=show_debug)
                
                for cand in candidates:
                    if len(self.position_mgr.positions) >= MAX_SIGNALS: break
                    
                    sym = cand['symbol']
                    direct = cand['direction']
                    
                    # Get data
                    df = pair_data[sym]
                    row = df.loc[current_time]
                    price = row['close']
                    atr = row['atr']
                    if atr <= 0: continue
                    
                    # Entry Logic
                    sl = price - (atr * INITIAL_SL_ATR) if direct == 'LONG' else price + (atr * INITIAL_SL_ATR)
                    dist = abs(price - sl)
                    risk_amt = equity * RISK_PER_ENTRY
                    notional = (risk_amt / dist) * price
                    pos_val = min(notional / LEVERAGE, equity * MAX_CAPITAL_PER_TRADE)
                    
                    if pos_val < 5: continue
                    
                    self.position_mgr.open_position(sym, direct, price, sl, pos_val, atr, current_time)
                    if direct == 'LONG': day_longs += 1
                    else: day_shorts += 1

        # End
        self.generate_report(daily_log, trades_log, equity)

    def score_universe_at_time(self, pair_data, timestamp, top_n=TOP_N, verbose=False):
        """
        Local implementation of Scanner logic for a specific slice of time.
        Efficiently checks valid pairs and scores them using pre-calculated metrics.
        """
        metrics = {}
        rejected_reasons = {} # For debug
        
        for symbol, df in pair_data.items():
            # 1. Check if symbol has data at this time
            if timestamp not in df.index:
                if verbose: rejected_reasons[symbol] = "No Data Index"
                continue
                
            # 2. Check HISTORY REQUIREMENT (Warmup)
            try:
                if timestamp < df.index[HISTORY_NEEDED]:
                    if verbose: rejected_reasons[symbol] = "Not Enough History"
                    continue
            except IndexError:
                continue
                
            # 3. Extract Pre-calculated Metrics
            try:
                row = df.loc[timestamp]
            except KeyError:
                continue
                
            # Valid row check (sometimes NaNs linger)
            # CRITICAL FIX: Ensure columns exist before access
            required_cols = ['vol_z', 'ret_24h', 'range_pct', 'is_accel_up', 'is_accel_down', 'vol_increasing']
            missing = [col for col in required_cols if col not in row.index]
            if missing:
                if verbose: rejected_reasons[symbol] = f"Missing Cols: {missing}"
                continue
                
            if pd.isna(row['close']) or pd.isna(row['vol_z']):
                if verbose: rejected_reasons[symbol] = "NaN Values"
                continue
                
            metrics[symbol] = {
                'vol_z': row['vol_z'],
                'ret_24h': row['ret_24h'],
                'range_pct': row['range_pct'],
                'is_accel_up': row['is_accel_up'],
                'is_accel_down': row['is_accel_down'],
                'vol_increasing': row['vol_increasing']
            }

        if verbose:
            print(f"DEBUG {timestamp}: Candidates {len(metrics)}/{len(pair_data)}. Rejections: {list(rejected_reasons.values())[:5]} ...")

        if not metrics: return []

        if not metrics: return []

        # ============================================
        # STEP 2: Relative ranking across ALL pairs
        # ============================================
        symbols = list(metrics.keys())
        n = len(symbols)
        if n < 5: return []

        # Rank by 24h return
        by_ret = sorted(symbols, key=lambda s: metrics[s]['ret_24h'], reverse=True)
        for i, s in enumerate(by_ret):
            metrics[s]['ret_rank_pct'] = (i / n) * 100

        # ============================================
        # STEP 3: Score Candidates (Replicating AnomalyScanner)
        # ============================================
        candidates = []
        
        for s in symbols:
            m = metrics[s]
            score = 0
            direction = None
            
            # --- LONG SCORING ---
            long_score = 0
            if m['vol_z'] >= 3.0: long_score += 40
            elif m['vol_z'] >= 2.0: long_score += 30
            elif m['vol_z'] >= 1.5: long_score += 15
            
            if m['ret_rank_pct'] <= 2: long_score += 40
            elif m['ret_rank_pct'] <= 5: long_score += 30
            elif m['ret_rank_pct'] <= 10: long_score += 20
            elif m['ret_rank_pct'] <= 15: long_score += 10
            
            if m['range_pct'] >= 95: long_score += 30
            elif m['range_pct'] >= 85: long_score += 20
            elif m['range_pct'] >= 75: long_score += 10
            
            if m['is_accel_up']: long_score += 15
            if m['vol_increasing'] and long_score >= 20: long_score += 10
            
            # Combo bonus Long
            sig_count = sum([m['vol_z'] >= 1.5, m['ret_rank_pct'] <= 15, m['range_pct'] >= 75])
            if sig_count >= 3: long_score += 20
            elif sig_count >= 2: long_score += 10
            
            # --- SHORT SCORING ---
            short_score = 0
            if m['vol_z'] >= 3.0: short_score += 40
            elif m['vol_z'] >= 2.0: short_score += 30
            elif m['vol_z'] >= 1.5: short_score += 15
            
            if m['ret_rank_pct'] >= 98: short_score += 40
            elif m['ret_rank_pct'] >= 95: short_score += 30
            elif m['ret_rank_pct'] >= 90: short_score += 20
            elif m['ret_rank_pct'] >= 85: short_score += 10
            
            if m['range_pct'] <= 5: short_score += 30
            elif m['range_pct'] <= 15: short_score += 20
            elif m['range_pct'] <= 25: short_score += 10
            
            if m['is_accel_down']: short_score += 15
            if m['vol_increasing'] and short_score >= 20: short_score += 10
            
            # Combo bonus Short
            sig_count = sum([m['vol_z'] >= 1.5, m['ret_rank_pct'] >= 85, m['range_pct'] <= 25])
            if sig_count >= 3: short_score += 20
            elif sig_count >= 2: short_score += 10
            
            # DECIDE DIRECTION
            if long_score > short_score and long_score > 0:
                candidates.append({'symbol': s, 'direction': 'LONG', 'score': long_score})
            elif short_score > long_score and short_score > 0:
                candidates.append({'symbol': s, 'direction': 'SHORT', 'score': short_score})
                
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_n]

    def generate_report(self, daily, trades, equity):
        report_path = os.path.join(self.reports_dir, "continuous_simulation.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# CONTINUOUS SIMULATION REPORT\n")
            f.write(f"Final Equity: ${equity:,.2f}\n")
            f.write(f"Total Trades: {len(trades)}\n\n")
            
            # Write Trades
            f.write("## Trade Log\n")
            lines = [f"| {t['symbol']} | {t['direction']} | {t['pnl_dollar']:.2f} | {t['reason']} |" for t in trades]
            f.write("| Sym | Dir | PnL | Reason |\n|---|---|---|---|\n")
            f.write("\n".join(lines))
            
        print(f"\nReport written to {report_path}")

if __name__ == "__main__":
    bot = ContinuousBacktest()
    bot.run()
