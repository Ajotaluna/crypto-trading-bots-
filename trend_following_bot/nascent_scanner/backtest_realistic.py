"""
Realistic Bot Simulator V4 — SCALED TREND RIDER

Based on X-Ray analysis of 190 scanner signals:
  - 92% of signals are winners (MFE >= 1 ATR)
  - Median MFE: 5.43 ATR (massive potential)
  - Median MAE: 4.28 ATR (deep pullbacks before moving)
  - SL 2.0 ATR killed 74% of winning trades
  - Shorts > Longs: 95% vs 89% winners
  - Low scores (60-100) outperform high scores (120+)

Strategy:
  - SCALE IN: 3 entries per signal (at signal, -1.5 ATR, -3.0 ATR)
  - WIDE SL: 5.0 ATR from first entry (covers 50% of winner MAE)
  - Trail at 2.0 ATR behind peak (~40% of median MFE)
  - Score filter: only enter score < 120 (data shows best performance)
  - Short bias: 40L/60S allocation
  - Max hold: 40h (75th percentile of peak time)
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nascent_scanner.scanner_anomaly import AnomalyScanner

# ================================================================
# CONFIGURATION — DATA-DRIVEN FROM X-RAY
# ================================================================
INITIAL_CAPITAL = 1000
LEVERAGE = 5
COMMISSION = 0.04 / 100       # 0.04% per side

# Risk: 1% per entry x 3 entries = 3% total per signal
RISK_PER_ENTRY = 0.01         # 1% equity per entry
MAX_CAPITAL_PER_TRADE = 0.10  # 10% max position size per entry
MAX_SIGNALS = 3               # Max concurrent signals being tracked
MAX_HOLD_CANDLES = 160        # 40h max hold
DAILY_LOSS_CAP = 0.08         # 8% daily loss cap

CANDLES_PER_DAY = 96
HISTORY_NEEDED = 480
WARMUP_DAYS = 3
TOP_N = 10

# SL/Trailing config — DATA-DRIVEN
INITIAL_SL_ATR = 5.0          # Wide SL: 50% of winning MAE survived
BE_LOCK_ATR = 1.5             # Lock BE after 1.5 ATR (MFE P25 = 2.28)
TRAIL_DISTANCE_ATR = 2.0      # Trail 2.0 ATR (~40% of median MFE 5.43)

# Scaled entry levels (ATR from first entry, in adverse direction)
SCALE_LEVEL_2 = 1.5           # Add 2nd entry at -1.5 ATR pullback
SCALE_LEVEL_3 = 3.0           # Add 3rd entry at -3.0 ATR pullback
MAX_SCALE = 3                 # Max 3 entries

# Score filter — low scores perform better
MAX_SCORE = 120               # Skip signals with score >= 120

# Entry windows
ENTRY_WINDOW_CANDLES = 48     # 12h entry window (trends develop slowly)


# ================================================================
# INDICATOR CALCULATOR
# ================================================================
def calculate_indicators(df):
    """Add all needed indicators."""
    df = df.copy()
    cols = {c: c.lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    close = pd.to_numeric(df['close'], errors='coerce')
    high = pd.to_numeric(df['high'], errors='coerce')
    low = pd.to_numeric(df['low'], errors='coerce')
    volume = pd.to_numeric(df['volume'], errors='coerce')

    # EMAs
    df['ema_9'] = close.ewm(span=9, adjust=False).mean()
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (14)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # ADX (14)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_14 = df['atr'].replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.rolling(14).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Volume MA
    df['vol_ma'] = volume.rolling(20).mean()

    df.fillna(0, inplace=True)
    return df


# ================================================================
# ENTRY CONFIRMATION (timing only — direction from scanner)
# ================================================================
def confirm_entry(df, direction):
    """
    Confirm entry TIMING only — direction comes from scanner.
    Simple checks: RSI not extreme + correct candle color.
    """
    if len(df) < 10:
        return False

    curr = df.iloc[-2]
    close_p = curr['close']
    atr = curr['atr']

    if atr <= 0:
        return False

    if direction == 'LONG':
        if close_p <= curr['open']:
            return False
        if curr['rsi'] > 72 or curr['rsi'] < 25:
            return False
        ema20 = curr['ema_20']
        if ema20 > 0 and (close_p - ema20) / atr > 3.0:
            return False
        return True

    elif direction == 'SHORT':
        if close_p >= curr['open']:
            return False
        if curr['rsi'] < 28 or curr['rsi'] > 75:
            return False
        ema20 = curr['ema_20']
        if ema20 > 0 and (ema20 - close_p) / atr > 3.0:
            return False
        return True

    return False


# ================================================================
# POSITION MANAGER — SCALED ENTRIES + WIDE SL + TRAILING
# ================================================================
class PositionManager:
    """
    Manages scaled positions: up to 3 entries per signal.
    Each signal tracks its entries and computes aggregate P&L.
    """

    def __init__(self):
        self.positions = {}  # symbol -> position dict

    def open_position(self, symbol, direction, entry_price, sl,
                      amount, atr_at_entry, candle_idx):
        """Open first entry of a scaled position."""
        self.positions[symbol] = {
            'symbol': symbol,
            'direction': direction,
            'entries': [
                {'price': entry_price, 'amount': amount, 'idx': candle_idx}
            ],
            'entry_price': entry_price,  # First entry price (reference)
            'avg_price': entry_price,    # Will update as we scale in
            'total_amount': amount,
            'sl': sl,
            'entry_idx': candle_idx,
            'be_locked': False,
            'best_price': entry_price,
            'atr_at_entry': atr_at_entry,
            'scale_level': 1,            # How many entries done (1-3)
        }

    def add_to_position(self, symbol, entry_price, amount, candle_idx):
        """Add a scaled entry to an existing position."""
        pos = self.positions[symbol]
        pos['entries'].append({
            'price': entry_price, 'amount': amount, 'idx': candle_idx
        })
        # Update average price
        total_cost = sum(e['price'] * e['amount'] for e in pos['entries'])
        pos['total_amount'] = sum(e['amount'] for e in pos['entries'])
        pos['avg_price'] = total_cost / pos['total_amount'] if pos['total_amount'] > 0 else entry_price
        pos['scale_level'] += 1

    def update_positions(self, pair_data, candle_idx):
        """Candle-by-candle: SL check, scaling, BE lock, trailing."""
        closed = []
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            if symbol not in pair_data:
                continue

            df = pair_data[symbol]
            
            # CRITICAL FIX: If data ends, close position to avoid "zombies"
            if candle_idx >= len(df):
                last_close = float(df.iloc[-1]['close'])
                # Force close at last known price
                closed.append(self._close_position(symbol, last_close, 'DATA_ENDED', len(df)-1))
                continue

            candle = df.iloc[candle_idx]
            c_high = float(candle['high'])
            c_low = float(candle['low'])
            c_close = float(candle['close'])

            atr_val = float(candle['atr']) if candle['atr'] > 0 else pos['atr_at_entry']
            if atr_val <= 0:
                atr_val = c_close * 0.02

            # Track best price for trailing
            if pos['direction'] == 'LONG':
                if c_high > pos['best_price']:
                    pos['best_price'] = c_high
                pnl_atr = (c_close - pos['avg_price']) / atr_val
            else:
                if c_low < pos['best_price']:
                    pos['best_price'] = c_low
                pnl_atr = (pos['avg_price'] - c_close) / atr_val

            # 1. CHECK STOP LOSS
            is_sl = False
            if pos['direction'] == 'LONG':
                if c_low <= pos['sl']:
                    is_sl = True
            else:
                if c_high >= pos['sl']:
                    is_sl = True

            if is_sl:
                exit_price = pos['sl']
                reason = 'TRAILING_STOP' if pos['be_locked'] else 'STOP_LOSS'
                closed.append(self._close_position(symbol, exit_price, reason, candle_idx))
                continue

            # 2. BREAKEVEN LOCK
            hold_candles = candle_idx - pos['entry_idx']
            if not pos['be_locked'] and pnl_atr >= BE_LOCK_ATR:
                buffer = pos['avg_price'] * 0.002
                if pos['direction'] == 'LONG':
                    pos['sl'] = pos['avg_price'] + buffer
                else:
                    pos['sl'] = pos['avg_price'] - buffer
                pos['be_locked'] = True

            # 3. SMOOTH TRAILING (after BE lock)
            if pos['be_locked']:
                if pnl_atr > 6.0:
                    trail_atr = 1.0
                elif pnl_atr > 4.0:
                    trail_atr = 1.5
                else:
                    trail_atr = TRAIL_DISTANCE_ATR

                if pos['direction'] == 'LONG':
                    trail_sl = pos['best_price'] - (atr_val * trail_atr)
                    if trail_sl > pos['sl']:
                        pos['sl'] = trail_sl
                else:
                    trail_sl = pos['best_price'] + (atr_val * trail_atr)
                    if trail_sl < pos['sl']:
                        pos['sl'] = trail_sl

            # 4. MAX HOLD TIME
            if hold_candles >= MAX_HOLD_CANDLES:
                closed.append(self._close_position(symbol, c_close, 'MAX_TIME', candle_idx))

        return closed

    def check_scale_opportunity(self, symbol, current_price, atr_val):
        """Check if we should add to an existing position."""
        if symbol not in self.positions:
            return False, 0

        pos = self.positions[symbol]
        if pos['scale_level'] >= MAX_SCALE:
            return False, 0
        if pos['be_locked']:
            return False, 0  # Already in profit, don't add

        ref_price = pos['entry_price']  # First entry price

        if pos['direction'] == 'LONG':
            # Price dropped from first entry — good to add
            adverse_atr = (ref_price - current_price) / atr_val
        else:
            adverse_atr = (current_price - ref_price) / atr_val

        if pos['scale_level'] == 1 and adverse_atr >= SCALE_LEVEL_2:
            return True, 2

        if pos['scale_level'] == 2 and adverse_atr >= SCALE_LEVEL_3:
            return True, 3

        return False, 0

    def _close_position(self, symbol, exit_price, reason, candle_idx):
        pos = self.positions.pop(symbol)

        # P&L on total position using average price
        if pos['direction'] == 'LONG':
            pnl_pct = (exit_price - pos['avg_price']) / pos['avg_price']
        else:
            pnl_pct = (pos['avg_price'] - exit_price) / pos['avg_price']

        notional = pos['total_amount'] * LEVERAGE
        fees = notional * COMMISSION * 2
        pnl_dollar = pnl_pct * notional - fees

        return {
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_price': pos['avg_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_dollar': pnl_dollar,
            'reason': reason,
            'hold_candles': candle_idx - pos['entry_idx'],
            'be_locked': pos['be_locked'],
            'scale_level': pos['scale_level'],
        }


# ================================================================
# MAIN SIMULATION
# ================================================================
class RealisticBacktest:
    def __init__(self):
        # Ask user which data to use
        print("\n📂 Select Data Source:")
        print("1. data (1 Year History) [Default]")
        print("2. data_monthly (January 2026)")
        choice = input("Choice (1/2): ").strip()
        
        folder_name = "data_monthly" if choice == "2" else "data"
        self.data_dir = os.path.join(os.path.dirname(__file__), folder_name)
        print(f"✅ Using data from: {folder_name}\n")
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.scanner = AnomalyScanner()
        self.position_mgr = PositionManager()

    def load_data(self):
        pair_data = {}
        files = [f for f in os.listdir(self.data_dir) if f.endswith("_15m.csv")]
        for f in files:
            symbol = f.replace("_15m.csv", "")
            try:
                df = pd.read_csv(os.path.join(self.data_dir, f))
                if len(df) >= 1488:
                    pair_data[symbol] = df
            except Exception:
                continue
        return pair_data

    def run(self):
        print("=" * 70)
        print("REALISTIC BOT V4 - SCALED TREND RIDER")
        print("Scanner -> Scale In on Pullbacks -> Wide SL + Trailing")
        print("=" * 70)

        pair_data_raw = self.load_data()
        print(f"Loaded {len(pair_data_raw)} pairs")

        print("Calculating indicators...", flush=True)
        pair_data_ind = {}
        for symbol, df in pair_data_raw.items():
            try:
                pair_data_ind[symbol] = calculate_indicators(df)
            except:
                continue
        print(f"Indicators ready for {len(pair_data_ind)} pairs")

        lengths = [len(df) for df in pair_data_raw.values()]
        median_len = int(np.median(lengths))
        warmup_candles = WARMUP_DAYS * CANDLES_PER_DAY + HISTORY_NEEDED
        test_start_idx = warmup_candles
        test_days = (median_len - test_start_idx) // CANDLES_PER_DAY

        print(f"Test period: {test_days} days")
        print(f"Capital: ${INITIAL_CAPITAL:,} | Leverage: {LEVERAGE}x | Max Signals: {MAX_SIGNALS}")
        print(f"Risk: {RISK_PER_ENTRY*100:.1f}%/entry x3 | SL: {INITIAL_SL_ATR} ATR | Trail: {TRAIL_DISTANCE_ATR} ATR")
        print(f"Score filter: < {MAX_SCORE} | Entry window: {ENTRY_WINDOW_CANDLES*15/60:.0f}h")
        print()

        equity = INITIAL_CAPITAL
        all_trades = []
        daily_results = []

        for day_num in range(test_days):
            day_start_idx = test_start_idx + day_num * CANDLES_PER_DAY

            try:
                sample_df = list(pair_data_raw.values())[0]
                day_label = str(sample_df.iloc[day_start_idx]['timestamp'])[:10]
            except:
                day_label = f"Day {day_num + 1}"

            # Scanner: watchlist WITH direction AND score
            watchlist_picks = self.scanner.score_universe(
                pair_data_raw, day_start_idx, top_n=TOP_N
            )
            # Filter by score and build watchlist
            watchlist = {}
            for p in watchlist_picks:
                if p.get('score', 999) < MAX_SCORE:
                    watchlist[p['symbol']] = {
                        'direction': p['direction'],
                        'score': p.get('score', 0),
                    }

            daily_equity_start = equity
            day_trades = []
            day_longs = 0
            day_shorts = 0
            day_stopped = False
            attempted_today = set()

            for candle_offset in range(CANDLES_PER_DAY):
                candle_idx = day_start_idx + candle_offset

                # Daily loss cap
                if equity < daily_equity_start * (1 - DAILY_LOSS_CAP):
                    day_stopped = True
                    break

                # 1. Manage existing positions
                closed = self.position_mgr.update_positions(pair_data_ind, candle_idx)
                for trade in closed:
                    equity += trade['pnl_dollar']
                    day_trades.append(trade)
                    all_trades.append(trade)

                # 2. Check scaling opportunities for existing positions
                for symbol in list(self.position_mgr.positions.keys()):
                    if symbol not in pair_data_ind:
                        continue
                    df = pair_data_ind[symbol]
                    if candle_idx >= len(df):
                        continue

                    current_price = float(df.iloc[candle_idx]['close'])
                    atr_val = float(df.iloc[candle_idx]['atr'])
                    if atr_val <= 0:
                        continue

                    should_scale, level = self.position_mgr.check_scale_opportunity(
                        symbol, current_price, atr_val
                    )
                    if should_scale:
                        # Size the add-on entry
                        risk_amount = equity * RISK_PER_ENTRY
                        pos = self.position_mgr.positions[symbol]
                        risk_distance = abs(current_price - pos['sl'])
                        if risk_distance > 0:
                            notional_needed = risk_amount / (risk_distance / current_price)
                            add_amount = notional_needed / LEVERAGE
                            max_allowed = equity * MAX_CAPITAL_PER_TRADE
                            add_amount = min(add_amount, max_allowed)
                            if add_amount >= 5:
                                self.position_mgr.add_to_position(
                                    symbol, current_price, add_amount, candle_idx
                                )

                # 3. New entries — only in entry window
                if candle_offset >= ENTRY_WINDOW_CANDLES:
                    continue

                # Count active signals (not entries)
                active_signals = len(self.position_mgr.positions)
                if active_signals >= MAX_SIGNALS:
                    continue

                for symbol in watchlist:
                    if symbol in self.position_mgr.positions:
                        continue
                    if symbol in attempted_today:
                        continue
                    if active_signals >= MAX_SIGNALS:
                        break
                    if symbol not in pair_data_ind:
                        continue

                    df = pair_data_ind[symbol]
                    if candle_idx >= len(df) or candle_idx < 200:
                        continue

                    df_slice = df.iloc[:candle_idx + 1]
                    direction = watchlist[symbol]['direction']

                    if not confirm_entry(df_slice, direction):
                        continue

                    # EXECUTE first entry
                    attempted_today.add(symbol)
                    entry_price = float(df_slice['close'].iloc[-1])
                    atr = float(df_slice['atr'].iloc[-1])
                    if atr <= 0:
                        continue

                    # Wide SL based on X-ray data
                    if direction == 'LONG':
                        sl = entry_price - (atr * INITIAL_SL_ATR)
                        day_longs += 1
                    else:
                        sl = entry_price + (atr * INITIAL_SL_ATR)
                        day_shorts += 1

                    # Risk-based sizing (correct: accounts for leverage)
                    risk_distance = abs(entry_price - sl)
                    if risk_distance == 0:
                        continue

                    risk_amount = equity * RISK_PER_ENTRY
                    notional_needed = risk_amount / (risk_distance / entry_price)
                    position_value = notional_needed / LEVERAGE
                    max_allowed = equity * MAX_CAPITAL_PER_TRADE
                    position_value = min(position_value, max_allowed)

                    if position_value < 5:
                        continue

                    self.position_mgr.open_position(
                        symbol, direction, entry_price, sl,
                        position_value, atr, candle_idx
                    )
                    active_signals += 1

            # Daily summary
            day_pnl = equity - daily_equity_start
            wins = sum(1 for t in day_trades if t['pnl_dollar'] > 0)
            losses = sum(1 for t in day_trades if t['pnl_dollar'] <= 0)

            daily_results.append({
                'day': day_num + 1,
                'date': day_label,
                'pnl': day_pnl,
                'equity': equity,
                'trades': len(day_trades),
                'wins': wins,
                'losses': losses,
                'longs': day_longs,
                'shorts': day_shorts,
                'stopped': day_stopped,
                'open_pos': len(self.position_mgr.positions),
            })

            emoji = "+" if day_pnl > 0 else ("-" if day_pnl < 0 else "=")
            stop_icon = " STOPPED" if day_stopped else ""
            open_str = f" [Open:{len(self.position_mgr.positions)}]" if self.position_mgr.positions else ""
            print(
                f"  {emoji} Day {day_num+1:2d} | {day_label} | "
                f"New: {day_longs}L/{day_shorts}S | "
                f"Closed: {len(day_trades)} ({wins}W/{losses}L) | "
                f"P&L: ${day_pnl:+7.2f} | "
                f"Equity: ${equity:,.2f}{open_str}{stop_icon}"
            )

        # Close remaining at end of test
        for symbol in list(self.position_mgr.positions.keys()):
            if symbol in pair_data_ind:
                df = pair_data_ind[symbol]
                end_idx = min(test_start_idx + test_days * CANDLES_PER_DAY - 1, len(df) - 1)
                exit_price = float(df.iloc[end_idx]['close'])
                trade = self.position_mgr._close_position(symbol, exit_price, 'END_OF_TEST', end_idx)
                equity += trade['pnl_dollar']
                all_trades.append(trade)

        self._generate_report(daily_results, all_trades)

    def _generate_report(self, daily_results, all_trades):
        if not daily_results or not all_trades:
            print("No results.")
            return

        report = []
        report.append("# REALISTIC BOT V4 - SCALED TREND RIDER")
        report.append(f"## Capital: ${INITIAL_CAPITAL:,} | Leverage: {LEVERAGE}x | Risk: {RISK_PER_ENTRY*100:.1f}%/entry x3")
        report.append(f"## SL: {INITIAL_SL_ATR} ATR | Trail: {TRAIL_DISTANCE_ATR} ATR | Max Hold: {MAX_HOLD_CANDLES*15/60:.0f}h")
        report.append(f"## Score filter: < {MAX_SCORE}")
        report.append("")

        final_equity = daily_results[-1]['equity']
        roi = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        total_trades = len(all_trades)
        winning_trades = [t for t in all_trades if t['pnl_dollar'] > 0]
        losing_trades = [t for t in all_trades if t['pnl_dollar'] <= 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        avg_win = np.mean([t['pnl_dollar'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_dollar'] for t in losing_trades]) if losing_trades else 0
        rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        avg_hold = np.mean([t['hold_candles'] for t in all_trades]) * 15 / 60 if all_trades else 0
        med_hold = np.median([t['hold_candles'] for t in all_trades]) * 15 / 60 if all_trades else 0

        win_days = sum(1 for d in daily_results if d['pnl'] > 0)
        total_longs = sum(1 for t in all_trades if t['direction'] == 'LONG')
        total_shorts = sum(1 for t in all_trades if t['direction'] == 'SHORT')
        long_wins = sum(1 for t in all_trades if t['direction'] == 'LONG' and t['pnl_dollar'] > 0)
        short_wins = sum(1 for t in all_trades if t['direction'] == 'SHORT' and t['pnl_dollar'] > 0)

        trades_per_day = total_trades / len(daily_results) if daily_results else 0

        # Scale level stats
        scale_levels = {}
        for t in all_trades:
            sl = t.get('scale_level', 1)
            if sl not in scale_levels:
                scale_levels[sl] = {'count': 0, 'pnl': 0, 'wins': 0}
            scale_levels[sl]['count'] += 1
            scale_levels[sl]['pnl'] += t['pnl_dollar']
            if t['pnl_dollar'] > 0:
                scale_levels[sl]['wins'] += 1

        # Max drawdown
        peak = INITIAL_CAPITAL
        max_dd = 0
        for d in daily_results:
            if d['equity'] > peak:
                peak = d['equity']
            dd = (d['equity'] - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

        best = max(all_trades, key=lambda t: t['pnl_dollar'])
        worst = min(all_trades, key=lambda t: t['pnl_dollar'])

        # Exit reasons
        exit_reasons = {}
        for t in all_trades:
            r = t['reason']
            if r not in exit_reasons:
                exit_reasons[r] = {'count': 0, 'pnl': 0, 'wins': 0}
            exit_reasons[r]['count'] += 1
            exit_reasons[r]['pnl'] += t['pnl_dollar']
            if t['pnl_dollar'] > 0:
                exit_reasons[r]['wins'] += 1

        # Summary
        report.append("## RESUMEN")
        report.append("| Metrica | Valor |")
        report.append("|---------|-------|")
        report.append(f"| Capital Final | ${final_equity:,.2f} |")
        report.append(f"| ROI Total | {roi:+.1f}% |")
        report.append(f"| Total Trades | {total_trades} ({trades_per_day:.1f}/dia) |")
        report.append(f"| Win Rate | {win_rate:.0f}% ({len(winning_trades)}/{total_trades}) |")
        report.append(f"| Avg Win | ${avg_win:+,.2f} |")
        report.append(f"| Avg Loss | ${avg_loss:+,.2f} |")
        report.append(f"| Risk/Reward | {rr_ratio:.2f}R |")
        report.append(f"| Dias Ganadores | {win_days}/{len(daily_results)} |")
        report.append(f"| Max Drawdown | {max_dd:.1f}% |")
        report.append(f"| Avg/Med Hold | {avg_hold:.1f}h / {med_hold:.1f}h |")
        if total_longs > 0:
            report.append(f"| Longs | {total_longs} (WR: {long_wins/total_longs*100:.0f}%) |")
        if total_shorts > 0:
            report.append(f"| Shorts | {total_shorts} (WR: {short_wins/total_shorts*100:.0f}%) |")
        report.append(f"| Mejor Trade | {best['symbol']} {best['direction']} ${best['pnl_dollar']:+,.2f} ({best['hold_candles']*15/60:.1f}h) |")
        report.append(f"| Peor Trade | {worst['symbol']} {worst['direction']} ${worst['pnl_dollar']:+,.2f} ({worst['hold_candles']*15/60:.1f}h) |")
        report.append("")

        # Exit reasons
        report.append("## RAZONES DE SALIDA")
        report.append("| Razon | Count | WR | P&L |")
        report.append("|-------|-------|-----|-----|")
        for r, data in sorted(exit_reasons.items(), key=lambda x: x[1]['pnl'], reverse=True):
            wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
            report.append(f"| {r} | {data['count']} | {wr:.0f}% | ${data['pnl']:+,.2f} |")
        report.append("")

        # Scale level breakdown
        report.append("## NIVEL DE ESCALADO")
        report.append("| Entries | Count | WR | P&L |")
        report.append("|---------|-------|-----|-----|")
        for sl_level in sorted(scale_levels.keys()):
            data = scale_levels[sl_level]
            wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
            report.append(f"| {sl_level} entry{'s' if sl_level > 1 else ' '} | {data['count']} | {wr:.0f}% | ${data['pnl']:+,.2f} |")
        report.append("")

        # Long vs Short
        report.append("## LONG vs SHORT")
        report.append("| Direction | Trades | WR | P&L |")
        report.append("|-----------|--------|----|-----|")
        long_pnl = sum(t['pnl_dollar'] for t in all_trades if t['direction'] == 'LONG')
        short_pnl = sum(t['pnl_dollar'] for t in all_trades if t['direction'] == 'SHORT')
        if total_longs > 0:
            report.append(f"| LONG | {total_longs} | {long_wins/total_longs*100:.0f}% | ${long_pnl:+,.2f} |")
        if total_shorts > 0:
            report.append(f"| SHORT | {total_shorts} | {short_wins/total_shorts*100:.0f}% | ${short_pnl:+,.2f} |")
        report.append("")

        # Daily breakdown
        report.append("## DIA POR DIA")
        report.append("| Dia | Fecha | New L/S | Closed W/L | Open | P&L | Equity |")
        report.append("|-----|-------|---------|------------|------|-----|--------|")
        for d in daily_results:
            emoji = "+" if d['pnl'] > 0 else ("-" if d['pnl'] < 0 else "=")
            stop = " STOP" if d['stopped'] else ""
            report.append(
                f"| {emoji} {d['day']:2d} | {d['date']} | "
                f"{d['longs']}L/{d['shorts']}S | "
                f"{d['wins']}W/{d['losses']}L | "
                f"{d['open_pos']} | "
                f"${d['pnl']:+,.2f} | ${d['equity']:,.2f}{stop} |"
            )
        report.append("")

        report.append("## VEREDICTO")
        report.append(f"> Capital: ${INITIAL_CAPITAL:,} -> ${final_equity:,.2f} ({roi:+.1f}%) en {len(daily_results)} dias")
        report.append(f"> {total_trades} trades ({trades_per_day:.1f}/dia) | WR: {win_rate:.0f}% | RR: {rr_ratio:.2f}R")

        full_report = "\n".join(report)
        print("\n" + full_report)

        report_path = os.path.join(self.reports_dir, "realistic_simulation.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\nReport: {report_path}")


if __name__ == "__main__":
    bt = RealisticBacktest()
    bt.run()
