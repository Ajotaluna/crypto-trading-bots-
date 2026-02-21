"""
Realistic Bot Simulator V4 — PROGRESSIVE CALIBRATION
Based on RealisticBot V4 logic with 5-day warmup for scanner calibration.

VERSION: Faithfully restored from backtest_realistic.py with ONLY 
data loading and timeline logic modified to align all pairs chronologically.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nascent_scanner.scanner_anomaly import AnomalyScanner
# NEW: Import strategy logic and constants from independent file
from nascent_scanner.trading_strategy import (
    calculate_indicators,
    confirm_entry,
    PositionManager,
    INITIAL_CAPITAL,
    LEVERAGE,
    COMMISSION,
    RISK_PER_ENTRY,
    MAX_CAPITAL_PER_TRADE,
    MAX_SIGNALS,
    MAX_HOLD_CANDLES,
    DAILY_LOSS_CAP,
    CANDLES_PER_DAY,
    HISTORY_NEEDED,
    WARMUP_DAYS,
    TOP_N,
    INITIAL_SL_ATR,
    BE_LOCK_ATR,
    TRAIL_DISTANCE_ATR,
    SCALE_LEVEL_2,
    SCALE_LEVEL_3,
    MAX_SCALE,
    MAX_SCORE,
    ENTRY_WINDOW_CANDLES
)

# ================================================================
# MAIN SIMULATION
# ================================================================
class ProgressiveBacktest:
    def __init__(self):
        # Interactive Data Selection
        print("\n--- DATA SOURCE SELECTION ---")
        print("1: Production (nascent_scanner/data) [Full - Risks included]")
        print("2: Testing (nascent_scanner/data_monthly) [Limited History]")
        print("3: Production FILTERED (nascent_scanner/data) [Full History + Quality Filter]")
        choice = input("Select [1/2/3] (Default 1): ").strip()

        self.whitelist = None # Default: All files

        if choice == "2":
            folder_name = "data_monthly"
        elif choice == "3":
            # Use Production Data but filter by Monthly Whitelist
            folder_name = "data"
            monthly_dir = os.path.join(os.path.dirname(__file__), "data_monthly")
            if os.path.exists(monthly_dir):
                self.whitelist = set(os.listdir(monthly_dir))
                print(f"✅ Whitelist loaded: {len(self.whitelist)} files from data_monthly")
            else:
                print("⚠️ Warning: data_monthly not found for whitelist. Using full data.")
        else:
            folder_name = "data"

        self.data_dir = os.path.join(os.path.dirname(__file__), folder_name)
        print(f"✅ Using data from: {folder_name}\n")
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.scanner = AnomalyScanner()
        self.position_mgr = PositionManager()

    def load_aligned_data(self):
        """
        Loads all CSVs, sorts by timestamp, finds the global timeline,
        and REINDEXES all dataframes to align them.
        """
        raw_data = {}
        if not os.path.exists(self.data_dir):
            print(f"❌ Error: Data directory not found: {self.data_dir}")
            return {}, []

        # 1. Load raw data
        all_timestamps = set()
        files = [f for f in os.listdir(self.data_dir) if f.endswith("_15m.csv")]
        
        # Apply Whitelist Filter
        if self.whitelist:
            original_count = len(files)
            files = [f for f in files if f in self.whitelist]
            print(f"🔍 Whitelist Applied: Filtered {original_count} -> {len(files)} files.")
            
        print(f"Loading {len(files)} files...")
        
        for f in files:
            symbol = f.replace("_15m.csv", "")
            try:
                df = pd.read_csv(os.path.join(self.data_dir, f))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.sort_values('timestamp', inplace=True)
                
                # Check minimum length
                if len(df) < (WARMUP_DAYS * CANDLES_PER_DAY + HISTORY_NEEDED):
                    continue
                
                df = calculate_indicators(df)
                raw_data[symbol] = df
                all_timestamps.update(df['timestamp'])
            except:
                continue

        if not raw_data:
            return {}, []

        # 2. Creating Global Timeline
        timeline = sorted(list(all_timestamps))
        full_index = pd.DatetimeIndex(timeline)
        print(f"Global Timeline: {len(timeline)} candles ({timeline[0]} to {timeline[-1]})")

        # 3. Align DataFrames
        aligned_data = {}
        for symbol, df in raw_data.items():
            # Set index to timestamp for reindexing
            df_reindexed = df.set_index('timestamp').reindex(full_index)
            # We do NOT fillna here because we want to know when data is missing
            # But we reset index to have 'timestamp' as a column again if needed, or keeping it as index?
            # PositionManager expects column access usually via iloc, but let's see.
            # PositionManager does `candle = df.iloc[candle_idx]` then `candle['high']`.
            # If index is datetime, `iloc` still works by position.
            # BUT `candle['high']` works on Series.
            # So reset_index is safer to match original structure where 0..N is row index.
            df_reindexed.reset_index(inplace=True)
            df_reindexed.rename(columns={'index': 'timestamp'}, inplace=True) # reindex might name it 'index'
            aligned_data[symbol] = df_reindexed

        return aligned_data, timeline

    def run(self):
        print("=" * 70)
        print("REALISTIC BOT V4 - PROGRESSIVE CHRONOLOGICAL BACKTEST")
        print("=" * 70)

        pair_data, timeline = self.load_aligned_data()
        if not pair_data:
            print("No data.")
            return

        equity = INITIAL_CAPITAL
        all_trades = []
        
        # We need to track when each pair actually starts to apply WARMUP
        pair_start_times = {}
        for s, df in pair_data.items():
            valid_idx = df['close'].first_valid_index()
            if valid_idx is not None:
                pair_start_times[s] = df.iloc[valid_idx]['timestamp']
            else:
                pair_start_times[s] = pd.Timestamp.max

        # Global Warmup: Skip first 10 days (960 candles)
        GLOBAL_WARMUP = (WARMUP_DAYS * CANDLES_PER_DAY) + HISTORY_NEEDED
        
        # Stats trackers
        daily_logs = []
        daily_watchlist = []
        day_count = 0
        current_day_stats = {'new_l': 0, 'new_s': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'open': 0}
        
        # Iterate GLOBAL index
        for t_idx, current_time in enumerate(timeline):
            # Skip global warmup
            if t_idx < GLOBAL_WARMUP:
                continue
                
            # START OF DAY SCANNING
            if (t_idx - GLOBAL_WARMUP) % CANDLES_PER_DAY == 0:
                # Log Previous Day
                if day_count > 0:
                    daily_logs.append({
                        'day': day_count,
                        'date': str(timeline[t_idx - CANDLES_PER_DAY])[:10],
                        'new_l': current_day_stats['new_l'],
                        'new_s': current_day_stats['new_s'],
                        'wins': current_day_stats['wins'],
                        'losses': current_day_stats['losses'],
                        'pnl': current_day_stats['pnl'],
                        'equity': equity,
                        'open': current_day_stats['open']
                    })

                day_count += 1
                current_day_stats = {'new_l': 0, 'new_s': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'open': len(self.position_mgr.positions)}
                
                print(f"  + Day {day_count:2d} | {str(current_time)[:10]} | Equity: ${equity:,.2f} | Pos: {len(self.position_mgr.positions)}")
                
                # 0. Identify eligible pairs (Warmup Check)
                eligible_pairs = {}
                for s, start_time in pair_start_times.items():
                    if current_time >= start_time + pd.Timedelta(days=WARMUP_DAYS):
                        # Optimization: Pass logic dense slices
                        df_aligned = pair_data[s]
                        eligible_pairs[s] = df_aligned.iloc[:t_idx+1].dropna(subset=['close'])

                # 1. Run Scanner ONCE per day
                if eligible_pairs:
                    daily_picks = self.scanner.score_universe(eligible_pairs, -1, top_n=TOP_N)
                    daily_watchlist = [p for p in daily_picks if p['score'] < MAX_SCORE]
                else:
                    daily_watchlist = []

            # 2. Update positions
            closed = self.position_mgr.update_positions(pair_data, t_idx)
            for trade in closed:
                trade['exit_time'] = current_time
                equity += trade['pnl_dollar']
                all_trades.append(trade)
                
                current_day_stats['pnl'] += trade['pnl_dollar']
                if trade['pnl_dollar'] > 0: current_day_stats['wins'] += 1
                else: current_day_stats['losses'] += 1

            # 3. Check scaling
            for symbol in list(self.position_mgr.positions.keys()):
                if symbol not in pair_data: continue
                df = pair_data[symbol]
                if t_idx >= len(df): continue
                candle = df.iloc[t_idx]
                if pd.isna(candle['close']): continue
                
                should_scale, level = self.position_mgr.check_scale_opportunity(symbol, candle['close'], candle['atr'])
                if should_scale:
                    pos = self.position_mgr.positions[symbol]
                    risk_amt = equity * RISK_PER_ENTRY
                    risk_dist = abs(candle['close'] - pos['sl'])
                    if risk_dist > 0:
                        v = (risk_amt / (risk_dist / candle['close'])) / LEVERAGE
                        v = min(v, equity * MAX_CAPITAL_PER_TRADE)
                        if v >= 5: self.position_mgr.add_to_position(symbol, candle['close'], v, t_idx)

            # 4. Check Entries
            if len(self.position_mgr.positions) < MAX_SIGNALS:
                for pick in daily_watchlist:
                    symbol = pick['symbol']
                    if symbol in self.position_mgr.positions: continue
                    if len(self.position_mgr.positions) >= MAX_SIGNALS: break

                    if symbol not in pair_data: continue
                    df = pair_data[symbol]
                    if t_idx >= len(df): continue
                    
                    # Confirm Entry with dense slice
                    df_slice = df.iloc[:t_idx+1].dropna(subset=['close'])
                    if len(df_slice) < 10: continue

                    if confirm_entry(df_slice, pick['direction']):
                        curr = df_slice.iloc[-1]
                        if pd.isna(df.iloc[t_idx]['close']): continue
                        
                        price = float(curr['close'])
                        atr = float(curr['atr'])
                        if atr <= 0: continue
                        
                        sl = price - (atr * INITIAL_SL_ATR) if pick['direction'] == 'LONG' else price + (atr * INITIAL_SL_ATR)
                        
                        risk_amt = equity * RISK_PER_ENTRY
                        risk_dist = abs(price - sl)
                        if risk_dist > 0:
                            v = (risk_amt / (risk_dist / price)) / LEVERAGE
                            v = min(v, equity * MAX_CAPITAL_PER_TRADE)
                            if v >= 5:
                                self.position_mgr.open_position(symbol, pick['direction'], price, sl, v, atr, t_idx)
                                if pick['direction'] == 'LONG': current_day_stats['new_l'] += 1
                                else: current_day_stats['new_s'] += 1
                                
                                # CRITICAL: Remove from watchlist to prevent re-entry today
                                daily_watchlist.remove(pick)

        # Log Final Partial Day
        daily_logs.append({
            'day': day_count,
            'date': str(timeline[-1])[:10],
            'new_l': current_day_stats['new_l'],
            'new_s': current_day_stats['new_s'],
            'wins': current_day_stats['wins'],
            'losses': current_day_stats['losses'],
            'pnl': current_day_stats['pnl'],
            'equity': equity,
            'open': current_day_stats['open']
        })

        self._final_report(equity, all_trades, daily_logs)

    def _final_report(self, equity, trades, daily_logs):
        print(f"\nFinal Equity: ${equity:,.2f}")
        print(f"Total Trades: {len(trades)}")
        
        roi = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        wr = 0
        avg_win, avg_loss = 0, 0
        if trades:
            wins = [t for t in trades if t['pnl_dollar'] > 0]
            losses = [t for t in trades if t['pnl_dollar'] <= 0]
            wr = len(wins) / len(trades) * 100
            avg_win = np.mean([t['pnl_dollar'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl_dollar'] for t in losses]) if losses else 0
        
        lines = []
        lines.append(f"# REALISTIC BOT V4 - PROGRESSIVE REPORT")
        lines.append(f"## Capital: ${INITIAL_CAPITAL:,.0f} | Leverage: {LEVERAGE}x | Risk: {RISK_PER_ENTRY*100}%/entry")
        lines.append(f"## SL: {INITIAL_SL_ATR} ATR | Trail: {TRAIL_DISTANCE_ATR} ATR | Max Hold: {MAX_HOLD_CANDLES} candles")
        lines.append(f"## Filter: Kalman Trend + Slope")
        lines.append(f"")
        lines.append(f"## RESUMEN")
        lines.append(f"| Metrica | Valor |")
        lines.append(f"|---------|-------|")
        lines.append(f"| Capital Final | ${equity:,.2f} |")
        lines.append(f"| ROI Total | {roi:+.1f}% |")
        lines.append(f"| Total Trades | {len(trades)} |")
        lines.append(f"| Win Rate | {wr:.1f}% ({len([t for t in trades if t['pnl_dollar']>0])}/{len(trades)}) |")
        lines.append(f"| Avg Win | ${avg_win:+.2f} |")
        lines.append(f"| Avg Loss | ${avg_loss:+.2f} |")
        lines.append(f"| Max Drawdown | N/A |") # Simplified
        
        lines.append(f"")
        lines.append(f"## RAZONES DE SALIDA")
        lines.append(f"| Razon | Count | P&L |")
        lines.append(f"|-------|-------|-----|")
        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons: reasons[r] = {'count': 0, 'pnl': 0}
            reasons[r]['count'] += 1
            reasons[r]['pnl'] += t['pnl_dollar']
        for r, d in reasons.items():
            lines.append(f"| {r} | {d['count']} | ${d['pnl']:+.2f} |")

        lines.append(f"")
        lines.append(f"## DIA POR DIA")
        lines.append(f"| Dia | Fecha | New L/S | Closed W/L | Open | P&L | Equity |")
        lines.append(f"|-----|-------|---------|------------|------|-----|--------|")
        for d in daily_logs:
            lines.append(f"| {d['day']:3d} | {d['date']} | {d['new_l']}L/{d['new_s']}S | {d['wins']}W/{d['losses']}L | {d['open']} | ${d['pnl']:+6.2f} | ${d['equity']:,.2f} |")

        path = os.path.join(self.reports_dir, "progressive_simulation_final.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"Report: {path}")

if __name__ == "__main__":
    ProgressiveBacktest().run()
