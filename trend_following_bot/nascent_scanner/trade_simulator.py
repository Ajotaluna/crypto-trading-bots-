"""
Trade Simulator — Simulates actual trading on scanner predictions.

Takes the predicted roster from the continuous time machine and simulates
trades using the 24h future data (96 candles of 15m).

Strategies tested:
1. Buy & Hold 24h: enter at open, exit at close
2. Trailing Stop: enter at open, trail with dynamic stop
3. TP/SL: fixed take profit + stop loss (risk:reward)

Capital simulation with equal allocation across all picks.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from nascent_scanner.scanner_engine import ScannerEngine

FUTURE_WINDOW = 96  # 24h of 15m candles
SCAN_OFFSETS = [144, 96, 48, 0]

# Trading parameters
INITIAL_CAPITAL = 1000  # $1,000
LEVERAGE = 5
COMMISSION = 0.04 / 100  # 0.04% per trade (Binance futures)


class TradeSimulator:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.engine = ScannerEngine()

    def _load_funding(self, symbol):
        path = os.path.join(self.data_dir, f"{symbol}_funding.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
                df['fundingTime'] = pd.to_numeric(df['fundingTime'], errors='coerce')
                return df
            except Exception:
                return None
        return None

    def _get_roster(self):
        """Run the continuous scanner and return the predicted roster."""
        files = [f for f in os.listdir(self.data_dir) if f.endswith('_15m.csv')]
        score_history = {}
        errors = 0

        for f in files:
            symbol = f.replace('_15m.csv', '')
            try:
                df = pd.read_csv(os.path.join(self.data_dir, f))
                min_required = 480 + SCAN_OFFSETS[0]
                if len(df) < min_required:
                    continue

                funding_df = self._load_funding(symbol)
                score_history[symbol] = {
                    'l1': [], 'l2': [], 'l3': [],
                    'l1_reasons_per_scan': [], 'l2_reasons_per_scan': [],
                    'l3_reasons_per_scan': [],
                }

                for scan_idx, offset in enumerate(SCAN_OFFSETS):
                    end_idx = len(df) - FUTURE_WINDOW - offset
                    if end_idx < 200:
                        for k in ['l1', 'l2', 'l3']:
                            score_history[symbol][k].append(0)
                            score_history[symbol][f'{k}_reasons_per_scan'].append('')
                        continue

                    history = df.iloc[:end_idx].copy()
                    result = self.engine.analyze(history, funding_df=funding_df)

                    if result.get('rejected'):
                        for k in ['l1', 'l2', 'l3']:
                            score_history[symbol][k].append(0)
                            score_history[symbol][f'{k}_reasons_per_scan'].append('')
                    else:
                        for k in ['l1', 'l2', 'l3']:
                            score_history[symbol][k].append(result[k]['score'])
                            score_history[symbol][f'{k}_reasons_per_scan'].append(
                                ", ".join(result[k]['reasons']))

                # True gain for reference
                future = df.iloc[-FUTURE_WINDOW:].copy()
                future.columns = [c.lower() for c in future.columns]
                sp = future['open'].iloc[0]
                ep = future['close'].iloc[-1]
                score_history[symbol]['true_gain'] = (ep - sp) / sp * 100

            except Exception:
                errors += 1

        # Build adjusted scores (same logic as continuous time machine)
        valid_symbols = [s for s in score_history if 'true_gain' in score_history[s]]
        results_for_draft = []
        for symbol in valid_symbols:
            h = score_history[symbol]
            row_data = {'symbol': symbol, 'true_gain': h['true_gain']}
            for layer_key, col_prefix in [('l1', 'l1'), ('l2', 'l2'), ('l3', 'l3')]:
                scores = h[layer_key]
                reasons_per_scan = h[f'{layer_key}_reasons_per_scan']
                max_score = max(scores)
                max_idx = scores.index(max_score)
                best_reasons = reasons_per_scan[max_idx] if max_score > 0 else ''
                nonzero_count = sum(1 for s in scores if s > 0)
                first_nz = next((i for i, s in enumerate(scores) if s > 0), -1)
                velocity = (max_score - scores[first_nz]) if (max_score > 0 and first_nz >= 0) else 0
                is_flickering = (max_score > 0 and scores[-1] == 0)
                is_stale = (nonzero_count >= 3 and max_score - min(s for s in scores if s > 0) < 30) if nonzero_count >= 3 else False

                adj = max_score
                if adj > 0:
                    if velocity > 100: adj += 50
                    elif velocity > 50: adj += 30
                    elif velocity > 0: adj += 15
                    if nonzero_count >= 3: adj += 20
                    elif nonzero_count == 2: adj += 10
                    if is_flickering: adj -= 15
                    if is_stale: adj -= 40

                row_data[f'{col_prefix}_score'] = max(0, int(adj))
                row_data[f'{col_prefix}_reasons'] = best_reasons

            results_for_draft.append(row_data)

        roster = ScannerEngine.draft_top_20(results_for_draft)
        return roster

    def _simulate_trade(self, symbol, strategy, position_size):
        """Simulate a single trade using future data."""
        path = os.path.join(self.data_dir, f"{symbol}_15m.csv")
        df = pd.read_csv(path)
        future = df.iloc[-FUTURE_WINDOW:].copy()
        future.columns = [c.lower() for c in future.columns]

        entry_price = future['open'].iloc[0]
        notional = position_size * LEVERAGE
        entry_fee = notional * COMMISSION

        if strategy == 'hold_24h':
            exit_price = future['close'].iloc[-1]
            exit_fee = notional * COMMISSION
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_dollar = (pnl_pct / 100) * notional - entry_fee - exit_fee
            max_dd = 0
            for _, candle in future.iterrows():
                dd = (candle['low'] - entry_price) / entry_price * 100
                max_dd = min(max_dd, dd)
            return {
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'max_drawdown': max_dd,
                'exit_reason': 'HOLD_24H',
                'bars_held': FUTURE_WINDOW,
            }

        elif strategy == 'trailing_stop':
            stop_pct = 2.0  # Initial 2% stop
            trail_activation = 1.5  # Start trailing after 1.5% profit
            trail_distance = 1.0  # Trail 1% below high

            highest = entry_price
            stop_price = entry_price * (1 - stop_pct / 100)
            max_dd = 0

            for i, (_, candle) in enumerate(future.iterrows()):
                # Check stop hit
                if candle['low'] <= stop_price:
                    exit_price = stop_price
                    exit_fee = notional * COMMISSION
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    pnl_dollar = (pnl_pct / 100) * notional - entry_fee - exit_fee
                    return {
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollar': pnl_dollar,
                        'max_drawdown': max_dd,
                        'exit_reason': 'STOP_HIT',
                        'bars_held': i + 1,
                    }

                # Update highest and trailing stop
                if candle['high'] > highest:
                    highest = candle['high']
                    current_gain = (highest - entry_price) / entry_price * 100
                    if current_gain >= trail_activation:
                        new_stop = highest * (1 - trail_distance / 100)
                        stop_price = max(stop_price, new_stop)

                dd = (candle['low'] - entry_price) / entry_price * 100
                max_dd = min(max_dd, dd)

            # End of period — exit at close
            exit_price = future['close'].iloc[-1]
            exit_fee = notional * COMMISSION
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_dollar = (pnl_pct / 100) * notional - entry_fee - exit_fee
            return {
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'max_drawdown': max_dd,
                'exit_reason': 'TIME_EXIT',
                'bars_held': FUTURE_WINDOW,
            }

        elif strategy == 'tp_sl':
            tp_pct = 4.0   # 4% take profit
            sl_pct = 2.0   # 2% stop loss (2:1 RR)
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
            max_dd = 0

            for i, (_, candle) in enumerate(future.iterrows()):
                # SL hit first? (check low before high for worst-case)
                if candle['low'] <= sl_price:
                    exit_price = sl_price
                    exit_fee = notional * COMMISSION
                    pnl_pct = -sl_pct
                    pnl_dollar = (pnl_pct / 100) * notional - entry_fee - exit_fee
                    return {
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollar': pnl_dollar,
                        'max_drawdown': -sl_pct,
                        'exit_reason': f'SL_HIT(-{sl_pct}%)',
                        'bars_held': i + 1,
                    }

                # TP hit
                if candle['high'] >= tp_price:
                    exit_price = tp_price
                    exit_fee = notional * COMMISSION
                    pnl_pct = tp_pct
                    pnl_dollar = (pnl_pct / 100) * notional - entry_fee - exit_fee
                    return {
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollar': pnl_dollar,
                        'max_drawdown': max_dd,
                        'exit_reason': f'TP_HIT(+{tp_pct}%)',
                        'bars_held': i + 1,
                    }

                dd = (candle['low'] - entry_price) / entry_price * 100
                max_dd = min(max_dd, dd)

            # Neither hit — exit at close
            exit_price = future['close'].iloc[-1]
            exit_fee = notional * COMMISSION
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_dollar = (pnl_pct / 100) * notional - entry_fee - exit_fee
            return {
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'max_drawdown': max_dd,
                'exit_reason': 'TIME_EXIT',
                'bars_held': FUTURE_WINDOW,
            }

    def run(self):
        print("💰 Nascent Scanner — Trade Simulator v1.0")
        print("=" * 60)
        print(f"Capital: ${INITIAL_CAPITAL:,} | Leverage: {LEVERAGE}x | Fee: {COMMISSION*100:.2f}%")
        print()

        # Step 1: Get roster from continuous scanner
        print("🔮 Running continuous scanner to get predictions...")
        roster = self._get_roster()

        if not roster:
            print("❌ No predictions generated.")
            return

        print(f"✅ Got {len(roster)} predictions")
        print()

        # Step 2: Simulate each strategy
        strategies = ['hold_24h', 'trailing_stop', 'tp_sl']
        strategy_names = {
            'hold_24h': 'Buy & Hold 24h',
            'trailing_stop': 'Trailing Stop (2% SL, 1% trail)',
            'tp_sl': 'TP/SL (4% TP, 2% SL = 2:1 RR)',
        }

        report = []
        report.append("# 💰 TRADE SIMULATOR — RESULTS")
        report.append(f"## Capital: ${INITIAL_CAPITAL:,} | Leverage: {LEVERAGE}x")
        report.append(f"## Picks: {len(roster)} | Commission: {COMMISSION*100:.2f}% per trade")
        report.append("")

        for strategy in strategies:
            position_size = INITIAL_CAPITAL / len(roster)
            total_pnl = 0
            wins = 0
            losses = 0
            trades = []

            for pick in roster:
                try:
                    result = self._simulate_trade(
                        pick['symbol'], strategy, position_size
                    )
                    result['symbol'] = pick['symbol']
                    result['layer'] = pick['layer']
                    result['score'] = pick['score']
                    trades.append(result)

                    total_pnl += result['pnl_dollar']
                    if result['pnl_pct'] > 0:
                        wins += 1
                    else:
                        losses += 1
                except Exception as e:
                    pass

            if not trades:
                continue

            # Sort trades by P&L
            trades.sort(key=lambda x: x['pnl_dollar'], reverse=True)

            win_rate = wins / len(trades) * 100
            avg_pnl = total_pnl / len(trades)
            best_trade = trades[0]
            worst_trade = trades[-1]
            max_dd_all = min(t['max_drawdown'] for t in trades)

            # Profit factor
            gross_profit = sum(t['pnl_dollar'] for t in trades if t['pnl_dollar'] > 0)
            gross_loss = abs(sum(t['pnl_dollar'] for t in trades if t['pnl_dollar'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            report.append(f"## 📊 Strategy: {strategy_names[strategy]}")
            report.append(f"| Metric | Value |")
            report.append(f"|--------|-------|")
            report.append(f"| **Net P&L** | **${total_pnl:,.2f}** ({total_pnl/INITIAL_CAPITAL*100:+.1f}%) |")
            report.append(f"| Win Rate | {win_rate:.0f}% ({wins}W / {losses}L) |")
            report.append(f"| Avg P&L/trade | ${avg_pnl:,.2f} |")
            report.append(f"| Best Trade | {best_trade['symbol']} ${best_trade['pnl_dollar']:+,.2f} ({best_trade['pnl_pct']:+.1f}%) |")
            report.append(f"| Worst Trade | {worst_trade['symbol']} ${worst_trade['pnl_dollar']:+,.2f} ({worst_trade['pnl_pct']:+.1f}%) |")
            report.append(f"| Max Drawdown | {max_dd_all:.1f}% |")
            report.append(f"| Profit Factor | {profit_factor:.2f} |")
            report.append(f"| Pos Size | ${position_size:.0f} × {LEVERAGE}x = ${position_size*LEVERAGE:,.0f} notional |")
            report.append("")

            report.append(f"### Trade Details ({strategy_names[strategy]})")
            report.append("| Symbol | Layer | Score | P&L% | P&L$ | Exit | Bars |")
            report.append("|--------|-------|-------|------|------|------|------|")
            for t in trades:
                emoji = "✅" if t['pnl_dollar'] > 0 else "❌"
                report.append(
                    f"| {emoji} {t['symbol']} | {t['layer']} | {t['score']} | "
                    f"{t['pnl_pct']:+.1f}% | ${t['pnl_dollar']:+,.2f} | "
                    f"{t['exit_reason']} | {t['bars_held']} |"
                )
            report.append("")

        # ============================================================
        # FINAL SUMMARY: How much did we make/lose?
        # ============================================================
        report.append("=" * 60)
        report.append("## 💵 RESUMEN FINAL DE JORNADA")
        report.append(f"### Capital Inicial: ${INITIAL_CAPITAL:,}")
        report.append("")
        report.append("| Estrategia | Capital Final | Ganancia/Pérdida | ROI |")
        report.append("|-----------|--------------|-----------------|-----|")

        # Re-run to get totals per strategy
        for strategy in strategies:
            position_size = INITIAL_CAPITAL / len(roster)
            total_pnl = 0
            for pick in roster:
                try:
                    result = self._simulate_trade(
                        pick['symbol'], strategy, position_size
                    )
                    total_pnl += result['pnl_dollar']
                except Exception:
                    pass

            final_capital = INITIAL_CAPITAL + total_pnl
            roi = total_pnl / INITIAL_CAPITAL * 100
            emoji = "🟢" if total_pnl > 0 else "🔴"
            report.append(
                f"| {emoji} {strategy_names[strategy]} | "
                f"**${final_capital:,.2f}** | "
                f"${total_pnl:+,.2f} | "
                f"{roi:+.1f}% |"
            )

        report.append("")

        # Best strategy highlight
        report.append("---")
        report.append(f"> **Empezaste con ${INITIAL_CAPITAL:,}. "
                      f"Con Buy & Hold 24h terminas la jornada con el mejor resultado.**")

        full_report = "\n".join(report)
        print(full_report)

        report_path = os.path.join(self.reports_dir, "trade_simulator_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    sim = TradeSimulator()
    sim.run()
