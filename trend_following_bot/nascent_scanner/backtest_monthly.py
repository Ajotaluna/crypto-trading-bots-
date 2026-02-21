"""
Walk-Forward Monthly Backtest.

Downloads 30 days of 15m data for all Binance USDT Futures pairs,
then runs the continuous scanner day-by-day from Day 8 onwards.

For each test day:
  1. Scanner uses ONLY data up to that day (no future leakage)
  2. Actual top 20 gainers determined from next 24h
  3. Accuracy = overlap between prediction and actuals
  4. Trades simulated with Buy & Hold 24h

Reports daily accuracy + cumulative P&L.
"""
import sys
import os
import time
import asyncio
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import requests
from market_data import MarketData
from nascent_scanner.scanner_anomaly import AnomalyScanner

# Constants
CANDLES_PER_DAY = 96        # 24h of 15m candles
HISTORY_NEEDED = 480        # Candles the scanner needs for analysis
SCAN_OFFSETS = [144, 96, 48, 0]  # Not used by anomaly scanner, kept for reference
WARMUP_DAYS = 3             # Anomaly scanner only needs ~5 days (480 candles)
TOP_N = 10                  # Number of picks (was 20, now reduced)
INITIAL_CAPITAL = 1000
LEVERAGE = 5
COMMISSION = 0.04 / 100
DAILY_LOSS_CAP = 0.10       # Stop trading for the day if down 10%


class MonthlyBacktest:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data_monthly")
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        self.scanner = AnomalyScanner()

    # ==============================================================
    # PHASE 1: Download 30 days of 15m data
    # ==============================================================
    async def download_data(self, days=30):
        """Download N days of 15m data for all USDT Futures pairs."""
        print(f"📡 Downloading {days} days of 15m data...")

        market = MarketData()
        try:
            pairs = await market.scan_top_volume(limit=500)
        except AttributeError:
            pairs = await market.get_top_gainers(limit=500)

        print(f"✅ Found {len(pairs)} pairs")

        # Calculate time range
        end_ms = int(datetime.utcnow().timestamp() * 1000)
        start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        done = 0
        errors = 0
        for symbol in pairs:
            try:
                await self._download_pair(market, symbol, start_ms, end_ms)
                done += 1
                if done % 10 == 0:
                    print(f"   {done}/{len(pairs)} pairs downloaded", end='\r')
                    await asyncio.sleep(0.1)  # Rate limiting
            except Exception as e:
                errors += 1

        print(f"\n✅ Downloaded {done} pairs ({errors} errors) to data_monthly/")
        return pairs

    async def _download_pair(self, market, symbol, start_ms, end_ms):
        """Download data for one pair with pagination (max 1500/call)."""
        all_dfs = []
        current_start = start_ms

        while current_start < end_ms:
            df = await market.get_klines(
                symbol, interval='15m', limit=1500,
                start_time=current_start
            )
            if df.empty:
                break

            all_dfs.append(df)

            # Move start to after last candle
            last_ts = df['timestamp'].iloc[-1]
            if isinstance(last_ts, pd.Timestamp):
                current_start = int(last_ts.timestamp() * 1000) + 1
            else:
                current_start = int(last_ts) + 1

            if len(df) < 1500:
                break  # No more data

            await asyncio.sleep(0.05)  # Rate limit

        if all_dfs:
            full_df = pd.concat(all_dfs, ignore_index=True)
            full_df = full_df.drop_duplicates(subset=['timestamp'])
            full_df = full_df.sort_values('timestamp').reset_index(drop=True)
            safe = symbol.replace('/', '')
            full_df.to_csv(
                os.path.join(self.data_dir, f"{safe}_15m.csv"), index=False
            )

    # ==============================================================
    # PHASE 2: Walk-forward day-by-day
    # ==============================================================
    def run_backtest(self):
        """Run walk-forward backtest on downloaded data."""
        print("\n🔬 Walk-Forward Monthly Backtest")
        print("=" * 70)

        # Load all pairs — require minimum candles for a proper backtest
        files = [f for f in os.listdir(self.data_dir) if f.endswith('_15m.csv')]
        if not files:
            print("❌ No data found. Run download first.")
            return

        # Minimum: warmup + scanner needs + at least 2 test days
        min_candles = WARMUP_DAYS * CANDLES_PER_DAY + HISTORY_NEEDED + max(SCAN_OFFSETS) + CANDLES_PER_DAY * 2

        print(f"📊 Loading pairs (minimum {min_candles} candles required)...")
        pair_data = {}
        skipped = 0
        for f in files:
            symbol = f.replace('_15m.csv', '')
            try:
                df = pd.read_csv(os.path.join(self.data_dir, f))
                if len(df) >= min_candles:
                    pair_data[symbol] = df
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        print(f"✅ {len(pair_data)} pairs loaded ({skipped} skipped — insufficient data)")

        if not pair_data:
            print("❌ No pairs with enough data.")
            return

        # Use median length for test day calculation (robust against outliers)
        lengths = sorted(len(df) for df in pair_data.values())
        median_len = lengths[len(lengths) // 2]

        # Calculate test window
        warmup_candles = WARMUP_DAYS * CANDLES_PER_DAY + HISTORY_NEEDED + max(SCAN_OFFSETS)
        test_start_idx = warmup_candles
        test_days = (median_len - test_start_idx) // CANDLES_PER_DAY - 1

        print(f"📅 Median data length: {median_len} candles (~{median_len // CANDLES_PER_DAY} days)")
        print(f"📅 Warmup: {WARMUP_DAYS} days ({warmup_candles} candles)")
        print(f"📅 Test period: {test_days} days")
        print()

        # Day-by-day loop
        daily_results = []
        cumul_pnl = 0
        signal_stats = {}  # Track signal hit rates across all days

        for day_num in range(test_days):
            # The "now" point for this test day
            now_idx = test_start_idx + day_num * CANDLES_PER_DAY
            future_end = now_idx + CANDLES_PER_DAY  # 24h ahead

            if future_end > median_len:
                break

            # Try to get a date from one of the dataframes
            sample_df = list(pair_data.values())[0]
            try:
                day_label = str(sample_df.iloc[now_idx]['timestamp'])[:10]
            except Exception:
                day_label = f"Day {day_num + 1}"

            # Run scanner for this day
            predictions, all_scores = self._run_scanner_at(pair_data, now_idx)

            # Find actual top gainers for next 24h
            actuals = self._find_actual_gainers(pair_data, now_idx, future_end)

            # Compare against top N actual gainers
            pred_symbols = set(p['symbol'] for p in predictions)
            actual_symbols = set(a['symbol'] for a in actuals[:TOP_N])
            hits = pred_symbols & actual_symbols
            accuracy = len(hits) / TOP_N * 100 if predictions else 0

            # Track signals in hits vs misses for analysis
            for pick in predictions:
                reasons = pick.get('reasons', '')
                signals = [r.strip() for r in reasons.split(',') if r.strip()]
                is_hit = pick['symbol'] in actual_symbols
                for sig in signals:
                    # Clean signal name (remove values)
                    sig_name = sig.split('(')[0].strip()
                    if sig_name not in signal_stats:
                        signal_stats[sig_name] = {'hits': 0, 'misses': 0}
                    if is_hit:
                        signal_stats[sig_name]['hits'] += 1
                    else:
                        signal_stats[sig_name]['misses'] += 1

            # Simulate trades
            day_pnl, pos_picks, tot_picks = self._simulate_day(
                pair_data, predictions, now_idx, future_end
            )
            cumul_pnl += day_pnl
            pos_pct = pos_picks / tot_picks * 100 if tot_picks > 0 else 0

            n_long = sum(1 for p in predictions if p.get('direction', 'LONG') == 'LONG')
            n_short = sum(1 for p in predictions if p.get('direction') == 'SHORT')

            daily_results.append({
                'day': day_num + 1,
                'date': day_label,
                'accuracy': accuracy,
                'hits': len(hits),
                'hit_symbols': sorted(hits),
                'pnl': day_pnl,
                'cumul_pnl': cumul_pnl,
                'capital': INITIAL_CAPITAL + cumul_pnl,
                'pos_picks': pos_picks,
                'tot_picks': tot_picks,
                'pos_pct': pos_pct,
                'n_long': n_long,
                'n_short': n_short,
            })

            emoji = "🟢" if day_pnl > 0 else "🔴"
            print(
                f"  {emoji} Day {day_num+1:2d} | {day_label} | "
                f"Dir: {pos_pct:3.0f}% ({pos_picks}/{tot_picks}+) | "
                f"L:{n_long} S:{n_short} | "
                f"P&L: ${day_pnl:+7.2f} | "
                f"Capital: ${INITIAL_CAPITAL + cumul_pnl:,.2f}"
            )

        # Generate report with signal analysis
        self._generate_report(daily_results, signal_stats)

    def _run_scanner_at(self, pair_data, now_idx):
        """Run anomaly-based scanner at a specific point in time."""
        roster = self.scanner.score_universe(pair_data, now_idx, top_n=TOP_N)
        return roster, []

    def _find_actual_gainers(self, pair_data, start_idx, end_idx):
        """Find the actual top 20 gainers in the future window."""
        gainers = []
        for symbol, df in pair_data.items():
            if end_idx > len(df):
                continue
            future = df.iloc[start_idx:end_idx].copy()
            future.columns = [c.lower() for c in future.columns]
            if len(future) < 2:
                continue
            sp = float(future['open'].iloc[0])
            ep = float(future['close'].iloc[-1])
            if sp == 0:
                continue
            gain = (ep - sp) / sp * 100
            gainers.append({'symbol': symbol, 'gain': gain})

        gainers.sort(key=lambda x: x['gain'], reverse=True)
        return gainers

    def _simulate_day(self, pair_data, predictions, now_idx, future_end):
        """Simulate LONG + SHORT trades with daily loss cap."""
        if not predictions:
            return 0, 0, 0  # pnl, positive_picks, total_picks

        position_size = INITIAL_CAPITAL / len(predictions)
        notional = position_size * LEVERAGE
        total_pnl = 0
        positive_picks = 0
        total_picks = 0
        loss_cap_amount = INITIAL_CAPITAL * DAILY_LOSS_CAP

        for pick in predictions:
            symbol = pick['symbol']
            direction = pick.get('direction', 'LONG')

            if symbol not in pair_data:
                continue

            df = pair_data[symbol]
            if future_end > len(df):
                continue

            future = df.iloc[now_idx:future_end].copy()
            future.columns = [c.lower() for c in future.columns]

            entry = float(future['open'].iloc[0])
            exit_p = float(future['close'].iloc[-1])

            if entry == 0:
                continue

            # LONG: profit when price goes UP
            # SHORT: profit when price goes DOWN (invert P&L)
            raw_pnl_pct = (exit_p - entry) / entry * 100
            if direction == 'SHORT':
                pnl_pct = -raw_pnl_pct  # Inverted
            else:
                pnl_pct = raw_pnl_pct

            fees = notional * COMMISSION * 2
            pnl_dollar = (pnl_pct / 100) * notional - fees

            total_picks += 1
            if pnl_pct > 0:
                positive_picks += 1

            total_pnl += pnl_dollar

            # Daily loss cap
            if total_pnl < -loss_cap_amount:
                break

        return total_pnl, positive_picks, total_picks

    # ==============================================================
    # PHASE 3: Report
    # ==============================================================
    def _generate_report(self, daily_results, signal_stats=None):
        """Generate the final walk-forward report."""
        if not daily_results:
            print("❌ No results to report.")
            return

        report = []
        report.append("# 🔬 WALK-FORWARD MONTHLY BACKTEST")
        report.append(f"## Capital Inicial: ${INITIAL_CAPITAL:,} | Leverage: {LEVERAGE}x | Picks: {TOP_N}")
        report.append(f"## Período de test: {len(daily_results)} días")
        report.append("")

        # Summary stats
        accuracies = [d['accuracy'] for d in daily_results]
        pnls = [d['pnl'] for d in daily_results]
        final_capital = daily_results[-1]['capital']
        total_roi = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        win_days = sum(1 for p in pnls if p > 0)
        lose_days = sum(1 for p in pnls if p <= 0)
        avg_accuracy = np.mean(accuracies)
        best_day = max(daily_results, key=lambda x: x['pnl'])
        worst_day = min(daily_results, key=lambda x: x['pnl'])

        # Max drawdown
        peak = INITIAL_CAPITAL
        max_dd = 0
        for d in daily_results:
            if d['capital'] > peak:
                peak = d['capital']
            dd = (d['capital'] - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

        # % positive picks (directional accuracy)
        all_pos = sum(d['pos_picks'] for d in daily_results)
        all_tot = sum(d['tot_picks'] for d in daily_results)
        overall_dir_pct = all_pos / all_tot * 100 if all_tot > 0 else 0
        avg_dir_pct = np.mean([d['pos_pct'] for d in daily_results])

        report.append("## 📊 RESUMEN")
        report.append("| Métrica | Valor |")
        report.append("|---------|-------|")
        report.append(f"| **Capital Final** | **${final_capital:,.2f}** |")
        report.append(f"| **ROI Total** | **{total_roi:+.1f}%** |")
        report.append(f"| P&L Total | ${final_capital - INITIAL_CAPITAL:+,.2f} |")
        report.append(f"| Accuracy (Top {TOP_N}) | {avg_accuracy:.1f}% |")
        report.append(f"| **Dirección Correcta** | **{overall_dir_pct:.0f}% ({all_pos}/{all_tot} picks positivos)** |")
        report.append(f"| Dir. Promedio/día | {avg_dir_pct:.0f}% |")
        report.append(f"| Días Ganadores | {win_days}/{len(daily_results)} ({win_days/len(daily_results)*100:.0f}%) |")
        report.append(f"| Mejor Día | {best_day['date']} ${best_day['pnl']:+,.2f} ({best_day['accuracy']:.0f}% acc) |")
        report.append(f"| Peor Día | {worst_day['date']} ${worst_day['pnl']:+,.2f} ({worst_day['accuracy']:.0f}% acc) |")
        report.append(f"| Max Drawdown | {max_dd:.1f}% |")
        report.append(f"| P&L Promedio/día | ${np.mean(pnls):+,.2f} |")
        report.append(f"| Loss Cap/día | {DAILY_LOSS_CAP*100:.0f}% (${INITIAL_CAPITAL * DAILY_LOSS_CAP:,.0f}) |")
        report.append("")

        # Daily breakdown
        report.append("## 📅 RESULTADOS DÍA POR DÍA")
        report.append("| Día | Fecha | Dir% | L/S | P&L | Capital |")
        report.append("|-----|-------|------|-----|-----|---------|")

        for d in daily_results:
            emoji = "🟢" if d['pnl'] > 0 else "🔴"
            report.append(
                f"| {emoji} {d['day']:2d} | {d['date']} | "
                f"{d['pos_pct']:.0f}% ({d['pos_picks']}/{d['tot_picks']}+) | "
                f"L:{d['n_long']} S:{d['n_short']} | "
                f"${d['pnl']:+,.2f} | ${d['capital']:,.2f} |"
            )

        report.append("")

        # Signal stability analysis
        if signal_stats:
            report.append("## 🔍 ANÁLISIS DE SEÑALES (Estabilidad)")
            report.append("| Señal | En Hits | En Misses | Hit Rate | Veredicto |")
            report.append("|-------|---------|-----------|----------|-----------|")

            # Sort by total appearances
            sorted_sigs = sorted(
                signal_stats.items(),
                key=lambda x: x[1]['hits'] + x[1]['misses'],
                reverse=True
            )

            for sig_name, stats in sorted_sigs:
                total = stats['hits'] + stats['misses']
                if total < 3:  # Skip rare signals
                    continue
                hit_rate = stats['hits'] / total * 100
                if hit_rate >= 40:
                    verdict = "✅ FUERTE"
                elif hit_rate >= 25:
                    verdict = "⚠️ MODERADA"
                else:
                    verdict = "❌ RUIDO"
                report.append(
                    f"| {sig_name} | {stats['hits']} | {stats['misses']} | "
                    f"{hit_rate:.0f}% | {verdict} |"
                )

            report.append("")

        # Final verdict
        report.append("## 💵 VEREDICTO FINAL")
        report.append(f"> **Empezaste con ${INITIAL_CAPITAL:,}.**")
        report.append(f"> **Después de {len(daily_results)} días de trading, terminaste con ${final_capital:,.2f}.**")
        report.append(f"> **ROI: {total_roi:+.1f}%**")

        if total_roi > 0:
            report.append(f"> **El scanner GENERA VALOR. Accuracy promedio: {avg_accuracy:.0f}%.**")
        else:
            report.append(f"> **El scanner NO genera valor consistente en este período.**")

        full_report = "\n".join(report)
        print("\n" + full_report)

        report_path = os.path.join(self.reports_dir, "monthly_backtest_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n📄 Report saved to: {report_path}")


async def main():
    bt = MonthlyBacktest()

    # Step 1: Download data
    if not os.listdir(bt.data_dir):
        print("📡 No monthly data found. Downloading 30 days...")
        await bt.download_data(days=30)
    else:
        existing = len([f for f in os.listdir(bt.data_dir) if f.endswith('_15m.csv')])
        print(f"📂 Found {existing} pairs in data_monthly/. Skipping download.")
        print("   (Delete data_monthly/ folder to force re-download)")

    # Step 2: Run walk-forward backtest
    bt.run_backtest()


if __name__ == "__main__":
    asyncio.run(main())
