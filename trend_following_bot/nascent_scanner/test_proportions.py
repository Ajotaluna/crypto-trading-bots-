"""
Proportion Tester — Runs the walk-forward backtest with different
LONG/SHORT ratios and compares results side by side.

Tests: 10L/0S, 8L/2S, 7L/3S, 6L/4S, 5L/5S, 4L/6S, 3L/7S
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
# Import from root (sys.path insertion handles this)
from scanner_anomaly import AnomalyScanner

# Constants (same as backtest)
CANDLES_PER_DAY = 96
HISTORY_NEEDED = 480
WARMUP_DAYS = 3
TOP_N = 10
INITIAL_CAPITAL = 1000
LEVERAGE = 5
COMMISSION = 0.04 / 100
DAILY_LOSS_CAP = 0.10

# Ratios to test
RATIOS = [
    (None, "Dynamic (Meritocratic)"),
    (1.0,  "10L/0S"),
    (0.8,  "8L/2S"),
    (0.7,  "7L/3S"),
    (0.6,  "6L/4S"),
    (0.5,  "5L/5S"),
    (0.4,  "4L/6S"),
    (0.3,  "3L/7S"),
]


def load_data():
    """Load monthly data from CSVs."""
    data_dir = os.path.join(os.path.dirname(__file__), "data_monthly")
    pair_data = {}

    files = [f for f in os.listdir(data_dir) if f.endswith("_15m.csv")]
    for f in files:
        symbol = f.replace("_15m.csv", "")
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            if len(df) >= 1488:
                pair_data[symbol] = df
        except Exception:
            continue

    return pair_data


def simulate_day(pair_data, predictions, now_idx, future_end):
    """Simulate one day of trading."""
    if not predictions:
        return 0, 0, 0

    position_size = INITIAL_CAPITAL / len(predictions)
    notional = position_size * LEVERAGE
    total_pnl = 0
    positive = 0
    total = 0
    loss_cap = INITIAL_CAPITAL * DAILY_LOSS_CAP

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

        raw_pnl = (exit_p - entry) / entry * 100
        pnl_pct = -raw_pnl if direction == 'SHORT' else raw_pnl
        fees = notional * COMMISSION * 2
        pnl_dollar = (pnl_pct / 100) * notional - fees

        total += 1
        if pnl_pct > 0:
            positive += 1
        total_pnl += pnl_dollar

        if total_pnl < -loss_cap:
            break

    return total_pnl, positive, total


def run_ratio(pair_data, long_ratio, label):
    """Run full walk-forward for one ratio."""
    scanner = AnomalyScanner()

    lengths = [len(df) for df in pair_data.values()]
    median_len = int(np.median(lengths))

    warmup_candles = WARMUP_DAYS * CANDLES_PER_DAY + HISTORY_NEEDED
    test_start_idx = warmup_candles
    test_days = (median_len - test_start_idx) // CANDLES_PER_DAY

    daily_pnls = []
    cumul_pnl = 0
    total_pos = 0
    total_all = 0

    for day_num in range(test_days):
        now_idx = test_start_idx + day_num * CANDLES_PER_DAY
        future_end = now_idx + CANDLES_PER_DAY

        if future_end > median_len:
            break

        predictions = scanner.score_universe(
            pair_data, now_idx, top_n=TOP_N, long_ratio=long_ratio
        )
        day_pnl, pos, tot = simulate_day(pair_data, predictions, now_idx, future_end)
        cumul_pnl += day_pnl
        total_pos += pos
        total_all += tot
        daily_pnls.append(day_pnl)

    final_cap = INITIAL_CAPITAL + cumul_pnl
    roi = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_days = sum(1 for p in daily_pnls if p > 0)
    dir_pct = total_pos / total_all * 100 if total_all > 0 else 0

    # Max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    running = INITIAL_CAPITAL
    for p in daily_pnls:
        running += p
        if running > peak:
            peak = running
        dd = (running - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    return {
        'label': label,
        'ratio': long_ratio,
        'roi': roi,
        'final_cap': final_cap,
        'dir_pct': dir_pct,
        'win_days': win_days,
        'total_days': len(daily_pnls),
        'max_dd': max_dd,
        'avg_pnl': np.mean(daily_pnls),
        'total_pnl': cumul_pnl,
    }


def main():
    print("📊 PROPORTION COMPARISON TEST")
    print("=" * 70)
    print("Loading data...")

    pair_data = load_data()
    print(f"✅ Loaded {len(pair_data)} pairs\n")

    results = []
    for ratio, label in RATIOS:
        print(f"  ⏳ Testing {label}...", end="", flush=True)
        r = run_ratio(pair_data, ratio, label)
        results.append(r)
        emoji = "🟢" if r['roi'] > 0 else "🔴"
        print(f" {emoji} ROI: {r['roi']:+.1f}% | Dir: {r['dir_pct']:.0f}% | DD: {r['max_dd']:.1f}%")

    # Print comparison table
    print("\n" + "=" * 70)
    print("## 📊 COMPARACIÓN DE PROPORCIONES LONG/SHORT")
    print(f"## Capital: ${INITIAL_CAPITAL:,} | Leverage: {LEVERAGE}x | Picks: {TOP_N}")
    print("=" * 70)

    print(f"\n| Proporción | ROI | Capital Final | Dir% | Win Days | Max DD | P&L/día |")
    print(f"|-----------|-----|---------------|------|----------|--------|---------|")

    best = max(results, key=lambda x: x['roi'])
    for r in results:
        star = " 🏆" if r == best else ""
        print(
            f"| **{r['label']}** | "
            f"**{r['roi']:+.1f}%** | "
            f"${r['final_cap']:,.0f} | "
            f"{r['dir_pct']:.0f}% | "
            f"{r['win_days']}/{r['total_days']} | "
            f"{r['max_dd']:.1f}% | "
            f"${r['avg_pnl']:+,.0f} |{star}"
        )

    print(f"\n🏆 **Mejor proporción: {best['label']}** con ROI {best['roi']:+.1f}%")

    # Save report
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "proportion_comparison.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Proportion Comparison Results\n\n")
        f.write(f"| Proporción | ROI | Capital Final | Dir% | Win Days | Max DD | P&L/día |\n")
        f.write(f"|-----------|-----|---------------|------|----------|--------|---------||\n")
        for r in results:
            star = " 🏆" if r == best else ""
            f.write(
                f"| {r['label']} | {r['roi']:+.1f}% | "
                f"${r['final_cap']:,.0f} | {r['dir_pct']:.0f}% | "
                f"{r['win_days']}/{r['total_days']} | "
                f"{r['max_dd']:.1f}% | ${r['avg_pnl']:+,.0f} |{star}\n"
            )

    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
