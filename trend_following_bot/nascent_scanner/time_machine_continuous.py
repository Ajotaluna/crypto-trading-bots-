"""
Time Machine Continuous v3 — MAX SCORE MEMORY + WHALE ACCURACY.

Runs 4 scans. Uses the MAXIMUM score across all scans as base.
Now includes L4 (WHALE) and L5 (MANIPULATION) layers.
Reports individual accuracy per layer including whale signals.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from nascent_scanner.scanner_engine import ScannerEngine

# Scan offsets from the end of history (in 15m candles)
# Each scan uses data up to: total - FUTURE_WINDOW - offset
FUTURE_WINDOW = 96   # 24h reserved for validation
SCAN_OFFSETS = [144, 96, 48, 0]  # T-36h, T-24h, T-12h, T-0h
WATCHLIST_SIZE = 20


class ContinuousTimeMachine:
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

    def run(self):
        print("🔮 Nascent Scanner — CONTINUOUS Time Machine v2.0 (MAX MEMORY)")
        print("=" * 60)

        files = [f for f in os.listdir(self.data_dir) if f.endswith('_15m.csv')]
        if not files:
            print("❌ No data files found.")
            return

        print(f"📊 {len(files)} pairs × {len(SCAN_OFFSETS)} scans = {len(files) * len(SCAN_OFFSETS)} analyses")

        # ============================================================
        # PHASE 1: Multi-scan — score each pair at 4 time points
        # ============================================================
        # score_history[symbol] = [score_scan1, score_scan2, score_scan3, score_scan4]
        # per layer
        score_history = {}
        all_scan_results = {i: [] for i in range(len(SCAN_OFFSETS))}
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
                    'l1': [], 'l2': [], 'l3': [], 'l4': [], 'l5': [],
                    'l1_reasons_per_scan': [], 'l2_reasons_per_scan': [],
                    'l3_reasons_per_scan': [], 'l4_reasons_per_scan': [],
                    'l5_reasons_per_scan': [],
                }

                for scan_idx, offset in enumerate(SCAN_OFFSETS):
                    # Cut data: remove future + offset
                    end_idx = len(df) - FUTURE_WINDOW - offset
                    if end_idx < 200:
                        for lk in ['l1', 'l2', 'l3', 'l4', 'l5']:
                            score_history[symbol][lk].append(0)
                        continue

                    history = df.iloc[:end_idx].copy()
                    result = self.engine.analyze(history, funding_df=funding_df)

                    if result.get('rejected'):
                        for lk in ['l1', 'l2', 'l3', 'l4', 'l5']:
                            score_history[symbol][lk].append(0)
                            score_history[symbol][f'{lk}_reasons_per_scan'].append('')
                    else:
                        for lk in ['l1', 'l2', 'l3', 'l4', 'l5']:
                            # L4/L5 may not exist in older engine results — degrade gracefully
                            layer_result = result.get(lk, {'score': 0, 'reasons': []})
                            score_history[symbol][lk].append(layer_result['score'])
                            score_history[symbol][f'{lk}_reasons_per_scan'].append(
                                ", ".join(layer_result['reasons']))

                # Calculate true gain from future data
                future = df.iloc[-FUTURE_WINDOW:].copy()
                future.columns = [c.lower() for c in future.columns]
                start_price = future['open'].iloc[0]
                end_price = future['close'].iloc[-1]
                score_history[symbol]['true_gain'] = (
                    (end_price - start_price) / start_price * 100
                )

            except Exception as e:
                errors += 1

        valid_symbols = [s for s in score_history if 'true_gain' in score_history[s]]
        print(f"✅ Analyzed {len(valid_symbols)} pairs across {len(SCAN_OFFSETS)} scans ({errors} errors)")

        if not valid_symbols:
            return

        # ============================================================
        # PHASE 2: MAX SCORE MEMORY + velocity
        # ============================================================
        velocity_data = []
        for symbol in valid_symbols:
            h = score_history[symbol]
            for layer_key, layer_name in [
                ('l1', 'TREND'), ('l2', 'ENERGY'), ('l3', 'MOMENTUM'),
                ('l4', 'WHALE'), ('l5', 'MANIPULATION'),
            ]:
                scores = h[layer_key]
                reasons_per_scan = h[f'{layer_key}_reasons_per_scan']

                # MAX SCORE across all scans (not just final)
                max_score = max(scores)
                max_scan_idx = scores.index(max_score)
                best_reasons = reasons_per_scan[max_scan_idx] if max_score > 0 else ''

                # How many scans had nonzero scores?
                nonzero_count = sum(1 for s in scores if s > 0)

                # Velocity: score growth pattern
                first_nonzero_idx = next((i for i, s in enumerate(scores) if s > 0), -1)
                if max_score > 0 and first_nonzero_idx >= 0:
                    # Measure growth from first appearance to max
                    velocity = max_score - scores[first_nonzero_idx]
                else:
                    velocity = 0

                # Flickering: appeared then disappeared
                is_flickering = (max_score > 0 and scores[-1] == 0)

                # Stale: constant high score across 3+ scans
                is_stale = (nonzero_count >= 3 and
                           max_score - min(s for s in scores if s > 0) < 30)

                # Tag which scan had the max
                scan_labels = ['T-36h', 'T-24h', 'T-12h', 'T-0h']
                max_scan_label = scan_labels[max_scan_idx] if max_score > 0 else ''

                velocity_data.append({
                    'symbol': symbol,
                    'layer': layer_name,
                    'layer_key': layer_key,
                    'max_score': max_score,
                    'final_score': scores[-1],
                    'velocity': velocity,
                    'nonzero_count': nonzero_count,
                    'is_flickering': is_flickering,
                    'is_stale': is_stale,
                    'max_scan_label': max_scan_label,
                    'score_history': scores,
                    'true_gain': h['true_gain'],
                    'reasons': best_reasons,
                })

        df_vel = pd.DataFrame(velocity_data)

        # ============================================================
        # PHASE 3: Rolling Watchlist Draft
        # ============================================================
        # Rule: draft by layer, but BOOST pairs with high velocity
        # and PENALIZE stale pairs

        # Create adjusted scores using MAX SCORE as base
        df_vel['adjusted_score'] = df_vel.apply(
            lambda row: self._adjust_score(row), axis=1
        )

        # Build results for draft
        results_for_draft = []
        for symbol in valid_symbols:
            h = score_history[symbol]
            sym_data = df_vel[df_vel['symbol'] == symbol]

            def _get_adj(layer_name):
                rows = sym_data[sym_data['layer'] == layer_name]['adjusted_score'].values
                return rows[0] if len(rows) else 0

            def _get_row(layer_name):
                rows = sym_data[sym_data['layer'] == layer_name]
                return rows.iloc[0] if len(rows) else None

            def make_reason(row):
                if row is None:
                    return ""
                r = str(row['reasons'])
                if row['is_flickering']:
                    r += f", 👁️MEMORY(@{row['max_scan_label']})"
                if row['is_stale']:
                    r += ", ⚠️STALE"
                hist = row['score_history']
                r += f" [{'>'.join(str(int(s)) for s in hist)}]"
                return r

            results_for_draft.append({
                'symbol':     symbol,
                'l1_score':   int(_get_adj('TREND')),
                'l1_reasons': make_reason(_get_row('TREND')),
                'l2_score':   int(_get_adj('ENERGY')),
                'l2_reasons': make_reason(_get_row('ENERGY')),
                'l3_score':   int(_get_adj('MOMENTUM')),
                'l3_reasons': make_reason(_get_row('MOMENTUM')),
                'l4_score':   int(_get_adj('WHALE')),
                'l4_reasons': make_reason(_get_row('WHALE')),
                'l5_score':   int(_get_adj('MANIPULATION')),
                'l5_reasons': make_reason(_get_row('MANIPULATION')),
                'true_gain':  h['true_gain'],
            })

        # Use standard draft
        roster = ScannerEngine.draft_top_20(results_for_draft)

        # ============================================================
        # PHASE 4: Report
        # ============================================================
        df_all = pd.DataFrame(results_for_draft)
        actual_top20 = set(
            df_all.sort_values('true_gain', ascending=False).head(20)['symbol']
        )

        hits = [r for r in roster if r['symbol'] in actual_top20]
        accuracy = len(hits) / 20 * 100

        report = []
        report.append("# 🔮 CONTINUOUS SCANNER — TIME MACHINE RESULT")
        report.append(f"## Accuracy: {accuracy:.0f}% ({len(hits)}/20 hits)")
        report.append(f"## Scans: {len(SCAN_OFFSETS)} (T-36h → T-0h)")
        report.append("")

        report.append("## 🎯 Hits (Predicted AND in Real Top 20)")
        if hits:
            for h in hits:
                report.append(
                    f"- **{h['symbol']}** [{h['layer']}] Score {h['score']} "
                    f"→ Gain **{h['true_gain']:.1f}%** | {h['reasons']}"
                )
        else:
            report.append("- No hits :(")
        report.append("")

        missed = actual_top20 - set(r['symbol'] for r in roster)
        report.append("## ❌ Missed (Real Top 20 we didn't pick)")
        missed_details = df_all[df_all['symbol'].isin(missed)].sort_values(
            'true_gain', ascending=False
        )
        for _, row in missed_details.iterrows():
            scores = f"T:{row['l1_score']} E:{row['l2_score']} M:{row['l3_score']}"
            report.append(f"- {row['symbol']} → **{row['true_gain']:.1f}%** | Scores: {scores}")
        report.append("")

        report.append("## 📋 Full Predicted Roster")
        df_roster = pd.DataFrame(roster)
        if not df_roster.empty:
            report.append(df_roster[['symbol', 'layer', 'score', 'true_gain', 'reasons']].to_string(index=False))
        report.append("")

        report.append("## 🏆 Actual Top 20 Gainers")
        actual_df = df_all.sort_values('true_gain', ascending=False).head(20)
        report.append(actual_df[['symbol', 'true_gain', 'l1_score', 'l2_score', 'l3_score']].to_string(index=False))

        report.append("")
        report.append("## 📊 Layer Contribution")
        if not df_roster.empty:
            for layer in ['TREND', 'ENERGY', 'MOMENTUM', 'WHALE', 'MANIPULATION']:
                layer_picks = [r for r in roster if layer in r['layer']]
                layer_hits = [r for r in layer_picks if r['symbol'] in actual_top20]
                avg_gain = (sum(r['true_gain'] for r in layer_picks) / len(layer_picks)
                           if layer_picks else 0)
                pct = len(layer_hits) / max(1, len(layer_picks)) * 100
                emoji = '🦈' if layer in ('WHALE', 'MANIPULATION') else ''
                report.append(
                    f"- {emoji}**{layer}**: {len(layer_picks)} picks, "
                    f"{len(layer_hits)} hits ({pct:.0f}%), avg gain: {avg_gain:.1f}%"
                )

        # ============================================================
        # WHALE ACCURACY SECTION
        # Any asset with L4 or L5 > 0 that appeared in real top 20?
        # ============================================================
        report.append("")
        report.append("## 🦈 WHALE Signal Accuracy")
        df_all_full = pd.DataFrame(results_for_draft)
        whale_active = df_all_full[(df_all_full['l4_score'] > 0) | (df_all_full['l5_score'] > 0)]
        whale_hits = whale_active[whale_active['symbol'].isin(actual_top20)]
        whale_acc = len(whale_hits) / max(1, len(whale_active)) * 100
        report.append(
            f"Assets with ANY whale signal: **{len(whale_active)}** " 
            f"→ {len(whale_hits)} were real top-20 gainers = **{whale_acc:.1f}% precision**"
        )
        report.append("")
        report.append("### Whale Signals that HIT (real top-20 gainers):")
        for _, row in whale_hits.sort_values('true_gain', ascending=False).iterrows():
            report.append(
                f"- **{row['symbol']}** → +{row['true_gain']:.1f}% | "
                f"🐋W={row['l4_score']} ⚠️M={row['l5_score']}"
            )
        report.append("")
        report.append("### Whale Signals that MISSED:")
        whale_misses = whale_active[~whale_active['symbol'].isin(actual_top20)]
        for _, row in whale_misses.sort_values('true_gain', ascending=False).head(10).iterrows():
            report.append(
                f"- {row['symbol']} → {row['true_gain']:+.1f}% | "
                f"W={row['l4_score']} M={row['l5_score']}"
            )

        # Velocity analysis for top winners
        report.append("")
        report.append("## 🚀 Score Velocity Analysis (Top 10 Actual Gainers)")
        top_gainers = df_all.sort_values('true_gain', ascending=False).head(10)
        for _, row in top_gainers.iterrows():
            sym = row['symbol']
            if sym in score_history:
                h = score_history[sym]
                e_hist = h['l2']
                t_hist = h['l1']
                m_hist = h['l3']
                report.append(
                    f"- **{sym}** (+{row['true_gain']:.1f}%): "
                    f"T[{'>'.join(str(int(s)) for s in t_hist)}] "
                    f"E[{'>'.join(str(int(s)) for s in e_hist)}] "
                    f"M[{'>'.join(str(int(s)) for s in m_hist)}]"
                )

        full_report = "\n".join(report)
        print(full_report)

        report_path = os.path.join(self.reports_dir, "continuous_v2_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n📄 Report saved to: {report_path}")

    def _adjust_score(self, row):
        """
        V2: Use MAX score as base. Add bonuses for growth patterns.
        Flickering pairs (scored then lost) get their max score preserved.
        """
        max_score = row['max_score']
        if max_score == 0:
            return 0

        score = max_score  # START from max, not final
        velocity = row['velocity']
        nonzero_count = row['nonzero_count']
        is_stale = row['is_stale']
        is_flickering = row['is_flickering']

        # Velocity bonus: growing across scans
        if velocity > 100:
            score += 50
        elif velocity > 50:
            score += 30
        elif velocity > 0:
            score += 15

        # Sustained presence bonus: scored in 3+ scans = confirmed signal
        if nonzero_count >= 3:
            score += 20
        elif nonzero_count == 2:
            score += 10

        # Flickering: appeared then disappeared. Still valid (memory)
        # but slight discount since signal wasn't sustained
        if is_flickering:
            score -= 15  # Small discount, NOT full removal

        # Stale penalty: constant high, no evolution
        if is_stale:
            score -= 40

        return max(0, score)


if __name__ == "__main__":
    tm = ContinuousTimeMachine()
    tm.run()
