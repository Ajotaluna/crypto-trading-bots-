"""
Time Machine — V14: 3-layer + funding rate signal.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from nascent_scanner.scanner_engine import ScannerEngine


class TimeMachine:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.engine = ScannerEngine()

    def _load_funding(self, symbol):
        """Load funding rate CSV if it exists."""
        safe = symbol.replace('/', '')
        path = os.path.join(self.data_dir, f"{safe}_funding.csv")
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
        print("🔮 Nascent Scanner — Time Machine v14.0")
        print("=" * 60)

        files = [f for f in os.listdir(self.data_dir) if f.endswith('_15m.csv')]
        if not files:
            print("❌ No data files found. Run scanner_data.py first.")
            return

        # Check for funding data
        funding_files = [f for f in os.listdir(self.data_dir) if f.endswith('_funding.csv')]
        print(f"📊 Analyzing {len(files)} pairs ({len(funding_files)} with funding data)...")

        all_results = []
        errors = 0

        for f in files:
            symbol = f.replace('_15m.csv', '')
            try:
                df = pd.read_csv(os.path.join(self.data_dir, f))
                if len(df) < 480:
                    continue

                history = df.iloc[:-96].copy()
                future = df.iloc[-96:].copy()

                # Load funding data
                funding_df = self._load_funding(symbol)

                result = self.engine.analyze(history, funding_df=funding_df)

                if result.get('rejected'):
                    continue

                future.columns = [c.lower() for c in future.columns]
                start_price = future['open'].iloc[0]
                end_price = future['close'].iloc[-1]
                true_gain = (end_price - start_price) / start_price * 100

                all_results.append({
                    'symbol': symbol,
                    'l1_score': result['l1']['score'],
                    'l1_reasons': ", ".join(result['l1']['reasons']),
                    'l2_score': result['l2']['score'],
                    'l2_reasons': ", ".join(result['l2']['reasons']),
                    'l3_score': result['l3']['score'],
                    'l3_reasons': ", ".join(result['l3']['reasons']),
                    'true_gain': true_gain,
                })

            except Exception as e:
                errors += 1

        if not all_results:
            print("❌ No valid results.")
            return

        print(f"✅ Analyzed {len(all_results)} valid pairs ({errors} errors)")

        roster = ScannerEngine.draft_top_20(all_results)
        df_roster = pd.DataFrame(roster)

        df_all = pd.DataFrame(all_results)
        actual_top20 = set(df_all.sort_values('true_gain', ascending=False).head(20)['symbol'])

        hits = [r for r in roster if r['symbol'] in actual_top20]
        accuracy = len(hits) / 20 * 100

        report = []
        report.append(f"# 🔮 NASCENT SCANNER V14 — TIME MACHINE RESULT")
        report.append(f"## Accuracy: {accuracy:.0f}% ({len(hits)}/20 hits)")
        report.append("")

        report.append("## 🎯 Hits (Predicted AND in Real Top 20)")
        if hits:
            for h in hits:
                report.append(f"- **{h['symbol']}** [{h['layer']}] Score {h['score']} → Gain **{h['true_gain']:.1f}%** | {h['reasons']}")
        else:
            report.append("- No hits :(")
        report.append("")

        missed = actual_top20 - set(r['symbol'] for r in roster)
        report.append("## ❌ Missed (Real Top 20 we didn't pick)")
        missed_details = df_all[df_all['symbol'].isin(missed)].sort_values('true_gain', ascending=False)
        for _, row in missed_details.iterrows():
            scores = f"T:{row['l1_score']} E:{row['l2_score']} M:{row['l3_score']}"
            report.append(f"- {row['symbol']} → **{row['true_gain']:.1f}%** | Scores: {scores}")
        report.append("")

        report.append("## 📋 Full Predicted Roster")
        if not df_roster.empty:
            report.append(df_roster[['symbol', 'layer', 'score', 'true_gain', 'reasons']].to_string(index=False))
        report.append("")

        report.append("## 🏆 Actual Top 20 Gainers")
        actual_df = df_all.sort_values('true_gain', ascending=False).head(20)
        report.append(actual_df[['symbol', 'true_gain', 'l1_score', 'l2_score', 'l3_score']].to_string(index=False))

        report.append("")
        report.append("## 📊 Layer Contribution")
        if not df_roster.empty:
            for layer in ['TREND', 'ENERGY', 'MOMENTUM']:
                layer_picks = [r for r in roster if layer in r['layer']]
                layer_hits = [r for r in layer_picks if r['symbol'] in actual_top20]
                avg_gain = sum(r['true_gain'] for r in layer_picks) / len(layer_picks) if layer_picks else 0
                report.append(f"- **{layer}**: {len(layer_picks)} picks, {len(layer_hits)} hits, avg gain: {avg_gain:.1f}%")

        full_report = "\n".join(report)
        print(full_report)

        report_path = os.path.join(self.reports_dir, "time_machine_v14_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    tm = TimeMachine()
    tm.run()
