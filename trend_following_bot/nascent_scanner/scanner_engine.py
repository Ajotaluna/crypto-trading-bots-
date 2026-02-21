"""
Scanner Engine — V15: Integración Whale (L4) + Manipulation (L5).
"""
import pandas as pd
from . import layer_trend
from . import layer_energy
from . import layer_momentum
from . import layer_whale
from . import layer_manipulation


class ScannerEngine:

    def analyze(self, df, funding_df=None):
        if len(df) < 200:
            return {'rejected': True, 'reason': 'Insufficient data'}

        df.columns = [c.lower() for c in df.columns]
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            return {'rejected': True, 'reason': 'Missing columns'}

        close = df['close']
        last = close.iloc[-1]

        high_24h = df['high'].iloc[-96:].max()
        low_24h = df['low'].iloc[-96:].min()
        vol_pct = (high_24h - low_24h) / low_24h * 100 if low_24h > 0 else 0
        quote_vol = df['volume'].iloc[-96:].sum() * last

        if quote_vol < 500_000:
            return {'rejected': True, 'reason': 'Illiquid'}
        if vol_pct < 0.5:
            return {'rejected': True, 'reason': 'Flatline'}

        l1 = layer_trend.score(df)
        l2 = layer_energy.score(df, funding_df=funding_df)
        l3 = layer_momentum.score(df)
        l4 = layer_whale.score(df, funding_df=funding_df)
        l5 = layer_manipulation.score(df, funding_df=funding_df)

        # Multi-layer bonus (V15: applied to TREND, MOMENTUM, WHALE, MANIPULATION)
        # ENERGY excluded (same as V14 design)
        scorable = [l1, l3, l4, l5]
        active_layers = sum(1 for l in scorable if l['score'] > 0)
        if active_layers >= 2:
            bonus = (active_layers - 1) * 25
            for layer in scorable:
                if layer['score'] > 0:
                    layer['score'] += bonus
                    layer['reasons'].append(f"MULTI_x{active_layers}(+{bonus})")

        # Whale + Manipulation convergence mega-bonus
        if l4['score'] > 0 and l5['score'] > 0:
            mega_bonus = 50
            l4['score'] += mega_bonus
            l4['reasons'].append(f"WHALE_MANIP_CONVERGENCE(+{mega_bonus})")

        return {
            'rejected': False,
            'l1': l1,
            'l2': l2,
            'l3': l3,
            'l4': l4,
            'l5': l5,
        }

    @staticmethod
    def draft_top_20(results, total=20):
        """
        V17 Draft: Configurable total picks.
        Round 1: TREND   (35% of slots)
        Round 2: ENERGY  (20% of slots)
        Round 3: WHALE   (10% of slots)  ← NEW
        Round 4: MANIPULATION (10% of slots)  ← NEW
        Round 5: BEST AVAILABLE by SUM (remaining slots)
        """
        df = pd.DataFrame(results)
        if df.empty:
            return []

        trend_slots  = max(1, int(total * 0.35))
        energy_slots = max(1, int(total * 0.20))
        whale_slots  = max(1, int(total * 0.10))
        manip_slots  = max(1, int(total * 0.10))

        roster = []
        selected = set()

        fixed_rounds = [
            ('TREND',        'l1_score', 'l1_reasons', trend_slots),
            ('ENERGY',       'l2_score', 'l2_reasons', energy_slots),
            ('WHALE',        'l4_score', 'l4_reasons', whale_slots),
            ('MANIPULATION', 'l5_score', 'l5_reasons', manip_slots),
        ]

        for layer_name, score_col, reasons_col, count in fixed_rounds:
            if score_col not in df.columns:
                continue
            pool = df[~df['symbol'].isin(selected)].copy()
            pool = pool[pool[score_col] > 0]
            pool = pool.sort_values(by=score_col, ascending=False)

            picked = 0
            for _, row in pool.iterrows():
                if picked >= count:
                    break
                roster.append({
                    'symbol': row['symbol'],
                    'layer': layer_name,
                    'score': row[score_col],
                    'reasons': row[reasons_col],
                    'true_gain': row.get('true_gain', 0),
                })
                selected.add(row['symbol'])
                picked += 1

        # Round 5: BEST AVAILABLE — top remaining by SUM of all layer scores
        remaining = total - len(roster)
        if remaining > 0:
            pool = df[~df['symbol'].isin(selected)].copy()

            best_rows = []
            for _, row in pool.iterrows():
                layers = [
                    ('T', row.get('l1_score', 0), row.get('l1_reasons', [])),
                    ('E', row.get('l2_score', 0), row.get('l2_reasons', [])),
                    ('M', row.get('l3_score', 0), row.get('l3_reasons', [])),
                    ('W', row.get('l4_score', 0), row.get('l4_reasons', [])),
                    ('X', row.get('l5_score', 0), row.get('l5_reasons', [])),
                ]
                score_sum = sum(s for _, s, _ in layers if s > 0)
                if score_sum == 0:
                    continue

                best_layer = max(layers, key=lambda x: x[1])
                active = [tag for tag, s, _ in layers if s > 0]
                label = f"BEST({''.join(active)})"

                best_rows.append({
                    'symbol': row['symbol'],
                    'layer': label,
                    'score': score_sum,
                    'reasons': best_layer[2],
                    'true_gain': row.get('true_gain', 0),
                })

            best_rows.sort(key=lambda x: x['score'], reverse=True)
            for entry in best_rows[:remaining]:
                roster.append(entry)
                selected.add(entry['symbol'])

        return roster
