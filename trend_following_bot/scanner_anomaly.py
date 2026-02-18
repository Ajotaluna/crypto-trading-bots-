"""
Anomaly Scanner V2B — Bidirectional: LONG + SHORT signals.

LONG signals (same as V2):
- Volume anomaly + top relative strength + range breakout = BUY

SHORT signals (mirror/inverse):
- Volume anomaly + BOTTOM relative strength + range BREAKDOWN = SELL

Philosophy: Market anomalies work in BOTH directions. If a pair is
unusually weak with heavy volume, it's likely to continue falling.
"""
import numpy as np
import pandas as pd


class AnomalyScanner:
    """Scores all pairs RELATIVE to the market — both LONG and SHORT."""

    @staticmethod
    def score_universe(pair_data, now_idx, top_n=10, long_ratio=None):
        """
        Score all pairs for both LONG and SHORT opportunities.

        Args:
            long_ratio: 0.0-1.0, fraction of slots for longs (rest = shorts)
                        1.0 = all longs, 0.0 = all shorts
        Returns: list of top_n picks with direction ('LONG' or 'SHORT'),
                 scores, and reasons.
        """
        metrics = {}

        for symbol, df in pair_data.items():
            if now_idx > len(df):
                continue

            history = df.iloc[:now_idx].copy()
            if len(history) < 480:
                continue

            try:
                close = pd.to_numeric(history['close'], errors='coerce')
                volume = pd.to_numeric(history['volume'], errors='coerce')
                high = pd.to_numeric(history['high'], errors='coerce')
                low = pd.to_numeric(history['low'], errors='coerce')

                if close.isna().sum() > 10 or len(close) < 480:
                    continue

                # ============================================
                # METRIC 1: Volume Z-Score (direction-neutral)
                # ============================================
                lookback = min(len(volume), 2880)
                vol_history = volume.iloc[-lookback:]

                daily_vols = []
                for i in range(0, len(vol_history) - 96, 96):
                    daily_vols.append(vol_history.iloc[i:i+96].sum())

                vol_24h = volume.iloc[-96:].sum()

                if daily_vols and len(daily_vols) >= 3:
                    vol_avg = np.mean(daily_vols)
                    vol_std = np.std(daily_vols)
                    vol_z = (vol_24h - vol_avg) / vol_std if vol_std > 0 else 0
                else:
                    vol_z = 0

                # ============================================
                # METRIC 2: Returns (multiple timeframes)
                # ============================================
                ret_24h = (close.iloc[-1] - close.iloc[-96]) / close.iloc[-96] * 100 if len(close) >= 96 else 0
                ret_12h = (close.iloc[-1] - close.iloc[-48]) / close.iloc[-48] * 100 if len(close) >= 48 else 0
                ret_6h = (close.iloc[-1] - close.iloc[-24]) / close.iloc[-24] * 100 if len(close) >= 24 else 0

                # ============================================
                # METRIC 3: Range Position (7-day range)
                # >90% = near top (long breakout)
                # <10% = near bottom (short breakdown)
                # ============================================
                range_candles = min(len(high), 672)
                high_7d = high.iloc[-range_candles:].max()
                low_7d = low.iloc[-range_candles:].min()
                range_7d = high_7d - low_7d

                if range_7d > 0:
                    range_pct = (close.iloc[-1] - low_7d) / range_7d * 100
                else:
                    range_pct = 50

                # ============================================
                # METRIC 4: Momentum direction
                # ============================================
                # Long acceleration
                is_accel_up = (
                    ret_6h > 0 and ret_12h > 0 and
                    ret_6h > ret_12h * 0.6
                )
                # Short acceleration (falling faster)
                is_accel_down = (
                    ret_6h < 0 and ret_12h < 0 and
                    ret_6h < ret_12h * 0.6  # 6h drop is >60% of 12h drop
                )

                # Volume trend
                vol_6h = volume.iloc[-24:].sum()
                vol_prev_6h = volume.iloc[-48:-24].sum()
                vol_increasing = vol_6h > vol_prev_6h * 1.2 if vol_prev_6h > 0 else False

                metrics[symbol] = {
                    'vol_z': vol_z,
                    'ret_24h': ret_24h,
                    'ret_12h': ret_12h,
                    'ret_6h': ret_6h,
                    'range_pct': range_pct,
                    'is_accel_up': is_accel_up,
                    'is_accel_down': is_accel_down,
                    'vol_increasing': vol_increasing,
                }

            except Exception:
                continue

        if not metrics:
            return []

        # ============================================
        # STEP 2: Relative ranking across ALL pairs
        # ============================================
        symbols = list(metrics.keys())
        n = len(symbols)

        # Rank by 24h return (0% = best gainer, 100% = worst loser)
        by_ret = sorted(symbols, key=lambda s: metrics[s]['ret_24h'], reverse=True)
        for i, s in enumerate(by_ret):
            metrics[s]['ret_rank_pct'] = (i / n) * 100

        # ============================================
        # STEP 2.5: MARKET REGIME DETECTION
        # Detect if market is mostly bearish/bullish
        # and apply bonus/penalty to align signals.
        # ============================================
        all_returns = [metrics[s]['ret_24h'] for s in symbols]
        median_ret = np.median(all_returns) if all_returns else 0
        
        # Regime classification
        if median_ret < -1.0:
            regime = 'BEARISH'
            regime_bonus_short = 15
            regime_penalty_long = 10
        elif median_ret > 1.0:
            regime = 'BULLISH'
            regime_bonus_short = 0
            regime_penalty_long = 0
        else:
            regime = 'NEUTRAL'
            regime_bonus_short = 0
            regime_penalty_long = 0
        
        # Bullish bonus (mirror logic)
        regime_bonus_long = 15 if regime == 'BULLISH' else 0
        regime_penalty_short = 10 if regime == 'BULLISH' else 0

        # ============================================
        # STEP 3: Score LONG candidates
        # ============================================
        long_candidates = []
        for s in symbols:
            m = metrics[s]
            score = 0
            reasons = []

            # Volume Anomaly (0-40 pts)
            if m['vol_z'] >= 3.0:
                score += 40
                reasons.append(f"VOL_EXTREME(z={m['vol_z']:.1f})")
            elif m['vol_z'] >= 2.0:
                score += 30
                reasons.append(f"VOL_SPIKE(z={m['vol_z']:.1f})")
            elif m['vol_z'] >= 1.5:
                score += 15
                reasons.append(f"VOL_ELEVATED(z={m['vol_z']:.1f})")

            # Relative Strength — TOP performers (0-40 pts)
            if m['ret_rank_pct'] <= 2:
                score += 40
                reasons.append(f"RS_TOP2({m['ret_24h']:+.1f}%)")
            elif m['ret_rank_pct'] <= 5:
                score += 30
                reasons.append(f"RS_TOP5({m['ret_24h']:+.1f}%)")
            elif m['ret_rank_pct'] <= 10:
                score += 20
                reasons.append(f"RS_TOP10({m['ret_24h']:+.1f}%)")
            elif m['ret_rank_pct'] <= 15:
                score += 10
                reasons.append(f"RS_TOP15({m['ret_24h']:+.1f}%)")

            # Range Breakout — near 7-day HIGH (0-30 pts)
            if m['range_pct'] >= 95:
                score += 30
                reasons.append("BREAKOUT_7D")
            elif m['range_pct'] >= 85:
                score += 20
                reasons.append("NEAR_BREAKOUT")
            elif m['range_pct'] >= 75:
                score += 10
                reasons.append("UPPER_RANGE")

            # Acceleration up (0-15 pts)
            if m['is_accel_up']:
                score += 15
                reasons.append("ACCEL_UP")

            # Volume confirmation (0-10 pts)
            if m['vol_increasing'] and score >= 20:
                score += 10
                reasons.append("VOL_RISING")

            # Combo bonus
            signal_count = sum([
                m['vol_z'] >= 1.5,
                m['ret_rank_pct'] <= 15,
                m['range_pct'] >= 75,
            ])
            if signal_count >= 3:
                score += 20
                reasons.append("TRIPLE_LONG")
            elif signal_count >= 2:
                score += 10
                reasons.append("DOUBLE_LONG")

            # Market Regime Adjustment
            if regime_bonus_long > 0:
                score += regime_bonus_long
                reasons.append(f"REGIME_BULL(+{regime_bonus_long})")
            if regime_penalty_long > 0:
                score -= regime_penalty_long
                reasons.append(f"REGIME_BEAR(-{regime_penalty_long})")

            if score > 0:
                long_candidates.append({
                    'symbol': s,
                    'score': score,
                    'reasons': ", ".join(reasons),
                    'direction': 'LONG',
                    'layer': 'LONG',
                    'ret_24h': m['ret_24h'],
                    'vol_z': m['vol_z'],
                    'breakout_pct': m['range_pct'],
                })

        # ============================================
        # STEP 4: Score SHORT candidates (INVERSE)
        # ============================================
        short_candidates = []
        for s in symbols:
            m = metrics[s]
            score = 0
            reasons = []

            # Volume Anomaly — same threshold (0-40 pts)
            if m['vol_z'] >= 3.0:
                score += 40
                reasons.append(f"VOL_EXTREME(z={m['vol_z']:.1f})")
            elif m['vol_z'] >= 2.0:
                score += 30
                reasons.append(f"VOL_SPIKE(z={m['vol_z']:.1f})")
            elif m['vol_z'] >= 1.5:
                score += 15
                reasons.append(f"VOL_ELEVATED(z={m['vol_z']:.1f})")

            # Relative Weakness — BOTTOM performers (0-40 pts)
            if m['ret_rank_pct'] >= 98:  # Bottom 2%
                score += 40
                reasons.append(f"RW_BOT2({m['ret_24h']:+.1f}%)")
            elif m['ret_rank_pct'] >= 95:  # Bottom 5%
                score += 30
                reasons.append(f"RW_BOT5({m['ret_24h']:+.1f}%)")
            elif m['ret_rank_pct'] >= 90:  # Bottom 10%
                score += 20
                reasons.append(f"RW_BOT10({m['ret_24h']:+.1f}%)")
            elif m['ret_rank_pct'] >= 85:  # Bottom 15%
                score += 10
                reasons.append(f"RW_BOT15({m['ret_24h']:+.1f}%)")

            # Range Breakdown — near 7-day LOW (0-30 pts)
            if m['range_pct'] <= 5:
                score += 30
                reasons.append("BREAKDOWN_7D")
            elif m['range_pct'] <= 15:
                score += 20
                reasons.append("NEAR_BREAKDOWN")
            elif m['range_pct'] <= 25:
                score += 10
                reasons.append("LOWER_RANGE")

            # Acceleration DOWN (0-15 pts)
            if m['is_accel_down']:
                score += 15
                reasons.append("ACCEL_DOWN")

            # Volume confirmation (0-10 pts)
            if m['vol_increasing'] and score >= 20:
                score += 10
                reasons.append("VOL_RISING")

            # Combo bonus
            signal_count = sum([
                m['vol_z'] >= 1.5,
                m['ret_rank_pct'] >= 85,
                m['range_pct'] <= 25,
            ])
            if signal_count >= 3:
                score += 20
                reasons.append("TRIPLE_SHORT")
            elif signal_count >= 2:
                score += 10
                reasons.append("DOUBLE_SHORT")

            # Market Regime Adjustment
            if regime_bonus_short > 0:
                score += regime_bonus_short
                reasons.append(f"REGIME_BEAR(+{regime_bonus_short})")
            if regime_penalty_short > 0:
                score -= regime_penalty_short
                reasons.append(f"REGIME_BULL(-{regime_penalty_short})")

            if score > 0:
                short_candidates.append({
                    'symbol': s,
                    'score': score,
                    'reasons': ", ".join(reasons),
                    'direction': 'SHORT',
                    'layer': 'SHORT',
                    'ret_24h': m['ret_24h'],
                    'vol_z': m['vol_z'],
                    'breakout_pct': m['range_pct'],
                })

        # ============================================
        # STEP 5: Build mixed roster (LONG + SHORT)
        # ============================================
        
        # New "Meritocratic" Mode (Dynamic Ratio)
        if long_ratio is None:
            # Combine all
            all_candidates = long_candidates + short_candidates
            # Sort by score desc
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            roster = []
            selected_symbols = set()
            
            for pick in all_candidates:
                if len(roster) >= top_n:
                    break
                if pick['symbol'] not in selected_symbols:
                    roster.append(pick)
                    selected_symbols.add(pick['symbol'])
                    
            return roster

        # Legacy "Fixed Quota" Mode
        else:
            long_slots = max(0, int(top_n * long_ratio))
            if long_ratio >= 1.0:
                long_slots = top_n
            short_slots = top_n - long_slots

            long_candidates.sort(key=lambda x: x['score'], reverse=True)
            short_candidates.sort(key=lambda x: x['score'], reverse=True)

            roster = []
            # Add top longs
            for pick in long_candidates[:long_slots]:
                roster.append(pick)

            # Add top shorts (avoid duplicates)
            selected_symbols = set(p['symbol'] for p in roster)
            for pick in short_candidates:
                if len(roster) >= top_n:
                    break
                if pick['symbol'] not in selected_symbols:
                    roster.append(pick)
                    selected_symbols.add(pick['symbol'])

            return roster
