"""
Point 0 Scanner (Nascent Trend) - Bidirectional: LONG + SHORT signals.

LONG signals:
- Compression (squeeze in recent 12h) + Ignition (sudden volume and price breakout upwards) = BUY

SHORT signals:
- Compression (squeeze in recent 12h) + Ignition (sudden volume and price breakdown downwards) = SELL

Philosophy: Catch trends exactly at the origin (Point 0) when they break a tight consolidation,
rather than chasing assets that have already moved a lot.
"""
import numpy as np
import pandas as pd


class AnomalyScannerPointZero:
    """Scores all pairs for nascent trends (Point 0 break of compression)."""

    @staticmethod
    def score_universe(pair_data, now_idx, top_n=10, long_ratio=None):
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
                # METRIC 1: Compression Score (The Coil)
                # ============================================
                # We measure the range of the last 48 candles (12h)
                # and compare it to a longer 288-candle (3 days) baseline ATR
                
                # Baseline Volatility (14-candle ATR averaged over ~3 days)
                tr = pd.DataFrame({
                    'hl': high - low,
                    'hc': (high - close.shift(1)).abs(),
                    'lc': (low - close.shift(1)).abs()
                }).max(axis=1)
                baseline_atr = tr.iloc[-288:].mean()

                # Recent Compression Range (Last 12 hours, EXCLUDING the ignition candles)
                high_12h = high.iloc[-48:-3].max()
                low_12h = low.iloc[-48:-3].min()
                range_12h = high_12h - low_12h

                # Ratio: How tight is the 12h range compared to the baseline ATR?
                if baseline_atr > 0:
                    squeeze_factor = range_12h / baseline_atr
                else:
                    squeeze_factor = 999 

                is_squeezed = squeeze_factor < 2.5  

                # ============================================
                # METRIC 2: Ignition Score (The Breakout)
                # ============================================
                # We check the most recent 1-3 candles to see if they are exploding
                # out of the compression range.
                
                # Ignition Price Move
                ret_15m = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
                ret_45m = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100
                
                # Ignition Volume
                vol_history = volume.iloc[-2880:]
                daily_vols = []
                for i in range(0, len(vol_history) - 96, 96):
                    daily_vols.append(vol_history.iloc[i:i+96].sum())
                
                # Volume z-score for the last 1 HOUR (4 candles) instead of 24h
                vol_1h = volume.iloc[-4:].sum()
                if daily_vols and len(daily_vols) >= 3:
                    vol_avg_24h = np.mean(daily_vols)
                    # Approx average 1h volume = vol_avg_24h / 24
                    vol_avg_1h = vol_avg_24h / 24
                    vol_std_1h = np.std([v/24 for v in daily_vols])
                    vol_z_1h = (vol_1h - vol_avg_1h) / vol_std_1h if vol_std_1h > 0 else 0
                else:
                    vol_z_1h = 0

                # ============================================
                # METRIC 3: Position Relative to Compression
                # ============================================
                if range_12h > 0:
                    range_pct_12h = (close.iloc[-1] - low_12h) / range_12h * 100
                else:
                    range_pct_12h = 50

                metrics[symbol] = {
                    'squeeze_factor': squeeze_factor,
                    'is_squeezed': is_squeezed,
                    'ret_15m': ret_15m,
                    'ret_45m': ret_45m,
                    'vol_z_1h': vol_z_1h,
                    'range_pct_12h': range_pct_12h,
                    'range_12h': range_12h,
                }

            except Exception:
                continue

        if not metrics:
            return []

        symbols = list(metrics.keys())
        
        # Determine Market Regime to lightly sway the scores (optional but good for alignment)
        all_ret_45m = [metrics[s]['ret_45m'] for s in symbols]
        median_ret = np.median(all_ret_45m) if all_ret_45m else 0
        
        if median_ret < -0.2:
            regime = 'BEARISH'
            regime_bonus_short = 15
            regime_penalty_long = 10
        elif median_ret > 0.2:
            regime = 'BULLISH'
            regime_bonus_short = 0
            regime_penalty_long = 0
        else:
            regime = 'NEUTRAL'
            regime_bonus_short = 0
            regime_penalty_long = 0
            
        regime_bonus_long = 15 if regime == 'BULLISH' else 0
        regime_penalty_short = 10 if regime == 'BULLISH' else 0

        # ============================================
        # SCORE LONG CANDIDATES (Point 0 Breakout UP)
        # ============================================
        long_candidates = []
        for s in symbols:
            m = metrics[s]
            score = 0
            reasons = []

            # 1. Provide massive bonus for breaking out of a tight squeeze
            if m['squeeze_factor'] < 1.5:
                score += 30
                reasons.append(f"MEGA_SQUEEZE({m['squeeze_factor']:.1f}x)")
            elif m['squeeze_factor'] < 2.5:
                score += 20
                reasons.append(f"SQUEZED({m['squeeze_factor']:.1f}x)")
            elif m['squeeze_factor'] < 4.0:
                score += 10
                reasons.append(f"MILD_SQUEEZE")
            else:
                score -= 20 # Penalize already expanded trends!
                reasons.append("EXTENDED")

            # 2. Sudden Volume Ignition (Now, not 24h ago)
            if m['vol_z_1h'] >= 3.0:
                score += 40
                reasons.append(f"VOL_IGNITION_EXTREME(z={m['vol_z_1h']:.1f})")
            elif m['vol_z_1h'] >= 1.5:
                score += 25
                reasons.append(f"VOL_IGNITION(z={m['vol_z_1h']:.1f})")

            # 3. Sudden Momentum (15m to 45m)
            if m['ret_45m'] > 1.5:
                score += 30
                reasons.append(f"MOMENTUM_45M(+{m['ret_45m']:.1f}%)")
            elif m['ret_45m'] > 0.8:
                score += 15
                reasons.append(f"MOMENTUM_45M(+{m['ret_45m']:.1f}%)")

            # 4. Breakout of the 12h box
            if m['range_pct_12h'] >= 95:
                score += 20
                reasons.append("BREAKING_12H_HIGH")

            # Combo bonus: Squeeze + Volume + Momentum + Breakout
            signal_count = sum([
                m['is_squeezed'],
                m['vol_z_1h'] >= 1.5,
                m['ret_45m'] > 0.8,
                m['range_pct_12h'] >= 90
            ])
            if signal_count >= 4:
                score += 30
                reasons.append("PERFECT_POINT_0_LONG")
            elif signal_count >= 3:
                score += 15
                reasons.append("STRONG_POINT_0_LONG")

            # Regime Adjustment
            if regime_bonus_long > 0:
                score += regime_bonus_long
                reasons.append(f"REGIME_BULL(+{regime_bonus_long})")
            if regime_penalty_long > 0:
                score -= regime_penalty_long
                reasons.append(f"REGIME_BEAR(-{regime_penalty_long})")

            # Filter: Must have AT LEAST some squeeze AND some upward momentum
            if score > 0 and m['ret_45m'] > 0 and m['squeeze_factor'] < 6.0:
                long_candidates.append({
                    'symbol': s,
                    'score': score,
                    'reasons': ", ".join(reasons),
                    'direction': 'LONG',
                    'layer': 'POINT_0',
                    'ret_15m': m['ret_15m'],
                    'ret_45m': m['ret_45m'],
                    'vol_z_1h': m['vol_z_1h'],
                    'squeeze': m['squeeze_factor']
                })

        # ============================================
        # SCORE SHORT CANDIDATES (Point 0 Breakdown DOWN)
        # ============================================
        short_candidates = []
        for s in symbols:
            m = metrics[s]
            score = 0
            reasons = []

            # 1. Squeeze is direction-neutral
            if m['squeeze_factor'] < 1.5:
                score += 30
                reasons.append(f"MEGA_SQUEEZE({m['squeeze_factor']:.1f}x)")
            elif m['squeeze_factor'] < 2.5:
                score += 20
                reasons.append(f"SQUEZED({m['squeeze_factor']:.1f}x)")
            elif m['squeeze_factor'] < 4.0:
                score += 10
                reasons.append(f"MILD_SQUEEZE")
            else:
                score -= 20
                reasons.append("EXTENDED")

            # 2. Sudden Volume Ignition
            if m['vol_z_1h'] >= 3.0:
                score += 40
                reasons.append(f"VOL_IGNITION_EXTREME(z={m['vol_z_1h']:.1f})")
            elif m['vol_z_1h'] >= 1.5:
                score += 25
                reasons.append(f"VOL_IGNITION(z={m['vol_z_1h']:.1f})")

            # 3. Sudden Negative Momentum (15m to 45m)
            if m['ret_45m'] < -1.5:
                score += 30
                reasons.append(f"MOMENTUM_45M({m['ret_45m']:.1f}%)")
            elif m['ret_45m'] < -0.8:
                score += 15
                reasons.append(f"MOMENTUM_45M({m['ret_45m']:.1f}%)")

            # 4. Breakdown of the 12h box
            if m['range_pct_12h'] <= 5:
                score += 20
                reasons.append("BREAKING_12H_LOW")

            # Combo bonus
            signal_count = sum([
                m['is_squeezed'],
                m['vol_z_1h'] >= 1.5,
                m['ret_45m'] < -0.8,
                m['range_pct_12h'] <= 10
            ])
            if signal_count >= 4:
                score += 30
                reasons.append("PERFECT_POINT_0_SHORT")
            elif signal_count >= 3:
                score += 15
                reasons.append("STRONG_POINT_0_SHORT")

            # Regime Adjustment
            if regime_bonus_short > 0:
                score += regime_bonus_short
                reasons.append(f"REGIME_BEAR(+{regime_bonus_short})")
            if regime_penalty_short > 0:
                score -= regime_penalty_short
                reasons.append(f"REGIME_BULL(-{regime_penalty_short})")

            if score > 0 and m['ret_45m'] < 0 and m['squeeze_factor'] < 6.0:
                short_candidates.append({
                    'symbol': s,
                    'score': score,
                    'reasons': ", ".join(reasons),
                    'direction': 'SHORT',
                    'layer': 'POINT_0',
                    'ret_15m': m['ret_15m'],
                    'ret_45m': m['ret_45m'],
                    'vol_z_1h': m['vol_z_1h'],
                    'squeeze': m['squeeze_factor']
                })

        # ============================================
        # COMPUTE ROSTER
        # ============================================
        if long_ratio is None:
            all_candidates = long_candidates + short_candidates
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
        else:
            long_slots = max(0, int(top_n * long_ratio))
            if long_ratio >= 1.0:
                long_slots = top_n
            short_slots = top_n - long_slots

            long_candidates.sort(key=lambda x: x['score'], reverse=True)
            short_candidates.sort(key=lambda x: x['score'], reverse=True)

            roster = []
            for pick in long_candidates[:long_slots]:
                roster.append(pick)

            selected_symbols = set(p['symbol'] for p in roster)
            for pick in short_candidates:
                if len(roster) >= top_n:
                    break
                if pick['symbol'] not in selected_symbols:
                    roster.append(pick)
                    selected_symbols.add(pick['symbol'])

            return roster
