"""
Raw Macro Scanner — Indiscriminate Market Observer.

Phase 1: REJECTION (Filters out trends that are >4h old)
Phase 2: SCORING (Grades the surviving point-0 candidates based on Kick, Fuel, and Breakout)
"""
import numpy as np
import pandas as pd


class AnomalyScanner:
    """Passes all pairs indiscriminately EXCEPT those with established trends."""

    @staticmethod
    def _is_trend_established(history_df, direction):
        """
        Determines if a trend in the given direction has already been
        established for more than 4 hours (16 candles of 15m).
        """
        if len(history_df) < 17:
            return True # Not enough data

        close = pd.to_numeric(history_df['close'], errors='coerce')
        ret_4h = (close.iloc[-1] - close.iloc[-17]) / close.iloc[-17] * 100
        
        if direction == 'LONG':
            # If it already rallied >4% in 4h, the trend is mature.
            if ret_4h >= 4.0: return True
                
            # If price has been floating 3% above its 4h moving average, it's mature.
            ma_4h = close.iloc[-17:].mean()
            if close.iloc[-1] > ma_4h * 1.03: return True
                
        elif direction == 'SHORT':
            # If it already dumped >4% in 4h, the trend is mature.
            if ret_4h <= -4.0: return True
                
            # If price has been sinking 3% below its 4h moving average, it's mature.
            ma_4h = close.iloc[-17:].mean()
            if close.iloc[-1] < ma_4h * 0.97: return True
                
        return False

    @staticmethod
    def score_universe(pair_data, now_idx, top_n=10, long_ratio=None):
        """
        Scores purely raw pairs using Origin Physics: 
        Initiator (Kick) + Fuel (RVol) + Barrier Break (Micro-shift)
        """
        metrics = {}

        for symbol, df in pair_data.items():
            if now_idx > len(df): continue
            
            # Need some history to calculate baselines (~3 days = 288 candles)
            history = df.iloc[:now_idx].copy()
            if len(history) < 96: continue

            try:
                close = pd.to_numeric(history['close'], errors='coerce')
                open_p = pd.to_numeric(history['open'], errors='coerce')
                high = pd.to_numeric(history['high'], errors='coerce')
                low = pd.to_numeric(history['low'], errors='coerce')
                volume = pd.to_numeric(history['volume'], errors='coerce')

                if close.isna().sum() > 5 or len(close) < 96:
                    continue

                # ============================================
                # PHASE 1: INDISCRIMINATE REJECTION
                # ============================================
                long_rejected = AnomalyScanner._is_trend_established(history, 'LONG')
                short_rejected = AnomalyScanner._is_trend_established(history, 'SHORT')

                if long_rejected and short_rejected:
                    continue # Skip if it's already trending hard in both directions (choppy/crazy) 

                # ============================================
                # PHASE 2: CALCULATING ORIGIN PHYSICS
                # ============================================
                
                # 1. The Kick (Candle Range Anomaly)
                # How big is the current/previous candle relative to the last 24h average body?
                bodies = (close - open_p).abs()
                avg_body_24h = bodies.iloc[-96:-2].mean() # Average body excluding the current spike
                
                # Check the most explosive of the last 2 candles
                current_kick = bodies.iloc[-1]
                prev_kick = bodies.iloc[-2]
                max_kick = max(current_kick, prev_kick)
                
                if avg_body_24h > 0:
                    kick_multiplier = max_kick / avg_body_24h
                else:
                    kick_multiplier = 0

                # 2. The Fuel (Relative Volume - RVol)
                # Volume of the last 1 hour vs average historical 1h volume
                vol_last_1h = volume.iloc[-4:].sum()
                avg_vol_1h = volume.iloc[-96:].sum() / 24 # Crude but fast 24h hourly average
                
                if avg_vol_1h > 0:
                    rvol = vol_last_1h / avg_vol_1h
                else:
                    rvol = 0

                # 3. Micro-Barrier (Is it breaking the 12h box?)
                high_12h = high.iloc[-48:-2].max()
                low_12h = low.iloc[-48:-2].min()
                range_12h = high_12h - low_12h

                if range_12h > 0:
                    range_pct_12h = (close.iloc[-1] - low_12h) / range_12h * 100
                else:
                    range_pct_12h = 50

                metrics[symbol] = {
                    'kick_multiplier': kick_multiplier,
                    'rvol': rvol,
                    'range_pct_12h': range_pct_12h,
                    'long_rejected': long_rejected,
                    'short_rejected': short_rejected
                }

            except Exception:
                continue

        # ============================================
        # PHASE 3: SCORING TRANSLATION
        # ============================================
        symbols = list(metrics.keys())
        long_candidates = []
        short_candidates = []

        for s in symbols:
            m = metrics[s]
            
            # --- EVALUATE LONG POINT 0 ---
            if not m['long_rejected']:
                score = 0
                reasons = []

                # Kick (0-40 pts)
                if m['kick_multiplier'] > 5.0:
                    score += 40
                    reasons.append(f"KICK_MASSIVE({m['kick_multiplier']:.1f}x)")
                elif m['kick_multiplier'] > 3.0:
                    score += 25
                    reasons.append(f"KICK_STRONG({m['kick_multiplier']:.1f}x)")
                elif m['kick_multiplier'] > 1.5:
                    score += 10
                    reasons.append(f"KICK_MILD({m['kick_multiplier']:.1f}x)")

                # Fuel / RVol (0-35 pts)
                if m['rvol'] > 4.0:
                    score += 35
                    reasons.append(f"RVOL_EXPLOSIVE({m['rvol']:.1f}x)")
                elif m['rvol'] > 2.0:
                    score += 20
                    reasons.append(f"RVOL_HIGH({m['rvol']:.1f}x)")

                # Barrier Break (0-25 pts)
                if m['range_pct_12h'] >= 95:
                    score += 25
                    reasons.append("MICRO_BREAKOUT_UP")
                elif m['range_pct_12h'] >= 85:
                    score += 10
                    reasons.append("TESTING_HIGHS")

                # You MUST have some kick and some fuel to be valid (noise reduction)
                if score > 0 and m['kick_multiplier'] > 1.2:
                    long_candidates.append({
                        'symbol': s,
                        'score': score,
                        'reasons': ", ".join(reasons),
                        'direction': 'LONG',
                        'layer': 'RAW'
                    })

            # --- EVALUATE SHORT POINT 0 ---
            if not m['short_rejected']:
                score = 0
                reasons = []

                # Kick (0-40 pts) - Direction agnostic since bodies are absolute lengths
                if m['kick_multiplier'] > 5.0:
                    score += 40
                    reasons.append(f"KICK_MASSIVE({m['kick_multiplier']:.1f}x)")
                elif m['kick_multiplier'] > 3.0:
                    score += 25
                    reasons.append(f"KICK_STRONG({m['kick_multiplier']:.1f}x)")
                elif m['kick_multiplier'] > 1.5:
                    score += 10
                    reasons.append(f"KICK_MILD({m['kick_multiplier']:.1f}x)")

                # Fuel / RVol (0-35 pts)
                if m['rvol'] > 4.0:
                    score += 35
                    reasons.append(f"RVOL_EXPLOSIVE({m['rvol']:.1f}x)")
                elif m['rvol'] > 2.0:
                    score += 20
                    reasons.append(f"RVOL_HIGH({m['rvol']:.1f}x)")

                # Barrier Break DOWN (0-25 pts)
                if m['range_pct_12h'] <= 5:
                    score += 25
                    reasons.append("MICRO_BREAKDOWN_DOWN")
                elif m['range_pct_12h'] <= 15:
                    score += 10
                    reasons.append("TESTING_LOWS")

                if score > 0 and m['kick_multiplier'] > 1.2:
                    short_candidates.append({
                        'symbol': s,
                        'score': score,
                        'reasons': ", ".join(reasons),
                        'direction': 'SHORT',
                        'layer': 'RAW'
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
            if long_ratio >= 1.0: long_slots = top_n
            short_slots = top_n - long_slots

            long_candidates.sort(key=lambda x: x['score'], reverse=True)
            short_candidates.sort(key=lambda x: x['score'], reverse=True)

            roster = []
            for pick in long_candidates[:long_slots]: roster.append(pick)
            selected_symbols = set(p['symbol'] for p in roster)
            
            for pick in short_candidates:
                if len(roster) >= top_n: break
                if pick['symbol'] not in selected_symbols:
                    roster.append(pick)
                    selected_symbols.add(pick['symbol'])

            return roster
