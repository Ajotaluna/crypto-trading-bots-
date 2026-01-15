"""
Pattern Detection & Analysis
Core logic for Trend Following: Breakouts, Reversals, Major Resistance.
"""
import pandas as pd
import numpy as np
# FIX: Robust Import for MathEngine
try:
    from math_engine import MathEngine
except ImportError:
    try:
        from .math_engine import MathEngine
    except ImportError:
        # Fallback for when running as a module from outside
        from trend_following_bot.math_engine import MathEngine

class SentimentAnalyzer:
    """
    TITAN STRATEGY: Analyze Market Sentiment (OI, L/S Ratios)
    """
    @staticmethod
    def analyze_sentiment(oi_data, top_ls_data, global_ls_data):
        """
        Analyze Sentiment Data.
        Returns: { 'signal': 'BULLISH'/'BEARISH'/'NEUTRAL', 'score': 0-100, 'reason': ... }
        """
        if not oi_data or not top_ls_data:
            return {'signal': 'NEUTRAL', 'score': 0, 'reason': 'No Data'}

        # 1. Parse Open Interest Trend (Last 3 hours)
        # We want to see if Money is entering or leaving
        try:
            oi_df = pd.DataFrame(oi_data)
            oi_df['sumOpenInterestValue'] = pd.to_numeric(oi_df['sumOpenInterestValue'])
            current_oi = oi_df.iloc[-1]['sumOpenInterestValue']
            prev_oi = oi_df.iloc[0]['sumOpenInterestValue'] # 30 periods ago ~ 30 hours if 1h? Limit is 30.
            
            # Simple check over the fetched window
            oi_change_pct = (current_oi - prev_oi) / prev_oi * 100
        except:
            oi_change_pct = 0

        # 2. Parse Top Trader L/S Ratio (Whales)
        try:
            top_ls_df = pd.DataFrame(top_ls_data)
            top_ls_df['longShortRatio'] = pd.to_numeric(top_ls_df['longShortRatio'])
            current_top_ratio = top_ls_df.iloc[-1]['longShortRatio']
        except:
            current_top_ratio = 1.0

        # 3. Parse Global L/S Ratio (Crowd) - Optional but good for contrarian
        try:
            global_ls_df = pd.DataFrame(global_ls_data)
            global_ls_df['longShortRatio'] = pd.to_numeric(global_ls_df['longShortRatio'])
            current_global_ratio = global_ls_df.iloc[-1]['longShortRatio']
        except:
            current_global_ratio = 1.0

        # --- TITAN LOGIC ---
        signal = 'NEUTRAL'
        score = 50
        reasons = []

        # BULLISH SETUP
        # Whales are Long (> 1.2) AND OI is increasing (Money backing the move)
        if current_top_ratio > 1.2 and oi_change_pct > 0:
            signal = 'BULLISH'
            score = 80
            reasons.append(f"Whales Long ({current_top_ratio:.2f})")
            reasons.append(f"OI Rising (+{oi_change_pct:.1f}%)")
            
            # SQUEEZE BONUS: If Crowd is Short (< 0.8) while Whales are Long -> HUGE SQUEEZE POTENTIAL
            if current_global_ratio < 0.9:
                score += 15
                reasons.append(f"Crowd Short ({current_global_ratio:.2f}) -> Squeeze Potential")

        # BEARISH SETUP
        # Whales are Short (< 0.8) AND OI is increasing
        elif current_top_ratio < 0.8 and oi_change_pct > 0:
            signal = 'BEARISH'
            score = 80
            reasons.append(f"Whales Short ({current_top_ratio:.2f})")
            reasons.append(f"OI Rising (+{oi_change_pct:.1f}%)")
            
            # SQUEEZE BONUS: If Crowd is Long (> 1.2) while Whales are Short -> LONG SQUEEZE
            if current_global_ratio > 1.1:
                score += 15
                reasons.append(f"Crowd Long ({current_global_ratio:.2f}) -> Long Squeeze Potential")

        return {
            'signal': signal,
            'score': score,
            'reason': ", ".join(reasons),
            'metrics': {
                'oi_change': oi_change_pct,
                'top_ratio': current_top_ratio,
                'global_ratio': current_global_ratio
            }
        }


class TechnicalAnalysis:
    
    @staticmethod
    def calculate_indicators(df):
        """Add technical indicators to DataFrame"""
        if len(df) < 50: return df
        
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']
        
        # Bollinger Bands (for breakouts)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_bb'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_bb'] = df['sma_20'] - (df['std_20'] * 2)
        
        # Volume MA
        df['vol_ma'] = df['volume'].rolling(window=20).mean()

        # ADX (Average Directional Index) - Safe Mode
        try:
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                  np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                             abs(df['low'] - df['close'].shift(1))))
            
            df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                                     np.maximum(df['high'] - df['high'].shift(1), 0), 0)
            df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                                      np.maximum(df['low'].shift(1) - df['low'], 0), 0)
            
            alpha = 1/14
            df['tr_s'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
            df['dm_plus_s'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
            df['dm_minus_s'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()
            
            denom = df['tr_s']
            df['di_plus'] = np.where(denom != 0, (df['dm_plus_s'] / denom) * 100, 0)
            df['di_minus'] = np.where(denom != 0, (df['dm_minus_s'] / denom) * 100, 0)
            
            dx_denom = df['di_plus'] + df['di_minus']
            df['dx'] = np.where(dx_denom != 0, (abs(df['di_plus'] - df['di_minus']) / dx_denom) * 100, 0)
            df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
            df['adx'] = df['adx'].fillna(0)
            
            # VOLUME DELTA (Net Buy/Sell Pressure)
            # taker_buy_vol is passed from MarketData
            if 'taker_buy_vol' in df.columns:
                 # Delta = Buy Vol - Sell Vol (Sell Vol = Total - Buy Vol)
                 # Delta = Buy - (Total - Buy) = 2*Buy - Total
                 df['volume_delta'] = (2 * df['taker_buy_vol']) - df['volume']
                 df['cum_delta'] = df['volume_delta'].cumsum() # For divergence?
            else:
                 df['volume_delta'] = 0
            
        except:
            df['adx'] = 0
        
        return df

class PatternDetector:
    
    def analyze_daily_structure(self, df_daily):
        """
        THE HISTORIAN: Analyze 90-day Daily Context.
        Returns: { 'trend': 'BULLISH'/'BEARISH'/'NEUTRAL', 'reason': ..., 'major_support': ..., 'major_resistance': ... }
        """
        if df_daily is None or len(df_daily) < 30:
            return {'trend': 'NEUTRAL', 'reason': 'Insufficient Data'}
            
        # 1. EMAs on Daily
        df_daily['ema_50'] = df_daily['close'].ewm(span=50, adjust=False).mean()
        df_daily['ema_200'] = df_daily['close'].ewm(span=200, adjust=False).mean()
        
        current = df_daily.iloc[-1]
        
        # 2. Identify Market Structure (High/Low of last 30 days)
        last_30 = df_daily.iloc[-30:]
        highest_high = last_30['high'].max()
        lowest_low = last_30['low'].min()
        
        # 3. Determine Bias
        trend = 'NEUTRAL'
        reason = []
        
        # BULLISH BIAS: Strong Uptrend (Price > EMA50 AND Price > EMA200)
        # This prevents buying "Rebounds" in a Bear Market.
        if current['close'] > current['ema_50'] and current['close'] > current['ema_200']:
            trend = 'BULLISH'
            reason.append("Price > Daily EMA50 & EMA200")
        
        # BEARISH BIAS: Strong Downtrend (Price < EMA50 AND Price < EMA200)
        elif current['close'] < current['ema_50'] and current['close'] < current['ema_200']:
            trend = 'BEARISH'
            reason.append("Price < Daily EMA50 & EMA200")
            
        # 4. Major Levels (Support/Resistance)
        levels = self.find_major_levels(df_daily)
        
        return {
            'trend': trend,
            'reason': ", ".join(reason),
            'levels': levels,
            'high_30d': highest_high,
            'low_30d': lowest_low
        }

    def analyze(self, df_15m, df_daily=None, btc_trend=0.0):
        """Analyze for signals with Historical Context"""
        if len(df_15m) < 50: return None
        
        # 1. GET HISTORICAL CONTEXT
        context = {'trend': 'NEUTRAL'}
        if df_daily is not None:
             context = self.analyze_daily_structure(df_daily)
        
        # KING'S GUARD: Inject BTC Trend
        context['btc_trend'] = btc_trend
        
        df = TechnicalAnalysis.calculate_indicators(df_15m)
        
        # USE CLOSED CANDLE (Index -2) to avoid Repainting/Wick Fakeouts.
        # Index -1 is the live forming candle. We wait for confirmation.
        curr = df.iloc[-2]
        
        signal = {
            'type': None,
            'score': 0,
            'direction': None,
            'reason': [],
            'context': context
        }
        
        # 2. STRICT TREND FILTER ( The Historian )
        # If Daily says BULLISH, we FORBID Shorts.
        allowed_direction = 'BOTH'
        if context['trend'] == 'BULLISH': allowed_direction = 'LONG'
        if context['trend'] == 'BEARISH': allowed_direction = 'SHORT'
        
        # ----------------------------------------------------
        # THE KING'S GUARD (BTC Veto)
        # ----------------------------------------------------
        btc_trend = context.get('btc_trend', 0.0) # % Change 1H
        
        if btc_trend < -0.01: # BTC Down > 1% in hour
            if allowed_direction == 'LONG':
                 # VETO LONG
                 allowed_direction = 'NONE'
                 signal['reason'].append(f"VETO: BTC Crash ({btc_trend*100:.2f}%)")
        
        elif btc_trend > 0.01: # BTC Up > 1% in hour
             if allowed_direction == 'SHORT':
                 # VETO SHORT
                 allowed_direction = 'NONE'
                 signal['reason'].append(f"VETO: BTC Pump ({btc_trend*100:.2f}%)")
                 
        if allowed_direction == 'NONE':
             return None # Safety First
        
        try:
            from config import config # Late import to avoid circular dependency
        except ImportError:
            try:
                from trend_following_bot.config import config
            except ImportError:
                from config import config
        
        # 3. BREAKOUT DETECTION (15m) matching Macro Trend
        # Calculate Volume Velocity (Speculative Bulla)
        vol_velocity = curr['volume'] / (curr['vol_ma'] + 1) # +1 to avoid div by zero
        is_vol_shock = vol_velocity > 4.0 # 400% Volume Spike (Hyper-Strict)
        
        is_vol_surge = curr['volume'] > (curr['vol_ma'] * 1.5)
        # CONFIG CHECK: If Volume Surge is NOT required, we treat it as confirmed even if False.
        surge_confirmed = is_vol_surge or (not config.REQUIRE_VOLUME_SURGE)
        
        is_trend_strong = curr['adx'] > 25
        
        # LONG SETUP
        if allowed_direction in ['LONG', 'BOTH']:
            if curr['close'] > curr['upper_bb'] and surge_confirmed:
                # Require ADX and Alignment
                if is_trend_strong and (curr['close'] > curr['ema_20'] > curr['ema_50']):
                    # Final Check: Not hitting Daily Resistance
                    safe = True
                    for level in context.get('levels', []):
                        if level > curr['close'] and (level - curr['close']) / curr['close'] < 0.02:
                            safe = False # Too close to resistance (<2%)
                            
                    if safe:
                        signal['type'] = 'BREAKOUT'
                        signal['direction'] = 'LONG'
                        signal['score'] = 90
                        if is_vol_shock:
                            signal['score'] += 20 # SUPER BOOST
                            signal['reason'].append(f'speculative BULLA (Vol {vol_velocity:.1f}x)')
                        
                        signal['reason'].append(f'Macro {context["trend"]} + Breakout')
        
        # SHORT SETUP
        if allowed_direction in ['SHORT', 'BOTH']:
            if curr['close'] < curr['lower_bb'] and is_vol_surge:
                if is_trend_strong and (curr['close'] < curr['ema_20'] < curr['ema_50']):
                    # Final Check: Not hitting Daily Support
                    safe = True
                    for level in context.get('levels', []):
                        if level < curr['close'] and (curr['close'] - level) / curr['close'] < 0.02:
                            safe = False 
                            
                    if safe:
                        signal['type'] = 'BREAKOUT'
                        signal['direction'] = 'SHORT'
                        signal['score'] = 90
                        signal['reason'].append(f'Macro {context["trend"]} + Breakdown')
        
        # ----------------------------------------------------
        # 4. MATH ENGINE: STATISTICAL VALIDATION (The Quants)
        # ----------------------------------------------------
        if signal['score'] > 0:
            math_score = 0
            reasons = []
            
            # A. Z-SCORE (Mean Reversion Check)
            z_score = MathEngine.calculate_z_score(df['close'])
            
            if allowed_direction == 'LONG':
                # Don't buy if price is > 2.5 Sigma (Overbought)
                if z_score > 2.5:
                    math_score -= 25 
                    reasons.append(f"Z-Score {z_score:.2f} (Overbought)")
                elif z_score < -1.0:
                    math_score += 10 # Buying Dip
                    reasons.append(f"Z-Score {z_score:.2f} (Dip)")
            elif allowed_direction == 'SHORT':
                if z_score < -2.5:
                    math_score -= 25 
                    reasons.append(f"Z-Score {z_score:.2f} (Oversold)")
                elif z_score > 1.0:
                    math_score += 10 
                    reasons.append(f"Z-Score {z_score:.2f} (Rip)")
                    
            # B. REGRESSION SLOPE
            reg = MathEngine.calculate_linear_regression(df['close'])
            slope = reg['slope']
            if allowed_direction == 'LONG' and slope > 0:
                math_score += 10
                if reg['r_squared'] > 0.7: math_score += 5
                reasons.append(f"Reg Slope +{slope:.4f}")
            elif allowed_direction == 'SHORT' and slope < 0:
                math_score += 10
                if reg['r_squared'] > 0.7: math_score += 5
                reasons.append(f"Reg Slope {slope:.4f}")
                
            # C. HURST (Trend Quality)
            hurst = MathEngine.calculate_hurst(df['close'])
            if hurst > 0.55:
                math_score += 10
                reasons.append(f"Hurst {hurst:.2f}")
            elif hurst < 0.45:
                math_score -= 10
                reasons.append(f"Choppy (H={hurst:.2f})")
                
            # D. ORDER FLOW (Volume Delta)
            vol_delta = df['volume_delta'].iloc[-2] # Closed candle
            if allowed_direction == 'LONG':
                if vol_delta > 0:
                     math_score += 10 # Buying Pressure Confirmed
                     reasons.append("Delta+")
                else:
                     math_score -= 10 # Divergence (Price Up, Net Sell)
                     reasons.append("Delta- (Div)")
            elif allowed_direction == 'SHORT':
                 if vol_delta < 0:
                     math_score += 10 # Selling Pressure Confirmed
                     reasons.append("Delta-")
                 else:
                     math_score -= 10 # Divergence
                     reasons.append("Delta+ (Div)")

            # Apply Math Score
            signal['score'] += math_score
            if math_score != 0:
                signal['reason'].append(f"Math: {math_score:+d} ({', '.join(reasons)})")

        if signal['score'] > 0:
            # --- TITAN V3: SMART CONTEXT (Risk & Levels) ---
            
            # 1. Dynamic Risk Sizing (AGGRESSIVE MODE)
            # Base Risk is 1.0 (Standard)
            # A+ Setup (>90): Risk 3.0% (Increased from 2.0%)
            # B Setup (80-89): Risk 2.0% (Increased from 1.0%)
            # C Setup (70-79): Risk 1.0% (Increased from 0.5%)
            
            risk_pct = 1.0 # Default
            if signal['score'] >= 90: risk_pct = 3.0
            elif signal['score'] >= 80: risk_pct = 2.0
            elif signal['score'] < 75: risk_pct = 1.0 # Even speculative calls get 1% now
            
            signal['risk_pct'] = risk_pct
            signal['reason'].append(f"Smart Risk: {risk_pct}% (Score {signal['score']})")
            
            # 2. Dynamic SL/TP (ATR Based)
            # We calculate this here so the Executor doesn't have to guess.
            # Better: use proper ATR
            df_atr = TechnicalAnalysis.calculate_indicators(df)
            if 'tr' not in df_atr.columns: # Fallback
                current_atr = df_atr.iloc[-1]['close'] * 0.01 
            else:
                # Rolling ATR
                current_atr = df_atr['tr'].rolling(14).mean().iloc[-1]
            
            sl_price = 0.0
            tp_price = 0.0
            current_price = curr['close']
            
            # Volatility Factor: If ATR is HUGE (>2% of price), widen stops
            vol_pct = current_atr / current_price
            atr_mult_sl = 2.0 # Standard Wide Stop
            if vol_pct > 0.02: atr_mult_sl = 3.0 # Volatile -> Wider
            if vol_pct < 0.005: atr_mult_sl = 1.5 # Stable -> Tighter
            
            # 3. Strategy Mode (Titan V4 Adaptive)
            # TREND: High Confidence (>85) AND Good Volatility -> Let it run
            # SCALP: Standard Confidence (75-84) OR Low Volatility -> Hit & Run
            
            strategy_mode = 'SCALP' # Default safety
            if signal['score'] >= 85 and vol_pct > 0.008: # >0.8% ATR is decent movement
                 strategy_mode = 'TREND'
            
            # Special Case: Momentum Override is always TREND
            if signal['score'] >= 90: strategy_mode = 'TREND'
            
            signal['strategy_mode'] = strategy_mode
            signal['reason'].append(f"Mode: {strategy_mode}")

            if allowed_direction == 'LONG':
                sl_price = current_price - (current_atr * atr_mult_sl)
                # TP Target: 
                # If SCALP: Aim for 1.5R (Risk Unit) dynamic target
                # If TREND: Infinity
                tp_price = current_price + (current_atr * atr_mult_sl * 3)
                
                # Dynamic Partial TP (For Scalp Mode)
                # target = Entry + (1.5 * ATR)
                partial_tp_price = current_price + (current_atr * 1.5)
                
            elif allowed_direction == 'SHORT':
                sl_price = current_price + (current_atr * atr_mult_sl)
                tp_price = current_price - (current_atr * atr_mult_sl * 3)
                
                partial_tp_price = current_price - (current_atr * 1.5)
                
            signal['sl_price'] = sl_price
            signal['tp_price'] = tp_price
            signal['partial_tp_price'] = partial_tp_price
            signal['reason'].append(f"Smart Levels (ATR {current_atr:.2f})")

        return signal if signal['score'] > 0 else None

    def find_major_levels(self, df, tolerance=0.01):
        """Find price levels tested 3+ times"""
        levels = []
        highs = df['high'].values
        lows = df['low'].values
        
        # Simple clustering
        all_points = np.concatenate([highs, lows])
        all_points.sort()
        
        current_cluster = [all_points[0]]
        major_levels = []
        
        for i in range(1, len(all_points)):
            if all_points[i] <= current_cluster[0] * (1 + tolerance):
                current_cluster.append(all_points[i])
            else:
                if len(current_cluster) >= 3: # 3+ touches
                    avg_level = sum(current_cluster) / len(current_cluster)
                    major_levels.append(avg_level)
                current_cluster = [all_points[i]]
                
        return major_levels

    @staticmethod
    def calculate_atr(df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def check_exhaustion(self, df, position_side):
        """Check if trend is dying"""
        if len(df) < 50: return False
        
        df = TechnicalAnalysis.calculate_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        if position_side == 'LONG':
            # MACD Bearish Cross or RSI Overbought divergence
            if curr['hist'] < 0 and prev['hist'] > 0: return True
            if curr['rsi'] > 80: return True
            if curr['close'] < curr['ema_20']: return True # Structure break
            
        elif position_side == 'SHORT':
            if curr['hist'] > 0 and prev['hist'] < 0: return True
            if curr['rsi'] < 20: return True
            if curr['close'] > curr['ema_20']: return True
            
        return False

    def calculate_dynamic_levels(self, df, direction):
        """
        Calculate Dynamic TP/SL based on recent Highs/Lows.
        SL = Recent Swing Low (Long) / High (Short)
        TP = Recent Swing High (Long) / Low (Short)
        """
        # We need to import config here to avoid circular imports if config imports patterns
        try:
            from config import config
        except ImportError:
            try:
                from trend_following_bot.config import config
            except ImportError:
                from config import config
        
        if len(df) < config.LOOKBACK_WINDOW_TP:
            return None, None
            
        current_price = df.iloc[-1]['close']
        
        # Get recent windows
        window_sl = df.iloc[-config.LOOKBACK_WINDOW_SL:]
        window_tp = df.iloc[-config.LOOKBACK_WINDOW_TP:]
        
        if direction == 'LONG':
            # SL = Lowest Low of last N candles
            sl_price = window_sl['low'].min()
            
            # TP = Highest High of last M candles
            # If current price is already at High, project it forward or use a multiplier
            # For Trend Following, we want to target the NEXT resistance or a new High
            # Simple approach: Target the recent high. If we are AT the high, use volatility expansion.
            recent_high = window_tp['high'].max()
            
            if recent_high <= current_price * 1.01:
                # We are breaking out, target 1.5x ATR or similar? 
                # User asked for "Highs of recent", so let's stick to that but ensure it's above entry
                # If recent high is too close, we might need a fallback.
                # Let's use the standard logic: Target the recent high.
                tp_price = current_price * 1.05 # Fallback if breaking out
            else:
                tp_price = recent_high
                
            # Safety & Clamping: Ensure SL is below price but NOT too far
            max_sl_dist = config.STOP_LOSS_PCT / 100
            hard_sl_price = current_price * (1 - max_sl_dist)
            
            # If dynamic SL is too far down (risk > max), pull it up to hard SL
            if sl_price < hard_sl_price:
                sl_price = hard_sl_price
                
            # If dynamic SL is above current price (impossible), fix it
            if sl_price >= current_price:
                sl_price = hard_sl_price
                
        else: # SHORT
            # SL = Highest High of last N candles
            sl_price = window_sl['high'].max()
            
            # TP = Lowest Low of last M candles
            recent_low = window_tp['low'].min()
            
            if recent_low >= current_price * 0.99:
                tp_price = current_price * 0.95 # Fallback
            else:
                tp_price = recent_low
                
            # Safety & Clamping for SHORT
            max_sl_dist = config.STOP_LOSS_PCT / 100
            hard_sl_price = current_price * (1 + max_sl_dist)
            
            # If dynamic SL is too high up (risk > max), pull it down
            if sl_price > hard_sl_price:
                sl_price = hard_sl_price
                
            if sl_price <= current_price:
                sl_price = hard_sl_price
                
        return sl_price, tp_price
