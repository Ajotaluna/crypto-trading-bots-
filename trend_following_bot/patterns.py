"""
Pattern Detection & Analysis
Core logic for Trend Following: Breakouts, Reversals, Major Resistance.
"""
import pandas as pd
import numpy as np

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
        except:
            df['adx'] = 0
        
        return df

class PatternDetector:
    
    def analyze(self, df):
        """Analyze for signals"""
        if len(df) < 50: return None
        
        df = TechnicalAnalysis.calculate_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = {
            'type': None,
            'score': 0,
            'direction': None,
            'reason': []
        }
        
        # 1. BREAKOUT DETECTION
        # Logic: Price breaks BB, Volume confirm, AND Trend is Strong (ADX>25)
        is_vol_surge = curr['volume'] > (curr['vol_ma'] * 1.5)
        is_trend_strong = curr['adx'] > 25
        
        if curr['close'] > curr['upper_bb'] and is_vol_surge:
            # STRICT CONFIRMATION:
            # 1. Strong Trend (ADX > 25)
            # 2. Bullish Alignment (Price > EMA20 > EMA50)
            # 3. Not Exhausted (RSI < 75)
            if is_trend_strong and (curr['close'] > curr['ema_20'] > curr['ema_50']) and (curr['rsi'] < 75):
                signal['type'] = 'BREAKOUT'
                signal['direction'] = 'LONG'
                signal['score'] = 85 # High Base Score
                signal['reason'].append(f'Strong Breakout (ADX {curr["adx"]:.1f})')
            else:
                signal['score'] = 0 # Reject weak setups
                
        elif curr['close'] < curr['lower_bb'] and is_vol_surge:
            # STRICT SHORT:
            # 1. Strong Trend (ADX > 25)
            # 2. Bearish Alignment (Price < EMA20 < EMA50)
            # 3. Not Oversold (RSI > 25)
            if is_trend_strong and (curr['close'] < curr['ema_20'] < curr['ema_50']) and (curr['rsi'] > 25):
                signal['type'] = 'BREAKOUT'
                signal['direction'] = 'SHORT'
                signal['score'] = 85
                signal['reason'].append(f'Strong Breakdown (ADX {curr["adx"]:.1f})')
            else:
                signal['score'] = 0
            
            if curr['ema_20'] < curr['ema_50']:
                signal['score'] += 30
                signal['reason'].append('Trend Aligned')

        # 2. TREND REVERSAL (EMA Cross + MACD)
        # Bullish Cross
        if (prev['ema_20'] <= prev['ema_50']) and (curr['ema_20'] > curr['ema_50']):
            if curr['hist'] > 0:
                signal['type'] = 'REVERSAL'
                signal['direction'] = 'LONG'
                signal['score'] += 40
                signal['reason'].append('EMA Golden Cross')
                if curr['rsi'] > 50: signal['score'] += 20
                
        # Bearish Cross
        elif (prev['ema_20'] >= prev['ema_50']) and (curr['ema_20'] < curr['ema_50']):
            if curr['hist'] < 0:
                signal['type'] = 'REVERSAL'
                signal['direction'] = 'SHORT'
                signal['score'] += 40
                signal['reason'].append('EMA Death Cross')
                if curr['rsi'] < 50: signal['score'] += 20

        # 3. MAJOR RESISTANCE / SUPPORT CHECK
        # We scan past 100 candles for levels hit 3+ times
        major_levels = self.find_major_levels(df)
        
        # If we are near a major level, REDUCE score (don't buy into resistance)
        price = curr['close']
        for level in major_levels:
            if abs(price - level) / price < 0.01: # Within 1%
                signal['score'] -= 50
                signal['reason'].append('Near Major Level')
        
        return signal

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
