"""
Scalping Strategy Engine
Precision indicators for 1m/5m scalping.
"""
import pandas as pd
import numpy as np

class ScalperStrategy:
    
    def analyze(self, df):
        if len(df) < 30: return None
        
        # Indicators
        df['close'] = df['close'].astype(float)
        
        # 1. Bollinger Bands (20, 2)
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['upper'] = sma + (std * 2)
        df['lower'] = sma - (std * 2)
        
        # 2. RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. Stochastic RSI (Fast)
        min_rsi = df['rsi'].rolling(14).min()
        max_rsi = df['rsi'].rolling(14).max()
        df['stoch_k'] = ((df['rsi'] - min_rsi) / (max_rsi - min_rsi)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # 4. ADX (Strength Check) - Safe Mode
        try:
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                  np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                             abs(df['low'] - df['close'].shift(1))))
            alpha = 1/14
            df['tr_s'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
            
            df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                                     np.maximum(df['high'] - df['high'].shift(1), 0), 0)
            df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                                      np.maximum(df['low'].shift(1) - df['low'], 0), 0)
            
            df['dm_plus_s'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
            df['dm_minus_s'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()
            
            # Avoid Division by Zero
            denominator = df['tr_s']
            df['di_plus'] = np.where(denominator != 0, (df['dm_plus_s'] / denominator) * 100, 0)
            df['di_minus'] = np.where(denominator != 0, (df['dm_minus_s'] / denominator) * 100, 0)
            
            dx_denom = df['di_plus'] + df['di_minus']
            df['dx'] = np.where(dx_denom != 0, (abs(df['di_plus'] - df['di_minus']) / dx_denom) * 100, 0)
            df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
            df['adx'] = df['adx'].fillna(0) # Ensure no NaNs
        except Exception as e:
            # Fallback
            df['adx'] = 0

        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = {'score': 0, 'direction': None, 'reason': []}
        
        # --- LOGIC: MEAN REVERSION SCALPING ---
        # Requirement: Volatility present (ADX > 20)
        if curr['adx'] < 20:
            return None

        # LONG: Deep Oversold (RSI < 25) + Stoch Crossing Up
        if curr['rsi'] < 25 and curr['stoch_k'] < 20 and curr['stoch_k'] > curr['stoch_d']:
            signal['direction'] = 'LONG'
            signal['score'] = 85
            signal['reason'].append(f"Oversold Reversal (RSI {curr['rsi']:.1f})")
            
            if curr['close'] < curr['lower']:
                signal['score'] += 10
                signal['reason'].append("Below Lower BB")
                
        # SHORT: Deep Overbought (RSI > 75) + Stoch Crossing Down
        elif curr['rsi'] > 75 and curr['stoch_k'] > 80 and curr['stoch_k'] < curr['stoch_d']:
            signal['direction'] = 'SHORT'
            signal['score'] = 85
            signal['reason'].append(f"Overbought Reversal (RSI {curr['rsi']:.1f})")
            
            if curr['close'] > curr['upper']:
                signal['score'] += 10
                signal['reason'].append("Above Upper BB")
                
        return signal
