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
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = {'score': 0, 'direction': None, 'reason': []}
        
        # --- LOGIC: MEAN REVERSION SCALPING ---
        
        # LONG: Oversold + Reversal
        if curr['rsi'] < 30 and curr['stoch_k'] < 20 and curr['stoch_k'] > curr['stoch_d']:
            signal['direction'] = 'LONG'
            signal['score'] = 80
            signal['reason'].append("Oversold StochRSI Cross")
            
            if curr['close'] < curr['lower']:
                signal['score'] += 10
                signal['reason'].append("Below Lower BB")
                
        # SHORT: Overbought + Reversal
        elif curr['rsi'] > 70 and curr['stoch_k'] > 80 and curr['stoch_k'] < curr['stoch_d']:
            signal['direction'] = 'SHORT'
            signal['score'] = 80
            signal['reason'].append("Overbought StochRSI Cross")
            
            if curr['close'] > curr['upper']:
                signal['score'] += 10
                signal['reason'].append("Above Upper BB")
                
        return signal
