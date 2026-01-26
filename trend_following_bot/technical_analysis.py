import pandas as pd
import numpy as np
import pandas_ta as ta

class TechnicalAnalysis:
    """
    Centralized Indicator Calculation.
    """
    @staticmethod
    def calculate_indicators(df):
        """ Calculates standard indicators for the bot """
        # Ensure we have data
        if df is None or len(df) < 50: return df

        # 1. Trend: EMA 200, EMA 50, EMA 5
        df['ema_200'] = ta.ema(df['close'], length=200)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['ema_5'] = ta.ema(df['close'], length=5)
        
        # 2. Oscillator: RSI 14
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # 3. Volatility: ATR 14
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # 4. Momentum: MACD (12, 26, 9)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        # macd columns: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        # Rename for easier access
        df['macd'] = macd['MACD_12_26_9']
        df['signal'] = macd['MACDs_12_26_9']
        df['hist'] = macd['MACDh_12_26_9']
        
        # 5. Bands: Bollinger (20, 2)
        bb = ta.bbands(df['close'], length=20, std=2)
        
        # Robust Column Extraction
        if bb is not None and not bb.empty:
            lower_col = next((c for c in bb.columns if c.startswith('BBL')), None)
            upper_col = next((c for c in bb.columns if c.startswith('BBU')), None)
            mid_col   = next((c for c in bb.columns if c.startswith('BBM')), None)
            
            if lower_col and upper_col and mid_col:
                df['lower_bb'] = bb[lower_col]
                df['upper_bb'] = bb[upper_col]
                df['sma_20'] = bb[mid_col]
            else:
                 # Fallback if TA fails
                df['sma_20'] = df['close'].rolling(20).mean()
                df['upper_bb'] = df['sma_20'] + (df['close'].rolling(20).std() * 2)
                df['lower_bb'] = df['sma_20'] - (df['close'].rolling(20).std() * 2)
        
        # 6. Trend Strength: ADX 14
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx['ADX_14']
        
        # 7. Volume Moving Average
        df['vol_ma'] = ta.sma(df['volume'], length=20)
        
        return df
