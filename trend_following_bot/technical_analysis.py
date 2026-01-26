import pandas as pd
import numpy as np
import ta

class TechnicalAnalysis:
    """
    Centralized Indicator Calculation (Using 'ta' library).
    Fixes SettingWithCopyWarning by forcing a copy.
    """
    @staticmethod
    def calculate_indicators(df):
        """ Calculates standard indicators for the bot """
        # Ensure we have data
        if df is None or len(df) < 50: return df

        # FORCE COPY to avoid SettingWithCopyWarning
        df = df.copy()

        # 1. Trend: EMA 200, EMA 50, EMA 5, EMA 20
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_5'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
        
        # 2. Oscillator: RSI 14
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # 3. Volatility: ATR 14
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # 4. Momentum: MACD (12, 26, 9)
        macd_ind = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd_ind.macd()
        df['signal'] = macd_ind.macd_signal()
        df['hist'] = macd_ind.macd_diff()
        
        # 5. Bands: Bollinger (20, 2)
        bb_ind = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['upper_bb'] = bb_ind.bollinger_hband()
        df['lower_bb'] = bb_ind.bollinger_lband()
        df['sma_20'] = bb_ind.bollinger_mavg()
        
        # 6. Trend Strength: ADX 14
        adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_ind.adx()
        
        # 7. Volume Moving Average
        df['vol_ma'] = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
        
        # Fill NaN (Backtest safety)
        # Fix FutureWarning: use ffill/bfill directly
        df.bfill(inplace=True) 
        
        return df
