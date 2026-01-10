
import numpy as np
import pandas as pd
from scipy.stats import linregress

class MathEngine:
    """
    The Quant Brain 🧮
    Provides statistical probabilities and projections to replace heuristics.
    """

    @staticmethod
    def calculate_linear_regression(series, period=30):
        """
        Calculates the Slope and Quality (R-Squared) of the trend.
        Returns: { 'slope': float, 'r_squared': float, 'projection': float }
        """
        if len(series) < period:
            return {'slope': 0, 'r_squared': 0, 'projection': series.iloc[-1]}

        y = series.iloc[-period:].values
        x = np.arange(len(y))
        
        # Linear Regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Projection (Where should price be in 5 candles?)
        latest_x = len(y) - 1
        current_projected = slope * latest_x + intercept
        future_projected = slope * (latest_x + 6) + intercept # 6 candles ahead
        
        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'current_projected': current_projected,
            'future_projection': future_projected,
            'divergence': series.iloc[-1] - current_projected # Is price above/below line?
        }

    @staticmethod
    def calculate_z_score(series, period=50):
        """
        Calculates Z-Score (Standard Deviations from Mean).
        Z > 2.0 = 95% outlier (Overbought).
        Z > 3.0 = 99.7% outlier (Extreme).
        """
        if len(series) < period: return 0.0
        
        recent = series.iloc[-period:]
        mean = recent.mean()
        std = recent.std()
        
        if std == 0: return 0.0
        
        current = series.iloc[-1]
        z_score = (current - mean) / std
        return z_score

    @staticmethod
    def calculate_volatility_percentile(series, period=100):
        """
        Determines if current volatility is historically high.
        Returns: percentile (0.0 to 1.0)
        """
        if len(series) < period: return 0.5
        
        # Calculate TR or simple returns std dev
        returns = series.pct_change().abs()
        recent_vol = returns.rolling(window=14).mean().iloc[-1]
        
        history = returns.rolling(window=14).mean().iloc[-period:]
        if history.empty: return 0.5
        
        # Rank
        rank = (history < recent_vol).sum()
        percentile = rank / len(history)
        return percentile

    @staticmethod
    def calculate_hurst(series, max_lag=20):
        """
        Hurst Exponent to determine Trend vs Mean Reversion.
        H < 0.5 = Mean Reverting (Choppy)
        H = 0.5 = Random Walk
        H > 0.5 = Trending
        """
        try:
            if len(series) < 100: return 0.5
            
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            
            # Filter out zero variance (logs fail on 0)
            valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
            if len(valid) < 3: return 0.5
            
            lags_v, tau_v = zip(*valid)
            
            # Use polyfit to estimate Hurst
            poly = np.polyfit(np.log(lags_v), np.log(tau_v), 1)
            return poly[0] * 2.0
        except:
            return 0.5
