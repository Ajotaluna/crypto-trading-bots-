"""
Scanner Math — Pure Stateless Indicators
V4: Added Mann-Kendall, Variance Ratio, Autocorrelation, Fisher Transform, KL Divergence.
"""
import pandas as pd
import numpy as np
from itertools import combinations


def z_score(series, window=96):
    """Standard Score: (Value - Mean) / StdDev"""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std().replace(0, 1e-10)
    return (series - mean) / std


def rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, 1e-10))
    return 100 - (100 / (1 + rs))


def hurst_exponent(series, max_lag=20):
    """Simplified Hurst. H<0.5: Reverting | H=0.5: Random | H>0.5: Trending"""
    try:
        values = series.values if hasattr(series, 'values') else series
        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            diff = np.subtract(values[lag:], values[:-lag])
            std = np.std(diff)
            tau.append(std if std > 0 else 1e-10)
        x = np.log(list(lags))
        y = np.log(tau)
        valid = np.isfinite(y)
        if not np.any(valid): return 0.5
        poly = np.polyfit(x[valid], y[valid], 1)
        return poly[0] * 2.0
    except Exception:
        return 0.5


def obv_slope(df, window=24):
    """Normalized On-Balance Volume slope."""
    change = df['close'].diff()
    direction = np.where(change > 0, 1, -1)
    direction[0] = 0
    obv = (direction * df['volume'].values).cumsum()
    if len(obv) < window: return 0.0
    y = obv[-window:]
    x = np.arange(len(y))
    try:
        slope, _ = np.polyfit(x, y, 1)
        vol_mean = df['volume'].iloc[-window:].mean()
        return slope / vol_mean if vol_mean > 0 else 0.0
    except Exception:
        return 0.0


def keltner_squeeze(df, window=20, mult_bb=2.0, mult_kc=1.5):
    """Returns True if Bollinger Bands are inside Keltner Channels."""
    basis = df['close'].rolling(window=window).mean()
    dev = df['close'].rolling(window=window).std()
    upper_bb = basis + mult_bb * dev
    lower_bb = basis - mult_bb * dev
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    upper_kc = basis + mult_kc * atr
    lower_kc = basis - mult_kc * atr
    squeeze_on = (upper_bb < upper_kc) & (lower_bb > lower_kc)
    return bool(squeeze_on.iloc[-1])


def atr_trend(df, window=20):
    """ATR ratio: current vs 24h ago. < 1.0 = contracting."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr_series = true_range.rolling(window=window).mean()
    if len(atr_series) < 96 + window: return 1.0
    current_atr = atr_series.iloc[-1]
    past_atr = atr_series.iloc[-96]
    if past_atr == 0: return 1.0
    return current_atr / past_atr


def higher_lows(df, window=48, segments=4):
    """Check if price makes progressively higher lows."""
    try:
        lows = df['low'].iloc[-window:].values
        if len(lows) < window: return False, 0.0
        chunk_size = window // segments
        mins = []
        for i in range(segments):
            start = i * chunk_size
            end = start + chunk_size
            mins.append(lows[start:end].min())
        is_hl = all(mins[i] >= mins[i - 1] * 0.998 for i in range(1, segments))
        strength = (mins[-1] - mins[0]) / mins[0] * 100 if mins[0] > 0 else 0
        return is_hl, strength
    except Exception:
        return False, 0.0


# ============================================================
# V4: ADVANCED STATISTICAL FORMULAS
# ============================================================

def mann_kendall(series, window=96):
    """
    Mann-Kendall Trend Test (non-parametric).
    Tests if a monotonic trend exists in the data.
    Returns: (S_statistic, p_value)
    - S > 0: upward trend | S < 0: downward trend
    - p < 0.05: statistically significant
    """
    try:
        data = series.iloc[-window:].values
        n = len(data)
        s = 0
        for k in range(n - 1):
            for j in range(k + 1, n):
                diff = data[j] - data[k]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1

        # Variance of S
        var_s = n * (n - 1) * (2 * n + 5) / 18.0

        # Z-statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # Two-sided p-value (using normal CDF approximation)
        from math import erfc
        p_value = erfc(abs(z) / np.sqrt(2))

        return s, p_value
    except Exception:
        return 0, 1.0


def variance_ratio(series, lag=5, window=96):
    """
    Lo-MacKinlay Variance Ratio Test.
    Tests if returns follow a random walk.
    VR = 1: Random Walk | VR > 1: Momentum | VR < 1: Mean Reverting
    Returns: (VR, deviation_from_1)
    """
    try:
        data = series.iloc[-window:].values
        returns = np.diff(np.log(data))
        returns = returns[np.isfinite(returns)]

        if len(returns) < lag * 2:
            return 1.0, 0.0

        # Variance of 1-period returns
        var_1 = np.var(returns, ddof=1)
        if var_1 == 0:
            return 1.0, 0.0

        # Variance of lag-period returns
        lagged_returns = np.array([
            sum(returns[i:i + lag]) for i in range(len(returns) - lag + 1)
        ])
        var_q = np.var(lagged_returns, ddof=1)

        # VR = Var(q-period) / (q * Var(1-period))
        vr = var_q / (lag * var_1)
        deviation = abs(vr - 1.0)

        return vr, deviation
    except Exception:
        return 1.0, 0.0


def autocorrelation(series, lag=1, window=48):
    """
    Lag-N autocorrelation of returns.
    Positive = momentum (self-reinforcing) | Negative = reverting
    Returns: float (-1 to 1)
    """
    try:
        data = series.iloc[-window:].values
        returns = np.diff(data) / data[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) < lag + 2:
            return 0.0

        r1 = returns[lag:]
        r2 = returns[:-lag]
        n = min(len(r1), len(r2))
        r1, r2 = r1[:n], r2[:n]

        mean = np.mean(returns)
        numerator = np.sum((r1 - mean) * (r2 - mean))
        denominator = np.sum((returns - mean) ** 2)

        if denominator == 0:
            return 0.0
        return numerator / denominator
    except Exception:
        return 0.0


def fisher_transform_rsi(series, period=14):
    """
    Fisher Transform applied to RSI.
    Produces sharper extremes for better turning point detection.
    Returns: float (typically -3 to +3)
    """
    try:
        rsi_val = rsi(series, period).iloc[-1]

        # Normalize RSI to (-1, +1) range
        x = (rsi_val / 100.0) * 2.0 - 1.0

        # Clamp to avoid log(0)
        x = max(-0.999, min(0.999, x))

        # Fisher Transform
        fisher = 0.5 * np.log((1.0 + x) / (1.0 - x))
        return fisher
    except Exception:
        return 0.0


def kl_divergence_volume(df, window_recent=48, window_hist=192):
    """
    Kullback-Leibler Divergence of volume distribution.
    Measures how "unusual" recent volume pattern is vs historical.
    High KL = volume behavior is significantly different from normal.
    Returns: float (0 = identical, higher = more unusual)
    """
    try:
        vol = df['volume'].values

        if len(vol) < window_hist:
            return 0.0

        recent = vol[-window_recent:]
        historical = vol[-window_hist:-window_recent]

        # Create normalized histograms (probability distributions)
        bins = 20
        all_data = np.concatenate([recent, historical])
        bin_edges = np.linspace(all_data.min(), all_data.max() + 1e-10, bins + 1)

        p_recent, _ = np.histogram(recent, bins=bin_edges, density=True)
        q_hist, _ = np.histogram(historical, bins=bin_edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p_recent = p_recent + eps
        q_hist = q_hist + eps

        # Normalize to proper probability distributions
        p_recent = p_recent / p_recent.sum()
        q_hist = q_hist / q_hist.sum()

        # KL Divergence: D_KL(P || Q)
        kl = np.sum(p_recent * np.log(p_recent / q_hist))

        return max(0, kl)
    except Exception:
        return 0.0


# ============================================================
# V13: NASCENT BIRTH SIGNALS
# ============================================================

def taker_buy_trend(df, window=96, segments=8):
    """
    Detects GRADUAL increase in taker buy ratio over time.
    This is a NASCENT signal: buyers are quietly accumulating
    before the price has moved significantly.
    
    Divides the window into segments and calculates buy ratio
    for each. Then runs Mann-Kendall to detect upward trend.
    
    Returns: (trend_slope, p_value)
    - Positive slope + low p = buyers gradually taking over
    """
    try:
        if 'taker_buy_base' not in df.columns:
            return 0.0, 1.0
        
        data = df.iloc[-window:]
        chunk_size = window // segments
        ratios = []
        
        for i in range(segments):
            start = i * chunk_size
            end = start + chunk_size
            chunk = data.iloc[start:end]
            buy = chunk['taker_buy_base'].sum()
            total = chunk['volume'].sum()
            if total > 0:
                ratios.append(buy / total)
            else:
                ratios.append(0.5)
        
        ratio_series = pd.Series(ratios)
        s_stat, p_val = mann_kendall(ratio_series, window=len(ratios))
        
        # Also compute slope direction
        slope = ratios[-1] - ratios[0]
        
        return slope, p_val
    except Exception:
        return 0.0, 1.0


def squeeze_tightness_rate(df, window=48, bb_window=20, mult_bb=2.0, mult_kc=1.5):
    """
    Measures HOW FAST a Keltner squeeze is tightening.
    
    Calculates BB_width / KC_width ratio at multiple points.
    If this ratio is DECREASING = squeeze tightening = energy building.
    
    Returns: (tightness_acceleration, current_tightness)
    - tightness_acceleration < 0 = squeeze getting tighter (GOOD)
    - current_tightness < 0.8 = very tight squeeze
    """
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        
        if len(df) < window + bb_window:
            return 0.0, 1.0
        
        # Calculate BB and KC widths at each point
        basis = close.rolling(bb_window).mean()
        dev = close.rolling(bb_window).std()
        bb_width = 2.0 * mult_bb * dev
        
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(bb_window).mean()
        kc_width = 2.0 * mult_kc * atr
        
        # Ratio at each point (lower = tighter squeeze)
        ratio = (bb_width / kc_width.replace(0, 1e-10)).iloc[-window:]
        ratio = ratio.dropna()
        
        if len(ratio) < 10:
            return 0.0, 1.0
        
        current_tightness = ratio.iloc[-1]
        
        # Measure acceleration: compare last quarter vs first quarter
        q1 = ratio.iloc[:len(ratio)//4].mean()
        q4 = ratio.iloc[-len(ratio)//4:].mean()
        acceleration = q4 - q1  # Negative = tightening faster
        
        return acceleration, current_tightness
    except Exception:
        return 0.0, 1.0


def funding_rate_trend(funding_df, history_end_ts=None):
    """
    Analyzes funding rate trend to detect nascent long positioning.
    
    Funding rate going negative → positive = longs building up.
    This is a NASCENT signal: positions being built BEFORE the move.
    
    Returns: (trend_direction, avg_recent_rate, is_transitioning)
    - trend_direction > 0: funding trending positive (longs building)
    - is_transitioning: True if funding crossed from neg→pos recently
    """
    try:
        if funding_df is None or funding_df.empty or len(funding_df) < 4:
            return 0.0, 0.0, False
        
        rates = funding_df['fundingRate'].values
        
        # If history_end_ts provided, filter to only history period
        if history_end_ts is not None and 'fundingTime' in funding_df.columns:
            mask = funding_df['fundingTime'] <= history_end_ts
            rates = funding_df.loc[mask, 'fundingRate'].values
            if len(rates) < 4:
                return 0.0, 0.0, False
        
        # Split into halves: first half vs second half
        mid = len(rates) // 2
        first_half = rates[:mid]
        second_half = rates[mid:]
        
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        avg_recent = np.mean(rates[-3:])  # Last 3 funding periods (24h)
        
        # Trend direction: positive = funding going up
        trend = avg_second - avg_first
        
        # Transition detection: first half negative, second half positive
        is_transitioning = (avg_first < 0 and avg_second > 0)
        
        # Also detect: overall negative becoming less negative
        is_recovering = (avg_first < -0.0001 and avg_second > avg_first * 0.5)
        
        return trend, avg_recent, (is_transitioning or is_recovering)
    except Exception:
        return 0.0, 0.0, False


