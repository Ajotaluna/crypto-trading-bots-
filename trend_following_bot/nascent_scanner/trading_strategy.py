"""
Trading Strategy Logic
Encapsulates indicators, entry confirmation, and position management.
Shared between backtesting and live execution.
"""
import pandas as pd
import numpy as np

# ================================================================
# CONFIGURATION
# ================================================================
INITIAL_CAPITAL = 1000
LEVERAGE = 5
COMMISSION = 0.04 / 100       # 0.04% per side

# Risk: 1% per entry x 3 entries = 3% total per signal
# Risk: 1% per entry x 3 entries = 3% total per signal
RISK_PER_ENTRY = 0.01         # 1% equity per entry
MAX_CAPITAL_PER_TRADE = 0.10  # 10% max position size per entry
MAX_SIGNALS = 3               # Max concurrent signals being tracked
MAX_HOLD_CANDLES = 96         # 24h max hold (reduced from 160)
DAILY_LOSS_CAP = 0.08         # 8% daily loss cap

CANDLES_PER_DAY = 96
HISTORY_NEEDED = 480
WARMUP_DAYS = 5               # 5 days calibration per pair
TOP_N = 10

# SL/Trailing config — DATA-DRIVEN (High Win Rate Config)
INITIAL_SL_ATR = 5.0          # Wide SL: 50% of winning MAE survived
BE_LOCK_ATR = 1.5             # Lock BE after 1.5 ATR
TRAIL_DISTANCE_ATR = 2.0      # Trail 2.0 ATR

# Scaled entry levels
SCALE_LEVEL_2 = 1.5           # Add 2nd entry at -1.5 ATR pullback
SCALE_LEVEL_3 = 3.0           # Add 3rd entry at -3.0 ATR pullback
MAX_SCALE = 3                 # Max 3 entries

# Score filter — low scores perform better
MAX_SCORE = 120               # Skip signals with score >= 120

# Entry windows
ENTRY_WINDOW_CANDLES = 48     # 12h entry window (trends develop slowly)


# ================================================================
# KALMAN FILTER LOGIC
# ================================================================
def apply_kalman_filter(prices, q=0.05, r=5.0):
    """
    Apply a 1D Kalman Filter (Constant Velocity Model).
    q: Process noise covariance (sensitivity to change)
       - Higher q = faster reaction, more noise.
       - Lower q = smoother, more lag.
    r: Measurement noise covariance (trust in raw price)
       - Higher r = trust model more (smoother).
       - Lower r = trust price more (reactive).
    Returns: (smoothed_prices, slopes)
    """
    n = len(prices)
    if n == 0:
        return np.zeros(0), np.zeros(0)

    # State vector [price, velocity]
    x = np.zeros((2, 1)) 
    x[0] = prices[0] # Initial price
    
    # Covariance matrix
    P = np.eye(2) 
    
    # State transition matrix (dt=1)
    F = np.array([[1, 1], 
                  [0, 1]])
    
    # Measurement matrix
    H = np.array([[1, 0]])
    
    # Measurement noise uncertainty
    R = np.array([[r]])
    
    # Process noise uncertainty
    Q = np.array([[q, 0], 
                  [0, q]])
    
    smoothed = np.zeros(n)
    slopes = np.zeros(n)
    
    # Pre-convert to numpy array if it's a Series
    obs = np.array(prices)
    
    for i in range(n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        
        # Update
        z = obs[i]
        if np.isnan(z):
            # Missing data: trust prediction only
            # No update step, just prediction carried forward
            pass
        else:
            y = z - H @ x # Innovation
            S = H @ P @ H.T + R # Innovation covariance
            try:
                K = P @ H.T @ np.linalg.inv(S) # Kalman gain
            except:
                K = np.zeros((2,1))

            x = x + K @ y
            P = (np.eye(2) - K @ H) @ P
        
        smoothed[i] = x[0, 0]
        slopes[i] = x[1, 0]
        
    return smoothed, slopes


# ================================================================
# ENTROPY CALCULATION
# ================================================================
def calculate_rolling_entropy(close_series, window=48):
    """
    Calculate Shannon Entropy of log-returns over a rolling window.
    H(X) = -sum(p * log2(p))
    Normalized to [0,1] by dividing by log2(bins).
    Low Entropy (~0) = Order/Trend.
    High Entropy (~1) = Chaos/Noise.
    """
    # 1. Log Returns
    log_ret = np.log(close_series / close_series.shift(1))
    
    # 2. Rolling Entropy
    # Helper to apply on window
    def _entropy(x):
        x = x[~np.isnan(x)]
        if len(x) < 10: return 1.0 # Default high entropy if excessive nan
        
        # Discretize returns into 20 bins
        counts, _ = np.histogram(x, bins=20, density=False)
        
        # Probabilities
        p = counts / counts.sum()
        p = p[p > 0] # Avoid log(0)
        
        # Shannon Entropy
        ent = -np.sum(p * np.log2(p))
        
        # Normalize (Max entropy for 20 bins is log2(20) ~= 4.32)
        norm_ent = ent / np.log2(20)
        return norm_ent

    # raw=True passes numpy array to function -> faster
    return log_ret.rolling(window).apply(_entropy, raw=True)


# ================================================================
# INDICATOR CALCULATOR
# ================================================================
def calculate_indicators(df):
    """Add all needed indicators."""
    df = df.copy()
    cols = {c: c.lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Note: Sorting here helps ensure indicator correctness
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        # We do NOT reset index here if we want to preserve alignment later, 
        # but calculate_indicators expects a range index usually or handles it?
        # RealisticBot usually does reset_index. 
        # For alignment later, we will be re-indexing anyway.
        df.reset_index(drop=True, inplace=True)

    close = pd.to_numeric(df['close'], errors='coerce')
    high = pd.to_numeric(df['high'], errors='coerce')
    low = pd.to_numeric(df['low'], errors='coerce')
    volume = pd.to_numeric(df['volume'], errors='coerce')

    # EMAs (still calculated for other logic if needed, but primary is KF)
    df['ema_9'] = close.ewm(span=9, adjust=False).mean()
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()

    # == KALMAN FILTER ==
    # Using q=0.02, r=10.0 for distinct trend following on 15m
    kf_price, kf_slope = apply_kalman_filter(close.values, q=0.02, r=10.0)
    df['kf_price'] = kf_price
    df['kf_price'] = kf_price
    df['kf_slope'] = kf_slope

    # == SHANNON ENTROPY (Market Quality) ==
    # Window 48 (~12h) to gauge regime
    df['entropy'] = calculate_rolling_entropy(close, window=48)

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (14)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # ADX (14)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_14 = df['atr'].replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.rolling(14).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Volume MA
    df['vol_ma'] = volume.rolling(20).mean()

    # FillNA before returning to prevent indicator nans at start
    df.fillna(0, inplace=True)
    return df


# ================================================================
# ENTRY CONFIRMATION
# ================================================================
def confirm_entry(df, direction):
    """
    Confirm entry TIMING using KALMAN FILTER.
    """
    if len(df) < 10:
        return False

    curr = df.iloc[-2]
    close_p = curr['close']
    atr = curr['atr']
    kf_price = curr['kf_price']
    kf_price = curr['kf_price']
    kf_slope = curr['kf_slope']
    entropy = curr['entropy']

    if atr <= 0:
        return False

    if direction == 'LONG':
        # 1. Price Action: Close > Open (Green Candle)
        if close_p <= curr['open']:
            return False
            
        # 2. RSI Check: Not overbought/oversold extremes
        if curr['rsi'] > 72 or curr['rsi'] < 25:
            return False
            
        # 3. KALMAN TREND CHECK (Replaces EMA)
        # Price must be above Kalman baseline
        if close_p < kf_price:
            return False
        # Trend velocity must be positive
        if kf_slope < 0:
            return False
            
        # 4. Extension check (don't buy if too far extended from Kalman)
        if kf_slope < 0:
            return False

        # 4. ENTROPY FILTER (Chaos Check)
        # 0.78 allows more activity but still blocks total chaos (1.0).
        if entropy > 0.78:
            return False
            
        # 5. Extension check (don't buy if too far extended from Kalman)
        if kf_price > 0 and (close_p - kf_price) / atr > 3.0:
            return False
            
        return True

    elif direction == 'SHORT':
        # 1. Price Action: Close < Open (Red Candle)
        if close_p >= curr['open']:
            return False
            
        # 2. RSI Check
        if curr['rsi'] < 28 or curr['rsi'] > 75:
            return False
            
        # 3. KALMAN TREND CHECK
        # Price below Kalman
        if close_p > kf_price:
            return False
        # Negative velocity
        if kf_slope > 0:
            return False
            
        # 4. Extension check
        if kf_slope > 0:
            return False

        # 4. ENTROPY FILTER
        if entropy > 0.78:
            return False
            
        # 5. Extension check
        if kf_price > 0 and (kf_price - close_p) / atr > 3.0:
            return False
            
        return True

    return False


# ================================================================
# PID CONTROLLER LOGIC
# ================================================================
class PIDController:
    """
    Standard PID Controller.
    """
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        self._prev_error = 0
        self._integral = 0
        
    def update(self, measurement):
        """
        Calculate PID output value for given reference feedback
        """
        error = self.setpoint - measurement
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term
        self._integral += error
        # Clamp integral to avoid windup? Simple clamp for now
        # limit integral to +/- 100 or something reasonable relative to error scope
        # For price/ATR errors, integral might not be huge.
        I = self.Ki * self._integral
        
        # Derivative term
        D = self.Kd * (error - self._prev_error)
        self._prev_error = error
        
        output = P + I + D
        
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)
            
        return output

# ================================================================
# POSITION MANAGER
# ================================================================
class PositionManager:
    """
    Manages scaled positions: up to 3 entries per signal.
    Each signal tracks its entries and computes aggregate P&L.
    NOW WITH PID CONTROL FOR DYNAMIC TRAILING.
    """

    def __init__(self):
        self.positions = {}  # symbol -> position dict

    def open_position(self, symbol, direction, entry_price, sl,
                      amount, atr_at_entry, candle_idx):
        """Open first entry of a scaled position."""
        
        # Initialize PID: 
        # Goal: Keep Trailing Stop optimal.
        # We process 'error' as (Price - Kalman) / ATR.
        # If Price moves far from Kalman (High Profit), Error is Large Negative (if Long).
        # We want PID output to Reduce Trail Distance.
        # So Setpoint = 0 (Price == Kalman). Measurement = (Price - Kalman)/ATR.
        # If Long and Price > Kalman, Measurement > 0. Error < 0.
        # PID Output should be negative? 
        # Let's say we ADD PID output to Base Trail.
        # Base Trail = 2.0.
        # If High Profit (Error < 0), we want Trail = 1.0. So Output should be -1.0.
        # So we need Positive Kp? 
        # Error = 0 - (Positive). Error is Negative.
        # P = Kp * Neg = Neg. Correct.
        
        # Kp=0.4: If Price is 2 ATR above Kalman, Error=-2. Output=-0.8. Trail = 2.0 - 0.8 = 1.2.
        # Kd=0.1: Reactions to speed.
        
        pid = PIDController(Kp=0.4, Ki=0.0, Kd=0.1, setpoint=0, output_limits=(-0.8, 0.5))
        
        self.positions[symbol] = {
            'symbol': symbol,
            'direction': direction,
            'entries': [
                {'price': entry_price, 'amount': amount, 'idx': candle_idx}
            ],
            'entry_price': entry_price,  # First entry price (reference)
            'avg_price': entry_price,    # Will update as we scale in
            'total_amount': amount,
            'sl': sl,
            'entry_idx': candle_idx,
            'be_locked': False,
            'best_price': entry_price,
            'atr_at_entry': atr_at_entry,
            'scale_level': 1,            # How many entries done (1-3)
            'pid': pid                   # Attached PID Controller
        }

    def add_to_position(self, symbol, entry_price, amount, candle_idx):
        """Add a scaled entry to an existing position."""
        pos = self.positions[symbol]
        pos['entries'].append({
            'price': entry_price, 'amount': amount, 'idx': candle_idx
        })
        # Update average price
        total_cost = sum(e['price'] * e['amount'] for e in pos['entries'])
        pos['total_amount'] = sum(e['amount'] for e in pos['entries'])
        pos['avg_price'] = total_cost / pos['total_amount'] if pos['total_amount'] > 0 else entry_price
        pos['scale_level'] += 1

    def update_positions(self, pair_data, candle_idx):
        """Candle-by-candle: SL check, scaling, BE lock, PID TRAILING."""
        closed = []
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            if symbol not in pair_data:
                continue

            df = pair_data[symbol]
            
            # Use 'iloc' based on global aligned index
            if candle_idx >= len(df):
                # Data ended
                last_close = float(df.iloc[-1]['close'])
                closed.append(self._close_position(symbol, last_close, 'DATA_ENDED', len(df)-1))
                continue

            candle = df.iloc[candle_idx]
            
            # Skip if data is NaN (pair not active yet or gap)
            if pd.isna(candle['close']) or candle['close'] == 0:
                continue

            c_high = float(candle['high'])
            c_low = float(candle['low'])
            c_close = float(candle['close'])
            kf_price = float(candle['kf_price']) if 'kf_price' in candle else c_close # Fallback

            atr_val = float(candle['atr']) if candle['atr'] > 0 else pos['atr_at_entry']
            if atr_val <= 0:
                atr_val = c_close * 0.02

            # Track best price for trailing
            if pos['direction'] == 'LONG':
                if c_high > pos['best_price']:
                    pos['best_price'] = c_high
                pnl_atr = (c_close - pos['avg_price']) / atr_val
                
                # PID Input: Distance from Kalman Trend (in ATR units)
                # If Price >> Kalman, we are "extended" -> Tighten Trail
                dist_kalman_atr = (c_close - kf_price) / atr_val
                
            else:
                if c_low < pos['best_price']:
                    pos['best_price'] = c_low
                pnl_atr = (pos['avg_price'] - c_close) / atr_val
                
                # PID Input: Distance from Kalman Trend
                # If Price << Kalman (Short), we are "extended" (dist is positive for short profit?)
                # We want "Extended in Profit" -> Tighten.
                # Kalman > Price. (Kalman - Price) > 0.
                dist_kalman_atr = (kf_price - c_close) / atr_val

            # 1. CHECK STOP LOSS
            is_sl = False
            if pos['direction'] == 'LONG':
                if c_low <= pos['sl']:
                    is_sl = True
            else:
                if c_high >= pos['sl']:
                    is_sl = True

            if is_sl:
                exit_price = pos['sl']
                reason = 'TRAILING_STOP' if pos['be_locked'] else 'STOP_LOSS'
                closed.append(self._close_position(symbol, exit_price, reason, candle_idx))
                continue

            # 2. BREAKEVEN LOCK
            hold_candles = candle_idx - pos['entry_idx']
            if not pos['be_locked'] and pnl_atr >= BE_LOCK_ATR:
                buffer = pos['avg_price'] * 0.002
                if pos['direction'] == 'LONG':
                    pos['sl'] = pos['avg_price'] + buffer
                else:
                    pos['sl'] = pos['avg_price'] - buffer
                pos['be_locked'] = True

            # 3. PID DYNAMIC TRAILING (after BE lock)
            if pos['be_locked']:
                # Update PID with "Distance from Center"
                # If we are 3 ATR away, Error = 0 - 3 = -3.
                # Output ~= 0.4 * -3 = -1.2.
                # Trail = Base(2.0) + Output(-1.2) = 0.8 ATR. (Super Tight!)
                # If we are at Center (0), Output = 0. Trail = 2.0. (Standard)
                # If we are adverse (-1 away), Error = 0 - (-1) = +1.
                # Output = +0.4. Trail = 2.4. (Looser, give room).
                
                pid_adjust = pos['pid'].update(dist_kalman_atr)
                
                # Base is user config
                current_trail_atr = TRAIL_DISTANCE_ATR + pid_adjust
                
                # Hard Limits for sanity
                current_trail_atr = max(0.5, min(4.0, current_trail_atr))

                if pos['direction'] == 'LONG':
                    trail_sl = pos['best_price'] - (atr_val * current_trail_atr)
                    if trail_sl > pos['sl']:
                        pos['sl'] = trail_sl
                else:
                    trail_sl = pos['best_price'] + (atr_val * current_trail_atr)
                    if trail_sl < pos['sl']:
                        pos['sl'] = trail_sl

            # 4. MAX HOLD TIME
            if hold_candles >= MAX_HOLD_CANDLES:
                closed.append(self._close_position(symbol, c_close, 'MAX_TIME', candle_idx))

        return closed

    def check_scale_opportunity(self, symbol, current_price, atr_val):
        """Check if we should add to an existing position."""
        if symbol not in self.positions:
            return False, 0

        pos = self.positions[symbol]
        if pos['scale_level'] >= MAX_SCALE:
            return False, 0
        if pos['be_locked']:
            return False, 0  # Already in profit, don't add
            
        if atr_val <= 0 or np.isnan(atr_val):
            return False, 0

        ref_price = pos['entry_price']  # First entry price

        if pos['direction'] == 'LONG':
            # Price dropped from first entry — good to add
            adverse_atr = (ref_price - current_price) / atr_val
        else:
            adverse_atr = (current_price - ref_price) / atr_val

        if pos['scale_level'] == 1 and adverse_atr >= SCALE_LEVEL_2:
            return True, 2

        if pos['scale_level'] == 2 and adverse_atr >= SCALE_LEVEL_3:
            return True, 3

        return False, 0

    def _close_position(self, symbol, exit_price, reason, candle_idx):
        pos = self.positions.pop(symbol)

        # P&L on total position using average price
        if pos['direction'] == 'LONG':
            pnl_pct = (exit_price - pos['avg_price']) / pos['avg_price']
        else:
            pnl_pct = (pos['avg_price'] - exit_price) / pos['avg_price']

        notional = pos['total_amount'] * LEVERAGE
        fees = notional * COMMISSION * 2
        pnl_dollar = pnl_pct * notional - fees

        return {
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_price': pos['avg_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_dollar': pnl_dollar,
            'reason': reason,
            'hold_candles': candle_idx - pos['entry_idx'],
            'be_locked': pos['be_locked'],
            'scale_level': pos['scale_level'],
        }
