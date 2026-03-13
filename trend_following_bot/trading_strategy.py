"""
Trading Strategy Logic
Encapsulates indicators, entry confirmation, and position management.
Shared between backtesting and live execution.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("ConfirmEntry")


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
MAX_SIGNALS = 5               # Max concurrent signals being tracked
MAX_HOLD_CANDLES = 144        # 36h max hold — tendencias fuertes duran más de 24h
DAILY_LOSS_CAP = 0.08         # 8% daily loss cap

CANDLES_PER_DAY = 96
HISTORY_NEEDED = 480
WARMUP_DAYS = 5               # 5 days calibration per pair
TOP_N = 10

# SL/Trailing config — optimizado agresivamente para asegurar ganancias
INITIAL_SL_ATR = 5.0          # SL de ~7% para absorber caídas brutas sin ser barrido
BE_LOCK_ATR = 3.0             # Mover a Break-Even solo tras profit gigante
TRAIL_DISTANCE_ATR = 4.0      # Trail súper holgado (estilo swing)

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
def apply_kalman_filter(prices, q=0.01, r=8.0):
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
    # q=0.01 (suave) para que el slope sea una referencia de tendencia limpia
    kf_price, kf_slope = apply_kalman_filter(close.values, q=0.01, r=8.0)
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
# ENTRY CONFIRMATION - DYNAMIC CONTEXTUAL FILTERS
# ================================================================
def confirm_entry(df, direction):
    """
    Confirma el TIMING de entrada usando FILTROS DINAMICOS POR PAR.
    """
    LOOKBACK = 100

    sym = df['symbol'].iloc[-1] if 'symbol' in df.columns else '???'
    def _reject(reason):
        logger.info(f"   ❌ confirm_entry {sym} ({direction}): {reason}")
        return False

    if len(df) < LOOKBACK:
        return _reject(f"datos insuficientes ({len(df)}<{LOOKBACK} velas)")

    curr = df.iloc[-2]
    hist = df.iloc[-LOOKBACK:-1]

    close_p   = float(curr['close'])
    open_p    = float(curr['open'])
    high_p    = float(curr['high'])
    low_p     = float(curr['low'])
    atr       = float(curr['atr'])
    kf_price  = float(curr['kf_price'])
    kf_slope  = float(curr['kf_slope'])
    entropy   = float(curr['entropy'])
    rsi       = float(curr['rsi'])
    adx       = float(curr['adx'])
    volume    = float(curr['volume'])
    macd      = float(curr['macd'])
    macd_sig  = float(curr['macd_signal'])
    macd_hist = macd - macd_sig

    if atr <= 0 or close_p <= 0:
        return _reject(f"ATR={atr:.6f} o close={close_p:.6f} inválido")

    # -- 1. ADX PERCENTIL PROPIO
    adx_hist = hist['adx'].replace(0, np.nan).dropna()
    if len(adx_hist) < 20:
        return _reject(f"ADX: historial insuficiente ({len(adx_hist)}<20)")
    adx_p40 = float(adx_hist.quantile(0.40))
    if adx <= adx_p40:
        return _reject(f"ADX={adx:.1f} ≤ p40={adx_p40:.1f}")


    # -- 2. VOLUMEN Z-SCORE
    vol_hist = hist['volume'].replace(0, np.nan).dropna()
    if len(vol_hist) >= 20:
        vol_mean = float(vol_hist.mean())
        vol_std  = float(vol_hist.std())
        if vol_std > 0:
            vol_z = (volume - vol_mean) / vol_std
            if vol_z < 0.6:
                return _reject(f"Vol z-score={vol_z:.2f} < 0.6")

    # -- 3. FUERZA RELATIVA DE LA VELA
    candle_range = high_p - low_p
    if candle_range > 0:
        body_pct      = abs(close_p - open_p) / candle_range
        hist_ranges   = (hist['high'] - hist['low']).replace(0, np.nan)
        hist_bodies   = (hist['close'] - hist['open']).abs()
        hist_body_pct = (hist_bodies / hist_ranges).dropna()
        if len(hist_body_pct) >= 20:
            body_p40 = float(hist_body_pct.quantile(0.40))
            if body_pct < body_p40:
                return _reject(f"Cuerpo vela={body_pct:.2f} < p40={body_p40:.2f}")

        close_pos = (close_p - low_p) / candle_range
        if direction == 'LONG'  and close_pos < 0.55:
            return _reject(f"LONG: cierre bajo en rango ({close_pos:.2f} < 0.55)")
        if direction == 'SHORT' and close_pos > 0.45:
            return _reject(f"SHORT: cierre alto en rango ({close_pos:.2f} > 0.45)")

    # -- 4. KF SLOPE NORMALIZADO
    slope_hist = hist['kf_slope'].abs().replace(0, np.nan).dropna()
    if len(slope_hist) >= 20:
        slope_p55 = float(slope_hist.quantile(0.55))
        if direction == 'LONG'  and kf_slope < slope_p55:
            return _reject(f"LONG: kf_slope={kf_slope:+.5f} < p55={slope_p55:.5f}")
        if direction == 'SHORT' and kf_slope > -slope_p55:
            return _reject(f"SHORT: kf_slope={kf_slope:+.5f} > -p55={-slope_p55:.5f}")

    # -- 5. RSI MOMENTUM ZONE DINAMICA
    rsi_hist = hist['rsi'].replace(0, np.nan).dropna()
    if len(rsi_hist) >= 20:
        rsi_min   = float(rsi_hist.quantile(0.20))
        rsi_max   = float(rsi_hist.quantile(0.80))
        rsi_range = rsi_max - rsi_min
        if rsi_range > 5:
            rsi_lo = rsi_min + rsi_range * 0.45
            rsi_hi = rsi_min + rsi_range * 0.55
            if direction == 'LONG'  and rsi < rsi_lo:
                return _reject(f"LONG: RSI={rsi:.1f} < zona {rsi_lo:.1f}")
            if direction == 'SHORT' and rsi > rsi_hi:
                return _reject(f"SHORT: RSI={rsi:.1f} > zona {rsi_hi:.1f}")

    # -- 6. MACD DIRECCION
    if direction == 'LONG'  and macd_hist <= 0:
        return _reject(f"LONG: MACD hist={macd_hist:.4f} ≤ 0")
    if direction == 'SHORT' and macd_hist >= 0:
        return _reject(f"SHORT: MACD hist={macd_hist:.4f} ≥ 0")

    # -- 7. EXTENSION PERCENTIL PROPIO
    if kf_price > 0 and atr > 0:
        curr_ext    = abs(close_p - kf_price) / atr
        ext_history = ((hist['close'] - hist['kf_price']).abs() /
                       hist['atr'].replace(0, np.nan)).dropna()
        if len(ext_history) >= 20:
            ext_p80 = float(ext_history.quantile(0.80))
            if curr_ext > ext_p80:
                return _reject(f"Sobreextendido: ext={curr_ext:.2f} > p80={ext_p80:.2f}")

    # -- 8. DIRECCION vs KALMAN
    if direction == 'LONG'  and close_p < kf_price:
        return _reject(f"LONG: close={close_p:.4f} < kf_price={kf_price:.4f}")
    if direction == 'SHORT' and close_p > kf_price:
        return _reject(f"SHORT: close={close_p:.4f} > kf_price={kf_price:.4f}")

    # -- 9. ENTROPY ADAPTATIVA
    entropy_hist = hist['entropy'].replace(0, np.nan).dropna()
    if len(entropy_hist) >= 20:
        ent_p70 = float(entropy_hist.quantile(0.70))
        if entropy > ent_p70:
            return _reject(f"Entropy={entropy:.3f} > p70={ent_p70:.3f} (caótico)")

    return True



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
        
        # Kp=0.05: Reactividad bajísima
        # output_limits=(-2.0, 0.5): Puede reducir 2 ATR al trail original de 4.0
        pid = PIDController(Kp=0.05, Ki=0.0, Kd=0.01, setpoint=0, output_limits=(-2.0, 0.5))
        
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
            'worst_price': entry_price,
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

            # Track best AND worst price
            if pos['direction'] == 'LONG':
                if c_high > pos['best_price']:
                    pos['best_price'] = c_high
                if c_low < pos['worst_price']:
                    pos['worst_price'] = c_low
                pnl_atr = (c_close - pos['avg_price']) / atr_val
                
                # PID Input: Distance from Kalman Trend (in ATR units)
                # If Price >> Kalman, we are "extended" -> Tighten Trail
                dist_kalman_atr = (c_close - kf_price) / atr_val
                
            else:
                if c_low < pos['best_price']:
                    pos['best_price'] = c_low
                if c_high > pos['worst_price']:
                    pos['worst_price'] = c_high
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

            # 2. BREAKEVEN LOCK — activar cuando se supera el BE_LOCK_ATR configurado
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
                
                # Clamp: mín 0.7 ATR (permite apretarse al máximo en parabolas) / máx 3.0 ATR
                current_trail_atr = max(0.7, min(3.0, current_trail_atr))

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
        """
        MOMENTUM SCALING — escalar cuando la posición YA ESTÁ EN PROFIT.

        Lógica anterior (pullback adverso): escalaba cuando precio bajaba -1.5 ATR
        del primer entry. Problema: si el trade es perdedor, tienes 3x de capital en
        una posición en contra → stops de -$30 en lugar de -$10.

        Nueva lógica: escalar SOLO cuando el precio se mueve A FAVOR >= 1.5 ATR
        del primer entry. Esto garantiza que la posición escalada ya está en profit
        antes de añadir más tamaño.
        """
        if symbol not in self.positions:
            return False, 0

        pos = self.positions[symbol]
        if pos['scale_level'] >= MAX_SCALE:
            return False, 0
        if not pos['be_locked']:
            return False, 0  # SOLO escalar después de que el BE esté bloqueado

        if atr_val <= 0 or np.isnan(atr_val):
            return False, 0

        ref_price = pos['entry_price']

        # Calcular cuántos ATR se ha movido A FAVOR
        if pos['direction'] == 'LONG':
            favor_atr = (current_price - ref_price) / atr_val
        else:
            favor_atr = (ref_price - current_price) / atr_val

        # Escalar 2ª entrada cuando el precio ya está 1.5 ATR a favor
        if pos['scale_level'] == 1 and favor_atr >= 1.5:
            return True, 2

        # Escalar 3ª entrada cuando el precio está 3.0 ATR a favor
        if pos['scale_level'] == 2 and favor_atr >= 3.0:
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

        # Calcular MAE (Maximum Adverse Excursion)
        if pos['direction'] == 'LONG':
            mae_pct = (pos['worst_price'] - pos['avg_price']) / pos['avg_price']
        else:
            mae_pct = (pos['avg_price'] - pos['worst_price']) / pos['avg_price']
        
        # MAE is always expressed as a negative number or zero (how much we suffered)
        mae_pct = min(0.0, mae_pct)

        return {
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_price': pos['avg_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_dollar': pnl_dollar,
            'mae_pct': mae_pct * 100, # El peor drawdown sufrido como % negativo
            'reason': reason,
            'hold_candles': candle_idx - pos['entry_idx'],
            'be_locked': pos['be_locked'],
            'scale_level': pos['scale_level'],
        }
