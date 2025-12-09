"""
Trend Following Bot Configuration
Strict parameters for high-quality swing trading.
"""

class Config:
    # --- Strategy Parameters ---
    STRATEGY_NAME = "TrendFollowing_V1"
    
    # Entry Criteria
    # Entry Criteria (STRICT)
    import os
    # LOOSE_MODE REMOVED - ONLY STRICT TRADING ALLOWED
    MIN_SIGNAL_SCORE = 80.0          # Only High Quality Quality
    REQUIRE_VOLUME_SURGE = True      # Mandatory Volume
    MIN_VOLUME_MULTIPLIER = 1.5      # Significant surge needed
    
    # Position Management
    MIN_POSITION_TIME_SEC = 3600     # 1 Hour Minimum Hold (Enforced)
    MAX_POSITION_TIME_SEC = 86400    # 24 Hours Maximum Hold
    
    # Profit Targets
    DAILY_PROFIT_TARGET_PCT = 10.0   # Stop after 10% daily gain
    
    # Risk Management
    # ROI Targets: SL -5% ROI, TP +20% ROI (at 5x Leverage)
    # Price Movement = ROI / Leverage
    STOP_LOSS_PCT = 1.0              # 1% move * 5x lev = 5% Loss
    TAKE_PROFIT_PCT = 4.0            # 4% move * 5x lev = 20% Gain
    
    MAX_OPEN_POSITIONS = 10          # Batch size of 10
    CAPITAL_PER_TRADE_PCT = 10.0     # 10% per trade (to fit 10 trades)
    LEVERAGE = 5                     # Fixed 5x Leverage
    
    # Dynamic TP/SL Lookback
    LOOKBACK_WINDOW_SL = 20          # Recent Low/High for Stop Loss
    LOOKBACK_WINDOW_TP = 50          # Recent High/Low for Take Profit
    
    # Trailing Stop
    USE_TRAILING_STOP = True
    TRAILING_ACTIVATION_PCT = 5.0    # Activate after 5% gain
    TRAILING_DISTANCE_PCT = 4.0      # Trail 4% behind price
    
    # Resistance Logic
    RESISTANCE_TOUCHES_MIN = 3       # Major resistance = 3+ touches
    IGNORE_MINOR_RESISTANCE = True   # Explicitly ignore weak levels
    
    # System
    TIMEFRAME = '15m'                # Analysis timeframe
    CHECK_INTERVAL = 60              # Slow scan interval
    CHECK_INTERVAL = 60              # Slow scan interval
    MONITOR_INTERVAL = 300           # Reporting interval (5 mins)
    SAFETY_CHECK_INTERVAL = 5        # Safety check interval (5 sec)
    WATCHLIST_SCORE_THRESHOLD = 50.0 # Score to add to watchlist
    LOG_LEVEL = "INFO"

config = Config()
