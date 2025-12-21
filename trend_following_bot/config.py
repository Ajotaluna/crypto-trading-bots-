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
    
    # Unleashed Mode (Verticality Capture)
    ALLOW_MOMENTUM_OVERRIDE = True   # If Score > 90, Ignore Titan/Sentiment
    MOMENTUM_SCORE_THRESHOLD = 90.0  # The "Verticality" Threshold
    
    # Smart Entry (Confirmation V3 - Sniper)
    SMART_ENTRY_ENABLED = True       # Wait for breakout before entering
    CONFIRMATION_TIMEOUT_MINS = 45   # Extended listening time
    
    # V3 Validation Params
    # V3 Validation Params
    CONFIRM_VOLUME_FACTOR = 1.5      # Volume must be 50% > Average (Stricter)
    CONFIRM_BUFFER_PCT = 0.4         # Price must break trigger by 0.4% (Reduce wick entries)
    USE_1M_CONFIRMATION = True       # REQUIRE 1m Candle Close > Trigger (No instant tick entries)
    CONFIRM_RSI_MAX = 70.0           # Long: Stricter Overbought Check
    CONFIRM_RSI_MIN = 30.0           # Short: Stricter Oversold Check
    
    # Stability Filters (Anti-Loss)
    MIN_VOLUME_USDT = 50000000       # 50M Minimum Volume (Majors Only)
    MAX_DAILY_CHANGE_PCT = 30.0      # Ignore coins that pumped > 30% (Too late)
    MIN_DAILY_CHANGE_PCT = 1.0       # Ignore dead coins (< 1% move)
    TREND_ALIGN_INTERVAL = '1h'      # Check 1H Trend Line
    TREND_ALIGN_EMA = 200            # Daily Trend Indicator
    
    # Position Management
    MIN_POSITION_TIME_SEC = 3600     # 1 Hour Minimum Hold (Enforced)
    MAX_POSITION_TIME_SEC = 86400    # 24 Hours Maximum Hold
    
    # Profit Targets
    DAILY_PROFIT_TARGET_PCT = 3.0    # Stop after 3% daily gain (Realistic)
    
    # Risk Management (The Risk Vault - SNIPER V4)
    RISK_PER_TRADE_PCT = 1.0         # Risk 1% of Equity per trade (Professional Standard)
    TARGET_RISK_REWARD = 3.0         # Aim for 3R (1% Risk -> 3% Profit)
    MAX_CAPITAL_PER_TRADE_PCT = 25.0 # Cap max position size to 25% of account (Safety Cap)
    
    # ROI Targets (Fallback/Initial)
    # Price Movement = ROI / Leverage
    STOP_LOSS_PCT = 1.5              # Base Stop distance (will be dynamic)
    TAKE_PROFIT_PCT = 4.5            # 1:3 RR base expectation
    
    MAX_OPEN_POSITIONS = 10          # Restore missing config
    CAPITAL_PER_TRADE_PCT = 0.0      # DEPRECATED in favor of Risk Sizing
    LEVERAGE = 10                    # 10x
    
    # Dynamic TP/SL Lookback
    LOOKBACK_WINDOW_SL = 20          # Recent Low/High for Stop Loss
    LOOKBACK_WINDOW_TP = 50          # Recent High/Low for Take Profit
    
    # Trailing Stop (The Bloodhound V4 - Sniper Mode)
    USE_TRAILING_STOP = True
    TRAILING_ACTIVATION_PCT = 1.0    # Activate Trailing immediately after 1R (1% Gain)
    TRAILING_DISTANCE_PCT = 4.0      # Dynamic (Will be overridden by ATR logic)
    
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
