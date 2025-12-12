"""
Scalping Bot V2 Configuration
High-frequency, precision-focused settings.
"""

class Config:
    STRATEGY_NAME = "Scalper_Pro_V2"
    
    # --- Scalping Specifics ---
    TIMEFRAME = '5m'                 # Fast timeframe
    CHECK_INTERVAL = 0.5             # Sub-second monitoring (Very Fast)
    SCAN_INTERVAL = 10               # Scan market every 10s
    
    # Entry Criteria (Strict)
    import os
    # LOOSE MODE DISABLED
    MIN_SCORE = 85.0                 # High quality Only
    MIN_VOL_MULTIPLIER = 1.5         # 1.5x Volume Spike
    
    # Risk Management (Tight)
    LEVERAGE = 10                    # 10x for Micro Account
    STOP_LOSS_ROI = -15.0            # -15% PnL (1.5% price move)
    TAKE_PROFIT_ROI = 30.0           # +30% PnL (3.0% price move)
    DAILY_PROFIT_TARGET_PCT = 3.0    # Stop after 3% daily gain (Realistic)
    
    # Fee Structure (Binance Futures)
    FEE_MAKER = 0.0002              # 0.02% (VIP0) - Using conservative 0.05% for safety
    FEE_TAKER = 0.0005              # 0.05% (VIP0)
    # User specified: 0.05% (Maker? usually lower) and 0.045% (Taker)
    # We will use user values for calculation safety
    CALC_FEE_MAKER = 0.0005          # 0.05%
    CALC_FEE_TAKER = 0.00045         # 0.045%
    
    # Position
    MAX_OPEN_POSITIONS = 5           # More concurrency
    CAPITAL_PER_TRADE_PCT = 15.0     # Smaller chunks
    MAX_HOLD_SECONDS = 3600          # Max 1 hour (Scalps shouldn't linger)
    
    # Trailing Stop (Aggressive)
    USE_TRAILING = True
    TRAILING_ACTIVATION_ROI = 10.0   # Activate at +10% ROI
    TRAILING_DISTANCE_ROI = 5.0      # Trail by 5% ROI

config = Config()
