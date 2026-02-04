import pandas as pd
import numpy as np
from technical_analysis import TechnicalAnalysis

class PatternDetector:
    """
    PatternDetector V2 (Clean Reset).
    Focus: SINGLE STRATEGY for MAJORS (BTC, ETH, BNB, XRP, DOT).
    Logic: Low-Level AI (Mean Reversion / Regression).
    """
    
    # Config will be loaded from JSON
    MAJOR_PAIRS = []
    GRINDER_PAIRS = [] 
    SCALPER_PAIRS = []
    
    SCALPER_GROUPS = {}
    MAJOR_GROUPS = {}
    GRINDER_GROUPS = {}

    @staticmethod
    def get_candle_limit(symbol):
        # Relaxed limit for all, as 300 covers all indicators safely (ema200 needs ~250)
        return 300

    def __init__(self):
        import os
        self.reasons = []
        self.strategy_map = {} 
        self.config_file = os.path.join(os.path.dirname(__file__), "market_config.json")
        
        # Load Config
        self._load_config()
        
    def _load_config(self):
        import json
        import os
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.MAJOR_PAIRS = data.get('vip_majors', [])
                    self.GRINDER_PAIRS = data.get('vip_grinders', [])
                    self.SCALPER_PAIRS = data.get('vip_scalpers', [])
                    
                    self.SCALPER_GROUPS = data.get('scalper_subgroups', {})
                    self.MAJOR_GROUPS = data.get('major_subgroups', {})
                    self.GRINDER_GROUPS = data.get('grinder_subgroups', {})
                    
            except Exception as e:
                print(f"Error loading config in PatternDetector: {e}")
        else:
            print("Warning: market_config.json not found in PatternDetector")
        
    def update_strategy_map(self, new_map):
        self.strategy_map = new_map
        
    def set_mock_variant(self, variant):
        """ 
        CALIBRATION MODE: Forces specific logic path for Tournament.
        Variants: MAJOR_VOLATILE, MAJOR_STABLE, SCALPER_A, SCALPER_B, SCALPER_C, GRINDER_DEFAULT, GRINDER_PROVEN
        """
        self.mock_variant = variant

    def analyze(self, df_15m, df_daily=None, btc_trend=0.0, symbol=None, force_strategy=None):
        """
        Main Enty Point.
        """
        self.reasons = [] # Reset reasons
        self.symbol = symbol
        
        # 1. Indicators
        df = TechnicalAnalysis.calculate_indicators(df_15m)
        self.curr = df.iloc[-1]
        self.df = df
        
        # 2. CALIBRATION / TOURNAMENT OVERRIDE
        if hasattr(self, 'mock_variant') and self.mock_variant:
            # Map Variant to Function Call
            if 'MAJOR' in self.mock_variant:
                return self._analyze_major_strategy(bypass_whitelist=True)
            elif 'SCALPER' in self.mock_variant:
                return self._analyze_scalper_strategy()
            elif 'GRINDER' in self.mock_variant:
                return self._analyze_grinder_strategy()

        # 3. STRATEGY ROUTING (Production)
        
        # A. CALIBRATION OVERRIDE (Legacy)
        if force_strategy == 'MAJOR_REVERSION':
            signal = self._analyze_major_strategy(bypass_whitelist=True)
        elif force_strategy == 'GRINDER':
            signal = self._analyze_grinder_strategy()
        elif force_strategy == 'SCALPER':
            signal = self._analyze_scalper_strategy()
            
        # B. PRODUCTION DYNAMIC ROUTING (Priority)
        elif self.symbol in self.strategy_map:
            assigned_strat = self.strategy_map[self.symbol]
            if assigned_strat == 'MAJOR_REVERSION':
                 signal = self._analyze_major_strategy(bypass_whitelist=True) 
            elif assigned_strat == 'SCALPER':
                 signal = self._analyze_scalper_strategy()
            elif assigned_strat == 'GRINDER':
                 signal = self._analyze_grinder_strategy()
            
        # C. FALLBACK LEGACY ROUTING
        else:
            if self.symbol in self.MAJOR_PAIRS:
                signal = self._analyze_major_strategy()
                if signal is None and self.symbol in self.FALLBACK_PAIRS:
                    signal = self._analyze_grinder_strategy()
            elif self.symbol in self.SCALPER_PAIRS:
                signal = self._analyze_scalper_strategy()
            else:
                signal = self._analyze_grinder_strategy()
                
        # 3. DYNAMIC SL/TP CALCULATION (The Risk Engine)
        if signal:
            atr = self.curr['atr'] if self.curr['atr'] > 0 else (self.curr['close'] * 0.02)
            entry_price = self.curr['close']
            
            # DEFAULT MULTIPLIERS (MATCHING BACKTEST_ANALYSIS.PY)
            # The backtester used: SL = 2.0 * ATR, TP = 3.0 * ATR
            # We strictly enforce this to match the validated results.
            sl_mult = 2.0
            tp_mult = 3.0
            
            # Note: Strategy nuances are disabled to ensure identity with Simulation.
            
            # Calculate Prices
            if signal['direction'] == 'LONG':
                 sl_price = entry_price - (atr * sl_mult)
                 tp_price = entry_price + (atr * tp_mult)
            else:
                 sl_price = entry_price + (atr * sl_mult)
                 tp_price = entry_price - (atr * tp_mult)
                 
            signal['sl_price'] = sl_price
            signal['tp_price'] = tp_price
            
        return signal

    def _analyze_scalper_strategy(self):
        """
        STRATEGY 3: SCALPER (The Rubber Band)
        """
        # PARAMETER CONFIGURATION
        require_macd = False
        rsi_limit = 30
        
        # GROUP A: The Champions (Proven > 10% Profit)
        # Logic: High trust, standard panic settings.
        if (hasattr(self, 'mock_variant') and self.mock_variant == 'SCALPER_A') or \
           (self.symbol in self.SCALPER_GROUPS.get('group_a', [])):
             rsi_limit = 30
             require_macd = False
             
        # GROUP B: The Proven New (Strict)
        # ORDI, MEME performed well.
        elif (hasattr(self, 'mock_variant') and self.mock_variant == 'SCALPER_B') or \
             (self.symbol in self.SCALPER_GROUPS.get('group_b', [])):
             rsi_limit = 25 
             require_macd = False
        
        # GROUP C: The Laggards (DOGE, SHIB) -> REMOVED FROM ELITE
        else:
             rsi_limit = 25 # Default fallback (SCALPER_C hits this by default logic flow)
             require_macd = True
             
        # 1. Setup: Price was outside Lower BB
        lower_bb = self.curr['lower_bb']
        close_p = self.curr['close']
        
        is_reversal = (self.curr['low'] < lower_bb) and (close_p > lower_bb)
        if not is_reversal: return None
        
        # 2. Filter: RSI Limit (Dynamic)
        if self.curr['rsi'] > rsi_limit: return None
        
        # 3. Filter: Volume Confirmation (Always On)
        if self.curr['volume'] < self.curr['vol_ma']: return None
        
        # 4. Filter: MACD Confirmation (Group C Only)
        if require_macd:
             # Logic: Histogram must be improving (ticking up)
             # Current Hist > Previous Hist
             prev_hist = self.df['hist'].iloc[-2]
             curr_hist = self.curr['hist']
             
             # Also, ideally we want to be in negative territory (oversold momentum)
             if curr_hist > 0: return None 
             
             if curr_hist <= prev_hist: return None # Momentum still falling
             
             self.reasons.append(f"MACD Confirmed: Hist {curr_hist:.2f} > {prev_hist:.2f}")
        
        self.reasons.append(f"Scalper({self.symbol}): Pinbar | RSI {self.curr['rsi']:.1f} < {rsi_limit}")
        
        return {
            'type': 'SCALP_REVERSION',
            'direction': 'LONG',
            'score': 90,
            'weight': 1.0,
            'reason': self.reasons,
            'strategy': 'SCALPER'
        }

    def _analyze_grinder_strategy(self):
        """
        STRATEGY 2: GRINDER (The Rescue)
        """
        # 0. Safety Check for Indicators
        if 'ema_200' not in self.curr or pd.isna(self.curr['ema_200']):
            return None
            
        # PARAMETER CONFIGURATION (Dynamic Defaults)
        
        # Default to Strict Settings (Toxic/Unknown)
        # This is safer for new pairs in calibration.
        rsi_threshold = 40
        adx_threshold = 25
        require_delta = True

        # IF it's a known Proven Winner, relax conditions
        if (hasattr(self, 'mock_variant') and self.mock_variant == 'GRINDER_PROVEN') or \
           (self.symbol in self.GRINDER_GROUPS.get('proven_winners', [])):
             rsi_threshold = 45
             adx_threshold = 20

        # 1. Trend Alignment (Price > EMA 200)
        if self.curr['close'] < self.curr['ema_200']:
            return None
            
        # 2. Trigger: RSI Crossing UP
        if self.curr['rsi'] < rsi_threshold: return None
        
        # 3. Filter: ADX
        if self.curr['adx'] < adx_threshold: return None
        
        # 4. Volume Delta
        if require_delta and 'volume_delta' in self.df.columns:
            if self.curr['volume_delta'] < 0: return None
            
        self.reasons.append(f"Grinder: Price > EMA200 | RSI {self.curr['rsi']:.1f}")
        
        return {
            'type': 'GRINDER',
            'direction': 'LONG',
            'score': 80,
            'weight': 1.0,
            'reason': self.reasons,
            'strategy': 'GRINDER'
        }

    def _analyze_major_strategy(self, bypass_whitelist=False):
        """
        STRATEGY: MAJOR REVERSION
        """
        lower_bb = self.curr['lower_bb']
        close_p = self.curr['close']
        
        # 0. STRICT WHITELIST (Unless in Calibration Mode)
        if not bypass_whitelist:
            ALLOWED_PAIRS = [
                'BTCUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT'
            ]
            if self.symbol not in ALLOWED_PAIRS:
                return None
        
        # PARAMETER CONFIGURATION
        
        # Default / New Pairs -> Treat as Volatile Major (Wait for dip)
        rsi_limit = 35
        adx_limit = 50
        require_volume = True

        # Proven Stable Majors
        if (hasattr(self, 'mock_variant') and self.mock_variant == 'MAJOR_STABLE') or \
           (self.symbol in self.MAJOR_GROUPS.get('stable_majors', [])):
            rsi_limit = 30
            adx_limit = 50
            require_volume = False
            
        # 1. TRIGGER: Price falls BELOW the Lower Band
        if close_p > lower_bb: return None
        
        # 2. FILTER: RSI Deep Oversold
        if self.curr['rsi'] > rsi_limit: return None
        
        # 3. FILTER: Trend Safety (ADX)
        if self.curr['adx'] > adx_limit: return None
        
        # 4. REFINEMENT
        if require_volume:
             if self.curr['volume'] < self.curr['vol_ma']: 
                 return None
        
        self.reasons.append(f"MajorStrategy: Price < BB | RSI={self.curr['rsi']:.1f}")
        
        return {
            'type': 'MEAN_REVERSION',
            'direction': 'LONG',
            'score': 90, 
            'weight': 1.0, 
            'reason': self.reasons,
            'strategy': 'MAJOR_REVERSION'
        }


