"""
Pattern Detection & Analysis
Consensus Engine: A 4-Module Voting System for High-Probability Entries.
"""
import pandas as pd
import numpy as np

# FIX: Robust Import for MathEngine
try:
    from math_engine import MathEngine
except ImportError:
    try:
        from .math_engine import MathEngine
    except ImportError:
        # Fallback for when running as a module from outside
        from trend_following_bot.math_engine import MathEngine

class TechnicalAnalysis:
    
    @staticmethod
    def calculate_indicators(df):
        """Add technical indicators to DataFrame"""
        if len(df) < 50: return df
        
        # EMAs
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean() # MOMENTUM TRIGGER
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean() # TREND ALIGNMENT
        
        # SuperTrend (Simplified) - Using volatility to determine trend
        # For this we need ATR first
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']
        
        # Bollinger Bands (for breakouts)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_bb'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_bb'] = df['sma_20'] - (df['std_20'] * 2)
        
        # Volume MA
        df['vol_ma'] = df['volume'].rolling(window=20).mean()

        # ADX (Average Directional Index) - Safe Mode
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join='outer').max(axis=1)
            atr = tr.rolling(14).mean()
            
            # Save ATR for Volatility Checks
            df['atr'] = atr
            
            plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
            minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
            dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
            adx = ((dx.shift(1) * (14 - 1)) + dx) / 14
            adx_smooth = adx.ewm(alpha=1/14).mean()
            df['adx'] = adx_smooth
            
            # VOLUME DELTA (Net Buy/Sell Pressure)
            # taker_buy_vol is passed from MarketData
            if 'taker_buy_vol' in df.columns:
                 # Delta = Buy Vol - Sell Vol (Total Vol - Buy Vol)
                 # Delta = Buy - (Total - Buy) = 2*Buy - Total
                 df['volume_delta'] = (2 * df['taker_buy_vol']) - df['volume']
            else:
                 df['volume_delta'] = 0
            
        except:
            df['adx'] = 0
            df['atr'] = 0
        
        return df


class ConsensusEngine:
    """ The 4-Module Voting System """

    def __init__(self, df_15m, df_daily=None, btc_trend=0.0, symbol=None):
        self.df = df_15m
        self.curr = df_15m.iloc[-2]
        self.daily = df_daily
        self.btc_trend = btc_trend
        self.symbol = symbol
        self.reasons = []
        self.total_score = 0
        self.veto = False
        
        self.category = self._classify_regime()
        self.is_major = (self.category == 'MAJOR') or (self.category == 'LAZY')

    def _classify_regime(self):
        """
        DYNAMIC ASSET CLASSIFIER
        Determines the asset's personality based on Volatility (ATR %).
        
        Thresholds (15m Timeframe):
        - > 0.8%: SCALPER (High Volatility, Elastic) -> Needs Deep RSI
        - < 0.4%: LAZY (Low Volatility, Grinder) -> Needs Tight Targets
        - 0.4% - 0.8%: MAJOR (Standard, Efficient) -> Standard Reversion
        """
        # Safety for backtest start or insufficient data
        if 'atr' not in self.df.columns or len(self.df) < 20:
             # Fallback based on known symbols if data missing
             if self.symbol in ['BTCUSDT', 'ETHUSDT']: return 'MAJOR'
             return 'SCALPER' # Default to safer/stricter logic
             
        # Calculate Volatility Ratio (ATR / Price)
        # We use the MEAN of the last 24 periods (6 hours) to stabilize the classification.
        # This prevents DOGE flipping between Major/Scalper every hour.
        curr_price = self.curr['close']
        recent_atr = self.df['atr'].iloc[-24:].mean()
        
        if curr_price == 0: return 'MAJOR'
        
        volatility_pct = (recent_atr / curr_price) * 100
        self.volatility_pct = volatility_pct
        
        # EFFICIENCY CHECK (To Distinguish DOGE from THETA)
        # Calculate ER (Efficiency Ratio) over last 24 periods
        period = 24
        if len(self.df) > period:
             change = abs(self.curr['close'] - self.df['close'].iloc[-period])
             vol_sum = self.df['close'].diff().abs().tail(period).sum()
             er = change / vol_sum if vol_sum > 0 else 0
        else:
             er = 1.0 # Default safe
             
        # CLASSIFICATION LOGIC
        if volatility_pct > 1.5: 
            return 'HYPER'
        elif volatility_pct > 0.8:
            # SPLIT: Clean Scalpers vs Messy Grinders
            # DOGE usually has ER > 0.4. THETA usually < 0.35.
            if er < 0.35:
                return 'GRINDER' # Tier 5: The Rescue
            return 'SCALPER'
        elif volatility_pct < 0.5:
            return 'LAZY'
        else:
            return 'MAJOR'

    def _analyze_trend_master(self):
        """ Module 1: The Navigator (+40 Max) """
        score = 0
        trend_aligned = False
        
        # ... (Previous code remains) ...
        # (Snippet truncated for brevity, assuming standard trend logic is same)
        # B. Local Trend (15m)
        if self.curr['close'] > self.curr['ema_200']:
            score += 15
            trend_aligned = True
            self.reasons.append("Price > EMA200")
        
        # C. Trend Strength (ADX)
        if self.curr['adx'] > 25:
            score += 10
            self.reasons.append(f"ADX Strong ({self.curr['adx']:.1f})")
            
        # D. Momentum (MACD)
        if self.curr['macd'] > self.curr['signal']:
            score += 5
            
        return score, trend_aligned

        return score

    def _check_toxic_efficiency(self):
        """
        Global Veto for 'Toxic' Assets (High Noise, No Direction).
        Calculates Kaufman Efficiency Ratio (ER).
        ER = Direction / Volatility
        """
        period = 10
        # Need at least 10 candles
        if len(self.df) < period + 1: return False
        
        # 1. Total Price Change (Direction)
        change = abs(self.curr['close'] - self.df['close'].iloc[-period])
        
        # 2. Sum of individual absolute changes (Volatility)
        # We sum the abstract difference of each candle in the window
        volatility = self.df['close'].diff().abs().tail(period).sum()
        
        if volatility == 0: return False
        
        er = change / volatility
        
        # ER Values:
        # > 0.6: Strong Trend (Parabolic)
        # 0.3 - 0.6: Normal Trend
        # < 0.25: CHOP / NOISE (Toxic)
        
        if er < 0.25:
            self.veto = True
            self.reasons.append(f"VETO: Toxic Efficiency ({er:.2f} < 0.25)")
            return True # Is Toxic
            
        return False # Is Safe

    def _analyze_quant_skeptic(self):
        """ Module 3: The Quant (+20 Max) """
        score = 0
        
        # GLOBAL EFFICIENCY CHECK (The Universal Filter)
        if self._check_toxic_efficiency():
            return 0
        
        # A. Hurst Exponent (Quality) - MANDATORY GATEKEEPER
        hurst = MathEngine.calculate_hurst(self.df['close'])
        
        # LEAGUE TUNING:
        # Majors: 0.48 (Mean Reverting)
        # Minors: 0.45 (Allow some chaos, but reject pure noise)
        hurst_threshold = 0.48 if self.is_major else 0.45
        
        # ADX OVERRIDE: If Trend is Strong (ADX > 25), ignore Hurst random walk.
        # Strong trends often look like random walks in short term due to volatility.
        adx = self.curr['adx']
        
        if hurst < hurst_threshold and adx < 25:
            self.veto = True
            self.reasons.append(f"VETO: Random Walk (H={hurst:.2f} < {hurst_threshold}) & ADX Weak")
            return 0
        else:
            score += 10
            self.reasons.append(f"Hurst {hurst:.2f} (Trending)")
            
        # ... (Z-Score remains same) ...
        # B. Z-Score (Overextension)
        z_score = MathEngine.calculate_z_score(self.df['close'])
        
        if z_score > 2.5:
             self.veto = True
             self.reasons.append(f"VETO: Z-Score {z_score:.2f} (Overbought)")
        elif z_score < -2.0:
            pass
        else:
            score += 10
            
        return score

    # ... (Other modules remain same) ...
    def _analyze_volatility_sniper(self):
        # Re-implementing strictly to include missed logic from replace
        score = 0
        breakout = False
        
        if self.curr['close'] > self.curr['upper_bb']:
            score += 15
            breakout = True
            self.reasons.append("Bollinger Breakout")
            
        if 'atr' in self.df.columns:
            recent_atr = self.df['atr'].iloc[-2]
            avg_atr = self.df['atr'].rolling(20).mean().iloc[-2]
            if recent_atr > avg_atr: score += 5
                
            price = self.curr['close']
            if price > 0 and (recent_atr / price) < 0.002:
                self.veto = True
                self.reasons.append("VETO: Zombie Coin (<0.2% Vol)")

        if self.curr['volume'] > (self.curr['vol_ma'] * 1.5):
            score += 10
            self.reasons.append("Vol Surge > 1.5x")
        return score, breakout

    def _analyze_flow_auditor(self):
        """ Module 4: The Auditor (+10 Max + Veto Power) """
        score = 0
        if 'volume_delta' in self.df.columns:
            delta = self.curr['volume_delta']
            if delta > 0:
                score += 10
                self.reasons.append("Delta Positive")
            else:
                self.reasons.append("Delta Negative")
                if self.is_major:
                    # Majors: Soft Penalty (Liquidity Grab possible)
                    score -= 10
                else:
                    # Alts: Hard Veto (Trap)
                    self.veto = True
                    self.reasons.append("VETO: Flow Divergence")
        return score

    def _analyze_regression_ai(self):
        """ 
        Special Logic for Majors (BTC/ETH).
        Strategy: Bollinger Reversion (The Trap).
        We buy when price snaps back inside the bands after a panic dump.
        """
        # 1. Bollinger Band Setup
        # We need to have pierced the Lower Band previously.
        # Since we only have current and daily, we check if LOW was below Lower BB 
        # and CLOSE is now back inside (or above).
        # Actually, better: Close > Lower BB AND Open < Lower BB (Green recovery candle).
        # Or Previous was outside? We can approximate with:
        # Low < Lower_BB AND Close > Lower_BB + (Limit deviations).
        
        close_p = self.curr['close']
        lower_bb = self.curr['lower_bb']
        
        # Trigger: We went below the band, but closed INSIDE it.
        # This represents "Rejection of Lower Prices".
        is_reversal = (self.curr['low'] < lower_bb) and (close_p > lower_bb)
        
        # 2. Indicators
        rsi = self.curr['rsi']
        adx = self.curr['adx']
        
        # 3. Order Flow (The Fuel)
        delta = 0
        if 'volume_delta' in self.df.columns:
            delta = self.curr['volume_delta']

        self.reasons.append(f"MajorReversal: BB_Trap={is_reversal}, Delta={delta:.0f}, RSI={rsi:.1f}")

        # DECISION MATRIX
        # A. Trigger
        if not is_reversal: return None
        
        # B. Deep Value Filter
        # We don't want just any touch. We want panic.
        # RSI < 35 ensures we are in a oversold territory.
        if rsi > 35: return None
        
        # C. Trend Safety
        # If trend is crashing HARD (ADX > 50), don't catch the knife even on a wick.
        if adx > 50: return None
        
        # D. Confirmation
        # Must have green close and positive delta
        if close_p <= self.curr['open']: return None
        if delta < 0: return None

        return {
            'type': 'MEAN_REVERSION',
            'direction': 'LONG',
            'score': 90, 
            'weight': 1.0, 
            'reason': self.reasons
        }

    def _analyze_scalp_reversion(self):
        """
        Special Logic for SCALPERS (DOGE, ADA).
        Strategy: Deep Scalp Reversion (Elastic Snap).
        """
        # 1. Setup: Price below Lower BB (Panic)
        close_p = self.curr['close']
        lower_bb = self.curr['lower_bb']
        
        # Trigger: Close back inside band OR Green candle pushing back
        is_reversal = (self.curr['low'] < lower_bb) and (close_p > lower_bb)
        
        if not is_reversal: return None

        # 2. Filters for Alts (The "Garbage" Filter)
        
        # A. Volume Check: Alts need FUEL.
        # Dead cat bounces usually have weak volume.
        # We demand Volume > Current Vol MA (Relative Strength)
        vol = self.curr['volume']
        vol_ma = self.curr['vol_ma']
        if vol < vol_ma: return None
        
        # B. RSI must be TRULY oversold.
        # Alts crash deeper. We need EXTREME fear.
        rsi = self.curr['rsi']
        if rsi > 25: return None
        
        # C. Delta Validation
        # Must have buyers stepping in.
        delta = 0
        if 'volume_delta' in self.df.columns:
            delta = self.curr['volume_delta']
        if delta < 0: return None

        self.reasons.append(f"MinorReversion: Scalp_Mode for {self.symbol}, RSI={rsi:.1f}")
        
        return {
            'type': 'SCALP_REVERSION', # Custom Type for Logic
            'direction': 'LONG',
            'score': 85,
            'weight': 1.0, 
            'reason': self.reasons
        }

    def _analyze_hyper_volatility(self):
        """
        Special Logic for HYPER Assets (SHIB, PEPE).
        Volatility > 1.5%.
        V-Shape Logic FAILED (25% WR).
        Strategy: EXTREME OVERSOLD (The Bottom Feeder).
        If it moves 1.5% in 15m, RSI 25 is not enough. We need RSI < 20.
        """
        # 1. Setup: Deep Crash 
        if self.curr['close'] >= self.curr['lower_bb']: return None
        
        # 2. Filter: RSI < 20 (EXTREME)
        if self.curr['rsi'] > 20: return None

        # 3. Filter: Volume at least average
        if self.curr['volume'] < self.curr['vol_ma']: return None
        
        # 4. Confirmation: Order Flow (Must be GREEN)
        # If we catch a falling knife, we need to see aggressive BUYERS.
        if 'volume_delta' in self.df.columns:
            if self.curr['volume_delta'] < 0: return None
        
        self.reasons.append(f"HyperReversal: Delta+ & RSI {self.curr['rsi']:.1f}")
        
        return {
            'type': 'HYPER_REVERSAL',
            'direction': 'LONG',
            'score': 90,
            'weight': 1.0,
            'reason': self.reasons
        }

    def _analyze_grinder_logic(self):
        """
        Special Logic for GRINDERS (THETA, ONE).
        High Volatility but Low Efficiency (Choppy).
        Standard Scalp Reversion fails because "The Dip" keeps dipping.
        Strategy: CONFIRMATION ENTRY (The Turn).
        We don't buy the bottom. We buy the specific momentum shift.
        """
        # 1. Setup: Recently Oversold
        # Look back 5 candles. Did we hit RSI < 25 recently?
        # This ensures we are acting on a "Recovery" from a crash.
        recent_rsi = self.df['rsi'].iloc[-5:]
        was_oversold = (recent_rsi < 25).any()
        
        if not was_oversold: return None
        
        # 2. Trigger: RSI Crossing UP over 30
        # Current RSI must be healthy (> 30) to prove strength.
        if self.curr['rsi'] < 30: return None
        
        # 3. Filter: Positive Delta (Buyers are back)
        if 'volume_delta' in self.df.columns:
            if self.curr['volume_delta'] < 0: return None
            
        self.reasons.append(f"GrinderRecovery: RSI>30 ({self.curr['rsi']:.1f}) after oversold")
        
        return {
            'type': 'GRINDER_RECOVERY',
            'direction': 'LONG',
            'score': 80, # Lower confidence than Majors
            'weight': 0.8, # Smaller size
            'reason': self.reasons
        }

    def compute_decision(self):
        # 0. Global Check
        if self.btc_trend < -0.01: return None 
        
        # CATEGORY ROUTING
        if self.category == 'MAJOR':
            return self._analyze_regression_ai()
        elif self.category == 'LAZY':
             # Lazy assets use Major logic but will have different sizing later
            return self._analyze_regression_ai()
        elif self.category == 'SCALPER':
            return self._analyze_scalp_reversion()
        elif self.category == 'HYPER':
            return self._analyze_hyper_volatility()
        elif self.category == 'GRINDER':
            return self._analyze_grinder_logic()
        else:
            # Default to Scalper logic for safety
            return self._analyze_scalp_reversion()

        # -- LEGACY CONSENSUS ENGINE (DEPRECATED) --
        # s_trend, is_aligned = self._analyze_trend_master()
        # ...
        
        total = s_trend + s_vol + s_quant + s_flow
        
        if total >= 65:
            return {
                'type': 'BREAKOUT',
                'direction': 'LONG',
                'score': total,
                'weight': (total - 65) / 25, 
                'reason': self.reasons
            }
        
        return None

class PatternDetector:
    """ Wrapper for ConsensusEngine to maintain API compatibility """
    
    def analyze(self, df_15m, df_daily=None, btc_trend=0.0, symbol=None):
        # Calculate Indicators first
        df = TechnicalAnalysis.calculate_indicators(df_15m)
        
        # Instantiate Engine
        engine = ConsensusEngine(df, df_daily, btc_trend, symbol)
        
        # Get Verdict
        signal = engine.compute_decision()
        
        if signal:
            # CALCULATE SL/TP (Required by Bot/Backtest)
            curr = df.iloc[-2]
            atr = curr['atr'] if curr['atr'] > 0 else (curr['close'] * 0.02)
            entry_price = curr['close']
            
            # LEAGUE TUNING: TP Multiplier
            # Now derived dynamically from the engine's classification
            category = engine.category
            
            # Dynamic TP based on Strategy Type & Category
            if signal['type'] == 'MEAN_REVERSION':
                if category == 'LAZY':
                     # Lazy Majors (Low Vol): Easy Targets
                     tp_mult = 1.0
                     sl_mult = 1.0
                else:
                     # Standard Majors (Efficient): Standard Targets
                     tp_mult = 1.5 
                     sl_mult = 1.0 
            elif signal['type'] == 'SCALP_REVERSION':
                # Alts Dead Cat Bounce: Quick In/Out
                tp_mult = 1.2 
                sl_mult = 0.8
            elif signal['type'] == 'HYPER_REVERSAL':
                # Hyper Volatility: Huge Risk, Huge Reward.
                # If we catch the bottom, it flies.
                tp_mult = 1.5 # Lowered from 2.0 to secure wins on dead cat bounces
                sl_mult = 1.5 # Wide SL to survive volatility
            elif signal['type'] == 'GRINDER_RECOVERY':
                # Grinder: Messy price action.
                # Take profit QUICKLY before it chops back down.
                tp_mult = 1.0 # Conservative target
                sl_mult = 2.0 # Wide SL because grinders wick hard
            elif signal['type'] == 'XRP_SURFER':
                # Trend Surfer: We are riding the EMA 20.
                # Standard Trend Following Risk.
                tp_mult = 2.0 
                sl_mult = 1.0
            elif signal['type'] == 'TREND_BREAKOUT':
                # Trend Following: Let it run, but tight SL on fakeout
                tp_mult = 3.0 # Big reward target
                sl_mult = 1.5
            elif is_major:
                # Conservative Trend for Majors
                tp_mult = 2.0
                sl_mult = 1.5
            else:
                # Aggressive Trend for Alts
                tp_mult = 2.5
                sl_mult = 1.5
            
            if signal['direction'] == 'LONG':
                 sl_price = entry_price - (atr * sl_mult)
                 tp_price = entry_price + (atr * tp_mult) 
            elif signal['direction'] == 'SHORT':
                 sl_price = entry_price + (atr * sl_mult)
                 tp_price = entry_price - (atr * tp_mult)
            
            return {
                'type': signal['type'],
                'direction': signal['direction'],
                'score': signal['score'],
                'reason': signal['reason'],
                'tp_price': tp_price,
                'sl_price': sl_price,
                'context': {'trend': 'BULLISH' if signal['score'] > 80 else 'NEUTRAL'}
            }
        return None
