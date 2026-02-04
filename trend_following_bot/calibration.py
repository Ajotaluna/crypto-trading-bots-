
import pandas as pd
import ccxt
import time
import os
from patterns_v2 import PatternDetector
from technical_analysis import TechnicalAnalysis

class CalibrationManager:
    """
    DYNAMIC MARKET CALIBRATOR (TOURNAMENT EDITION)
    Function: Auditions candidate pairs against 7 Strategy Variants to find the optimal fit.
    Result: Returns a list of 'Approved' pairs + their assigned strategy (and subgroup).
    
    Criteria:
    1. Win Rate >= 40% (Safety)
    2. Net Profit >= 10% (Performance)
    """
    
    def __init__(self, use_cache=False):
        """
        use_cache: If False (Production), downloads are in-memory only (Ephemeral).
                   If True (Dev), saves CSVs to calibration_data/ for speed.
        """
        self.exchange = ccxt.binance()
        self.detector = PatternDetector()
        self.use_cache = use_cache
        self.data_dir = os.path.join(os.getcwd(), "data_cache")
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
            except: pass
        self.config_file = os.path.join(os.path.dirname(__file__), "market_config.json")
        
        self.vip_majors = []
        self.vip_grinders = []
        self.vip_scalpers = []
        
        # Subgroup Memory
        self.major_subgroups = {}
        self.grinder_subgroups = {}
        self.scalper_subgroups = {}
        
        self.calib_settings = {} # Default Init
        self.approved_pairs = {} 
        
        self._load_market_config()
        
        if self.use_cache and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _load_market_config(self):
        """ Loads VIP lists from JSON """
        if os.path.exists(self.config_file):
            try:
                import json
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.vip_majors = data.get('vip_majors', [])
                    self.vip_grinders = data.get('vip_grinders', [])
                    self.vip_scalpers = data.get('vip_scalpers', [])
                    
                    self.major_subgroups = data.get('major_subgroups', {})
                    self.grinder_subgroups = data.get('grinder_subgroups', {})
                    self.scalper_subgroups = data.get('scalper_subgroups', {})
                    
                    self.calib_settings = data.get('calibration_settings', {})
                print(f">>> Loaded Market Config: {len(self.vip_majors)} Majors, {len(self.vip_grinders)} Grinders, {len(self.vip_scalpers)} Scalpers.")
            except Exception as e:
                print(f"Error loading market_config.json: {e}")
        else:
            print("Warning: market_config.json not found. Using empty VIP lists.")

    def _save_market_config(self):
        """ Writes current VIP lists to JSON (In-Memory Source of Truth) """
        try:
            import json
            data = {
                "vip_majors": self.vip_majors,
                "vip_grinders": self.vip_grinders,
                "vip_scalpers": self.vip_scalpers,
                
                "major_subgroups": self.major_subgroups,
                "grinder_subgroups": self.grinder_subgroups,
                "scalper_subgroups": self.scalper_subgroups,
                
                "calibration_settings": self.calib_settings
            }

            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f">>> Market Config Updated (Auto-Save).")
        except Exception as e:
            print(f"Error saving market_config.json: {e}")

    def promote_candidate(self, symbol, strategy):
        """ Legacy Helper - Not used in Tournament Mode but kept for compatibility """
        pass

    def ban_candidate(self, symbol):
        """ Removes a toxic pair from ALL VIP lists """
        self._load_market_config()
        
        removed = False
        if symbol in self.vip_majors: 
            self.vip_majors.remove(symbol)
            removed = True
        if symbol in self.vip_grinders: 
            self.vip_grinders.remove(symbol)
            removed = True
        if symbol in self.vip_scalpers: 
            self.vip_scalpers.remove(symbol)
            removed = True
            
        if removed:
            self._save_market_config()
            print(f"💀 PURGED {symbol} from VIP Lists.")

    def fetch_calibration_data(self, symbol, limit=None):
        """ 
        Downloads 15m (Analysis) and 1m (Precision) data. 
        """
        # Load Limit from Config (Default 6000 to match Backtest)
        if limit is None:
             limit = self.calib_settings.get('limit', 6000)
             
        try:
            print(f"   Fetching {limit} candles for calibration (Deep Scan)...")
            # 1. Fetch 15m Analysis Data
            ohlcv_15m = self.exchange.fetch_ohlcv(symbol, timeframe='15m', limit=limit)
            if not ohlcv_15m: return None, None
            
            df_15m = pd.DataFrame(ohlcv_15m, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df_15m['time'] = pd.to_datetime(df_15m['time'], unit='ms')
            df_15m['symbol'] = symbol
            
            # 2. Fetch 1m Precision Data (Covering the same time range)
            start_time = ohlcv_15m[0][0]
            end_time = ohlcv_15m[-1][0]
            
            # Batch fetch 1m data
            all_1m = []
            since = start_time
            while True:
                ohlcv_1m = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1000, since=since)
                if not ohlcv_1m: break
                
                all_1m.extend(ohlcv_1m)
                last_ts = ohlcv_1m[-1][0]
                since = last_ts + 60000 # +1 min
                
                if last_ts >= end_time: break
                if len(all_1m) > limit * 16: break # Safety cap
                time.sleep(0.5) # Rate limit protection (Safe Mode)

            df_1m = pd.DataFrame(all_1m, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df_1m['time'] = pd.to_datetime(df_1m['time'], unit='ms')
            
            # Indexing for speed
            df_1m.set_index('time', inplace=True)
            df_1m.sort_index(inplace=True)
                
            return df_15m, df_1m
            
        except Exception as e:
            print(f"[Calibration] Error fetching {symbol}: {e}")
            return None, None

    def load_strategy_map(self):
        """ Load the persistent strategy map from disk """
        map_file = os.path.join(self.data_dir, "strategy_licenses.json")
        if os.path.exists(map_file):
            try:
                import json
                with open(map_file, 'r') as f:
                    return json.load(f)
            except: 
                return {}
        return {}

    def save_strategy_map(self, strategy_map):
        """ Save the map with timestamps """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        map_file = os.path.join(self.data_dir, "strategy_licenses.json")
        try:
            import json
            with open(map_file, 'w') as f:
                json.dump(strategy_map, f, indent=4)
        except Exception as e:
            print(f"Failed to save strategy map: {e}")

    def calibrate(self, candidate_list, reset_lists=True):
        """
        Main Routine: The Strategy Tournament (TRUE CALIBRATION).
        Tests 7 Variants per Pair. Assigns Winner > 40% WR and > 10% PnL.
        """
        
        # 1. Load Existing Licenses (Cache check)
        license_map = self.load_strategy_map()
        current_time = time.time()
        expiry_seconds = self.calib_settings.get('license_expiry_hours', 4) * 3600
        
        # Load Thresholds
        min_pnl = float(self.calib_settings.get('min_pnl_pct', 10.0))
        min_wr = float(self.calib_settings.get('min_win_rate', 40.0))
        
        # 2. LIST MANAGEMENT (RESET vs INCREMENTAL)
        if reset_lists:
            # Full Reset (Weekly maintenance)
            self.vip_majors = []
            self.vip_grinders = []
            self.vip_scalpers = []
            self.major_subgroups = {"stable_majors": []}
            self.scalper_subgroups = {"group_a": [], "group_b": []}
            self.grinder_subgroups = {"proven_winners": []}
            self.approved_pairs = {}
        # else: Keep existing lists, just add new winners
        
        print(f">>> STARTING PRODUCTION TOURNAMENT (7 VARIANTS) FOR {len(candidate_list)} PAIRS")
        print(f"    Criteria: WR >= {min_wr}%, PnL >= {min_pnl}% | Mode: {'RESET' if reset_lists else 'INCREMENTAL'}")
        
        print(f">>> STARTING PRODUCTION TOURNAMENT (7 VARIANTS) FOR {len(candidate_list)} PAIRS")
        print(f"    Criteria: WR >= {min_wr}%, PnL >= {min_pnl}%")

        VARIANTS = [
            'MAJOR_VOLATILE', 'MAJOR_STABLE', 
            'SCALPER_A', 'SCALPER_B', 'SCALPER_C',
            'GRINDER_DEFAULT', 'GRINDER_PROVEN'
        ]
        
        # 2b. SORT CANDIDATES BY TREND STRENGTH (ADX) - User Priority
        print(">>> Analyzing Trend Strength to prioritize candidates...")
        scored_candidates = []
        for symbol in candidate_list:
            # Fetch minimal data for trend check
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
                if not ohlcv: continue
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df = TechnicalAnalysis.calculate_indicators(df)
                adx = df['adx'].iloc[-1]
                scored_candidates.append( (symbol, adx) )
            except:
                scored_candidates.append( (symbol, 0) )
        
        # Sort Descending (Highest ADX first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        candidate_list = [x[0] for x in scored_candidates]
        print(f"    Top Trending Candidates: {[x[0] for x in scored_candidates[:5]]}")
        
        # 3. TOURNAMENT LOOP
        for symbol in candidate_list:
            
            df_15m, df_1m = self.fetch_calibration_data(symbol, limit=6000)
            if df_15m is None or df_1m is None: continue
            
            best_stats = None
            best_pnl = -999.0
            best_variant = None
            
            for variant in VARIANTS:
                # Inject Mock
                self.detector.set_mock_variant(variant)
                
                # Determine Base Strategy for Simulation
                if 'MAJOR' in variant: base_strat = 'MAJOR_REVERSION'
                elif 'SCALPER' in variant: base_strat = 'SCALPER'
                elif 'GRINDER' in variant: base_strat = 'GRINDER'
                
                stats = self.run_simulation(df_15m, df_1m, symbol, base_strat)
                if not stats: continue
                
                # Check Criteria
                is_qualified = (stats['pnl_pct'] >= min_pnl) and (stats['wr'] >= min_wr)
                
                if is_qualified:
                    if stats['pnl_pct'] > best_pnl:
                        best_pnl = stats['pnl_pct']
                        best_stats = stats
                        best_variant = variant
            
            # 4. ASSIGN WINNER
            if best_variant:
                print(f"   [{symbol}] WINNER: {best_variant:<20} | PnL: {best_stats['pnl_pct']:>5.1f}% | WR: {best_stats['wr']:.1f}%")
                
                # Map to Structs
                if 'MAJOR' in best_variant:
                    self.vip_majors.append(symbol)
                    self.approved_pairs[symbol] = 'MAJOR_REVERSION'
                    if 'STABLE' in best_variant:
                         self.major_subgroups.setdefault('stable_majors', []).append(symbol)
                         
                elif 'SCALPER' in best_variant:
                    self.vip_scalpers.append(symbol)
                    self.approved_pairs[symbol] = 'SCALPER'
                    if 'SCALPER_A' in best_variant:
                        self.scalper_subgroups.setdefault('group_a', []).append(symbol)
                    elif 'SCALPER_B' in best_variant:
                        self.scalper_subgroups.setdefault('group_b', []).append(symbol)
                        
                elif 'GRINDER' in best_variant:
                    self.vip_grinders.append(symbol)
                    self.approved_pairs[symbol] = 'GRINDER'
                    if 'PROVEN' in best_variant:
                        self.grinder_subgroups.setdefault('proven_winners', []).append(symbol)
                        
                # Update License cache
                license_map[symbol] = {
                    'strategy': self.approved_pairs[symbol],
                    'variant': best_variant,
                    'timestamp': current_time,
                    'stats': best_stats
                }
                
                # INCREMENTAL SAVE (Safety First)
                self._save_market_config()
                self.save_strategy_map(license_map)
            else:
                print(f"   [{symbol}] REJECTED. Best PnL: {best_pnl:.1f}%")

        # 5. SAVE EVERYTHING
        self._save_market_config()
        self.save_strategy_map(license_map)
        
        return self.approved_pairs

    def run_simulation(self, df_15m, df_1m, symbol, strategy_type):
        """ 
        High-Precision 'Mini-Backtest' using 1m Granularity.
        """
        initial_balance = 1000
        current_balance = initial_balance
        trades = 0
        wins = 0
        
        required_history = 300 if strategy_type == 'GRINDER' else 100
        if len(df_15m) < required_history: return None
        
        # Calculate Indicators on 15m
        df = TechnicalAnalysis.calculate_indicators(df_15m)
        
        warmup = required_history
        active_trade = None
        
        for i in range(warmup, len(df)):
            curr = df.iloc[i]
            curr_time = curr['time']
            next_time = curr_time + pd.Timedelta(minutes=15)
            
            # 1. Manage Active Trade (HIGH PRECISION MODE)
            if active_trade:
                # Slice 1m data for this 15m candle period
                try:
                    micro_candles = df_1m.loc[curr_time:next_time]
                except:
                    micro_candles = pd.DataFrame()
                
                if not micro_candles.empty:
                    for idx, row in micro_candles.iterrows():
                         # Check Low vs SL
                        if row['low'] <= active_trade['sl']:
                            current_balance -= (active_trade['risk'] * 1.0) # Loss
                            active_trade = None
                            break
                        # Check High vs TP
                        elif row['high'] >= active_trade['tp']:
                            current_balance += (active_trade['risk'] * 1.5) # Win
                            wins += 1
                            active_trade = None
                            break
            
            # 2. Find Entry
            if active_trade is None:
                # Use subset ending at CURRENT candle
                subset = df.iloc[:i+1]
                signal = self.detector.analyze(subset, symbol=symbol, force_strategy=strategy_type)
                
                if signal:
                    atr = curr['atr']
                    tp = curr['close'] + (atr * 3.0)
                    sl = curr['close'] - (atr * 2.0)
                    
                    active_trade = {
                        'entry': curr['close'],
                        'tp': tp,
                        'sl': sl,
                        'risk': 20 
                    }
                    trades += 1
                    
        return {
            'strategy': strategy_type,
            'trades': trades,
            'wins': wins,
            'pnl_pct': (current_balance - initial_balance) / initial_balance * 100,
            'wr': (wins/trades*100) if trades > 0 else 0
        }
