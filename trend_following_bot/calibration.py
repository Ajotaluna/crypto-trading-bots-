
import pandas as pd
import ccxt
import time
import os
from patterns_v2 import PatternDetector
from technical_analysis import TechnicalAnalysis

class CalibrationManager:
    """
    DYNAMIC MARKET CALIBRATOR
    Function: Auditions candidate pairs on startup to see if they fit our strategies.
    Result: Returns a list of 'Approved' pairs + their assigned strategy.
    
    Criteria:
    1. Win Rate > 50%
    OR
    2. Net Profit > 10% (in the test window)
    """
    
    def __init__(self, use_cache=False):
        """
        use_cache: If False (Production), downloads are in-memory only (Ephemeral).
                   If True (Dev), saves CSVs to calibration_data/ for speed.
        """
        self.exchange = ccxt.binance()
        self.detector = PatternDetector()
        self.use_cache = use_cache
        self.data_dir = os.path.join(os.path.dirname(__file__), "calibration_data")
        self.config_file = os.path.join(os.path.dirname(__file__), "market_config.json")
        
        self.vip_majors = []
        self.vip_grinders = []
        self.vip_scalpers = []
        self.calib_settings = {} # Default Init
        
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
                    self.calib_settings = data.get('calibration_settings', {})
                print(f">>> Loaded Market Config: {len(self.vip_majors)} Majors, {len(self.vip_grinders)} Grinders, {len(self.vip_scalpers)} Scalpers.")
            except Exception as e:
                print(f"Error loading market_config.json: {e}")
        else:
            print("Warning: market_config.json not found. Using empty VIP lists.")

    def _save_market_config(self):
        """ Writes current VIP lists to JSON """
        try:
            import json
            data = {
                "vip_majors": self.vip_majors,
                "vip_grinders": self.vip_grinders,
                "vip_scalpers": self.vip_scalpers,
                # Preserve subgroups if they exist, or use internal defaults if missing?
                # Best effort: Load existing first to preserve other keys like subgroups
                "major_subgroups": {}, 
                "grinder_subgroups": {},
                "scalper_subgroups": {},
                "calibration_settings": self.calib_settings
            }
            
            # Try to preserve existing subgroups by reading first
            if os.path.exists(self.config_file):
                 with open(self.config_file, 'r') as f:
                    existing = json.load(f)
                    data['major_subgroups'] = existing.get('major_subgroups', {})
                    data['grinder_subgroups'] = existing.get('grinder_subgroups', {})
                    data['scalper_subgroups'] = existing.get('scalper_subgroups', {})

            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f">>> Market Config Updated (Auto-Save).")
        except Exception as e:
            print(f"Error saving market_config.json: {e}")

    def promote_candidate(self, symbol, strategy):
        """ Promotes a new winner to the VIP list """
        self._load_market_config() # Refresh first
        
        if strategy == 'MAJOR_REVERSION':
            if symbol not in self.vip_majors: self.vip_majors.append(symbol)
        elif strategy == 'GRINDER':
             if symbol not in self.vip_grinders: self.vip_grinders.append(symbol)
        elif strategy == 'SCALPER':
             if symbol not in self.vip_scalpers: self.vip_scalpers.append(symbol)
             
        self._save_market_config()
        print(f"🌟 PROMOTED {symbol} to VIP {strategy} List.")

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

    # ... (run_simulation remains same) ...

    def calibrate(self, candidate_list):
        """
        Main Routine: The Strategy Tournament (Optimized with Licenses).
        """
        
        # 1. Load Existing Licenses
        license_map = self.load_strategy_map()
        current_time = time.time()
        expiry_seconds = self.calib_settings.get('license_expiry_hours', 4) * 3600
        
        # Explicit Definition to avoid NameError
        min_pnl = float(self.calib_settings.get('min_pnl_pct', 10.0))
        min_wr = float(self.calib_settings.get('min_win_rate', 40.0))
        
        approved_pairs = {}
        
        # 2a. VIP MAJORS (Always Approved, Instant)
        # Strategy: MAJOR_REVERSION
        print(">>> Loading VIP Majors...")
        for pair in self.vip_majors:
            approved_pairs[pair] = 'MAJOR_REVERSION'

        # 2b. VIP GRINDERS (Preferred Trend Followers)
        # Strategy: GRINDER
        # These pairs are historically better at trending (TRX, SUI, etc)
        # ELITE WINNERS (>10% Profit)
        print(">>> Loading VIP Grinders...")
        for pair in self.vip_grinders:
            approved_pairs[pair] = 'GRINDER'
            
        # 2c. VIP SCALPERS (High Volatility)
        # Strategy: SCALPER
        # Winners: WIF, PEPE, FLOKI, etc.
        print(">>> Loading VIP Scalpers...")
        for pair in self.vip_scalpers:
            approved_pairs[pair] = 'SCALPER'
            
        # 3. Process Candidates
        to_calibrate = []
        
        for symbol in candidate_list:
            if symbol in approved_pairs: continue # VIPs already handled
            
            # CHECK LICENSE
            if symbol in license_map:
                data = license_map[symbol]
                age = current_time - data['timestamp']
                
                if age < expiry_seconds:
                    # Valid License: Reuse Strategy
                    approved_pairs[symbol] = data['strategy']
                    continue
                else:
                    print(f"   [{symbol}] License Expired. Re-calibrating.")
            
            # If not valid or new, add to queue
            to_calibrate.append(symbol)

        # 4. Run Tournament only for Queue
        if to_calibrate:
            print(f">>> Running Strategy Tournament for {len(to_calibrate)} New/Expired Candidates...")
            strategies_to_test = ['MAJOR_REVERSION', 'GRINDER', 'SCALPER']
            
            for symbol in to_calibrate:
                df_15m, df_1m = self.fetch_calibration_data(symbol)
                if df_15m is None or df_1m is None: continue
                
                best_result = None
                best_pnl = -999.0
                
                for strat in strategies_to_test:
                    stats = self.run_simulation(df_15m, df_1m, symbol, strat)
                    if not stats: continue
                    
                    # RIGOROUS CRITERIA
                    is_qualified = (stats['pnl_pct'] >= min_pnl) and (stats['wr'] >= min_wr)
                    
                    if is_qualified:
                        if stats['pnl_pct'] > best_pnl:
                            best_pnl = stats['pnl_pct']
                            best_result = stats
                
                if best_result:
                    print(f"   [{symbol}] WINNER: {best_result['strategy']:<15} | PnL: {best_result['pnl_pct']:>5.1f}%")
                    approved_pairs[symbol] = best_result['strategy']
                    
                    # UPDATE LICENSE
                    license_map[symbol] = {
                        'strategy': best_result['strategy'],
                        'timestamp': current_time,
                        'stats': best_result # Optional stats storage
                    }
                else:
                    print(f"   [{symbol}] REJECTED (No profitable strategy)")

            # Save updated map
            self.save_strategy_map(license_map)
            
        else:
            print(">>> All candidates have valid licenses. No new calibration needed.")
                
        return approved_pairs

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

    def calibrate(self, candidate_list):
        """
        Main Routine: The Strategy Tournament (Optimized with Licenses).
        """
        # 1. Load Existing Licenses
        license_map = self.load_strategy_map()
        current_time = time.time()
        expiry_seconds = self.calib_settings.get('license_expiry_hours', 4) * 3600
        
        # Explicit Definition (Retry)
        min_pnl = float(self.calib_settings.get('min_pnl_pct', 10.0))
        min_wr = float(self.calib_settings.get('min_win_rate', 40.0))
        
        approved_pairs = {}
        
        # 2a. VIP MAJORS (Always Approved, Instant)
        # Strategy: MAJOR_REVERSION
        print(">>> Loading VIP Majors...")
        for pair in self.vip_majors:
            approved_pairs[pair] = 'MAJOR_REVERSION'

        # 2b. VIP GRINDERS (Preferred Trend Followers)
        # Strategy: GRINDER
        # These pairs are historically better at trending (TRX, SUI, etc)
        # ELITE WINNERS (>10% Profit)
        print(">>> Loading VIP Grinders...")
        for pair in self.vip_grinders:
            approved_pairs[pair] = 'GRINDER'
            
        # 2c. VIP SCALPERS (High Volatility)
        # Strategy: SCALPER
        # Winners: WIF, PEPE, FLOKI, etc.
        print(">>> Loading VIP Scalpers...")
        for pair in self.vip_scalpers:
            approved_pairs[pair] = 'SCALPER'
            
        # 3. Process Candidates
        to_calibrate = []
        
        for symbol in candidate_list:
            if symbol in approved_pairs: continue # VIPs already handled
            
            # CHECK LICENSE
            if symbol in license_map:
                data = license_map[symbol]
                age = current_time - data['timestamp']
                
                if age < expiry_seconds:
                    # Valid License: Reuse Strategy
                    approved_pairs[symbol] = data['strategy']
                    continue
                else:
                    print(f"   [{symbol}] License Expired. Re-calibrating.")
            
            # If not valid or new, add to queue
            to_calibrate.append(symbol)

        # 4. Run Tournament only for Queue
        if to_calibrate:
            print(f">>> Running Strategy Tournament for {len(to_calibrate)} New/Expired Candidates...")
            strategies_to_test = ['MAJOR_REVERSION', 'GRINDER', 'SCALPER']
            
            for symbol in to_calibrate:
                df_15m, df_1m = self.fetch_calibration_data(symbol)
                if df_15m is None or df_1m is None: continue
                
                best_result = None
                best_pnl = -999.0
                
                for strat in strategies_to_test:
                    stats = self.run_simulation(df_15m, df_1m, symbol, strat)
                    if not stats: continue
                    
                    # RIGOROUS CRITERIA (Loaded from JSON)
                    # Requires BOTH Profit and Win Rate
                    is_qualified = (stats['pnl_pct'] >= min_pnl) and (stats['wr'] >= min_wr)
                    
                    if is_qualified:
                        if stats['pnl_pct'] > best_pnl:
                            best_pnl = stats['pnl_pct']
                            best_result = stats
                
                if best_result:
                    print(f"   [{symbol}] WINNER: {best_result['strategy']:<15} | PnL: {best_result['pnl_pct']:>5.1f}%")
                    approved_pairs[symbol] = best_result['strategy']
                    
                    # UPDATE LICENSE
                    license_map[symbol] = {
                        'strategy': best_result['strategy'],
                        'timestamp': current_time,
                        'stats': best_result 
                    }
                    
                    # AUTO-PROMOTE TO CONFIG (Make it official)
                    self.promote_candidate(symbol, best_result['strategy'])
                else:
                    print(f"   [{symbol}] REJECTED (No profitable strategy)")

            # Save updated map
            self.save_strategy_map(license_map)
            
        else:
            print(">>> All candidates have valid licenses. No new calibration needed.")
                
        return approved_pairs

# --- TEST BLOCK ---
if __name__ == "__main__":
    manager = CalibrationManager()
    
    # Test Candidates (Mix)
    candidates = [
        'TRXUSDT', 'PEPEUSDT', # Grinder Winners
        'LTCUSDT',             # The Loser
        'SHIBUSDT',            # Scalper Candidate
        'WIFUSDT'              # Scalper King (should verify Scalper auto-select)
    ]
    
    final_list = manager.calibrate(candidates)
    
    print("\n=== FINAL TRADING LIST ===")
    for k, v in final_list.items():
        print(f"{k}: {v}")
