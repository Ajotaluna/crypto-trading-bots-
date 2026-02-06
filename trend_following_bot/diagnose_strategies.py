
import asyncio
import logging
from market_data import MarketData
from patterns_v2 import PatternDetector
from calibration import CalibrationManager
from technical_analysis import TechnicalAnalysis

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Diagnose")

async def run_diagnosis():
    print(">>> 🕵️ STARTING STRATEGY DIAGNOSIS <<<")
    
    # 1. Setup Components
    market = MarketData(is_dry_run=True)
    detector = PatternDetector()
    calibrator = CalibrationManager(use_cache=False)
    
    # 2. Replicate Scanning (Volume)
    print("\n[1] Scanning Top 5 Volume Pairs (Fast Mode)...")
    candidates = await market.scan_top_volume(limit=5)
    print(f"    Found {len(candidates)} candidates: {candidates}...")
    
    # 3. Replicate Calibration
    print("\n[2] Running Calibration (Strict Mode: WR>50%, PnL>15%)...")
    # Note: calibrate is synchronous but uses network calls? 
    # Wait, calibration.py uses self.exchange (ccxt) which is blocking.
    # main.py runs it in executor. We can run it directly here for simplicity if it doesn't block async loop too bad.
    
    # Check if we should use execute logic
    approved_map = calibrator.calibrate(candidates, reset_lists=True)
    approved_pairs = list(approved_map.keys())
    
    print(f"\n>>> CALIBRATION RESULT: {len(approved_pairs)} Pairs Approved")
    for p in approved_pairs:
        print(f"    ✅ {p} -> {approved_map[p]}")
        
    if not approved_pairs:
        print("    ❌ NO PAIRS PASSED CALIBRATION. (This matches the 'Infinite Loop' if list is empty)")
        print("    ⚠️ SUGGESTION: The user mentioned '4 pairs' were found. If this script finds 0,")
        print("       it means the market changed or the previous run had different data.")
        return

    # 4. Analyze Signals & Trend (replicating main.py loop logic)
    print("\n[3] Checking Trend & Signals for Approved Pairs...")
    
    for symbol in approved_pairs:
        print(f"\n--- Analyzing {symbol} ---")
        
        # A. ADX Trend Check (Main Loop Filter)
        adx = await market.get_adx_now(symbol)
        print(f"    📊 ADX Score: {adx:.2f}")
        
        if adx < 20:
             print(f"    🛑 BLOCKED BY MAIN LOOP: ADX {adx:.2f} < 20 (Trend too weak)")
             # We continue analysis anyway to see if a SIGNAL exists, 
             # proving that 'Trend Filter' is the culprit.
        else:
             print(f"    ✅ TREND PASS: ADX {adx:.2f} >= 20")
             
        # B. Strategy Signal Check
        # Need Limit + Daily
        required_limit = detector.get_candle_limit(symbol)
        df_int = await market.get_klines(symbol, interval='15m', limit=required_limit)
        df_daily = await market.get_klines(symbol, interval='1d', limit=90)
        
        # Calculate Indicators
        df_int = TechnicalAnalysis.calculate_indicators(df_int)
        
        # Run Pattern Detector
        btc_trend = 0.0 # Assuming neutral for test
        signal = detector.analyze(df_int, df_daily, btc_trend)
        
        if signal:
            print(f"    🎯 SIGNAL GENERATED: {signal['direction']} (Score {signal['score']})")
            print(f"       Strategy: {signal.get('strategy', 'Unknown')}")
            
            # Additional Main Loop Checks
            if signal['score'] < 7.0: # Config default? main.py uses config.MIN_SIGNAL_SCORE
                print(f"       ⚠️ WEAK SIGNAL: Score {signal['score']} < Threshold")
            else:
                 print(f"       🚀 EXECUTABLE SIGNAL!")
        else:
            print(f"    💤 NO SIGNAL DETECTED via {approved_map[symbol]}")

if __name__ == "__main__":
    try:
        asyncio.run(run_diagnosis())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
