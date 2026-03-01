import pandas as pd
import numpy as np
import sys
import os

from scanner_point_zero import AnomalyScannerPointZero

def create_dummy_data(scenario):
    """
    Creates 480 15m candles of dummy data to test the Point 0 scanner.
    """
    base_price = 100.0
    dates = pd.date_range(end=pd.Timestamp.now(), periods=480, freq='15min')
    
    if scenario == "ALREADY_TRENDING":
        # Price went straight up steadily for 24h. No compression.
        prices = np.linspace(80, 100, 96)
        vol = np.random.normal(100, 10, 480)
        
    elif scenario == "PERFECT_POINT_0_LONG":
        # Flat and completely compressed for 3 days, then sudden spike in the last 2 candles
        prices = np.ones(96) * 100
        # Compression phase
        prices[-48:-2] = np.random.uniform(99.5, 100.5, 46)
        # Ignition phase
        prices[-2:] = [102, 105]
        
        # Volume was super low, but spikes massive in last 2 candles
        vol = np.random.normal(10, 2, 480)
        vol[-2:] = [500, 800]
        
    elif scenario == "PERFECT_POINT_0_SHORT":
        # Flat and completely compressed for 3 days, then sudden dump
        prices = np.ones(96) * 100
        # Compression phase
        prices[-48:-2] = np.random.uniform(99.5, 100.5, 46)
        # Ignition phase
        prices[-2:] = [98, 95]
        
        # Volume was super low, but spikes massive in last 2 candles
        vol = np.random.normal(10, 2, 480)
        vol[-2:] = [500, 800]
        
    elif scenario == "FAKE_OUT":
        # Spiked, but no previous compression (already volatile)
        prices = np.random.uniform(90, 110, 96) 
        prices[-2:] = [112, 115]
        
        vol = np.random.normal(200, 50, 480)
        vol[-2:] = [500, 800]
        
    elif scenario == "FLAT_COIN":
        prices = np.random.uniform(99.5, 100.5, 96)
        vol = np.random.normal(10, 2, 480)
    else:
        prices = np.ones(96) * 100
        vol = np.random.normal(10, 2, 480)

    # Fill older data with flat line to help with long-term ATR
    full_prices = np.ones(480) * 100.0
    if scenario != "FLAT_COIN" and scenario != "PERFECT_POINT_0_LONG" and scenario != "PERFECT_POINT_0_SHORT":
         full_prices = np.random.uniform(95, 105, 480)
         
    full_prices[-96:] = prices
    
    # We need realistic high low
    df = pd.DataFrame({
        'close': full_prices,
        'high': full_prices * 1.002,
        'low': full_prices * 0.998,
        'volume': vol
    }, index=dates)
    
    return df

def run_test():
    print("🧪 Testing Nascent Trend Scanner Logic (POINT 0)...\n")
    
    pair_data = {
        'ALREADY_TRENDING': create_dummy_data("ALREADY_TRENDING"),
        'PERFECT_POINT_0_LONG': create_dummy_data("PERFECT_POINT_0_LONG"),
        'PERFECT_POINT_0_SHORT': create_dummy_data("PERFECT_POINT_0_SHORT"),
        'FAKE_OUT_HIGH_VOLATILITY': create_dummy_data("FAKE_OUT"),
        'FLAT_COIN': create_dummy_data("FLAT_COIN"),
    }
    
    scanner = AnomalyScannerPointZero()
    
    picks = scanner.score_universe(pair_data, 480, top_n=5)
    
    print("🏆 RESULTS (Ordered by Score):")
    for p in picks:
        print(f"\n{p['symbol']} ({p['direction']}) | Score: {p['score']}")
        print(f"Squeeze Factor: {p['squeeze']:.2f}x | 45m Ret: {p['ret_45m']:+.2f}% | Vol Z: {p['vol_z_1h']:.2f}")
        print(f"Reasons: {p['reasons']}")

if __name__ == "__main__":
    run_test()
