import pandas as pd
import numpy as np
import sys
import os

from scanner_raw import RawMarketScanner

def create_dummy_data(scenario):
    """
    Creates 480 15m candles of dummy data.
    """
    dates = pd.date_range(end=pd.Timestamp.now(), periods=480, freq='15min')
    
    # Baseline for all coins
    base_price = 100.0
    prices = np.ones(480) * base_price
    
    # We add tiny noise to make standard deviation calculations possible
    noise = np.random.uniform(-0.1, 0.1, 480)
    prices += noise
    
    vol = np.random.normal(10, 2, 480) # Baseline volume
    
    if scenario == "ALREADY_TRENDING_LONG":
        # Price steadily rose 6% in the last 4 hours (16 candles)
        # Should be REJECTED.
        prices[-17:] = np.linspace(100, 106, 17)
        vol[-17:] = np.random.normal(50, 5, 17)
        
    elif scenario == "ALREADY_TRENDING_SHORT":
        # Price steadily dropped 6% in the last 4 hours (16 candles)
        # Should be REJECTED.
        prices[-17:] = np.linspace(100, 94, 17)
        vol[-17:] = np.random.normal(50, 5, 17)

    elif scenario == "PERFECT_POINT_0_LONG":
        # Flat for days.
        # Sudden huge single candle kick right now (1.5% jump). Huge volume. Breakout.
        # Should be ACCEPTED and score HIGH.
        prices[-2:] = [100.5, 102.5] 
        vol[-4:] = [10, 15, 300, 450] # Fuel
        
    elif scenario == "FAKE_KICK_NO_FUEL":
        # Price jumped, but volume is flat (average).
        # Should be ACCEPTED (not trending yet), but score LOW.
        prices[-2:] = [100.5, 102.5] 
        vol[-4:] = [10, 11, 10, 12] # NO Fuel
        
    elif scenario == "FLAT_COIN":
        # Doing absolutely nothing. 
        # Should be ACCEPTED (not trending), but score ZERO (no kick).
        pass

    df = pd.DataFrame({
        'close': prices,
        'open': np.roll(prices, 1),
        'high': prices * 1.002,
        'low': prices * 0.998,
        'volume': vol
    }, index=dates)
    
    # Fix first open
    df.loc[df.index[0], 'open'] = prices[0]
    
    return df

def run_test():
    print("🧪 Testing RAW ORIGIN Scanner Logic...\n")
    
    pair_data = {
        'ALREADY_TRENDING_LONG': create_dummy_data("ALREADY_TRENDING_LONG"),
        'ALREADY_TRENDING_SHORT': create_dummy_data("ALREADY_TRENDING_SHORT"),
        'PERFECT_POINT_0_LONG': create_dummy_data("PERFECT_POINT_0_LONG"),
        'FAKE_KICK_NO_FUEL': create_dummy_data("FAKE_KICK_NO_FUEL"),
        'FLAT_COIN': create_dummy_data("FLAT_COIN"),
    }
    
    scanner = RawMarketScanner()
    
    picks = scanner.score_universe(pair_data, 480, top_n=5)
    
    print("🏆 SURVIVING RESULTS (Ordered by Score):")
    survivors = set()
    for p in picks:
        survivors.add(p['symbol'])
        print(f"\n{p['symbol']} ({p['direction']}) | Score: {p['score']}")
        print(f"Reasons: {p['reasons']}")
        
    print("\n💀 REJECTED PAIRS (>4h Exhaustion):")
    for s in pair_data.keys():
        if s not in survivors:
            print(f"- {s} was blocked by Phase 1.")

if __name__ == "__main__":
    run_test()
