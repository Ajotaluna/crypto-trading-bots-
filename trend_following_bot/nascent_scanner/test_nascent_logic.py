import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the scanner
from scanner_anomaly import AnomalyScanner

def create_dummy_data(scenario):
    """
    Creates 480 15m candles of dummy data to test the scanner.
    """
    base_price = 100.0
    dates = pd.date_range(end=pd.Timestamp.now(), periods=480, freq='15min')
    
    if scenario == "EXHAUSTED_GAINER":
        # Price went straight up for 24 hours (last 96 candles)
        # 24h return = +25%
        # 6h return = +1% (flat/exhausted)
        prices = np.linspace(80, 100, 96)
        prices[-24:] = np.linspace(99, 100, 24)
        vol = np.random.normal(100, 10, 480)
        
    elif scenario == "NASCENT_SPRING":
        # Price was flat for days, then spiked exactly in the last 6 hours
        # 24h return = +8% 
        # 6h return = +8% (all the move happened right now)
        prices = np.ones(96) * 92.5
        prices[-24:] = np.linspace(92.5, 100, 24)
        
        # Volume was super low but variable, but spiked massive in last 24h/6h
        vol = np.random.normal(10, 2, 480)
        vol[-24:] = 500 # Huge spike
        
    elif scenario == "EXHAUSTED_LOSER":
        # Price went straight down for 24 hours
        # 24h return = -25%
        prices = np.linspace(133, 100, 96)
        prices[-24:] = np.linspace(101, 100, 24)
        vol = np.random.normal(100, 10, 480)
        
    elif scenario == "NASCENT_DUMP":
        # Price was flat, then dumped violently in last 6h
        # 24h return = -8%
        prices = np.ones(96) * 108.6
        prices[-24:] = np.linspace(108.6, 100, 24)
        
        # Volume spike
        vol = np.random.normal(10, 2, 480)
        vol[-24:] = 500
        
    elif scenario == "FLAT_COIN":
        prices = np.ones(96) * 100
        vol = np.random.normal(10, 2, 480)
    else:
        prices = np.ones(96) * 100
        vol = np.random.normal(10, 2, 480)

    # Fill older data
    full_prices = np.ones(480) * prices[0]
    full_prices[-96:] = prices
    
    df = pd.DataFrame({
        'close': full_prices,
        'high': full_prices * 1.01,
        'low': full_prices * 0.99,
        'volume': vol
    }, index=dates)
    
    return df

def run_test():
    print("🧪 Testing Nascent Trend Scanner Logic...\n")
    
    pair_data = {
        'EXHAUSTED_COIN': create_dummy_data("EXHAUSTED_GAINER"),
        'SPRING_COIN': create_dummy_data("NASCENT_SPRING"),
        'DUMPED_COIN': create_dummy_data("EXHAUSTED_LOSER"),
        'FALLING_KNIFE_COIN': create_dummy_data("NASCENT_DUMP"),
        'FLAT_COIN_1': create_dummy_data("FLAT_COIN"),
        'FLAT_COIN_2': create_dummy_data("FLAT_COIN"),
        'FLAT_COIN_3': create_dummy_data("FLAT_COIN"),
        'FLAT_COIN_4': create_dummy_data("FLAT_COIN"),
        'FLAT_COIN_5': create_dummy_data("FLAT_COIN"),
    }
    
    scanner = AnomalyScanner()
    
    # We need to print the raw metrics to see what is happening inside
    metrics = {}
    for symbol, df in pair_data.items():
        # duplicate part of the scanner logic here just to see the raw metrics
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']
        
        ret_24h = (close.iloc[-1] - close.iloc[-96]) / close.iloc[-96] * 100
        ret_6h = (close.iloc[-1] - close.iloc[-24]) / close.iloc[-24] * 100
        
        lookback = min(len(volume), 2880)
        vol_history = volume.iloc[-lookback:]
        daily_vols = []
        for i in range(0, len(vol_history) - 96, 96): daily_vols.append(vol_history.iloc[i:i+96].sum())
        vol_avg = np.mean(daily_vols)
        vol_std = np.std(daily_vols)
        vol_24h = volume.iloc[-96:].sum()
        vol_z = (vol_24h - vol_avg) / vol_std if vol_std > 0 else 0
        
        long_score = 0
        if vol_z >= 3.0: long_score += 40
        elif vol_z >= 2.0: long_score += 30
        elif vol_z >= 1.5: long_score += 15
        
        if ret_6h >= 5.0: long_score += 30
        elif ret_6h >= 3.0: long_score += 20
        elif ret_6h >= 1.5: long_score += 10
        
        if ret_24h >= 15.0: long_score -= 50
        
        print(f"RAW {symbol} -> 24h: {ret_24h:.2f}%, 6h: {ret_6h:.2f}%, VolZ: {vol_z:.2f} | MOCK SCORE: {long_score}")

    # Puntuamos usando el índice len(df) para no cortar la historia
    picks = scanner.score_universe(pair_data, 480, top_n=4)
    
    print("🏆 RESULTS (Ordered by Score):")
    for p in picks:
        print(f"\n{p['symbol']} ({p['direction']}) | Score: {p['score']}")
        print(f"24h Ret: {p['ret_24h']:+.2f}% | Vol Z: {p['vol_z']:.2f}")
        print(f"Reasons: {p['reasons']}")

if __name__ == "__main__":
    run_test()
