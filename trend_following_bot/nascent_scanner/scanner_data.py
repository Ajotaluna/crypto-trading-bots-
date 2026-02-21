"""
Scanner Data — Downloads 7 days of 15m candles + funding rates
for all Binance USDT Futures pairs.
Saves to nascent_scanner/data/ as CSV files.
"""
import asyncio
import sys
import os
import time

# Path setup to import MarketData from parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_data import MarketData
import requests
import pandas as pd


class ScannerData:
    def __init__(self):
        self.market = MarketData()
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.data_dir, exist_ok=True)

    async def download_all(self):
        """Download 7 days of 15m data for all USDT Futures pairs."""
        print("📡 Nascent Scanner: Fetching pair list...")

        try:
            pairs = await self.market.scan_top_volume(limit=500)
        except AttributeError:
            pairs = await self.market.get_top_gainers(limit=500)

        print(f"✅ Found {len(pairs)} pairs. Downloading 7-day history...")

        chunk_size = 10
        done = 0
        for i in range(0, len(pairs), chunk_size):
            chunk = pairs[i:i + chunk_size]
            tasks = [self._fetch_one(sym) for sym in chunk]
            await asyncio.gather(*tasks)
            done += len(chunk)
            print(f"   {done}/{len(pairs)}", end='\r')

        print(f"\n✅ Download complete. {done} pairs saved to nascent_scanner/data/")

        # Download funding rates
        print("📡 Downloading funding rates...")
        self.download_funding_rates(pairs)

    async def _fetch_one(self, symbol):
        try:
            # Prepare for 1 year of data
            now_ms = int(time.time() * 1000)
            one_year_ms = 365 * 24 * 3600 * 1000
            start_ts = now_ms - one_year_ms
            
            all_klines = []
            
            while True:
                # Fetch 1000 at a time (Max safe limit)
                df = await self.market.get_klines(
                    symbol, 
                    interval='15m', 
                    limit=1000, 
                    start_time=start_ts
                )
                
                if df.empty:
                    break
                
                all_klines.append(df)
                
                # Update start_ts to the last candle's close time + 1ms
                last_time = df.iloc[-1]['close_time']
                start_ts = int(last_time) + 1
                
                # Break if we caught up to now
                if start_ts >= now_ms:
                    break
                
                # Safety break if fetching returned partial count (end of data)
                if len(df) < 1000:
                    break
                    
                # Small pause to be nice to API
                await asyncio.sleep(0.05)

            if not all_klines:
                return

            # Combine all chunks
            full_df = pd.concat(all_klines, ignore_index=True)
            
            # Remove duplicates just in case
            full_df.drop_duplicates(subset=['timestamp'], inplace=True)
            full_df.sort_values(by='timestamp', inplace=True)

            safe = symbol.replace('/', '')
            
            # Only save if we have a decent amount of data
            if len(full_df) > 100:
                full_df.to_csv(os.path.join(self.data_dir, f"{safe}_15m.csv"), index=False)
                # print(f"   Saved {symbol}: {len(full_df)} candles") 
        except Exception as e:
            pass  # Silent fail for individual pairs

    def download_funding_rates(self, pairs=None):
        """Download historical funding rates for all pairs."""
        if pairs is None:
            # Get pairs from existing CSV files
            files = [f for f in os.listdir(self.data_dir) if f.endswith('_15m.csv')]
            pairs = [f.replace('_15m.csv', '') for f in files]

        base_url = "https://fapi.binance.com"
        done = 0
        errors = 0

        for symbol in pairs:
            try:
                resp = requests.get(
                    f"{base_url}/fapi/v1/fundingRate",
                    params={'symbol': symbol, 'limit': 100},
                    timeout=5
                )
                if resp.status_code != 200:
                    errors += 1
                    continue

                data = resp.json()
                if not data:
                    continue

                df = pd.DataFrame(data)
                df['fundingRate'] = pd.to_numeric(df['fundingRate'])
                df['fundingTime'] = pd.to_numeric(df['fundingTime'])
                safe = symbol.replace('/', '')
                df.to_csv(
                    os.path.join(self.data_dir, f"{safe}_funding.csv"),
                    index=False
                )
                done += 1

                # Rate limiting: small pause every 10 requests
                if done % 10 == 0:
                    print(f"   Funding: {done}/{len(pairs)}", end='\r')
                    time.sleep(0.2)

            except Exception:
                errors += 1

        print(f"\n✅ Funding rates: {done} downloaded, {errors} errors")


if __name__ == "__main__":
    loader = ScannerData()

    if len(sys.argv) > 1 and sys.argv[1] == '--funding-only':
        # Just download funding rates for existing pairs
        print("📡 Downloading funding rates only...")
        loader.download_funding_rates()
    else:
        asyncio.run(loader.download_all())
