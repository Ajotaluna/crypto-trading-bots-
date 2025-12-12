import requests
import json

def check_coin(symbol):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    print(f"Fetching info for {symbol}...")
    try:
        res = requests.get(url).json()
        for s in res['symbols']:
            if s['symbol'] == symbol:
                print(f"\n--- {symbol} FILTERS ---")
                for f in s['filters']:
                    if f['filterType'] in ['LOT_SIZE', 'PRICE_FILTER', 'MIN_NOTIONAL']:
                        print(json.dumps(f, indent=2))
                return
        print(f"Symbol {symbol} not found!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_coin("COAIUSDT")
