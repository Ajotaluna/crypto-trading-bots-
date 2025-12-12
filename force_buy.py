import os
import asyncio
import time
import hmac
import hashlib
from urllib.parse import urlencode
import requests
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForceBuy")

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://fapi.binance.com" # PRODUCTION URL

def get_signature(params):
    query_string = urlencode(params)
    return hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def signed_request(method, endpoint, params=None):
    if params is None: params = {}
    params['timestamp'] = int(time.time() * 1000)
    params['signature'] = get_signature(params)
    headers = {'X-MBX-APIKEY': API_KEY}
    url = f"{BASE_URL}{endpoint}"
    
    try:
        logger.info(f"Sending {method} to {endpoint}...")
        if method == 'GET':
            resp = requests.get(url, headers=headers, params=params)
        elif method == 'POST':
            resp = requests.post(url, headers=headers, params=params)
        
        logger.info(f"Response ({resp.status_code}): {resp.text}")
        return resp.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None

async def force_trade():
    if not API_KEY or not API_SECRET:
        logger.error("MISSING KEYS")
        return

    symbol = "ADAUSDT" # Cheap symbol
    logger.info(f"--- FORCING REAL BUY ON {symbol} ---")
    
    # 1. Set Leverage to 10x
    signed_request('POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': 10})
    
    # 2. Get Price
    price_res = requests.get(f"{BASE_URL}/fapi/v1/ticker/price?symbol={symbol}").json()
    price = float(price_res['price'])
    logger.info(f"Current Price: {price}")
    
    # 3. Calculate Qty for $6 USD (Min is $5)
    # Positions size = $6.
    # Margin used = $0.60 (at 10x)
    qty = 6.0 / price
    qty = round(qty, 0) # ADA only accepts whole numbers or 1 decimal? distinct per asset. ADA is int usually for safe side let's check info.
    # actually ADA step size is 1. let's round to 0 decimals.
    
    logger.info(f"Buying {qty} {symbol} (Value ~$6 USD)")
    
    # 4. EXECUTE MARKET BUY
    res = signed_request('POST', '/fapi/v1/order', {
        'symbol': symbol,
        'side': 'BUY',
        'type': 'MARKET',
        'quantity': int(qty)
    })
    
    if res and 'orderId' in res:
        logger.info("SUCCESS! ORDER PLACED.")
        logger.info("CHECK BINANCE APP.")
    else:
        logger.error("FAILED.")

if __name__ == "__main__":
    asyncio.run(force_trade())
