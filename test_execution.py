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
logger = logging.getLogger("TestOrder")

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://fapi.binance.com"

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
        if method == 'GET':
            resp = requests.get(url, headers=headers, params=params)
        elif method == 'POST':
            resp = requests.post(url, headers=headers, params=params)
        
        logger.info(f"Req: {endpoint} | Status: {resp.status_code}")
        return resp.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None

async def test_execution_flow():
    if not API_KEY or not API_SECRET:
        logger.error("Please set API_KEY and API_SECRET env vars!")
        return

    symbol = "BTCUSDT"
    logger.info(f"1. Testing Leverage Setup for {symbol}...")
    # Try setting leverage to 5x (Safe test)
    res = signed_request('POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': 5})
    logger.info(f"Leverage Response: {res}")

    logger.info("2. Checking Account Balance...")
    res = signed_request('GET', '/fapi/v2/balance', {})
    if isinstance(res, list):
        for asset in res:
            if asset['asset'] == 'USDT':
                logger.info(f"USDT Balance: {asset['balance']}")
    else:
        logger.error(f"Balance check failed: {res}")

if __name__ == "__main__":
    asyncio.run(test_execution_flow())
