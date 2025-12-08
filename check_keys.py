import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

# Load keys from Env or use placeholders
API_KEY = os.getenv('API_KEY', '')
API_SECRET = os.getenv('API_SECRET', '')

BASE_URL = "https://fapi.binance.com"

def get_signature(params):
    query_string = urlencode(params)
    return hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def check_connection():
    print("--- CHECKING BINANCE CONNECTION ---")
    print(f"API Base: {BASE_URL}")
    print(f"Key: {API_KEY[:4]}...{API_KEY[-4:] if API_KEY else 'None'}")
    
    # 1. Public Endpoint Check (Time)
    try:
        r = requests.get(f"{BASE_URL}/fapi/v1/time", timeout=5)
        if r.status_code == 200:
            print(f"✅ Public API OK (Server Time: {r.json()['serverTime']})")
        else:
            print(f"❌ Public API Failed: {r.text}")
            return
    except Exception as e:
        print(f"❌ Connection Failed (Internet?): {e}")
        return

    # 2. Private Endpoint Check (Account Balance)
    if not API_KEY or not API_SECRET:
        print("⚠️ No API Keys provided. Skipping private check.")
        return

    try:
        params = {'timestamp': int(time.time() * 1000)}
        params['signature'] = get_signature(params)
        headers = {'X-MBX-APIKEY': API_KEY}
        
        r = requests.get(f"{BASE_URL}/fapi/v2/account", params=params, headers=headers, timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            balance = float(data['totalWalletBalance'])
            print(f"✅ Private API OK! Balance: {balance} USDT")
            print(">>> KEYS ARE VALID <<<")
        else:
            print(f"❌ Private API Failed (Using Keys): Code {r.status_code}")
            print(f"Error Message: {r.text}")
            print(">>> KEYS ARE LIKELY INVALID OR IP RESTRICTED <<<")
            
    except Exception as e:
        print(f"❌ Auth Request Error: {e}")

if __name__ == "__main__":
    check_connection()
