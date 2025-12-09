"""
Scalping Market Data
Handles data fetching, execution, and PRECISE Fee Calculation.
"""
import asyncio
import logging
import hmac
import hashlib
import time
from urllib.parse import urlencode
from datetime import datetime
import pandas as pd
import requests
from config import config

logger = logging.getLogger("ScalperData")

class MarketData:
    def __init__(self, is_dry_run=True, api_key=None, api_secret=None):
        self.is_dry_run = is_dry_run
        self.api_key = api_key
        self.api_secret = api_secret
        self.positions = {}
        self.balance = 1000.0
        self.base_url = "https://fapi.binance.com"
        
        if not is_dry_run and (not api_key or not api_secret):
            self.is_dry_run = True

    async def _signed_request(self, method, endpoint, params=None):
        if params is None: params = {}
        params['timestamp'] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        params['signature'] = signature
        headers = {'X-MBX-APIKEY': self.api_key}
        
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: requests.request(method, f"{self.base_url}{endpoint}", headers=headers, params=params, timeout=3))
            if resp.status_code == 200: return resp.json()
        except Exception: pass
        return None

    async def get_current_price(self, symbol):
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: requests.get(f"{self.base_url}/fapi/v1/ticker/price", params={'symbol': symbol}, timeout=2))
            if resp.status_code == 200: return float(resp.json()['price'])
        except: pass
        return 0.0

    async def get_klines(self, symbol, interval='5m', limit=100):
        loop = asyncio.get_running_loop()
        def _fetch():
            try:
                resp = requests.get(f"{self.base_url}/fapi/v1/klines", params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=3)
                if resp.status_code == 200:
                    df = pd.DataFrame(resp.json(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tbb', 'tbq', 'ignore'])
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
                    return df
            except: pass
            return pd.DataFrame()
        return await loop.run_in_executor(None, _fetch)

    async def get_top_vol_symbols(self):
        loop = asyncio.get_running_loop()
        def _fetch():
            try:
                resp = requests.get(f"{self.base_url}/fapi/v1/ticker/24hr", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    # Filter high volume USDT pairs
                    pairs = [x for x in data if x['symbol'].endswith('USDT') and float(x['quoteVolume']) > 50000000]
                    pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                    return [x['symbol'] for x in pairs]
            except: pass
            return []
        return await loop.run_in_executor(None, _fetch)

    # --- Fee & PnL Calculation ---
    def calculate_real_pnl(self, entry_price, exit_price, amount, side):
        """
        Calculate Net PnL after Fees.
        Fees are calculated on Notional Value (Size * Price).
        """
        position_size = amount # Quantity of coins
        
        # 1. Entry Fee (Taker usually)
        entry_notional = position_size * entry_price
        entry_fee = entry_notional * config.CALC_FEE_TAKER
        
        # 2. Exit Fee (Taker usually for stop loss/market close)
        exit_notional = position_size * exit_price
        exit_fee = exit_notional * config.CALC_FEE_TAKER
        
        # 3. Gross PnL
        if side == 'LONG':
            gross_pnl = (exit_price - entry_price) * position_size
        else:
            gross_pnl = (entry_price - exit_price) * position_size
            
        # 4. Net PnL
        net_pnl = gross_pnl - (entry_fee + exit_fee)
        return net_pnl, entry_fee + exit_fee

    # --- Execution ---
    async def open_position(self, symbol, side, amount_usdt, sl, tp):
        price = await self.get_current_price(symbol)
        if price == 0: return
        
        qty = amount_usdt / price
        
        if self.is_dry_run:
            self.positions[symbol] = {
                'symbol': symbol, 'side': side, 'entry_price': price, 'amount': qty,
                'sl': sl, 'tp': tp, 'entry_time': datetime.now(), 'max_roi': -100
            }
            logger.info(f"[MOCK SCALP] {side} {symbol} @ {price} | SL: {sl} | TP: {tp}")
        else:
            # Real execution logic (simplified for brevity, same as trend bot but faster)
            # ... (Safety checks would go here)
            pass

    async def close_position(self, symbol, reason):
        if symbol not in self.positions: return
        pos = self.positions[symbol]
        price = await self.get_current_price(symbol)
        
        net_pnl, fees = self.calculate_real_pnl(pos['entry_price'], price, pos['amount'], pos['side'])
        roi_pct = (net_pnl / (pos['amount'] * pos['entry_price'] / config.LEVERAGE)) * 100
        
        logger.info(f"💰 CLOSE {symbol} | {reason} | Net PnL: ${net_pnl:.2f} ({roi_pct:.2f}%) | Fees: ${fees:.2f}")
        del self.positions[symbol]
