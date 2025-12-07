"""
Market Data & Mock Client
Handles data fetching and simulation.
"""
import asyncio
import logging
import random
import hmac
import hashlib
import time
from urllib.parse import urlencode
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

logger = logging.getLogger("MarketData")

class MarketData:
    def __init__(self, is_dry_run=True, api_key=None, api_secret=None):
        self.is_dry_run = is_dry_run
        self.api_key = api_key
        self.api_secret = api_secret
        self.positions = {}
        self.balance = 1000.0
        self.base_url = "https://fapi.binance.com"
        
        if not is_dry_run and (not api_key or not api_secret):
            logger.warning("Production mode requested but missing keys! Reverting to Dry Run.")
            self.is_dry_run = True

    def _get_signature(self, params):
        """Generate HMAC SHA256 signature"""
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _signed_request(self, method, endpoint, params=None):
        """Execute signed request for production (Non-Blocking)"""
        if params is None: params = {}
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._get_signature(params)
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.base_url}{endpoint}"
        
        loop = asyncio.get_running_loop()
        
        def _req():
            try:
                if method == 'GET':
                    return requests.get(url, headers=headers, params=params, timeout=5)
                elif method == 'POST':
                    return requests.post(url, headers=headers, params=params, timeout=5)
                elif method == 'DELETE':
                    return requests.delete(url, headers=headers, params=params, timeout=5)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return None

        try:
            resp = await loop.run_in_executor(None, _req)
                
            if resp and resp.status_code == 200:
                return resp.json()
            else:
                if resp: logger.error(f"API Error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Executor failed: {e}")
            return None

    async def initialize_balance(self):
        """Get initial balance"""
        if self.is_dry_run:
            self.balance = 1000.0
        else:
            res = self._signed_request('GET', '/fapi/v2/balance')
            if res:
                for asset in res:
                    if asset['asset'] == 'USDT':
                        self.balance = float(asset['balance'])
                        break
        logger.info(f"Initial Balance: {self.balance:.2f} USDT ({'DRY RUN' if self.is_dry_run else 'PRODUCTION'})")
        
    async def get_top_symbols(self, limit=None):
        """Get top volume USDT pairs"""
        try:
            # Real API call for symbols to be realistic
            resp = requests.get(f"{self.base_url}/fapi/v1/ticker/24hr", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                # Filter USDT and sort by volume (Min vol 10M to avoid junk)
                usdt_pairs = [x for x in data if x['symbol'].endswith('USDT') and float(x['quoteVolume']) > 10000000]
                usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                
                if limit:
                    return [x['symbol'] for x in usdt_pairs[:limit]]
                else:
                    return [x['symbol'] for x in usdt_pairs]
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
        
        # Fallback
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

    async def get_klines(self, symbol, interval='15m', limit=100):
        """Fetch candlestick data (Non-Blocking)"""
        url = f"{self.base_url}/fapi/v1/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        
        loop = asyncio.get_running_loop()
        
        def _fetch_and_parse():
            try:
                resp = requests.get(url, params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'trades', 
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    # Numeric conversion is heavy, good to do in thread
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
            except Exception as e:
                logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

        return await loop.run_in_executor(None, _fetch_and_parse)

    async def get_current_price(self, symbol):
        """Get latest price"""
        try:
            url = f"{self.base_url}/fapi/v1/ticker/price"
            params = {'symbol': symbol}
            resp = requests.get(url, params=params, timeout=2)
            if resp.status_code == 200:
                return float(resp.json()['price'])
        except Exception:
            pass
        return 0.0

    # --- Execution Methods ---
    
    async def open_position(self, symbol, side, amount_usdt, sl_price, tp_price):
        """Open position (Mock or Real)"""
        price = await self.get_current_price(symbol)
        if price == 0: return None
        
        amount = amount_usdt / price
        
        if self.is_dry_run:
            # MOCK EXECUTION
            self.positions[symbol] = {
                'symbol': symbol,
                'side': side,
                'entry_price': price,
                'amount': amount,
                'sl': sl_price,
                'tp': tp_price,
                'entry_time': datetime.now()
            }
            self.balance -= amount_usdt
            logger.info(f"[MOCK] OPEN {side} {symbol} @ {price} | SL: {sl_price} | TP: {tp_price}")
            return self.positions[symbol]
        else:
            # REAL EXECUTION
            # 0. Enforce Safety Settings
            try:
                # Set Leverage
                self._signed_request('POST', '/fapi/v1/leverage', {
                    'symbol': symbol,
                    'leverage': 5
                })
                # Set Margin Type (ISOLATED)
                self._signed_request('POST', '/fapi/v1/marginType', {
                    'symbol': symbol,
                    'marginType': 'ISOLATED'
                })
            except Exception:
                pass # Ignore if already set

            # 1. Place Market Order
            side_param = 'BUY' if side == 'LONG' else 'SELL'
            # Adjust precision (simplified)
            qty = "{:.3f}".format(amount) 
            
            order = self._signed_request('POST', '/fapi/v1/order', {
                'symbol': symbol,
                'side': side_param,
                'type': 'MARKET',
                'quantity': qty
            })
            
            if order:
                entry_price = float(order.get('avgPrice', price))
                self.positions[symbol] = {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': entry_price,
                    'amount': float(qty),
                    'sl': sl_price,
                    'tp': tp_price,
                    'entry_time': datetime.now()
                }
                logger.info(f"[REAL] OPEN {side} {symbol} @ {entry_price} | Lev: 5x | SL: {sl_price:.4f} (-5% ROI) | TP: {tp_price:.4f} (+20% ROI)")
                return self.positions[symbol]
            return None

    async def close_position(self, symbol, reason):
        """Close position (Mock or Real)"""
        if symbol not in self.positions: return
        
        pos = self.positions[symbol]
        price = await self.get_current_price(symbol)
        
        if self.is_dry_run:
            # MOCK CLOSE
            if pos['side'] == 'LONG':
                pnl_pct = (price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - price) / pos['entry_price']
                
            pnl_usdt = (pos['amount'] * pos['entry_price']) * pnl_pct
            self.balance += (pos['amount'] * pos['entry_price']) + pnl_usdt
            
            duration = (datetime.now() - pos['entry_time']).total_seconds() / 60
            logger.info(f"[MOCK] CLOSE {symbol} @ {price} | {reason} | PnL: {pnl_pct*100:.2f}% (${pnl_usdt:.2f}) | Time: {duration:.0f}m")
            del self.positions[symbol]
            
        else:
            # REAL CLOSE
            side_param = 'SELL' if pos['side'] == 'LONG' else 'BUY'
            order = self._signed_request('POST', '/fapi/v1/order', {
                'symbol': symbol,
                'side': side_param,
                'type': 'MARKET',
                'quantity': pos['amount'],
                'reduceOnly': 'true'
            })
            
            if order:
                # Update balance
                await self.initialize_balance()
                logger.info(f"[REAL] CLOSE {symbol} | {reason}")
                del self.positions[symbol]
