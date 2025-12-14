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
from config import config

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
        
        # Cache for exchange info (Precision rules)
        self.exchange_info_cache = {}

    def _get_signature(self, params):
        """Generate HMAC SHA256 signature"""
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _get_symbol_precision(self, symbol):
        """Get quantity precision (stepSize) for symbol"""
        if not self.exchange_info_cache:
            try:
                # Fetch fresh info
                resp = requests.get(f"{self.base_url}/fapi/v1/exchangeInfo", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    for s in data['symbols']:
                        # Save stepSize (Qty) and tickSize (Price) for each symbol
                        q_step = 1.0
                        p_tick = 0.01
                        for f in s['filters']:
                            if f['filterType'] == 'LOT_SIZE':
                                q_step = float(f['stepSize'])
                            elif f['filterType'] == 'PRICE_FILTER':
                                p_tick = float(f['tickSize'])
                        
                        self.exchange_info_cache[s['symbol']] = {'q': q_step, 'p': p_tick}
            except Exception as e:
                logger.error(f"Failed to fetch Exchange Info: {e}")
        
        return self.exchange_info_cache.get(symbol, {'q': 1.0, 'p': 0.01})

    def _round_step_size(self, quantity, step_size):
        """Round quantity to nearest stepSize"""
        if step_size == 0: return quantity
        precision = int(round(-np.log10(step_size), 0))
        return float(round(quantity - (quantity % step_size), precision))

    def _round_price(self, price, tick_size):
        """Round price to nearest tickSize"""
        if tick_size == 0: return price
        precision = int(round(-np.log10(tick_size), 0))
        return float(round(price - (price % tick_size), precision))

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
                logger.error(f"Net Error: {e}")
                return None

        resp = await loop.run_in_executor(None, _req)
        
        if resp:
            try:
                if resp.status_code != 200:
                    logger.error(f"API Error {resp.status_code}: {resp.text}") # LOG FULL ERROR
                    return None
                return resp.json()
            except Exception:
                return None
        return None

    async def initialize_balance(self):
        """Get initial balance"""
        if self.is_dry_run:
            self.balance = 1000.0
        else:
            res = await self._signed_request('GET', '/fapi/v2/balance')
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
                
                # STABILITY FILTER V4:
                # 1. Volume > 50M (Top Tier)
                # 2. Change < 30% (Not Pumping)
                # 3. Change > 1% (Not Dead)
                valid_pairs = []
                for x in data:
                    if not x['symbol'].endswith('USDT'): continue
                    
                    try:
                        vol = float(x['quoteVolume'])
                        change = float(x['priceChangePercent'])
                        
                        if vol < config.MIN_VOLUME_USDT: continue
                        if abs(change) > config.MAX_DAILY_CHANGE_PCT: continue # Too volatile
                        if abs(change) < config.MIN_DAILY_CHANGE_PCT: continue # Dead
                        
                        valid_pairs.append(x)
                    except: continue

                valid_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                
                if limit:
                    return [x['symbol'] for x in valid_pairs[:limit]]
                else:
                    return [x['symbol'] for x in valid_pairs]
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
        
        # Fallback (Safe Majors)
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']

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
                await self._signed_request('POST', '/fapi/v1/leverage', {
                    'symbol': symbol,
                    'leverage': config.LEVERAGE if hasattr(config, 'LEVERAGE') else 5
                })
                # Set Margin Type (ISOLATED)
                await self._signed_request('POST', '/fapi/v1/marginType', {
                    'symbol': symbol,
                    'marginType': 'ISOLATED'
                })
            except Exception as e:
                logger.warning(f"Setup Warning (Lev/Margin): {e}") # Don't crash, but tell us why

            # 1. Place Market Order
            side_param = 'BUY' if side == 'LONG' else 'SELL'
            
            # Dynamic Precision Rounding
            info = await self._get_symbol_precision(symbol) # Returns dict {'q': step, 'p': tick}
            step_size = info['q']
            tick_size = info['p']
            
            qty_val = self._round_step_size(amount, step_size)
            qty = f"{qty_val}" # Auto format
            
            # Double check against min qty (optional but good)
            if qty_val <= 0:
                logger.error(f"Quantity {qty_val} too small for {symbol}")
                return None

            order = await self._signed_request('POST', '/fapi/v1/order', {
                'symbol': symbol,
                'side': side_param,
                'type': 'MARKET',
                'quantity': qty
            })
            
            if order and not isinstance(order, list): # Check if valid dict
                # Handle avgPrice=0 from Binance (Market orders)
                avg_price = float(order.get('avgPrice', 0.0))
                entry_price = avg_price if avg_price > 0 else price
                
                # --- PLACING HARD STOPS (SAFETY) ---
                try:
                    # Determine Exit Side
                    exit_side = 'SELL' if side == 'LONG' else 'BUY'
                    
                    # Round SL/TP Prices
                    sl_rounded = self._round_price(sl_price, tick_size)
                    tp_rounded = self._round_price(tp_price, tick_size)
                    
                    # 1. STOP LOSS
                    await self._signed_request('POST', '/fapi/v1/order', {
                        'symbol': symbol,
                        'side': exit_side,
                        'type': 'STOP_MARKET',
                        'stopPrice': f"{sl_rounded}",
                        'closePosition': 'true' # Closes entire position
                    })
                    
                    # 2. TAKE PROFIT
                    await self._signed_request('POST', '/fapi/v1/order', {
                        'symbol': symbol,
                        'side': exit_side,
                        'type': 'TAKE_PROFIT_MARKET',
                        'stopPrice': f"{tp_rounded}",
                        'closePosition': 'true' # Closes entire position
                    })
                    logger.info(f"Protective Orders (Hard SL/TP) placed for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to place Hard Stops for {symbol}: {e}")
                    # Continue anyway, bot will manage soft stops
                
                self.positions[symbol] = {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': entry_price,
                    'amount': float(qty),
                    'sl': sl_price,
                    'tp': tp_price,
                    'entry_time': datetime.now()
                }
                logger.info(f"[REAL] OPEN {side} {symbol} @ {entry_price} | Lev: {config.LEVERAGE if hasattr(config, 'LEVERAGE') else 5}x | SL: {sl_price:.4f} (-{config.STOP_LOSS_PCT}%) | TP: {tp_price:.4f}")
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
            order = await self._signed_request('POST', '/fapi/v1/order', {
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
