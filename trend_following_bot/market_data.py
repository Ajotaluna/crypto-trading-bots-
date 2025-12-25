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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
        
        # --- CONNECTION SHIELD (Anti-Disconnect) ---
        self.session = requests.Session()
        retries = Retry(
            total=3,                # Retry 3 times
            backoff_factor=0.5,     # Wait 0.5s, 1s, 2s
            status_forcelist=[429, 500, 502, 503, 504], # Retry on Server Errors
            allowed_methods=["GET", "POST", "DELETE"]    # Retry on these methods
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        if not is_dry_run and (not api_key or not api_secret):
            logger.warning("Production mode requested but missing keys! Reverting to Dry Run.")
            self.is_dry_run = True
        
        # Cache for exchange info (Precision rules)
        self.exchange_info_cache = {}
        
        # Cumulative Daily PnL Tracker
        self.cumulative_pnl_daily = 0.0

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
                resp = self.session.get(f"{self.base_url}/fapi/v1/exchangeInfo", timeout=10)
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
                    return self.session.get(url, headers=headers, params=params, timeout=10)
                elif method == 'POST':
                    return self.session.post(url, headers=headers, params=params, timeout=10)
                elif method == 'DELETE':
                    return self.session.delete(url, headers=headers, params=params, timeout=10)
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
            # LIFECYCLE TRACKER MODE: Limited Funds
            # User requested $1000 to test behavior with small capital.
            self.balance = 1000.0 
            logger.info("🧬 [TRACKER] Mode Active: Balance set to $1000.0 for Limited Funds Testing.")
        else:
            res = await self._signed_request('GET', '/fapi/v2/balance')
            if res:
                for asset in res:
                    if asset['asset'] == 'USDT':
                        self.balance = float(asset['balance'])
                        break
        logger.info(f"Initial Balance: {self.balance:.2f} USDT ({'DRY RUN' if self.is_dry_run else 'PRODUCTION'})")

    async def get_real_account_status(self):
        """
        Fetch REAL-TIME Account Status directly from Binance.
        Returns exact Equity, Balance, and PnL including fees/funding.
        """
        if self.is_dry_run:
            # Fallback for dry run (shouldn't be called, but safe)
            return None
            
        try:
            # /fapi/v2/account gives detailed margin/pnl info
            res = await self._signed_request('GET', '/fapi/v2/account')
            if res:
                # Key fields: 
                # totalWalletBalance: Balance including realized PnL/Funding
                # totalUnrealizedProfit: Floating PnL of open positions
                # totalMarginBalance: Equity (Wallet + Unrealized)
                
                return {
                    'balance': float(res['totalWalletBalance']),
                    'equity': float(res['totalMarginBalance']),
                    'unrealized_pnl': float(res['totalUnrealizedProfit'])
                }
        except Exception as e:
            logger.error(f"Failed to fetch Real Account Status: {e}")
            return None
        return None
        
    async def get_top_symbols(self, limit=None):
        """Get top volume USDT pairs"""
        try:
            # Real API call for symbols to be realistic
            resp = self.session.get(f"{self.base_url}/fapi/v1/ticker/24hr", timeout=10)
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
                # Use Session for Retry
                resp = self.session.get(url, params=params, timeout=10)
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
                # Suppress error log for common network blips to keep log clean
                if "Connection" not in str(e):
                    logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

        return await loop.run_in_executor(None, _fetch_and_parse)

    async def get_current_price(self, symbol):
        """Get latest price"""
        tk = await self.get_klines(symbol, limit=1)
        if not tk.empty:
            return float(tk.iloc[-1]['close'])
        return 0.0

    async def get_open_interest(self, symbol, period='1h'):
        """
        Get Open Interest Statistics.
        Returns: { 'sumOpenInterest': float, 'sumOpenInterestValue': float }
        """
        # Note: Binance API has distinct endpoints. For simple OI, we use openInterestHist
        # But for real-time decision, 'openInterest' endpoint is better but limited.
        # Let's use openInterestHist to see the TREND of OI.
        url = f"{self.base_url}/fapi/v1/openInterestHist"
        params = {'symbol': symbol, 'period': period, 'limit': 30}
        
        loop = asyncio.get_running_loop()
        def _fetch():
            try:
                r = self.session.get(url, params=params, timeout=10)
                if r.status_code == 200: return r.json()
            except: pass
            return []
            
        data = await loop.run_in_executor(None, _fetch)
        return data

    async def get_top_long_short_ratio(self, symbol, period='1h'):
        """Get Top Traders Long/Short Ratio (Accounts)"""
        url = f"{self.base_url}/fapi/v1/topLongShortAccountRatio"
        params = {'symbol': symbol, 'period': period, 'limit': 30}
        
        loop = asyncio.get_running_loop()
        def _fetch():
            try:
                r = self.session.get(url, params=params, timeout=10)
                if r.status_code == 200: return r.json()
            except: pass
            return []
            
        return await loop.run_in_executor(None, _fetch)

    async def get_global_long_short_ratio(self, symbol, period='1h'):
        """Get Global Long/Short Ratio (The Crowd)"""
        url = f"{self.base_url}/fapi/v1/globalLongShortAccountRatio"
        params = {'symbol': symbol, 'period': period, 'limit': 30}
        
        loop = asyncio.get_running_loop()
        def _fetch():
            try:
                r = self.session.get(url, params=params, timeout=10)
                if r.status_code == 200: return r.json()
            except: pass
            return []
            
        return await loop.run_in_executor(None, _fetch)

    # --- CRASH DETECTION LOGIC MOVED TO SENTIMENT ANALYZER (Keep simple here) ---
    async def get_btc_trend(self):
        """
        Check BTCUSDT Trend (Legacy + Sentiment Wrapper needed in Main).
        Returns: 'BULLISH', 'BEARISH', 'CRASH', or 'NEUTRAL'
        """
        # 1. Check 15m (Immediate Danger)
        df_15m = await self.get_klines('BTCUSDT', interval='15m', limit=50)
        if df_15m.empty: return 'NEUTRAL' # Assume Check failed
        
        # Simple EMA calculation
        df_15m['ema_20'] = df_15m['close'].ewm(span=20, adjust=False).mean()
        current_15m = df_15m.iloc[-1]
        
        # Calculate RSI for Crash Detection
        delta = df_15m['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Calculate % Drop in last candle
        open_price = float(current_15m['open'])
        close_price = float(current_15m['close'])
        pct_change = (close_price - open_price) / open_price * 100
        
        # CRASH DETECTION: Drop > 1% in 15m OR RSI < 25 (Panic)
        if pct_change < -1.0 or current_rsi < 25:
             return 'CRASH'

        # DANGER: BTC dumping on 15m (Standard Downtrend)
        if current_15m['close'] < current_15m['ema_20']:
            return 'BEARISH'
            
        # 2. Check 1H (General Health)
        df_1h = await self.get_klines('BTCUSDT', interval='1h', limit=50)
        if not df_1h.empty:
            df_1h['ema_20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
            current_1h = df_1h.iloc[-1]
            if current_1h['close'] > current_1h['ema_20']:
                return 'BULLISH'
                
        return 'NEUTRAL'

    def calculate_position_size(self, symbol, entry_price, sl_price):
        """
        THE RISK VAULT: Calculate Position Size based on Risk %.
        Formula: Size = (Balance * Risk%) / Distance%
        """
        if entry_price <= 0 or sl_price <= 0: return 0.0
        
        # 1. Calculate Risk Amount (e.g. 1% of $1000 = $10)
        risk_amount = self.balance * (config.RISK_PER_TRADE_PCT / 100)
        
        # 2. Calculate Stop Loss Distance %
        dist_pct = abs(entry_price - sl_price) / entry_price
        
        if dist_pct == 0: return 0.0
        
        # 3. Calculate Position Size (Notional Value)
        # $10 Risk / 0.05 Dist = $200 Position
        position_size_usdt = risk_amount / dist_pct
        
        # 4. Apply Safety Caps
        max_position_usdt = self.balance * (config.MAX_CAPITAL_PER_TRADE_PCT / 100) * config.LEVERAGE 
        # Note: Max Cap is usually unleveraged % of balance, but here we cap the Notional.
        # Let's cap the MARGIN used to 25% of balance.
        # Margin Used = Position / Leverage
        
        max_margin = self.balance * (config.MAX_CAPITAL_PER_TRADE_PCT / 100)
        max_allowed_notional = max_margin * config.LEVERAGE
        
        if position_size_usdt > max_allowed_notional:
            logger.warning(f"⚠️ RISK VAULT: Capping Position for {symbol}. Needed {position_size_usdt:.2f} but capped at {max_allowed_notional:.2f}")
            position_size_usdt = max_allowed_notional
            
        return position_size_usdt

    # --- Execution Methods ---
    
    async def open_position(self, symbol, side, amount_usdt, sl_price, tp_price):
        """Open position (Mock or Real)"""
        price = await self.get_current_price(symbol)
        if price == 0: return None
        
        amount = amount_usdt / price
        
        if self.is_dry_run:
            # MOCK EXECUTION (LIFECYCLE TRACKER MODE)
            # No Balance Deduction. Infinite Funds for Strategy Testing.
            self.positions[symbol] = {
                'symbol': symbol,
                'side': side,
                'entry_price': price,
                'amount': amount,
                'sl': sl_price,
                'tp': tp_price,
                'entry_time': datetime.now()
            }
            # Log purely as a Strategy Event
            logger.info(f"🧬 [TRACKER] OPEN {side} {symbol} @ {price:.4f} | Target: {tp_price:.4f} | Stop: {sl_price:.4f} | Tracking Lifecycle...")
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

    async def close_position(self, symbol, reason, params=None):
        """Close position (Mock or Real)"""
        if symbol not in self.positions: return
        if params is None: params = {}
        
        pos = self.positions[symbol]
        price = await self.get_current_price(symbol)
        
        # Determine quantity to close (Full or Partial)
        qty_to_close = params.get('qty', pos['amount'])
        is_partial = qty_to_close < pos['amount'] * 0.99
        
        if self.is_dry_run:
            # MOCK CLOSE (LIFECYCLE TRACKER)
            if pos['side'] == 'LONG':
                pnl_pct = (price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - price) / pos['entry_price']
                
            # No Financial Math (Balance updates removed)
            
            # Simple Outcome Reporting
            outcome = "WIN" if pnl_pct > 0 else "LOSS"
            if abs(pnl_pct) < 0.001: outcome = "BREAKEVEN"
            
            # Update position record if partial
            if is_partial:
                pos['amount'] -= qty_to_close
                logger.info(f"🧬 [TRACKER] PARTIAL CLOSE {symbol} | ROI: {pnl_pct*100:.2f}% | Secured Partial.")
            else:
                duration = (datetime.now() - pos['entry_time']).total_seconds() / 60
                logger.info(f"🧬 [TRACKER] CLOSED {symbol} | Result: {outcome} ({pnl_pct*100:.2f}%) | Reason: {reason} | Time: {duration:.1f}m")
                del self.positions[symbol]
            
        else:
            # REAL CLOSE
            side_param = 'SELL' if pos['side'] == 'LONG' else 'BUY'
            
            # ReduceOnly order
            order = await self._signed_request('POST', '/fapi/v1/order', {
                'symbol': symbol,
                'side': side_param,
                'type': 'MARKET',
                'quantity': qty_to_close,
                'reduceOnly': 'true'
            })
            
            if order:
                await self.initialize_balance()
                if is_partial:
                     pos['amount'] -= qty_to_close
                     logger.info(f"[REAL] PARTIAL CLOSE {symbol} | {reason}")
                else:
                    logger.info(f"[REAL] CLOSE {symbol} | {reason}")
                    del self.positions[symbol]
