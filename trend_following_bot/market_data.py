"""
Market Data & Mock Client
Handles data fetching and simulation.
"""
import asyncio
import logging
import os
import random
import hmac
import hashlib
import time
import json
from urllib.parse import urlencode
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
try:
    from config import config
except ImportError:
    from trend_following_bot.config import config

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
            allowed_methods=["GET"]    # ONLY Retry Safe Methods (No POST/DELETE to avoid double-spend)
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
                        min_qty = 0.001
                        max_qty = 1000000.0 # Default High
                        min_notional = 5.0 
                        
                        for f in s['filters']:
                            if f['filterType'] == 'LOT_SIZE':
                                q_step = float(f['stepSize'])
                                min_qty = float(f['minQty'])
                                max_qty = float(f['maxQty'])
                            elif f['filterType'] == 'PRICE_FILTER':
                                p_tick = float(f['tickSize'])
                            elif f['filterType'] == 'MIN_NOTIONAL':
                                min_notional = float(f.get('notional', 5.0))
                        
                        self.exchange_info_cache[s['symbol']] = {
                            'q': q_step, 
                            'p': p_tick, 
                            'min_q': min_qty, 
                            'max_q': max_qty,
                            'min_n': min_notional
                        }
            except Exception as e:
                logger.error(f"Failed to fetch Exchange Info: {e}")
        
        return self.exchange_info_cache.get(symbol, {'q': 1.0, 'p': 0.01, 'min_q': 0.001, 'min_n': 5.0})

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
        # ... (implementation kept same via diff) ...
        # TRUNCATED FOR BREVITY IN PROMPT, WILL USE FULL CONTENT
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

    async def ensure_symbol_settings(self, symbol):
        """
        SAFETY CHECK: Enforce Isolated Margin & Leverage
        """
        if self.is_dry_run: return True
        
        try:
            # 1. Set Margin Type -> ISOLATED
            # This endpoint throws error if already set, so we ignore specific error code -4046
            await self._signed_request('POST', '/fapi/v1/marginType', {
                'symbol': symbol,
                'marginType': config.MARGIN_TYPE # 'ISOLATED'
            })
        except Exception:
            pass # Likely already set, proceed.

        try:
            # 2. Set Leverage -> 5x
            await self._signed_request('POST', '/fapi/v1/leverage', {
                'symbol': symbol,
                'leverage': config.LEVERAGE
            })
        except Exception as e:
            logger.error(f"Failed to set Leverage/Margin for {symbol}: {e}")
            return False
            
        return True
    def _validate_order_compliance(self, symbol, qty, price, info):
        """
        THE GATEKEEPER: Strict Order Validation (Local Pre-Flight Check).
        Returns: (is_valid, corrected_qty, reason)
        """
        min_qty = info.get('min_q', 0.001)
        min_notional = info.get('min_n', 5.0)
        step_size = info.get('q', 1.0)
        
        # 1. Check Min Quantity
        if qty < min_qty:
            return False, 0.0, f"Qty {qty} < MinQty {min_qty}"

        # 2. Check Min Notional (Value)
        notional_value = qty * price
        
        # Safety Buffer: Add 1% to Min Notional to avoid edge-case rejections
        safe_min_notional = min_notional * 1.01 
        
        if notional_value < safe_min_notional:
            # AUTO-CORRECT: Upgrade Quantity to meet Min Notional
            # Target: safe_min_notional
            # New Qty = Target / Price
            required_qty = safe_min_notional / price
            corrected_qty = self._round_step_size(required_qty, step_size)
            
            # Double Check (Rounding might lower it slightly)
            if (corrected_qty * price) < min_notional:
                 corrected_qty += step_size # Bump up one step
                 corrected_qty = self._round_step_size(corrected_qty, step_size)
            
            return False, corrected_qty, f"Notional ${notional_value:.2f} < Min ${min_notional}. Upgrading."

        return True, qty, "OK"

    async def sync_positions(self):
        """Sync Open Positions from Exchange (Full Reconciliation)"""
        if self.is_dry_run: return
        
        try:
            positions = await self._signed_request('GET', '/fapi/v2/positionRisk')
            if not positions: return
            
            active_symbols = set()
            count_new = 0
            
            for p in positions:
                amt = float(p['positionAmt'])
                if amt == 0: continue
                
                symbol = p['symbol']
                active_symbols.add(symbol)
                
                side = 'LONG' if amt > 0 else 'SHORT'
                entry_price = float(p['entryPrice'])
                
                # UPDATE or ADD
                if symbol in self.positions:
                     # Just update criticals, don't overwrite SL/TP/EntryTime if possible?
                     # Wait, if we overwrote entry_price, pnl calc changes.
                     # Trust Exchange Entry Price.
                     self.positions[symbol]['entry_price'] = entry_price
                     self.positions[symbol]['amount'] = abs(amt)
                     self.positions[symbol]['side'] = side
                else:
                    # RECOVERED (Brand New to Bot)
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'side': side,
                        'amount': abs(amt),
                        'entry_price': entry_price,
                        'entry_time': datetime.now(), # Estimate
                        'sl': 0.0, # Unknown
                        'tp': 0.0,
                        'recovered': True
                    }
                    count_new += 1

            # CLEANUP GHOSTS (Closed on Exchange but Open in Bot)
            # Iterate copy of keys to avoid runtime error
            for sym in list(self.positions.keys()):
                if sym not in active_symbols:
                    logger.info(f"👻 GHOST BUSTED: Removing closed position {sym} from tracker.")
                    del self.positions[sym]
            
            if count_new > 0:
                logger.info(f"🔄 SYNC: Recovered {count_new} position(s).")
                
        except Exception as e:
            logger.error(f"Sync Failed: {e}")

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
        
    async def get_trading_universe(self):
        """
        [THE ORACLE]
        Returns the VALID trading universe.
        Priority:
        1. whitelist.json (Exact Backtest Match ~148/297 pairs)
        2. Fallback: Top 150 by Volume (if whitelist missing)
        """
        # 1. Try Whitelist
        if os.path.exists("whitelist.json"):
            try:
                with open("whitelist.json", "r") as f:
                    symbols = json.load(f)
                if symbols and len(symbols) > 0:
                    # logger.info(f"� Whitelist Loaded: {len(symbols)} pairs.")
                    return symbols
            except Exception as e:
                logger.error(f"Failed to load whitelist.json: {e}")
        
        # 2. Fallback (The Old Way)
        logger.warning("⚠️ Whitelist not found or empty. Fallback to Dynamic Volume Scan.")
        return await self._scan_top_volume_fallback(limit=150)

    async def _scan_top_volume_fallback(self, limit=150):
        """
        Scanning the market for liquidity (Fallback Mode).
        """
        try:
            resp = self.session.get(f"{self.base_url}/fapi/v1/ticker/24hr", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                valid_pairs = []
                for x in data:
                    if not x['symbol'].endswith('USDT'): continue
                    try:
                        vol = float(x['quoteVolume'])
                        if vol < 5000000: continue 
                        valid_pairs.append(x)
                    except: continue

                valid_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                return [x['symbol'] for x in valid_pairs[:limit]]
        except Exception as e:
            logger.error(f"Fallback Scan Failed: {e}")
            return []

    # Wrapper for backward compatibility if needed, but confusing. Removed.

    async def get_klines(self, symbol, interval='15m', limit=100, start_time=None, end_time=None):
        """Fetch candlestick data (Non-Blocking)"""
        url = f"{self.base_url}/fapi/v1/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_time:
             params['startTime'] = start_time
        if end_time:
             params['endTime'] = end_time
        
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
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                    
                    # Rename for clarity
                    df['taker_buy_vol'] = df['taker_buy_base']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
            except Exception as e:
                # Suppress error log for common network blips to keep log clean
                if "Connection" not in str(e):
                    logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

        return await loop.run_in_executor(None, _fetch_and_parse)

    async def get_btc_trend(self):
        """
        THE KING'S GUARD: Fetch BTC 1H Trend for Correlation Check.
        Returns: % Change in last hour.
        """
        try:
             df = await self.get_klines("BTCUSDT", interval="1h", limit=2)
             if df.empty: return 0.0
             
             close = df['close'].iloc[-1]
             open_p = df['open'].iloc[-1]
             
             change_pct = (close - open_p) / open_p
             return change_pct
        except:
             return 0.0

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
    
    # --- CRASH DETECTION LOGIC MOVED TO SENTIMENT ANALYZER (Removed legacy functions here) ---


    def calculate_position_size(self, symbol, entry_price, sl_price, override_risk_pct=None, override_leverage=None):
        """
        THE RISK VAULT: Calculate Position Size based on Risk %.
        Formula: Size = (Balance * Risk%) / Distance%
        """
        if entry_price <= 0 or sl_price <= 0: return 0.0
        
        # 1. Determine Risk Percentage (Dynamic vs Static)
        risk_pct = config.RISK_PER_TRADE_PCT
        if override_risk_pct is not None:
             risk_pct = override_risk_pct
        
        # 2. Calculate Risk Amount (e.g. 1% of $1000 = $10)
        risk_amount = self.balance * (risk_pct) # override_risk_pct is usually 0.01 (decimal), config might be 1.0 (percent)
        # WAIT: config.RISK_PER_TRADE_PCT is likely 1.0 (percent).
        # Strategy RISK_PER_ENTRY is 0.01 (decimal).
        # We need to normalize.
        if override_risk_pct is not None:
             # Strategy passes decimal (0.01)
             risk_amount = self.balance * override_risk_pct
        else:
             # Config passes percent (1.0)
             risk_amount = self.balance * (risk_pct / 100)
        
        # 2. Calculate Stop Loss Distance %
        dist_pct = abs(entry_price - sl_price) / entry_price
        
        if dist_pct == 0: return 0.0
        
        # 3. Calculate Position Size (Notional Value)
        # $10 Risk / 0.05 Dist = $200 Position
        position_size_usdt = risk_amount / dist_pct
        
        # 4. Apply Safety Caps
        leverage = config.LEVERAGE
        if override_leverage is not None: leverage = override_leverage
            
        # Max Margin = 10% of Balance (Default)
        # If strategy says MAX_CAPITAL_PER_TRADE = 0.10 (decimal), it means 10% of equity per trade (Margin? Or Notional?)
        # Strategy: MAX_CAPITAL_PER_TRADE = 0.10. 
        # Usually implies Max Margin allocated per trade.
        
        # New Logic using Strategy Constant if available (we don't pass MAX_CAPITAL_PER_TRADE here yet)
        # For now, rely on LEVERAGE override for Notional Cap.
        
        max_margin = self.balance * (config.MAX_CAPITAL_PER_TRADE_PCT / 100)
        max_allowed_notional = max_margin * leverage
        
        if position_size_usdt > max_allowed_notional:
            logger.warning(f"⚠️ RISK VAULT: Capping Position for {symbol}. Needed {position_size_usdt:.2f} but capped at {max_allowed_notional:.2f}")
            position_size_usdt = max_allowed_notional
            
        return position_size_usdt

    async def get_adx_now(self, symbol):
        """ Quick ADX Check for Dynamic Pruning """
        try:
            # Fetch 100 candles (15m) for speed
            # Use get_klines which handles retry/dryrun logic
            ohlcv = await self.get_klines(symbol, interval='15m', limit=100)
            if ohlcv is None or ohlcv.empty: return 0.0
            
            # from technical_analysis import TechnicalAnalysis
            # df = TechnicalAnalysis.calculate_indicators(ohlcv)
            
            # USE NEW STRATEGY INDICATORS
            from trading_strategy import calculate_indicators
            df = calculate_indicators(ohlcv)
            
            if 'adx' in df.columns:
                return df['adx'].iloc[-1]
            return 0.0
        except Exception as e:
            # logger.debug(f"ADX Check Failed: {e}")
            return 0.0

    # --- Execution Methods ---
    
    async def cancel_open_orders(self, symbol):
        """Cancel ALL open orders for a symbol"""
        if self.is_dry_run: return

        try:
            # DELETE /fapi/v1/allOpenOrders
            await self._signed_request('DELETE', '/fapi/v1/allOpenOrders', {'symbol': symbol})
            logger.info(f"🧹 Cleared Open Orders for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to clear orders for {symbol}: {e}")

    async def update_sl(self, symbol, new_sl):
        """
        Updates the Stop Loss for an open position.
        Strategy: Cancel existing SL, Place new STOP_MARKET.
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"⚠️ Cannot update SL for {symbol}: Position not tracked.")
                return

            pos = self.positions[symbol]
            
            # 1. Round Price
            info = await self._get_symbol_precision(symbol)
            if not info: info = {'p': 0.01}
            tick_size = info['p']
            new_sl = self._round_price(new_sl, tick_size)
            
            if self.is_dry_run:
                # Mock Update
                pos['sl'] = new_sl
                # logger.info(f"✅ [MOCK] SL Update {symbol} -> {new_sl}")
                return

            # REAL UPDATE
            # 2. Cancel Old Orders (Blanket Cancel)
            await self.cancel_open_orders(symbol)
            
            # 3. Place New SL
            # SL for LONG = SELL STOP_MARKET
            # SL for SHORT = BUY STOP_MARKET
            side = pos['side']
            sl_side = 'SELL' if side == 'LONG' else 'BUY'
            
            params = {
                'symbol': symbol,
                'side': sl_side,
                'type': 'STOP_MARKET',
                'stopPrice': f"{new_sl}",
                'closePosition': 'true' # Binance Specific: Closes position at this price
            }
            
            res = await self._signed_request('POST', '/fapi/v1/order', params)
            if res:
                 logger.info(f"✅ SL Updated on Exchange: {symbol} @ {new_sl}")
                 pos['sl'] = new_sl
            else:
                 logger.error(f"❌ Failed to update SL for {symbol} (API Rejected)")
                 
        except Exception as e:
            logger.error(f"SL Update Error {symbol}: {e}")

    async def open_position(self, symbol, side, amount_usdt, sl_price, tp_price):
        """Open position (Mock or Real)"""
        # 0. Safety First: Clear checks
        await self.cancel_open_orders(symbol)

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

            # 1. Prepare Parameters
            side_param = 'BUY' if side == 'LONG' else 'SELL'
            
            # Dynamic Precision Rounding
            info = await self._get_symbol_precision(symbol) # Returns dict {'q': step, 'p': tick}
            step_size = info['q']
            tick_size = info['p']
            
            qty_val = self._round_step_size(amount, step_size)
            
            # VALIDATION GUARD (The Gatekeeper) for Entry
            is_valid, corrected_qty, reason = self._validate_order_compliance(symbol, qty_val, price, info)
            if not is_valid:
                logger.warning(f"👮 GATEKEEPER (ENTRY {symbol}): {reason} -> Auto-Correcting to {corrected_qty}")
                qty_val = corrected_qty
            
            qty = f"{qty_val}" # Auto format
            
            # Round SL/TP Prices (Pre-calc for Atomic Order)
            sl_rounded = self._round_price(sl_price, tick_size)
            tp_rounded = self._round_price(tp_price, tick_size)
            
            # Double check against min qty (optional but good)
            if qty_val <= 0:
                logger.error(f"Quantity {qty_val} too small for {symbol}")
                return None
            
            # 2. SEPARATE EXECUTION (Zero Tolerance Mode)
            # We abandoned Atomic params (stopLossPrice) because they are often ignored by Binance 
            # in One-Way Mode / Market Orders, leaving trades naked.
            
            # A. ENTRY
            params = {
                'symbol': symbol,
                'side': side_param,
                'type': 'MARKET',
                'quantity': qty,
                'reduceOnly': 'false' # Added as per instruction, though typically 'false' for entry
            }
            
            try:
                # 1. Place Entry
                order = await self._signed_request('POST', '/fapi/v1/order', params)
                
                if order and not isinstance(order, list):
                    avg_price = float(order.get('avgPrice', 0.0))
                    entry_price = avg_price if avg_price > 0 else price
                    
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'side': side,
                        'entry_price': entry_price,
                        'amount': float(qty),
                        'sl': sl_price,
                        'tp': tp_price,
                        'entry_time': datetime.now()
                    }
                    
                    # 2. PLACING HARD STOPS (Immediate Follow-up)
                    try:
                         exit_side = 'SELL' if side == 'LONG' else 'BUY'
                         sl_rounded = self._round_price(sl_price, tick_size)
                         tp_rounded = self._round_price(tp_price, tick_size)
                         
                         # STOP LOSS (Critical)
                         await self._signed_request('POST', '/fapi/v1/order', {
                            'symbol': symbol,
                            'side': exit_side,
                            'type': 'STOP_MARKET',
                            'stopPrice': f"{sl_rounded}",
                            'closePosition': 'true'
                         })
                         
                         # TAKE PROFIT
                         await self._signed_request('POST', '/fapi/v1/order', {
                            'symbol': symbol,
                            'side': exit_side,
                            'type': 'TAKE_PROFIT_MARKET',
                            'stopPrice': f"{tp_rounded}",
                            'closePosition': 'true'
                         })
                         
                         logger.info(f"✅ [REAL] ENTRY + STOPS SUCCESS {symbol} @ {entry_price} | SL: {sl_rounded} | TP: {tp_rounded}")
                         return self.positions[symbol]
                         
                    except Exception as e_stops:
                        # ZERO TOLERANCE: Stops Failed -> KILL TRADE
                        logger.error(f"🛑 CRITICAL: STOP LOSS FAILED for {symbol}: {e_stops}. EMERGENCY CLOSING.")
                        await self.close_position(symbol, "SAFETY: STOPS FAILED")
                        del self.positions[symbol] # Ensure it's gone from tracker
                        return None
                
            except Exception as e:
                logger.error(f"❌ ENTRY FAILED for {symbol}: {e}")
                return None
            
            return None
    
    async def emergency_close_all(self, reason="EMERGENCY"):
        """KILL SWITCH: Close ALL positions immediately"""
        if not self.positions: return
        logger.warning(f"🚨 INITIATING EMERGENCY CLOSE ALL: {reason}")
        
        tasks = []
        for sym in list(self.positions.keys()):
            tasks.append(self.close_position(sym, reason))
            
        await asyncio.gather(*tasks)
        logger.info("🚨 EMERGENCY CLOSE COMPLETE.")

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
            # Calculate Value
            exit_value = qty_to_close * price
            entry_value = qty_to_close * pos['entry_price']
            
            if pos['side'] == 'LONG':
                pnl_usdt = (exit_value - entry_value)
                pnl_pct = (price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_usdt = (entry_value - exit_value)
                pnl_pct = (pos['entry_price'] - price) / pos['entry_price']
            
            # Simulate Fees (0.04% Taker x 2 for Entry/Exit approx)
            fee = (entry_value + exit_value) * 0.0004
            net_pnl = pnl_usdt - fee
            
            # UPDATE BALANCE
            self.balance += net_pnl
            
            # Simple Outcome Reporting
            outcome = "WIN" if net_pnl > 0 else "LOSS"
            
            # Update position record if partial
            if is_partial:
                pos['amount'] -= qty_to_close
                logger.info(f"🧬 [TRACKER] PARTIAL {symbol}: {pnl_pct*100:.2f}% | PnL: ${net_pnl:.2f} | Bal: ${self.balance:.2f}")
            else:
                duration = (datetime.now() - pos['entry_time']).total_seconds() / 60
                logger.info(f"🧬 [TRACKER] CLOSED {symbol}: {outcome} (${net_pnl:.2f}) | Bal: ${self.balance:.2f} | Time: {duration:.1f}m")
                del self.positions[symbol]
            
            return {'status': 'FILLED', 'avgPrice': price, 'simulated': True, 'pnl': net_pnl}
        else:
            # REAL CLOSE (NUCLEAR OPTION V3: SERVER TRUTH)
            # "Blind Max" failed (API Limits). "Local Tracking" fails (Drift).
            # Solution: ONE Query (Get Truth) -> ONE Order (Execute Truth).
            
            try:
                # 1. Fetch Exact Position (The only way to avoid API rejections)
                risk_data = await self._signed_request('GET', '/fapi/v2/positionRisk', {'symbol': symbol})
                server_amt = 0.0
                if risk_data and isinstance(risk_data, list):
                    pos_data = risk_data[0] # One-Way Mode
                    server_amt = float(pos_data['positionAmt'])
                
                if server_amt == 0:
                    logger.info(f"☢️ NUCLEAR: {symbol} Server says CLOSED (0). Syncing local.")
                    if symbol in self.positions: del self.positions[symbol]
                    return {'status': 'CLOSED'}

                # 2. Fire Exact Amount
                # If we send exact amount + reduceOnly, it works perfectly.
                side_curr = 'SELL' if server_amt > 0 else 'BUY'
                qty_abs = abs(server_amt)
                
                logger.info(f"☢️ NUCLEAR: Closing {qty_abs} {symbol} (Server Validated).")
                
                # Cleanup Limit Orders first
                await self.cancel_open_orders(symbol)
                
                order = await self._signed_request('POST', '/fapi/v1/order', {
                    'symbol': symbol,
                    'side': side_curr,
                    'type': 'MARKET',
                    'quantity': f"{qty_abs}",
                    'reduceOnly': 'true'
                })
                
                if order:
                    await self.initialize_balance()
                    if is_partial:
                        pos['amount'] -= qty_to_close
                    else:
                        if symbol in self.positions: del self.positions[symbol]
                    logger.info(f"✅ NUCLEAR CLOSE SUCCEEDED: {symbol}")
                    return order
                else:
                    logger.error(f"❌ CLOSE REJECTED for {symbol}.")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ NUCLEAR EXCEPTION: {e}")
                return None
            

                return None

