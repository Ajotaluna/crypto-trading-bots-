"""
Async Orderbook Streamer & Analyzer for Binance Futures.

Maintains live connections to Binance WebSockets ONLY for the symbols 
currently shortlisted by the RawMarketScanner (or active trades).
Calculates real-time Orderbook Imbalance (OBI) and detects Whale Walls.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
import aiohttp

try:
    import websockets
except ImportError:
    print("Warning: 'websockets' module not found. Run 'pip install websockets'")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrderbookStreamer")

class OrderbookStreamer:
    """
    Maintains active Data Streams for the TOP N candidates.
    If the bot wants to know if a Breakout is false, it asks this class.
    """
    def __init__(self, depth=20, update_speed="100ms"):
        # We must use COMBINED streams to get the stream name back so we know which symbol it is.
        self.base_url = "wss://fstream.binance.com/stream"
        self.depth = depth
        self.update_speed = update_speed
        
        # State
        self.active_symbols = set()
        self.ws_connection = None
        self.running = False
        
        # In-Memory Live Order Books
        # format: self.books[symbol] = {'bids': [(price, qty), ...], 'asks': [(price, qty), ...], 'updated': time.time(), 'deep_updated': 0}
        self.books = defaultdict(lambda: {'bids': [], 'asks': [], 'updated': 0, 'deep_updated': 0})
        
        # Analysis Settings
        # A single price level must hold > 35% of the entire visible book length to be considered a wall.
        self.wall_concentration_pct = 0.35
        # Minimum distance (%) to ignore statistical noise right near the spread
        self.min_wall_dist_pct = 0.15

    async def _manage_subscriptions(self, websocket, subscribe_list, unsubscribe_list):
        """Sends JSON payload to Binance to turn streams on/off."""
        if not websocket:
            return
            
        streams_to_add = [f"{sym.lower()}@depth{self.depth}@{self.update_speed}" for sym in subscribe_list]
        streams_to_remove = [f"{sym.lower()}@depth{self.depth}@{self.update_speed}" for sym in unsubscribe_list]

        if streams_to_add:
            sub_payload = {
                "method": "SUBSCRIBE",
                "params": streams_to_add,
                "id": int(time.time() * 1000)
            }
            await websocket.send(json.dumps(sub_payload))
            logger.info(f"Subscribed to Orderbooks: {subscribe_list}")
            
        if streams_to_remove:
            unsub_payload = {
                "method": "UNSUBSCRIBE",
                "params": streams_to_remove,
                "id": int(time.time() * 1000) + 1
            }
            await websocket.send(json.dumps(unsub_payload))
            logger.info(f"Unsubscribed from Orderbooks: {unsubscribe_list}")

    async def update_focus_list(self, new_symbols: list):
        """
        Called by the main bot every 15m. Updates the list of symbols we are listening to.
        Does not disconnect the websocket, just alters the subscriptions dynamically.
        """
        new_set = set(new_symbols)
        
        to_add = new_set - self.active_symbols
        to_remove = self.active_symbols - new_set
        
        self.active_symbols = new_set
        
        # Clean up old data from memory
        for sym in to_remove:
            if sym in self.books:
                del self.books[sym]
                
        if self.ws_connection and self.running:
            await self._manage_subscriptions(self.ws_connection, list(to_add), list(to_remove))

    async def _fetch_deep_book(self, session, symbol):
        """Fetches 500 levels of depth via REST API for distant whales."""
        try:
            url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol.upper()}&limit=500"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    bids = [(float(p), float(q)) for p, q in data.get('bids', [])]
                    asks = [(float(p), float(q)) for p, q in data.get('asks', [])]
                    
                    # Merge deep data if it's fresher or if websocket hasn't updated recently
                    if symbol in self.books:
                        self.books[symbol]['bids'] = bids
                        self.books[symbol]['asks'] = asks
                        self.books[symbol]['updated'] = time.time()
                        self.books[symbol]['deep_updated'] = time.time()
        except Exception as e:
            logger.debug(f"Deep Book Fetch Error {symbol}: {e}")

    async def _deep_polling_loop(self):
        """Periodically requests the full 500-level depth for active symbols."""
        while self.running:
            try:
                if self.active_symbols:
                    async with aiohttp.ClientSession() as session:
                        tasks = [self._fetch_deep_book(session, sym) for sym in self.active_symbols]
                        await asyncio.gather(*tasks)
                # Parse deep books every 5 seconds (avoid API bans)
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Deep Polling Error: {e}")
                await asyncio.sleep(5)

    async def run_stream(self):
        """
        The infinite loop that connects to Binance and listens forever.
        Also starts the deep polling background task.
        """
        self.running = True
        logger.info("Opening WebSocket connection to Binance Futures OB...")
        
        # Start background deep book poller
        asyncio.create_task(self._deep_polling_loop())
        
        while self.running:
            try:
                # We connect with no initial streams, we add them dynamically
                async with websockets.connect(self.base_url) as websocket:
                    self.ws_connection = websocket
                    
                    # If we already have symbols in mind when we connect (like after a restart drop)
                    if self.active_symbols:
                        await self._manage_subscriptions(websocket, list(self.active_symbols), [])
                    
                    while self.running:
                        msg = await websocket.recv()
                        data = json.loads(msg)
                        
                        # Process Depth Update
                        if 'e' in data and data['e'] == 'depthUpdate':
                            symbol = data['s']
                            # Binance partial depth format:
                            # {"e":"depthUpdate","E":1234,"T":1234,"s":"BTCUSDT","b":[["100.0","10"]],"a":[["100.1","10"]]}
                            # Since we subscribe to @depth10, it sends snapshots. 
                            # Wait, @depthX@100ms sends snapshots with "lastUpdateId", "bids", "asks".
                            pass
                            
                        if 'bids' in data and 'asks' in data:
                            # It's a partial depth snapshot (no 'e' field)
                            # E.g. {"lastUpdateId": ..., "bids": [...], "asks": [...]}
                            # We need to extract the symbol, but Binance's basic snapshot stream doesn't include the symbol!
                            # Workaround: We must parse the stream name from the wrapper if we use combined streams.
                            # Combined stream payload: {"stream": "btcusdt@depth10@100ms", "data": {"lastUpdateId":..., "bids":...}}
                            pass

                        if 'stream' in data and 'data' in data:
                            payload = data['data']
                            
                            # Futures partial depth stream puts bids/asks in 'b' and 'a'. 
                            # If it puts them in 'bids' and 'asks', handle both.
                            bids = payload.get('b') or payload.get('bids')
                            asks = payload.get('a') or payload.get('asks')
                            
                            if bids is not None and asks is not None:
                                # Standardize payload for our parser
                                standard_payload = {'bids': bids, 'asks': asks}
                                self.process_depth_message(data['stream'], standard_payload)
                        else:
                            # It's not a depth message (could be a connection success message)
                            pass
                            
            except Exception as e:
                logger.error(f"WebSocket Error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    def process_depth_message(self, stream_name, payload):
        """Parses the binance message and updates the local book memory."""
        # e.g. stream_name = "btcusdt@depth10@100ms"
        try:
            symbol = stream_name.split('@')[0].upper()
            
            bids = [(float(p), float(q)) for p, q in payload.get('bids', [])]
            asks = [(float(p), float(q)) for p, q in payload.get('asks', [])]
            
            self.books[symbol]['bids'] = bids
            self.books[symbol]['asks'] = asks
            self.books[symbol]['updated'] = time.time()
        except:
             pass

    # ================================================================
    # ANALYTICAL METHODS (Called synchronously or asynchronously by bot)
    # ================================================================

    def get_orderbook_imbalance(self, symbol):
        """
        Calculates OBI: (Total Bid Volume - Total Ask Volume) / Total Volume
        Returns float between -1.0 (100% sellers) and +1.0 (100% buyers)
        """
        if symbol not in self.books:
            return 0.0 # Neutral if no data
            
        book = self.books[symbol]
        if time.time() - book['updated'] > 10:
             return 0.0 # Data is stale
             
        # Calculate Total USD volume
        bid_vol_usd = sum(p * q for p, q in book['bids'])
        ask_vol_usd = sum(p * q for p, q in book['asks'])
        
        total_vol = bid_vol_usd + ask_vol_usd
        if total_vol == 0:
            return 0.0
            
        # OBI: Positive means Bids > Asks (Bullish support). Negative means Asks > Bids (Bearish walls).
        return (bid_vol_usd - ask_vol_usd) / total_vol

    def get_nearest_wall(self, symbol, direction='LONG'):
        """
        Finds the closest "Wall" (concentration of limit orders > threshold).
        If moving LONG, we look for ASK walls (resistances).
        If moving SHORT, we look for BID walls (supports).
        Returns: Tuple(Price of wall, Distance in %), or (None, None)
        """
        if symbol not in self.books:
            return None, None
            
        book = self.books[symbol]
        
        # We need the current "Mid Price" roughly to calculate distance
        if not book['bids'] or not book['asks']:
            return None, None
            
        mid_price = (book['bids'][0][0] + book['asks'][0][0]) / 2.0
        
        wall_price = None
        closest_dist = 999
        
        if direction == 'LONG':
            # Looking for Ask Walls that will block us
            total_asks_usd = sum(p * q for p, q in book['asks'])
            if total_asks_usd == 0: return None, None
            
            for price, qty in book['asks']:
                usd_val = price * qty
                if usd_val / total_asks_usd >= self.wall_concentration_pct:
                    dist_pct = ((price - mid_price) / mid_price) * 100
                    if dist_pct >= self.min_wall_dist_pct and dist_pct < closest_dist:
                        closest_dist = dist_pct
                        wall_price = price
                        
        elif direction == 'SHORT':
            # Looking for Bid Walls that will block us
            total_bids_usd = sum(p * q for p, q in book['bids'])
            if total_bids_usd == 0: return None, None
            
            for price, qty in book['bids']:
                usd_val = price * qty
                if usd_val / total_bids_usd >= self.wall_concentration_pct:
                    dist_pct = ((mid_price - price) / mid_price) * 100
                    if dist_pct >= self.min_wall_dist_pct and dist_pct < closest_dist:
                        closest_dist = dist_pct
                        wall_price = price
                        
        if wall_price is not None:
             return wall_price, closest_dist
        return None, None


# Example usage for testing standalone
async def _test():
    streamer = OrderbookStreamer(depth=20, update_speed="100ms")
    
    # Start listening in background
    task = asyncio.create_task(streamer.run_stream())
    
    # Wait a sec for connection
    await asyncio.sleep(2)
    
    # Subscribe to test pairs
    await streamer.update_focus_list(["BTCUSDT", "ETHUSDT"])
    
    # Watch live imbalance for 10 seconds
    for _ in range(10):
        obi = streamer.get_orderbook_imbalance("BTCUSDT")
        wall_p, wall_d = streamer.get_nearest_wall("BTCUSDT", 'LONG')
        
        logger.info(f"BTCUSDT OBI: {obi:+.2f} | Nearest Ask Wall: {wall_d if wall_d else 'None'}% away")
        await asyncio.sleep(1)
        
    await streamer.update_focus_list([])
    streamer.running = False
    task.cancel()

if __name__ == "__main__":
    asyncio.run(_test())
