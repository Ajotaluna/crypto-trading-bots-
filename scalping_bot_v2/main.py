"""
Scalping Bot V2 - MAIN
Sub-second monitoring loop.
"""
import asyncio
import logging
import sys
from config import config
from market_data import MarketData
from strategy import ScalperStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("ScalperV2")

import concurrent.futures
from datetime import datetime

class ScalpingBot:
    def __init__(self):
        self.market = MarketData()
        self.strategy = ScalperStrategy()
        self.running = True
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

    def __del__(self):
        self.executor.shutdown(wait=False)
        
    async def start(self):
        logger.info(">>> SCALPING BOT V2 STARTED <<<")
        logger.info(f"Fee Settings: Maker {config.CALC_FEE_MAKER*100}% | Taker {config.CALC_FEE_TAKER*100}%")
        await asyncio.gather(self.fast_loop(), self.scan_loop())
        
    async def fast_loop(self):
        """Sub-second monitor for exits"""
        while self.running:
            for symbol, pos in list(self.market.positions.items()):
                price = await self.market.get_current_price(symbol)
                if price == 0: continue
                
                # Check Duration
                duration_sec = (datetime.now() - pos['entry_time']).total_seconds()
                
                # Calculate ROI
                if pos['side'] == 'LONG':
                    roi = ((price - pos['entry_price']) / pos['entry_price']) * config.LEVERAGE * 100
                else:
                    roi = ((pos['entry_price'] - price) / pos['entry_price']) * config.LEVERAGE * 100
                
                # Update Max ROI for Trailing
                if roi > pos['max_roi']: pos['max_roi'] = roi
                
                # 1. Stop Loss
                if roi <= config.STOP_LOSS_ROI:
                    await self.market.close_position(symbol, f"STOP LOSS ({roi:.2f}%)")
                    continue
                    
                # 2. Take Profit
                if roi >= config.TAKE_PROFIT_ROI:
                    await self.market.close_position(symbol, f"TAKE PROFIT ({roi:.2f}%)")
                    continue
                    
                # 3. Trailing Stop
                if config.USE_TRAILING and pos['max_roi'] >= config.TRAILING_ACTIVATION_ROI:
                    if roi < (pos['max_roi'] - config.TRAILING_DISTANCE_ROI):
                        await self.market.close_position(symbol, f"TRAILING STOP (Max {pos['max_roi']:.2f}%)")
                        continue

                # 4. Stagnation Exit (Opportunity Cost)
                # If > 10 mins and ROI < 2% (Stuck), close it
                if duration_sec > 600 and roi < 2.0:
                    await self.market.close_position(symbol, f"STAGNATION (Time {duration_sec/60:.1f}m)")
                    continue

                # 5. Max Time Limit
                if duration_sec > config.MAX_HOLD_SECONDS:
                    await self.market.close_position(symbol, "MAX TIME LIMIT")
                        
            await asyncio.sleep(config.CHECK_INTERVAL)

    async def scan_loop(self):
        """Market Scanner"""
        loop = asyncio.get_running_loop()
        
        while self.running:
            if len(self.market.positions) < config.MAX_OPEN_POSITIONS:
                symbols = await self.market.get_top_vol_symbols()
                
                tasks = []
                # Fetch first
                pending_symbols = []
                for symbol in symbols[:30]:
                    if symbol in self.market.positions: continue
                    pending_symbols.append(symbol)
                
                if not pending_symbols: 
                    await asyncio.sleep(config.SCAN_INTERVAL)
                    continue

                for symbol in pending_symbols:
                    df = await self.market.get_klines(symbol, interval=config.TIMEFRAME)
                    if df.empty: continue
                    
                    # Parallel Analyze
                    tasks.append(
                        loop.run_in_executor(self.executor, self.strategy.analyze, df)
                    )
                
                if tasks:
                    results = await asyncio.gather(*tasks)
                    
                    for i, signal in enumerate(results):
                         # Re-map result to symbol (assuming order preservation in gather)
                         # This implies synchronous iteration order which is true for tasks list
                        if signal and signal['score'] >= config.MIN_SCORE:
                             # Re-fetch price for accuracy or use last close
                             # Ideally we pass 'df' to callback, but strategy just returns signal.
                             # We can infer symbol from loop index if we tracked it, 
                             # but let's assume 'analyze' returns minimal info. 
                             # Since we process 'pending_symbols' in order, we can map:
                             # Wait, pending_symbols loop fetched DF. 
                             # We need to map back. simpler:
                             pass # Logic complexity. Let's fix loop structure below.

                # Correct Parallel Loop Structure
                # We need to pair Symbol -> DF -> Task
                scan_candidates = []
                for symbol in pending_symbols:
                    df = await self.market.get_klines(symbol, interval=config.TIMEFRAME)
                    if not df.empty:
                        scan_candidates.append({'symbol': symbol, 'df': df})
                
                if scan_candidates:
                    logger.info(f"Analyzing {len(scan_candidates)} candidates...")
                    # Run analyses
                    analysis_tasks = [
                        loop.run_in_executor(self.executor, self.strategy.analyze, c['df']) 
                        for c in scan_candidates
                    ]
                    results = await asyncio.gather(*analysis_tasks)
                    
                    for i, signal in enumerate(results):
                        sym = scan_candidates[i]['symbol']
                        if signal:
                           logger.info(f"Analyzed {sym}: Score {signal['score']}")
                        else:
                           pass # Too verbose to log failures
                        
                        if signal and signal['score'] >= config.MIN_SCORE:
                        if signal and signal['score'] >= config.MIN_SCORE:
                            cand = scan_candidates[i]
                            # Execute
                            price = cand['df'].iloc[-1]['close']
                            move_pct = abs(config.STOP_LOSS_ROI / config.LEVERAGE / 100)
                            
                            sl, tp = 0, 0
                            if signal['direction'] == 'LONG':
                                sl = price * (1 - move_pct)
                                tp = price * (1 + (abs(config.TAKE_PROFIT_ROI) / config.LEVERAGE / 100))
                            else:
                                sl = price * (1 + move_pct)
                                tp = price * (1 - (abs(config.TAKE_PROFIT_ROI) / config.LEVERAGE / 100))
                                
                            amount = self.market.balance * (config.CAPITAL_PER_TRADE_PCT / 100)
                            await self.market.open_position(cand['symbol'], signal['direction'], amount, sl, tp)

            await asyncio.sleep(config.SCAN_INTERVAL)

if __name__ == "__main__":
    bot = ScalpingBot()
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt: pass
