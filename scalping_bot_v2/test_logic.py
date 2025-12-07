import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta

from main import ScalpingBot
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScalpTester")

async def test_stagnation():
    bot = ScalpingBot()
    symbol = "ETHUSDT"
    
    # Mock Position: 11 mins old, low ROI
    bot.market.positions[symbol] = {
        'symbol': symbol,
        'side': 'LONG',
        'entry_price': 3000.0,
        'amount': 1.0,
        'sl': 2900.0,
        'tp': 3100.0,
        'entry_time': datetime.now() - timedelta(minutes=11),
        'max_roi': 0.0
    }
    
    # Mock Price: Static (3001 = small profit < 2%)
    async def mock_get_price(sym):
        return 3001.0
    bot.market.get_current_price = mock_get_price
    
    # Mock Close
    closed_log = []
    async def mock_close(sym, reason):
        closed_log.append(reason)
        # Manually remove to simulate close
        if sym in bot.market.positions:
            del bot.market.positions[sym]
            
    bot.market.close_position = mock_close
    
    # Run ONE iteration of fast_loop logic
    # Since fast_loop is infinite, we can't call it directly easily without modification or async timeout.
    # But we can instantiate the task and cancel it? No, verify logic by extracting it? 
    # Or just run it for 0.6 seconds.
    
    bot.running = True
    task = asyncio.create_task(bot.fast_loop())
    await asyncio.sleep(0.6)
    bot.running = False
    try:
        await task
    except:
        pass
        
    if closed_log and "STAGNATION" in closed_log[0]:
        logger.info(f"✅ PASS: Stagnation Exit triggered. Reason: {closed_log[0]}")
    else:
        logger.error(f"❌ FAIL: Not closed. Log: {closed_log}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_stagnation())
