import sys
import os
import asyncio
import logging
from datetime import datetime

# Local import trickery isn't needed if we run FROM the dir
from main import TrendBot
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrendTester")

async def test_breakeven():
    bot = TrendBot(is_dry_run=True)
    symbol = "BTCUSDT"
    entry_price = 50000.0
    bot.market.positions[symbol] = {
        'symbol': symbol,
        'side': 'LONG',
        'entry_price': entry_price,
        'amount': 0.1,
        'sl': 45000.0,
        'tp': 60000.0,
        'entry_time': datetime.now()
    }
    
    # Mock Price: +1.6% (Trigger is > 1.5%)
    async def mock_get_price(sym):
        return 50800.0
    
    bot.market.get_current_price = mock_get_price
    
    await bot.manage_positions()
    
    pos = bot.market.positions[symbol]
    if pos['sl'] == entry_price:
        logger.info("✅ PASS: Breakeven triggered")
    else:
        logger.error(f"❌ FAIL: SL {pos['sl']} != Entry {entry_price}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_breakeven())
