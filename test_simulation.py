import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Tester")

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'trend_following_bot'))
sys.path.append(os.path.join(os.getcwd(), 'scalping_bot_v2'))

# Imports (wrapped in try/except to handle potential path issues)
try:
    from trend_following_bot.main import TrendBot
    from trend_following_bot.config import config as trend_config
    from scalping_bot_v2.main import ScalpingBot
    from scalping_bot_v2.config import config as scalp_config
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

async def test_trend_breakeven():
    logger.info("--- TESTING TREND BOT BREAKEVEN ---")
    bot = TrendBot(is_dry_run=True)
    
    # Mock Position
    symbol = "BTCUSDT"
    entry_price = 50000.0
    bot.market.positions[symbol] = {
        'symbol': symbol,
        'side': 'LONG',
        'entry_price': entry_price,
        'amount': 0.1,
        'sl': 45000.0, # Initial SL
        'tp': 60000.0,
        'entry_time': datetime.now()
    }
    
    # Mock Market Data Price (Surge to trigger BE)
    # Risk is 1% * 5x = 5% ROI? Config has STOP_LOSS_PCT = 1.0 (Price Move)
    # Breakeven trigger: pnl_pct > (risk_pct * 1.5)
    # Risk Pct = 1.0% (0.01)
    # Trigger = 1.5% (0.015) gain
    # Target Price = 50000 * 1.016 = 50800
    
    async def mock_get_price(sym):
        return 50800.0 # +1.6% move
    
    bot.market.get_current_price = mock_get_price
    
    # Run ONE pass of manage_positions
    await bot.manage_positions()
    
    # Verify
    pos = bot.market.positions[symbol]
    if pos['sl'] == entry_price:
        logger.info("✅ PASS: Trend Bot moved SL to Breakeven!")
    else:
        logger.error(f"❌ FAIL: SL is {pos['sl']}, expected {entry_price}")

async def test_scalp_stagnation():
    logger.info("\n--- TESTING SCALP BOT STAGNATION ---")
    bot = ScalpingBot()
    
    # Mock Position (Stagnant)
    symbol = "ETHUSDT"
    entry_price = 3000.0
    # Entry 11 mins ago
    entry_time = datetime.now() - timedelta(minutes=11)
    
    bot.market.positions[symbol] = {
        'symbol': symbol,
        'side': 'LONG',
        'entry_price': entry_price,
        'amount': 1.0,
        'sl': 2900.0,
        'tp': 3100.0,
        'entry_time': entry_time,
        'max_roi': 0.0
    }
    
    # Mock Price (Sideways/Low Profit)
    # ROI < 2.0 needed.
    # Price = 3001 (+0.03% move * 10x Lev = 0.3% ROI)
    async def mock_get_price(sym):
        return 3001.0
        
    bot.market.get_current_price = mock_get_price
    
    # Mock Close
    closed_check = []
    async def mock_close(sym, reason):
        closed_check.append((sym, reason))
        
    bot.market.close_position = mock_close
    
    # Run loop logic manually (extracting logic from fast_loop is hard, so we instantiate and run once? 
    # fast_loop is a while loop. We can verify logic by copying the condition or hopefully just running checking state if we can break loop.
    # Actually, we can just call the logic block if it was separated.
    # Since it's inside fast_loop, let's duplicate the logic condition check here for unit testing, 
    # OR redefine fast_loop to run once.
    
    # Let's redefine run loop for test
    bot.running = False # Don't loop
    
    # We'll copy-paste the logic chunk for "simulation" assurance 
    # or better, use the actual object method if I refactored it. 
    # I didn't separate `manage_positions` in Scalper (I did in Trend).
    # In Scalper it's inside `fast_loop`. 
    # I'll inject a "running" toggle hack if possible, but fast_loop has `while self.running`.
    # I will start it as a task, wait 0.1s, then stop it.
    
    bot.running = True
    task = asyncio.create_task(bot.fast_loop())
    await asyncio.sleep(0.6) # Wait for one iteration (interval is 0.5)
    bot.running = False
    await task 
    
    if len(closed_check) > 0 and "STAGNATION" in closed_check[0][1]:
        logger.info(f"✅ PASS: Scalp Bot closed stagnant trade! Reason: {closed_check[0][1]}")
    else:
        logger.error(f"❌ FAIL: Trade not closed. Closed: {closed_check}")

async def main():
    await test_trend_breakeven()
    await test_scalp_stagnation()

if __name__ == "__main__":
    asyncio.run(main())
