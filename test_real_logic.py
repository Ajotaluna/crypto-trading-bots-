import asyncio
import logging
import os
import sys

# Add project root and bot directory to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'trend_following_bot'))

from trend_following_bot.market_data import MarketData
from config import config

# Configure logging to show everything
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DeepDiag")

# Mock Config if needed or ensure env vars
os.environ['BOT_TYPE'] = 'trend'
os.environ['DRY_RUN'] = 'false' # FORCE REAL

async def test_logic():
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ NO API KEYS FOUND IN ENV")
        return

    logger.info("--- 1. INITIALIZING MARKET DATA (REAL MODE) ---")
    market = MarketData(is_dry_run=False, api_key=api_key, api_secret=api_secret)
    await market.initialize_balance()
    logger.info(f"Balance: {market.balance}")

    symbol = "COAIUSDT" # The problematic coin
    logger.info(f"\n--- 2. ANALYZING {symbol} ---")
    
    # 1. Fetch Precision
    info = await market._get_symbol_precision(symbol)
    logger.info(f"Precision Info: {info}")
    
    # 2. Get Price
    price = await market.get_current_price(symbol)
    logger.info(f"Current Price: {price}")
    
    if price == 0:
        logger.error("Price is 0, aborting.")
        return

    # 3. Simulate Math
    amount_usdt = 15.0 # Test amount
    amount = amount_usdt / price
    
    logger.info(f"Raw Amount: {amount} (for ${amount_usdt})")
    
    rounded_qty = market._round_step_size(amount, info['q'])
    logger.info(f"Rounded Qty: {rounded_qty} (Step: {info['q']})")
    
    # 4. Check Min Notional (Approx)
    notional = rounded_qty * price
    logger.info(f"Est. Notional Value: ${notional:.2f}")
    if notional < 5.0:
        logger.warning("⚠️ WARNING: Notional < $5.0. Binance might reject this.")

    # 5. ATTEMPT EXECUTION
    logger.info("\n--- 3. ATTEMPTING EXECUTION (REAL ORDER) ---")
    logger.warning("!!! THIS WILL OPEN A TRADE IF SUCCESSFUL !!!")
    
    # SL/TP simulation
    sl = price * 0.98
    tp = price * 1.02
    
    # Call open_position directly to test the method
    result = await market.open_position(symbol, 'LONG', amount_usdt, sl, tp)
    
    if result:
        logger.info("✅ SUCCESS! Order executed.")
        logger.info(f"Result: {result}")
    else:
        logger.error("❌ FAILURE. Inspect logs above for API rejection reason.")

if __name__ == "__main__":
    asyncio.run(test_logic())
