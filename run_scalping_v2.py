"""
RUN SCALPER V2
Launcher for the High-Frequency Scalping Bot.
"""
import sys
import os
import asyncio

# Add bot directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scalping_bot_v2'))

from main import ScalpingBot

print("="*60)
print("SCALPING BOT V2 - HIGH FREQUENCY")
print("="*60)
print("Strategy: StochRSI + BB Mean Reversion")
print("Leverage: 10x")
print("Fees Included: Yes (0.05% / 0.045%)")
print("="*60)

if __name__ == "__main__":
    bot = ScalpingBot()
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nBot stopped.")
