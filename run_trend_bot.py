"""
RUN TREND BOT
Interactive launcher for the Trend Following Bot.
"""
import sys
import os
import asyncio
import getpass

# Add bot directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trend_following_bot'))

from main import TrendBot

def get_user_mode():
    print("="*60)
    print("TREND FOLLOWING BOT - LAUNCHER")
    print("="*60)
    print("Select Mode:")
    print("1. TEST MODE (Dry Run - No Real Money)")
    print("2. PRODUCTION MODE (Real Trading - Requires API Keys)")
    print("="*60)
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == '1':
            return True, None, None
        elif choice == '2':
            print("\nWARNING: You are entering PRODUCTION mode.")
            print("Real trades will be executed on Binance Futures.")
            confirm = input("Type 'CONFIRM' to proceed: ")
            if confirm != 'CONFIRM':
                print("Aborted.")
                sys.exit()
                
            api_key = input("\nEnter Binance API Key: ").strip()
            api_secret = input("Enter Binance API Secret: ").strip()
            
            if len(api_key) < 10 or len(api_secret) < 10:
                print("Invalid keys provided.")
                sys.exit()
                
            return False, api_key, api_secret
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    try:
        is_dry_run, key, secret = get_user_mode()
        
        bot = TrendBot(is_dry_run=is_dry_run, api_key=key, api_secret=secret)
        asyncio.run(bot.start())
        
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        input("Press Enter to exit...")
