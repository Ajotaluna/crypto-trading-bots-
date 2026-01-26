import json
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger("BlacklistManager")

class BlacklistManager:
    def __init__(self, filename="blacklist.json"):
        self.filename = filename
        self.blacklist = {} # {symbol: {'banned_until': timestamp, 'reason': str}}
        self.loss_tracker = {} # {symbol: {'losses': int, 'last_loss': timestamp}}
        self.load()

    def load(self):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                self.blacklist = data.get('blacklist', {})
                self.loss_tracker = data.get('loss_tracker', {})
        except FileNotFoundError:
            self.save()
        except Exception as e:
            logger.error(f"Failed to load blacklist: {e}")

    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump({
                    'blacklist': self.blacklist,
                    'loss_tracker': self.loss_tracker
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save blacklist: {e}")

    def record_loss(self, symbol):
        """Register a loss. 2 losses in 24h = BAN."""
        now = time.time()
        
        if symbol not in self.loss_tracker:
            self.loss_tracker[symbol] = {'losses': 0, 'last_loss': 0}
            
        tracker = self.loss_tracker[symbol]
        
        # Reset tracker if last loss was > 24h ago
        if now - tracker['last_loss'] > 86400:
            tracker['losses'] = 0
            
        tracker['losses'] += 1
        tracker['last_loss'] = now
        
        logger.warning(f"📉 LOSS RECORDED for {symbol}. Total (24h): {tracker['losses']}")
        
        if tracker['losses'] >= 2:
            self.ban_symbol(symbol, duration_hours=48, reason="2 Consecutive Losses")
            
        self.save()

    def ban_symbol(self, symbol, duration_hours=24, reason="Manual"):
        """Ban a symbol temporarily"""
        ban_until = time.time() + (duration_hours * 3600)
        self.blacklist[symbol] = {
            'banned_until': ban_until,
            'reason': reason
        }
        logger.warning(f"🚫 BANNED {symbol} for {duration_hours}h. Reason: {reason}")
        
        # PERMANENT REMOVAL FROM VIP LIST (If repeated offender)
        if "Consecutive Losses" in reason:
             try:
                 from calibration import CalibrationManager
                 cm = CalibrationManager()
                 cm.ban_candidate(symbol)
             except Exception as e:
                 logger.error(f"Failed to remove {symbol} from Config: {e}")
                 
        self.save()

    def is_allowed(self, symbol):
        """Check if symbol is allowed to trade"""
        if symbol in self.blacklist:
            info = self.blacklist[symbol]
            if time.time() < info['banned_until']:
                return False
            else:
                # Expired ban
                del self.blacklist[symbol]
                self.save()
                return True
        return True
