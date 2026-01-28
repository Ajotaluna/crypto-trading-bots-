import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WinRateTracker:
    """
    SCOREBOARD: Persists Win/Loss stats across reboots.
    """
    
    def __init__(self, stats_file="stats.json"):
        self.stats_file = stats_file
        self.stats = self.load_stats()
        
    def load_stats(self):
        """Load stats from JSON or create new"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")
        
        # Default Scheme
        return {
            'wins': 0,
            'losses': 0,
            'total_trades': 0,
            'total_pnl_usdt': 0.0,
            'max_drawdown_usdt': 0.0,
            'best_win_usdt': 0.0,
            'start_date': datetime.now().isoformat()
        }
        
    def save_stats(self):
        """Save current stats to JSON"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
            
    def record_trade(self, pnl_usdt, symbol):
        """Record trade outcome"""
        self.stats['total_trades'] += 1
        self.stats['total_pnl_usdt'] += pnl_usdt
        
        # Verify Daily Reset
        self.reset_daily_pnl_if_new_day()
        current_daily = self.stats.get('daily_pnl_usdt', 0.0)
        self.stats['daily_pnl_usdt'] = current_daily + pnl_usdt
        
        if pnl_usdt > 0:
            self.stats['wins'] += 1
            if pnl_usdt > self.stats['best_win_usdt']:
                self.stats['best_win_usdt'] = pnl_usdt
            logger.info(f"🏆 SCOREBOARD: WIN Recorded! PnL: +${pnl_usdt:.2f}")
        else:
            self.stats['losses'] += 1
            logger.info(f"💀 SCOREBOARD: LOSS Recorded. PnL: ${pnl_usdt:.2f}")
            
        self.save_stats()
        self.log_summary()
        
    def log_summary(self):
        """Log the current Scoreboard"""
        rate = 0.0
        if self.stats['total_trades'] > 0:
            rate = (self.stats['wins'] / self.stats['total_trades']) * 100
            
        logger.info(
            f"SCOREBOARD: WINS: {self.stats['wins']} | LOSSES: {self.stats['losses']} | "
            f"RATE: {rate:.1f}% | TOTAL PNL: ${self.stats['total_pnl_usdt']:.2f} | "
            f"DAILY: ${self.stats.get('daily_pnl_usdt', 0.0):.2f}"
        )

    # --- DAILY PNL CONTROL ---
    
    def reset_daily_pnl_if_new_day(self):
        """ Checks if a new UTC day has started and resets counters """
        last_day = self.stats.get('last_trade_day', "")
        current_day = datetime.utcnow().strftime('%Y-%m-%d')
        
        if last_day != current_day:
            logger.info("📅 NEW DAY DETECTED (UTC). Resetting Daily PnL Target.")
            self.stats['daily_pnl_usdt'] = 0.0
            self.stats['last_trade_day'] = current_day
            self.save_stats()
            
    def get_daily_pnl(self):
        self.reset_daily_pnl_if_new_day() # Ensure fresh
        return self.stats.get('daily_pnl_usdt', 0.0)
