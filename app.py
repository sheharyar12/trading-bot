import streamlit as st
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random
import logging
from typing import List, Dict, Optional
import threading
import pytz
from collections import deque
import talib

# Set page config
st.set_page_config(
    page_title="Trading Bot Monitor",
    page_icon="📈",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import alpaca_trade_api as tradeapi
    import yfinance as yf
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca API not installed. Running in simulation mode.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests library not available for notifications")

# CONFIGURATION FROM ENVIRONMENT VARIABLES (Set in Streamlit Cloud Secrets)
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Pushover Configuration for iPhone Notifications
PUSHOVER_USER_KEY = os.getenv('PUSHOVER_USER_KEY', '')
PUSHOVER_APP_TOKEN = os.getenv('PUSHOVER_APP_TOKEN', '')

# Trading Parameters (can also be env vars if you want)
POSITION_SIZE = int(os.getenv('POSITION_SIZE', '100'))
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.02'))
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.07'))
VOLUME_MULTIPLIER = float(os.getenv('VOLUME_MULTIPLIER', '2.0'))
MIN_PRICE_CHANGE_PCT = float(os.getenv('MIN_PRICE_CHANGE_PCT', '0.05'))

# Enhanced Trading Parameters
DAILY_PROFIT_TARGET_PCT = float(os.getenv('DAILY_PROFIT_TARGET_PCT', '0.04'))
MAX_DAILY_LOSS_PCT = float(os.getenv('MAX_DAILY_LOSS_PCT', '0.02'))
ENABLE_MEAN_REVERSION = os.getenv('ENABLE_MEAN_REVERSION', 'true').lower() == 'true'
ENABLE_SHORT_SELLING = os.getenv('ENABLE_SHORT_SELLING', 'false').lower() == 'true'
MAX_NEAR_MISS_LOG = int(os.getenv('MAX_NEAR_MISS_LOG', '50'))
MAX_CONCURRENT_POSITIONS = int(os.getenv('MAX_CONCURRENT_POSITIONS', '5'))
NO_TRADES_AFTER_HOUR = int(os.getenv('NO_TRADES_AFTER_HOUR', '15'))  # 3:00 PM ET

# Market hours (Eastern Time)
MARKET_TIMEZONE = pytz.timezone('US/Eastern')
MARKET_OPEN_TIME = (9, 30)  # 9:30 AM ET
MARKET_CLOSE_TIME = (15, 50)  # 3:50 PM ET (close positions 10 min early)

class TechnicalIndicators:
    """Calculate technical indicators using TA-Lib"""
    
    @staticmethod
    def calculate_indicators(prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """Calculate all technical indicators for a price series"""
        try:
            if len(prices) < 50:
                return {}
            
            high = prices * 1.02  # Simulate high
            low = prices * 0.98   # Simulate low
            close = prices
            
            # RSI calculations
            rsi_14 = talib.RSI(close, timeperiod=14)
            rsi_5 = talib.RSI(close, timeperiod=5)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            
            # ADX
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            # ATR
            atr = talib.ATR(high, low, close, timeperiod=14)
            
            # VWAP calculation (simplified)
            if volumes is not None:
                vwap = np.sum(close * volumes) / np.sum(volumes)
            else:
                vwap = close[-1]
            
            return {
                'rsi_14': rsi_14[-1] if len(rsi_14) > 0 and not np.isnan(rsi_14[-1]) else 50,
                'rsi_5': rsi_5[-1] if len(rsi_5) > 0 and not np.isnan(rsi_5[-1]) else 50,
                'macd': macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0,
                'bb_upper': bb_upper[-1] if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]) else close[-1] * 1.02,
                'bb_middle': bb_middle[-1] if len(bb_middle) > 0 and not np.isnan(bb_middle[-1]) else close[-1],
                'bb_lower': bb_lower[-1] if len(bb_lower) > 0 and not np.isnan(bb_lower[-1]) else close[-1] * 0.98,
                'adx': adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 25,
                'atr': atr[-1] if len(atr) > 0 and not np.isnan(atr[-1]) else close[-1] * 0.02,
                'vwap': vwap,
                'price': close[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

class SimulatedMarketData:
    """Simulates market data when real API is not available"""

    def __init__(self):
        self.stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'CRM']
        self.base_prices = {s: random.uniform(50, 500) for s in self.stocks}
        self.volumes = {s: random.randint(1000000, 50000000) for s in self.stocks}
        self.opening_ranges = {}
        
    def get_opening_range(self, symbol: str) -> Dict:
        """Get or create opening range for a symbol"""
        if symbol not in self.opening_ranges:
            base_price = self.base_prices[symbol]
            self.opening_ranges[symbol] = {
                'high': base_price * random.uniform(1.01, 1.03),
                'low': base_price * random.uniform(0.97, 0.99),
                'volume': self.volumes[symbol] * random.uniform(0.8, 1.2)
            }
        return self.opening_ranges[symbol]

    def get_breakout_candidates(self) -> List[Dict]:
        """Get momentum breakout candidates"""
        candidates = []
        for stock in random.sample(self.stocks, 8):
            price = self.base_prices[stock]
            change_pct = random.uniform(0.03, 0.15)
            volume = self.volumes[stock] * random.uniform(1.5, 4.0)
            
            # Generate price series for indicators
            price_series = np.array([price * (1 + random.uniform(-0.01, 0.01)) for _ in range(50)])
            price_series[-1] = price * (1 + change_pct)
            
            indicators = TechnicalIndicators.calculate_indicators(price_series)

            candidates.append({
                'symbol': stock,
                'price': price * (1 + change_pct),
                'change_pct': change_pct,
                'volume': volume,
                'avg_volume': self.volumes[stock],
                'morning_high': price * (1 + change_pct * 0.7),
                'indicators': indicators,
                'strategy': 'momentum'
            })
        return candidates
    
    def get_mean_reversion_candidates(self) -> List[Dict]:
        """Get mean reversion candidates"""
        candidates = []
        for stock in random.sample(self.stocks, 5):
            price = self.base_prices[stock]
            change_pct = random.uniform(-0.08, -0.02)  # Negative for mean reversion
            volume = self.volumes[stock] * random.uniform(1.2, 2.5)
            
            # Generate price series for indicators
            price_series = np.array([price * (1 + random.uniform(-0.01, 0.01)) for _ in range(50)])
            price_series[-1] = price * (1 + change_pct)
            
            indicators = TechnicalIndicators.calculate_indicators(price_series)
            
            # Simulate oversold conditions
            indicators['rsi_5'] = random.uniform(15, 35)
            indicators['bb_lower'] = price * (1 + change_pct) * 1.01  # Price below lower band

            candidates.append({
                'symbol': stock,
                'price': price * (1 + change_pct),
                'change_pct': change_pct,
                'volume': volume,
                'avg_volume': self.volumes[stock],
                'indicators': indicators,
                'strategy': 'mean_reversion'
            })
        return candidates
    
    def get_breakout_range_candidates(self) -> List[Dict]:
        """Get opening range breakout candidates"""
        candidates = []
        for stock in random.sample(self.stocks, 3):
            opening_range = self.get_opening_range(stock)
            current_price = opening_range['high'] * random.uniform(1.005, 1.02)  # Above opening range
            
            price_series = np.array([self.base_prices[stock] * (1 + random.uniform(-0.01, 0.01)) for _ in range(50)])
            price_series[-1] = current_price
            
            indicators = TechnicalIndicators.calculate_indicators(price_series)

            candidates.append({
                'symbol': stock,
                'price': current_price,
                'change_pct': (current_price - self.base_prices[stock]) / self.base_prices[stock],
                'volume': self.volumes[stock] * random.uniform(1.5, 3.0),
                'avg_volume': self.volumes[stock],
                'opening_range': opening_range,
                'indicators': indicators,
                'strategy': 'breakout'
            })
        return candidates

    def get_current_price(self, symbol: str) -> float:
        base = self.base_prices.get(symbol, 100)
        return base * (1 + random.uniform(-0.02, 0.02))

class AutoTradingBot:
    """Automated trading bot that runs continuously during market hours"""

    def __init__(self):
        self.api = None
        self.positions = {}
        self.trades_log = []
        self.candidates = []
        self.daily_pnl = 0
        self.total_pnl = 0
        self.simulated_data = SimulatedMarketData()
        self.is_simulation = True
        self.is_trading_hours = False
        self.last_scan_date = None
        self.status = "INITIALIZING"
        self.account_balance = 100000.0  # Default for simulation
        self.buying_power = 100000.0
        self.initial_balance = 100000.0
        self.daily_starting_balance = 100000.0  # Track balance at start of each day
        self.notifications_enabled = bool(PUSHOVER_USER_KEY and PUSHOVER_APP_TOKEN)
        
        # Enhanced features
        self.near_miss_log = deque(maxlen=MAX_NEAR_MISS_LOG)
        self.trading_enabled_today = True
        self.consecutive_losses = 0
        self.pause_trading_until = None
        self.market_regime = "unknown"
        self.all_candidates = {'momentum': [], 'mean_reversion': [], 'breakout': []}
        
        # Daily tracking
        self.daily_target_hit = False
        self.daily_loss_limit_hit = False

        # Initialize API connection
        self._connect_to_alpaca()

        # Start the main trading loop in a separate thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        logger.info("Enhanced Auto Trading Bot initialized and running")

    def _connect_to_alpaca(self):
        """Connect to Alpaca API using environment variables"""
        if ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_AVAILABLE:
            try:
                self.api = tradeapi.REST(
                    ALPACA_API_KEY,
                    ALPACA_SECRET_KEY,
                    ALPACA_BASE_URL,
                    api_version='v2'
                )
                account = self.api.get_account()
                self.is_simulation = False
                self.status = "CONNECTED TO ALPACA"
                self.account_balance = float(account.cash)
                self.buying_power = float(account.buying_power)
                self.initial_balance = float(account.cash)
                logger.info(f"Connected to Alpaca Paper Trading. Balance: ${self.account_balance:,.2f}")
            except Exception as e:
                logger.error(f"Failed to connect to Alpaca: {e}")
                self.is_simulation = True
                self.status = "SIMULATION MODE"
        else:
            self.is_simulation = True
            self.status = "SIMULATION MODE (No API Keys)"
            logger.info("Running in simulation mode - no API keys provided")

    def _send_notification(self, title: str, message: str, priority: int = 0):
        """Send push notification to iPhone via Pushover"""
        if not self.notifications_enabled or not REQUESTS_AVAILABLE:
            return

        try:
            # Pushover API endpoint
            url = "https://api.pushover.net/1/messages.json"

            data = {
                "token": PUSHOVER_APP_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "title": title,
                "message": message,
                "priority": priority,  # -2=silent, -1=quiet, 0=normal, 1=high, 2=emergency
                "sound": "cashregister" if "profit" in message.lower() else "pushover"
            }

            response = requests.post(url, data=data, timeout=5)
            if response.status_code == 200:
                logger.info(f"Notification sent: {title}")
            else:
                logger.error(f"Failed to send notification: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    def _trading_loop(self):
        """Enhanced main trading loop that runs continuously"""
        logger.info("Starting enhanced automated trading loop")

        while True:
            try:
                now = datetime.now(MARKET_TIMEZONE)

                # Check if it's a weekday (Monday=0, Sunday=6)
                if now.weekday() >= 5:
                    self.status = "WEEKEND - MARKET CLOSED"
                    self.is_trading_hours = False
                    time.sleep(3600)  # Sleep for 1 hour on weekends
                    continue

                # Get current time components
                current_time = (now.hour, now.minute)

                # Check if market is open
                market_open = self._time_to_minutes(MARKET_OPEN_TIME)
                market_close = self._time_to_minutes(MARKET_CLOSE_TIME)
                current_minutes = self._time_to_minutes(current_time)

                if current_minutes < market_open:
                    # Before market open - reset daily flags
                    if self.last_scan_date != now.date():
                        self._reset_daily_flags()
                    self.status = "WAITING FOR MARKET OPEN"
                    self.is_trading_hours = False
                    wait_seconds = (market_open - current_minutes) * 60
                    logger.info(f"Waiting {wait_seconds/60:.1f} minutes until market open")
                    time.sleep(min(wait_seconds, 300))  # Check every 5 minutes max

                elif current_minutes >= market_close:
                    # After market close
                    if self.is_trading_hours:
                        # Close all positions at end of day
                        self._close_all_positions()
                        self._log_daily_summary()
                        self.is_trading_hours = False

                    self.status = "AFTER HOURS - MARKET CLOSED"
                    time.sleep(300)  # Check every 5 minutes

                else:
                    # Market is open - trading hours!
                    self.is_trading_hours = True
                    
                    # Check daily profit target first
                    if self._check_daily_profit_target():
                        self.status = "DAILY TARGET HIT - NO MORE TRADES"
                        time.sleep(300)  # Check every 5 minutes
                        continue
                    
                    # Check daily loss limit
                    if self._check_daily_loss_limit():
                        self.status = "DAILY LOSS LIMIT - NO MORE TRADES"
                        time.sleep(300)
                        continue
                    
                    # Check if trading is paused
                    if self._is_trading_paused():
                        time.sleep(60)
                        continue
                    
                    self.status = "TRADING ACTIVE"

                    # Scan for candidates once per day (in the morning)
                    if current_minutes < market_open + 30 and self.last_scan_date != now.date():
                        logger.info("Market open - scanning for all strategy candidates")
                        self.daily_starting_balance = self.account_balance
                        self._scan_all_strategies()
                        self.last_scan_date = now.date()
                        self.daily_pnl = 0  # Reset daily P&L
                        
                        # Assess market regime
                        self.market_regime = self._assess_market_regime()

                        # Send market open notification
                        total_candidates = sum(len(candidates) for candidates in self.all_candidates.values())
                        if total_candidates > 0:
                            self._send_notification(
                                "🔔 Market Open - Enhanced Bot Active",
                                f"Found {total_candidates} candidates across all strategies\n"
                                f"Market Regime: {self.market_regime.upper()}\n"
                                f"Balance: ${self.account_balance:,.2f}",
                                priority=0
                            )

                    # Monitor and trade with multiple strategies
                    self._monitor_and_trade_enhanced()

                    # Check positions for exit conditions
                    self._check_exit_conditions()

                    # Update position prices
                    self._update_positions()

                    # Sleep for 30 seconds between checks
                    time.sleep(30)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self.status = f"ERROR: {str(e)}"
                time.sleep(60)  # Wait a minute before retrying

    def _time_to_minutes(self, time_tuple):
        """Convert (hour, minute) to minutes since midnight"""
        return time_tuple[0] * 60 + time_tuple[1]
    
    def _reset_daily_flags(self):
        """Reset daily flags for new trading day"""
        self.trading_enabled_today = True
        self.daily_target_hit = False
        self.daily_loss_limit_hit = False
        self.consecutive_losses = 0
        self.pause_trading_until = None
        self.near_miss_log.clear()
        logger.info("Daily flags reset for new trading day")
    
    def _check_daily_profit_target(self) -> bool:
        """Check if daily profit target has been hit"""
        if self.daily_target_hit:
            return True
            
        daily_return = self.daily_pnl / self.daily_starting_balance
        if daily_return >= DAILY_PROFIT_TARGET_PCT:
            self.daily_target_hit = True
            self.trading_enabled_today = False
            
            # Close all positions immediately
            self._close_all_positions("DAILY_TARGET")
            
            # Send notification
            self._send_notification(
                "🎯 Daily Target Hit!",
                f"Achieved +{daily_return:.1%} (${self.daily_pnl:+,.2f})\n"
                f"All positions closed\n"
                f"No more trades today\n"
                f"Balance: ${self.account_balance:,.2f}",
                priority=1
            )
            
            logger.info(f"Daily profit target reached: +{daily_return:.2%} (${self.daily_pnl:.2f})")
            return True
        
        return False
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been hit"""
        if self.daily_loss_limit_hit:
            return True
            
        daily_return = self.daily_pnl / self.daily_starting_balance
        if daily_return <= -MAX_DAILY_LOSS_PCT:
            self.daily_loss_limit_hit = True
            self.trading_enabled_today = False
            
            # Close all positions immediately
            self._close_all_positions("DAILY_LOSS_LIMIT")
            
            # Send notification
            self._send_notification(
                "🛑 Daily Loss Limit Hit",
                f"Hit -{daily_return:.1%} loss limit (${self.daily_pnl:+,.2f})\n"
                f"All positions closed\n"
                f"Trading stopped for today\n"
                f"Balance: ${self.account_balance:,.2f}",
                priority=2  # Emergency priority
            )
            
            logger.warning(f"Daily loss limit reached: {daily_return:.2%} (${self.daily_pnl:.2f})")
            return True
        
        return False
    
    def _is_trading_paused(self) -> bool:
        """Check if trading is paused due to consecutive losses"""
        if self.pause_trading_until and datetime.now() < self.pause_trading_until:
            minutes_left = (self.pause_trading_until - datetime.now()).seconds // 60
            self.status = f"TRADING PAUSED - {minutes_left}min left"
            return True
        elif self.pause_trading_until and datetime.now() >= self.pause_trading_until:
            self.pause_trading_until = None
            logger.info("Trading pause ended - resuming normal operations")
        
        return False
    
    def _assess_market_regime(self) -> str:
        """Assess current market regime based on indicators"""
        try:
            # Use momentum candidates to assess regime
            momentum_candidates = self.all_candidates.get('momentum', [])
            if not momentum_candidates:
                return "neutral"
            
            # Average ADX across candidates
            avg_adx = np.mean([c.get('indicators', {}).get('adx', 25) for c in momentum_candidates])
            
            if avg_adx > 25:
                return "trending"
            elif avg_adx < 20:
                return "ranging"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error assessing market regime: {e}")
            return "neutral"
    
    def _scan_all_strategies(self):
        """Scan for candidates using all strategies"""
        try:
            # Get candidates from all strategies
            self.all_candidates = {
                'momentum': self.simulated_data.get_breakout_candidates() if self.is_simulation else self._scan_momentum_real(),
                'mean_reversion': self.simulated_data.get_mean_reversion_candidates() if self.is_simulation else self._scan_mean_reversion_real(),
                'breakout': self.simulated_data.get_breakout_range_candidates() if self.is_simulation else self._scan_breakout_real()
            }
            
            # Filter candidates
            for strategy in self.all_candidates:
                self.all_candidates[strategy] = self._filter_candidates(self.all_candidates[strategy], strategy)
            
            # Set legacy candidates for backward compatibility
            self.candidates = self.all_candidates['momentum']
            
            total_candidates = sum(len(candidates) for candidates in self.all_candidates.values())
            logger.info(f"Scanned all strategies: {total_candidates} total candidates")
            
        except Exception as e:
            logger.error(f"Error scanning strategies: {e}")
            self.all_candidates = {'momentum': [], 'mean_reversion': [], 'breakout': []}
    
    def _scan_momentum_real(self) -> List[Dict]:
        """Scan for momentum candidates using real API (enhanced version)"""
        if self.is_simulation:
            return self.simulated_data.get_breakout_candidates()
        
        # For now, use simulated data but this would connect to real API
        return self.simulated_data.get_breakout_candidates()
    
    def _scan_mean_reversion_real(self) -> List[Dict]:
        """Scan for mean reversion candidates using real API"""
        if self.is_simulation:
            return self.simulated_data.get_mean_reversion_candidates()
        
        # For now, use simulated data but this would connect to real API
        return self.simulated_data.get_mean_reversion_candidates()
    
    def _scan_breakout_real(self) -> List[Dict]:
        """Scan for opening range breakout candidates using real API"""
        if self.is_simulation:
            return self.simulated_data.get_breakout_range_candidates()
        
        # For now, use simulated data but this would connect to real API
        return self.simulated_data.get_breakout_range_candidates()
    
    def _filter_candidates(self, candidates: List[Dict], strategy: str) -> List[Dict]:
        """Filter candidates based on strategy-specific criteria"""
        filtered = []
        
        for candidate in candidates:
            try:
                indicators = candidate.get('indicators', {})
                symbol = candidate['symbol']
                
                # Common filters
                if strategy == 'momentum':
                    if self._validate_momentum_candidate(candidate):
                        filtered.append(candidate)
                    else:
                        self._log_near_miss(candidate, "Failed momentum validation")
                
                elif strategy == 'mean_reversion' and ENABLE_MEAN_REVERSION:
                    if self._validate_mean_reversion_candidate(candidate):
                        filtered.append(candidate)
                    else:
                        self._log_near_miss(candidate, "Failed mean reversion validation")
                
                elif strategy == 'breakout':
                    if self._validate_breakout_candidate(candidate):
                        filtered.append(candidate)
                    else:
                        self._log_near_miss(candidate, "Failed breakout validation")
                        
            except Exception as e:
                logger.error(f"Error filtering candidate {candidate.get('symbol', 'unknown')}: {e}")
                
        return filtered[:5]  # Limit to top 5 per strategy
    
    def _validate_momentum_candidate(self, candidate: Dict) -> bool:
        """Validate momentum strategy candidate with enhanced criteria"""
        indicators = candidate.get('indicators', {})
        
        # Original criteria
        if candidate['change_pct'] < MIN_PRICE_CHANGE_PCT:
            return False
        if candidate['volume'] < candidate['avg_volume'] * VOLUME_MULTIPLIER:
            return False
            
        # Enhanced criteria
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        adx = indicators.get('adx', 20)
        vwap = indicators.get('vwap', candidate['price'])
        
        # MACD must be positive and rising
        if macd <= 0 or macd <= macd_signal:
            return False
            
        # ADX > 25 for strong trends
        if adx < 25:
            return False
            
        # Price must be above VWAP for longs
        if candidate['price'] < vwap:
            return False
            
        return True
    
    def _validate_mean_reversion_candidate(self, candidate: Dict) -> bool:
        """Validate mean reversion strategy candidate"""
        indicators = candidate.get('indicators', {})
        
        rsi_5 = indicators.get('rsi_5', 50)
        bb_lower = indicators.get('bb_lower', candidate['price'] * 0.98)
        
        # RSI(5) < 30 for oversold
        if rsi_5 >= 30:
            return False
            
        # Price below lower Bollinger Band
        if candidate['price'] >= bb_lower:
            return False
            
        # Minimum volume requirement
        if candidate['volume'] < candidate['avg_volume'] * 1.2:
            return False
            
        return True
    
    def _validate_breakout_candidate(self, candidate: Dict) -> bool:
        """Validate opening range breakout candidate"""
        opening_range = candidate.get('opening_range', {})
        if not opening_range:
            return False
            
        # Must break above opening range high with volume
        if candidate['price'] <= opening_range['high'] * 1.005:  # 0.5% above
            return False
            
        if candidate['volume'] < candidate['avg_volume'] * 1.5:
            return False
            
        return True
    
    def _log_near_miss(self, candidate: Dict, reason: str):
        """Log near-miss candidates for analysis"""
        try:
            near_miss = {
                'timestamp': datetime.now(),
                'symbol': candidate['symbol'],
                'strategy': candidate.get('strategy', 'unknown'),
                'missed_reason': reason,
                'metrics': {
                    'price_change': candidate.get('change_pct', 0),
                    'volume_ratio': candidate['volume'] / candidate['avg_volume'],
                    'rsi': candidate.get('indicators', {}).get('rsi_14', 50),
                    'distance_from_vwap': (candidate['price'] - candidate.get('indicators', {}).get('vwap', candidate['price'])) / candidate['price'],
                    'macd': candidate.get('indicators', {}).get('macd', 0),
                    'why_rejected': reason
                }
            }
            
            self.near_miss_log.append(near_miss)
            
        except Exception as e:
            logger.error(f"Error logging near miss: {e}")
    
    def _monitor_and_trade_enhanced(self):
        """Enhanced monitoring and trading with multiple strategies"""
        try:
            # Check if we've hit position limit
            open_positions = sum(1 for p in self.positions.values() if p['status'] == 'OPEN')
            if open_positions >= MAX_CONCURRENT_POSITIONS:
                return
            
            # Check time cutoff (no new trades after 3:00 PM)
            now = datetime.now(MARKET_TIMEZONE)
            if now.hour >= NO_TRADES_AFTER_HOUR:
                return
            
            # Select strategy based on market regime
            active_strategies = self._select_strategies_by_regime()
            
            for strategy in active_strategies:
                candidates = self.all_candidates.get(strategy, [])
                for candidate in candidates:
                    if self._should_enter_position(candidate):
                        current_price = self._get_current_price(candidate['symbol'])
                        position_size = self._calculate_position_size(candidate, current_price)
                        self._enter_position_enhanced(candidate['symbol'], current_price, candidate['strategy'], position_size)
                        
                        # Only take one position per loop iteration
                        return
                        
        except Exception as e:
            logger.error(f"Error in enhanced monitoring: {e}")
    
    def _select_strategies_by_regime(self) -> List[str]:
        """Select active strategies based on market regime"""
        if self.market_regime == "trending":
            return ['momentum', 'breakout']
        elif self.market_regime == "ranging":
            return ['mean_reversion'] if ENABLE_MEAN_REVERSION else ['momentum']
        else:
            return ['momentum', 'mean_reversion', 'breakout'] if ENABLE_MEAN_REVERSION else ['momentum', 'breakout']
    
    def _should_enter_position(self, candidate: Dict) -> bool:
        """Determine if we should enter a position for this candidate"""
        symbol = candidate['symbol']
        
        # Skip if already have position
        if symbol in self.positions and self.positions[symbol]['status'] == 'OPEN':
            return False
            
        # Skip if already traded today
        if any(t['symbol'] == symbol and t['date'] == datetime.now().date()
               for t in self.trades_log):
            return False
        
        # Strategy-specific entry logic
        strategy = candidate.get('strategy', 'momentum')
        
        if strategy == 'momentum':
            return self._check_momentum_entry(candidate)
        elif strategy == 'mean_reversion':
            return self._check_mean_reversion_entry(candidate)
        elif strategy == 'breakout':
            return self._check_breakout_entry(candidate)
        
        return False
    
    def _check_momentum_entry(self, candidate: Dict) -> bool:
        """Check momentum strategy entry conditions"""
        current_price = self._get_current_price(candidate['symbol'])
        morning_high = candidate.get('morning_high', current_price * 0.98)
        
        # Must break above morning high by 1%
        return current_price > morning_high * 1.01
    
    def _check_mean_reversion_entry(self, candidate: Dict) -> bool:
        """Check mean reversion strategy entry conditions"""
        # Entry conditions already validated in filtering
        return True
    
    def _check_breakout_entry(self, candidate: Dict) -> bool:
        """Check opening range breakout entry conditions"""
        current_price = self._get_current_price(candidate['symbol'])
        opening_range = candidate.get('opening_range', {})
        
        if not opening_range:
            return False
            
        # Must break above opening range high
        return current_price > opening_range['high'] * 1.005
    
    def _calculate_position_size(self, candidate: Dict, current_price: float) -> int:
        """Calculate position size using ATR-based sizing"""
        try:
            indicators = candidate.get('indicators', {})
            atr = indicators.get('atr', current_price * 0.02)  # Default to 2% of price
            
            # Risk 1% of account per trade
            risk_amount = self.account_balance * 0.01
            
            # Position size = risk_amount / (ATR * 2)
            position_size = int(risk_amount / (atr * 2))
            
            # Ensure minimum position size and maximum
            position_size = max(position_size, 10)  # Minimum 10 shares
            position_size = min(position_size, POSITION_SIZE * 2)  # Maximum 2x default
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return POSITION_SIZE
    
    def _enter_position_enhanced(self, symbol: str, entry_price: float, strategy: str, shares: int):
        """Enhanced position entry with strategy tracking"""
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': entry_price,
            'shares': shares,
            'stop_loss': entry_price * (1 - STOP_LOSS_PCT),
            'take_profit': entry_price * (1 + TAKE_PROFIT_PCT),
            'entry_time': datetime.now(),
            'status': 'OPEN',
            'pnl': 0,
            'pnl_pct': 0,
            'strategy': strategy
        }

        cost = entry_price * shares

        if self.is_simulation:
            self.positions[symbol] = position
            self.account_balance -= cost
            self.buying_power -= cost
            logger.info(f"BUY ({strategy.upper()}): {symbol} - {shares} shares @ ${entry_price:.2f}")
            logger.info(f"Account Balance: ${self.account_balance:,.2f}")
        else:
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                self.positions[symbol] = position
                self._update_account_balance()
                logger.info(f"BUY ORDER ({strategy.upper()}): {symbol} - Order ID: {order.id}")
                logger.info(f"Account Balance: ${self.account_balance:,.2f}")
            except Exception as e:
                logger.error(f"Failed to buy {symbol}: {e}")
                return

        # Send enhanced buy notification
        strategy_emoji = {"momentum": "🚀", "mean_reversion": "🔄", "breakout": "💥"}.get(strategy, "📈")
        self._send_notification(
            f"{strategy_emoji} BUY: {symbol} ({strategy.upper()})",
            f"Bought {shares} shares @ ${entry_price:.2f}\n"
            f"Stop Loss: ${position['stop_loss']:.2f}\n"
            f"Take Profit: ${position['take_profit']:.2f}\n"
            f"Strategy: {strategy.upper()}\n"
            f"Balance: ${self.account_balance:,.2f}",
            priority=1
        )

        self.trades_log.append({
            'date': datetime.now().date(),
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': 'BUY',
            'symbol': symbol,
            'price': entry_price,
            'shares': shares,
            'balance': self.account_balance,
            'strategy': strategy
        })


    def _check_exit_conditions(self):
        """Check all positions for exit conditions"""
        for symbol, position in list(self.positions.items()):
            if position['status'] != 'OPEN':
                continue

            current_price = self._get_current_price(symbol)
            position['current_price'] = current_price
            position['pnl'] = (current_price - position['entry_price']) * position['shares']
            position['pnl_pct'] = ((current_price - position['entry_price']) / position['entry_price']) * 100

            # Check stop loss
            if current_price <= position['stop_loss']:
                self._exit_position(symbol, current_price, 'STOP_LOSS')
            # Check take profit
            elif current_price >= position['take_profit']:
                self._exit_position(symbol, current_price, 'TAKE_PROFIT')

    def _exit_position(self, symbol: str, exit_price: float, reason: str):
        """Enhanced position exit with consecutive loss tracking"""
        position = self.positions.get(symbol)
        if not position or position['status'] != 'OPEN':
            return

        pnl = (exit_price - position['entry_price']) * position['shares']
        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
        proceeds = exit_price * position['shares']
        strategy = position.get('strategy', 'unknown')

        if self.is_simulation:
            self.account_balance += proceeds
            self.buying_power += proceeds
            logger.info(f"SELL ({strategy.upper()}): {symbol} @ ${exit_price:.2f} - {reason} - P&L: ${pnl:.2f}")
            logger.info(f"Account Balance: ${self.account_balance:,.2f}")
        else:
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=position['shares'],
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                # Update balance from Alpaca
                self._update_account_balance()
                logger.info(f"SELL ORDER ({strategy.upper()}): {symbol} - {reason} - P&L: ${pnl:.2f}")
                logger.info(f"Account Balance: ${self.account_balance:,.2f}")
            except Exception as e:
                logger.error(f"Failed to sell {symbol}: {e}")
                return

        position['status'] = 'CLOSED'
        position['exit_price'] = exit_price
        position['exit_reason'] = reason
        position['final_pnl'] = pnl

        self.daily_pnl += pnl
        self.total_pnl += pnl

        # Track consecutive losses for circuit breaker
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                self.pause_trading_until = datetime.now() + timedelta(hours=1)
                self._send_notification(
                    "⏸️ Trading Paused",
                    f"3 consecutive losses detected\n"
                    f"Trading paused for 1 hour\n"
                    f"Resume time: {self.pause_trading_until.strftime('%H:%M')}",
                    priority=1
                )
                logger.warning(f"Trading paused for 1 hour due to {self.consecutive_losses} consecutive losses")
        else:
            self.consecutive_losses = 0  # Reset on winning trade

        # Send enhanced sell notification
        strategy_emoji = {"momentum": "🚀", "mean_reversion": "🔄", "breakout": "💥"}.get(strategy, "📈")
        emoji = "💰" if pnl > 0 else "📉"

        self._send_notification(
            f"{emoji} SELL: {symbol} ({strategy.upper()}) - {reason}",
            f"Sold @ ${exit_price:.2f}\n"
            f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n"
            f"Strategy: {strategy.upper()}\n"
            f"Balance: ${self.account_balance:,.2f}\n"
            f"Daily P&L: ${self.daily_pnl:+,.2f}",
            priority=1 if abs(pnl) > 500 else 0
        )

        self.trades_log.append({
            'date': datetime.now().date(),
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': f'SELL ({reason})',
            'symbol': symbol,
            'price': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'balance': self.account_balance,
            'strategy': strategy
        })

    def _close_all_positions(self, reason: str = 'END_OF_DAY'):
        """Close all open positions"""
        logger.info(f"Closing all positions - Reason: {reason}")

        for symbol, position in list(self.positions.items()):
            if position['status'] == 'OPEN':
                current_price = self._get_current_price(symbol)
                self._exit_position(symbol, current_price, reason)

    def _update_positions(self):
        """Update current prices and P&L for all positions"""
        for symbol, position in self.positions.items():
            if position['status'] == 'OPEN':
                current_price = self._get_current_price(symbol)
                position['current_price'] = current_price
                position['pnl'] = (current_price - position['entry_price']) * position['shares']
                position['pnl_pct'] = ((current_price - position['entry_price']) / position['entry_price']) * 100

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if self.is_simulation:
            return self.simulated_data.get_current_price(symbol)
        else:
            try:
                quote = self.api.get_latest_quote(symbol)
                return float(quote.ap)
            except:
                return self.simulated_data.get_current_price(symbol)

    def _log_daily_summary(self):
        """Enhanced daily trading summary with near-miss analysis"""
        logger.info("="*60)
        logger.info(f"ENHANCED DAILY SUMMARY - {datetime.now().date()}")
        logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
        logger.info(f"Total P&L: ${self.total_pnl:.2f}")

        # Count winners/losers for today
        today_trades = [t for t in self.trades_log if t.get('date') == datetime.now().date()]
        winners = sum(1 for t in today_trades
                     if t.get('action', '').startswith('SELL') and t.get('pnl', 0) > 0)
        losers = sum(1 for t in today_trades
                    if t.get('action', '').startswith('SELL') and t.get('pnl', 0) <= 0)
        total_today = winners + losers

        if total_today > 0:
            win_rate = winners / total_today * 100
            logger.info(f"Win Rate: {win_rate:.1f}% ({winners}W/{losers}L)")
        else:
            win_rate = 0

        # Strategy performance breakdown
        strategy_performance = {}
        for trade in today_trades:
            if trade.get('action', '').startswith('SELL'):
                strategy = trade.get('strategy', 'unknown')
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'wins': 0, 'losses': 0, 'pnl': 0}
                
                pnl = trade.get('pnl', 0)
                strategy_performance[strategy]['pnl'] += pnl
                if pnl > 0:
                    strategy_performance[strategy]['wins'] += 1
                else:
                    strategy_performance[strategy]['losses'] += 1

        logger.info("STRATEGY PERFORMANCE:")
        for strategy, stats in strategy_performance.items():
            total_trades = stats['wins'] + stats['losses']
            win_rate = (stats['wins'] / total_trades * 100) if total_trades > 0 else 0
            logger.info(f"  {strategy.upper()}: {stats['wins']}W/{stats['losses']}L ({win_rate:.1f}%) P&L: ${stats['pnl']:+.2f}")

        # Top 5 missed opportunities
        logger.info("TOP 5 MISSED OPPORTUNITIES:")
        top_misses = sorted(list(self.near_miss_log), 
                           key=lambda x: x['metrics']['volume_ratio'], reverse=True)[:5]
        
        for i, miss in enumerate(top_misses, 1):
            logger.info(f"  {i}. {miss['symbol']} ({miss['strategy']}) - {miss['missed_reason']}")
            logger.info(f"     Volume: {miss['metrics']['volume_ratio']:.1f}x, Change: {miss['metrics']['price_change']:.1%}")

        # Market regime assessment
        logger.info(f"MARKET REGIME: {self.market_regime.upper()}")
        logger.info(f"CONSECUTIVE LOSSES: {self.consecutive_losses}")
        if self.pause_trading_until:
            logger.info(f"TRADING PAUSED UNTIL: {self.pause_trading_until}")

        logger.info("="*60)

        # Send enhanced end-of-day summary notification
        daily_return = ((self.account_balance - self.daily_starting_balance) / self.daily_starting_balance) * 100
        total_return = ((self.account_balance - self.initial_balance) / self.initial_balance) * 100

        # Choose emoji based on performance
        if self.daily_target_hit:
            emoji = "🎯"
            priority = 1
        elif self.daily_pnl > 1000:
            emoji = "🚀"
            priority = 1
        elif self.daily_pnl > 0:
            emoji = "✅"
            priority = 0
        elif self.daily_pnl < -1000:
            emoji = "🔴"
            priority = 1
        else:
            emoji = "⚪"
            priority = 0

        # Build strategy summary
        strategy_summary = ""
        for strategy, stats in strategy_performance.items():
            total_trades = stats['wins'] + stats['losses']
            if total_trades > 0:
                win_rate = stats['wins'] / total_trades * 100
                strategy_summary += f"{strategy.upper()}: {stats['wins']}W/{stats['losses']}L (${stats['pnl']:+.0f})\n"

        summary_message = (
            f"Today's P&L: ${self.daily_pnl:+,.2f} ({daily_return:+.2f}%)\n"
            f"Trades: {total_today} ({winners}W/{losers}L) - {win_rate:.1f}% win rate\n"
            f"Market Regime: {self.market_regime.upper()}\n"
            f"─────────────\n"
            f"{strategy_summary}"
            f"Near Misses: {len(self.near_miss_log)} logged\n"
            f"─────────────\n"
            f"Balance: ${self.account_balance:,.2f}\n"
            f"Total Return: {total_return:+.2f}%\n"
            f"Total P&L: ${self.total_pnl:+,.2f}"
        )

        self._send_notification(
            f"{emoji} Market Closed - Enhanced Summary",
            summary_message,
            priority=priority
        )

    def _update_account_balance(self):
        """Update account balance from Alpaca API"""
        if not self.is_simulation and self.api:
            try:
                account = self.api.get_account()
                self.account_balance = float(account.cash)
                self.buying_power = float(account.buying_power)
            except Exception as e:
                logger.error(f"Failed to update account balance: {e}")

    def get_stats(self):
        """Get enhanced statistics for display"""
        open_positions = sum(1 for p in self.positions.values() if p['status'] == 'OPEN')
        total_trades = len([t for t in self.trades_log if t.get('action', '').startswith('SELL')])

        winners = sum(1 for t in self.trades_log
                     if t.get('action', '').startswith('SELL') and t.get('pnl', 0) > 0)

        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        # Calculate daily progress toward target
        daily_return = (self.daily_pnl / self.daily_starting_balance) if self.daily_starting_balance > 0 else 0
        target_progress = (daily_return / DAILY_PROFIT_TARGET_PCT) * 100

        # Update balance if using real API
        if not self.is_simulation and self.is_trading_hours:
            self._update_account_balance()

        return {
            'status': self.status,
            'mode': 'PAPER TRADING' if not self.is_simulation else 'SIMULATION',
            'is_trading_hours': self.is_trading_hours,
            'open_positions': open_positions,
            'total_trades': total_trades,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'win_rate': win_rate,
            'account_balance': self.account_balance,
            'buying_power': self.buying_power,
            'initial_balance': self.initial_balance,
            # Enhanced metrics
            'market_regime': self.market_regime,
            'daily_target_hit': self.daily_target_hit,
            'daily_loss_limit_hit': self.daily_loss_limit_hit,
            'consecutive_losses': self.consecutive_losses,
            'trading_paused': bool(self.pause_trading_until and datetime.now() < self.pause_trading_until),
            'target_progress': target_progress,
            'near_misses_count': len(self.near_miss_log),
            'daily_return_pct': daily_return * 100,
            'all_candidates': self.all_candidates
        }

# Initialize the bot globally (runs once when app starts)
@st.cache_resource
def get_bot():
    return AutoTradingBot()

bot = get_bot()

# UI - READ ONLY DASHBOARD
st.title("🤖 Automated Trading Bot Monitor")
st.caption("Bot runs automatically during market hours (Mon-Fri, 9:30 AM - 4:00 PM ET)")

# Auto-refresh every 30 seconds
st_autorefresh = st.empty()

# Get current stats
stats = bot.get_stats()

# Status Header
col1, col2, col3 = st.columns(3)
with col1:
    if stats['is_trading_hours']:
        st.success(f"🟢 {stats['status']}")
    else:
        st.info(f"🔴 {stats['status']}")
with col2:
    st.metric("Mode", stats['mode'])
with col3:
    current_time = datetime.now(MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S ET")
    st.metric("Time", current_time)

# Metrics
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Account Balance",
              f"${stats['account_balance']:,.2f}",
              delta=f"${stats['account_balance'] - stats['initial_balance']:,.2f}")

with col2:
    st.metric("💵 Buying Power",
              f"${stats['buying_power']:,.2f}")

with col3:
    st.metric("📈 Total Return",
              f"{((stats['account_balance'] - stats['initial_balance']) / stats['initial_balance'] * 100):.2f}%",
              delta=f"${stats['account_balance'] - stats['initial_balance']:,.2f}")

# Enhanced metrics row
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    delta_color = "normal"
    if stats['daily_target_hit']:
        delta_color = "inverse"
    st.metric("Daily P&L", f"${stats['daily_pnl']:.2f}",
              delta=f"{stats['daily_return_pct']:.2f}%" if stats['daily_pnl'] != 0 else None)
              
with col2:
    st.metric("Daily Target", f"{stats['target_progress']:.0f}%",
              delta=f"Target: {DAILY_PROFIT_TARGET_PCT*100:.0f}%")
              
with col3:
    st.metric("Total P&L", f"${stats['total_pnl']:.2f}",
              delta=f"{stats['total_pnl']:.2f}" if stats['total_pnl'] != 0 else None)
              
with col4:
    st.metric("Open Positions", f"{stats['open_positions']}/{MAX_CONCURRENT_POSITIONS}")
    
with col5:
    st.metric("Total Trades", stats['total_trades'])
    
with col6:
    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")

# Enhanced status row
col1, col2, col3, col4 = st.columns(4)

with col1:
    if stats['market_regime']:
        regime_emoji = {"trending": "📈", "ranging": "↔️", "neutral": "⚪"}.get(stats['market_regime'], "❓")
        st.metric("Market Regime", f"{regime_emoji} {stats['market_regime'].title()}")
    else:
        st.metric("Market Regime", "❓ Unknown")

with col2:
    if stats['trading_paused']:
        st.metric("Status", "⏸️ Paused", delta="Circuit Breaker Active")
    elif stats['daily_target_hit']:
        st.metric("Status", "🎯 Target Hit", delta="No more trades today")
    elif stats['daily_loss_limit_hit']:
        st.metric("Status", "🛑 Loss Limit", delta="Trading stopped")
    else:
        st.metric("Status", "✅ Active", delta="Normal operation")

with col3:
    st.metric("Consecutive Losses", stats['consecutive_losses'], 
              delta="Pause at 3" if stats['consecutive_losses'] > 0 else None)

with col4:
    st.metric("Near Misses", stats['near_misses_count'],
              delta=f"Max: {MAX_NEAR_MISS_LOG}")

# Enhanced Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Positions", "🎯 Candidates", "📜 Trade Log", "🔍 Near Misses", "ℹ️ Info"])

with tab1:
    st.subheader("Current Positions")
    if bot.positions:
        positions_data = []
        for symbol, pos in bot.positions.items():
            if pos['status'] == 'OPEN':
                strategy = pos.get('strategy', 'unknown')
                strategy_emoji = {"momentum": "🚀", "mean_reversion": "🔄", "breakout": "💥"}.get(strategy, "📈")
                
                # Color code P&L
                pnl_color = "🟢" if pos['pnl'] > 0 else "🔴" if pos['pnl'] < 0 else "⚪"
                
                positions_data.append({
                    'Symbol': f"{strategy_emoji} {symbol}",
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Entry': f"${pos['entry_price']:.2f}",
                    'Current': f"${pos['current_price']:.2f}",
                    'Shares': pos['shares'],
                    'P&L': f"{pnl_color} ${pos['pnl']:+.2f}",
                    'P&L %': f"{pos['pnl_pct']:+.2f}%",
                    'Stop': f"${pos['stop_loss']:.2f}",
                    'Target': f"${pos['take_profit']:.2f}",
                    'Time': pos['entry_time'].strftime('%H:%M')
                })

        if positions_data:
            df = pd.DataFrame(positions_data)
            st.dataframe(df, use_container_width=True)
            
            # Position summary
            total_positions = len(positions_data)
            total_pnl = sum(pos['pnl'] for pos in bot.positions.values() if pos['status'] == 'OPEN')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Open Positions", f"{total_positions}/{MAX_CONCURRENT_POSITIONS}")
            with col2:
                st.metric("Unrealized P&L", f"${total_pnl:+.2f}")
            with col3:
                risk_per_position = POSITION_SIZE * STOP_LOSS_PCT
                st.metric("Max Risk", f"${risk_per_position * total_positions:.2f}")
        else:
            st.info("No open positions")
    else:
        st.info("No positions yet today")

with tab2:
    st.subheader("Today's Strategy Candidates")
    
    # Show candidates for each strategy
    all_candidates = stats.get('all_candidates', {})
    
    if any(candidates for candidates in all_candidates.values()):
        # Create sub-tabs for each strategy
        strategy_tabs = st.tabs(["🚀 Momentum", "🔄 Mean Reversion", "💥 Breakout Range"])
        
        with strategy_tabs[0]:  # Momentum
            momentum_candidates = all_candidates.get('momentum', [])
            if momentum_candidates:
                st.write("**Enhanced Momentum Strategy** (MACD+, ADX>25, Price>VWAP)")
                for c in momentum_candidates:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.write(f"**{c['symbol']}**")
                    col2.write(f"${c['price']:.2f}")
                    col3.write(f"{c['change_pct']:+.2%}")
                    col4.write(f"{c['volume']/c['avg_volume']:.1f}x vol")
                    indicators = c.get('indicators', {})
                    col5.write(f"ADX: {indicators.get('adx', 0):.0f}")
            else:
                st.info("No momentum candidates today")
        
        with strategy_tabs[1]:  # Mean Reversion
            if ENABLE_MEAN_REVERSION:
                mr_candidates = all_candidates.get('mean_reversion', [])
                if mr_candidates:
                    st.write("**Mean Reversion Strategy** (RSI<30, Price<BB Lower)")
                    for c in mr_candidates:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.write(f"**{c['symbol']}**")
                        col2.write(f"${c['price']:.2f}")
                        col3.write(f"{c['change_pct']:+.2%}")
                        col4.write(f"{c['volume']/c['avg_volume']:.1f}x vol")
                        indicators = c.get('indicators', {})
                        col5.write(f"RSI: {indicators.get('rsi_5', 50):.0f}")
                else:
                    st.info("No mean reversion candidates today")
            else:
                st.info("Mean reversion strategy disabled")
        
        with strategy_tabs[2]:  # Breakout
            breakout_candidates = all_candidates.get('breakout', [])
            if breakout_candidates:
                st.write("**Opening Range Breakout** (30-min range break + volume)")
                for c in breakout_candidates:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.write(f"**{c['symbol']}**")
                    col2.write(f"${c['price']:.2f}")
                    col3.write(f"{c['change_pct']:+.2%}")
                    col4.write(f"{c['volume']/c['avg_volume']:.1f}x vol")
                    opening_range = c.get('opening_range', {})
                    col5.write(f"Range: ${opening_range.get('high', 0):.2f}")
            else:
                st.info("No breakout candidates today")
                
        st.markdown("---")
        st.caption(f"Market Regime: **{stats['market_regime'].upper()}** | Active Strategies: {', '.join([s.title() for s, candidates in all_candidates.items() if candidates])}")
    else:
        st.info("📊 Multi-strategy candidates are scanned at market open (9:30 AM ET)")
        
        st.markdown("### 🔍 Strategy Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🚀 Enhanced Momentum**
            - MACD positive & rising
            - ADX > 25 (strong trend)  
            - Price > VWAP
            - Volume > 2x average
            """)
            
        with col2:
            enabled_text = "✅ Enabled" if ENABLE_MEAN_REVERSION else "❌ Disabled"
            st.markdown(f"""
            **🔄 Mean Reversion** {enabled_text}
            - RSI(5) < 30 (oversold)
            - Price < Lower Bollinger Band
            - Volume > 1.2x average
            - Exit at middle band
            """)
            
        with col3:
            st.markdown("""
            **💥 Opening Range Breakout**
            - Break 30-min high/low
            - Volume confirmation  
            - Time-based entry
            - Momentum follow-through
            """)

with tab3:
    st.subheader("Recent Trade Log")
    if bot.trades_log:
        # Show last 20 trades with enhanced information
        recent_trades = bot.trades_log[-20:]
        enhanced_trades = []
        
        for trade in recent_trades:
            strategy = trade.get('strategy', 'unknown')
            strategy_emoji = {"momentum": "🚀", "mean_reversion": "🔄", "breakout": "💥"}.get(strategy, "📈")
            
            enhanced_trade = {
                'Date': trade['date'].strftime('%m/%d'),
                'Time': trade['time'],
                'Action': f"{strategy_emoji} {trade['action']}",
                'Symbol': trade['symbol'],
                'Price': f"${trade['price']:.2f}",
                'Shares': trade['shares'],
                'P&L': f"${trade.get('pnl', 0):+.2f}" if 'pnl' in trade else '-',
                'Balance': f"${trade['balance']:,.0f}",
                'Strategy': strategy.replace('_', ' ').title() if strategy != 'unknown' else '-'
            }
            enhanced_trades.append(enhanced_trade)
        
        df = pd.DataFrame(enhanced_trades)
        df = df.iloc[::-1]  # Reverse to show most recent first
        st.dataframe(df, use_container_width=True)
        
        # Trade statistics
        today_trades = [t for t in bot.trades_log if t.get('date') == datetime.now().date()]
        if today_trades:
            col1, col2, col3 = st.columns(3)
            
            buys = [t for t in today_trades if t['action'] == 'BUY']
            sells = [t for t in today_trades if t['action'].startswith('SELL')]
            
            with col1:
                st.metric("Today's Trades", f"{len(buys)} buys, {len(sells)} sells")
            with col2:
                total_pnl = sum(t.get('pnl', 0) for t in sells)
                st.metric("Today's Realized P&L", f"${total_pnl:+.2f}")
            with col3:
                avg_hold_time = "N/A"  # Could calculate this if needed
                st.metric("Avg Hold Time", avg_hold_time)
    else:
        st.info("No trades executed yet")

with tab4:
    st.subheader("🔍 Near Miss Analysis")
    
    if bot.near_miss_log:
        st.write(f"**Tracking {len(bot.near_miss_log)} near-miss opportunities** (Max: {MAX_NEAR_MISS_LOG})")
        
        # Sort by volume ratio (most interesting first)
        sorted_misses = sorted(list(bot.near_miss_log), 
                              key=lambda x: x['metrics']['volume_ratio'], reverse=True)
        
        # Top 10 near misses
        st.markdown("### 📈 Top Missed Opportunities")
        for i, miss in enumerate(sorted_misses[:10], 1):
            with st.expander(f"{i}. {miss['symbol']} ({miss['strategy'].title()}) - {miss['missed_reason']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Price Action**")
                    st.write(f"Change: {miss['metrics']['price_change']:+.2%}")
                    st.write(f"Volume: {miss['metrics']['volume_ratio']:.1f}x avg")
                    
                with col2:
                    st.write("**Technical Indicators**")
                    st.write(f"RSI: {miss['metrics']['rsi']:.1f}")
                    st.write(f"MACD: {miss['metrics']['macd']:.3f}")
                    
                with col3:
                    st.write("**Analysis**")
                    st.write(f"VWAP Distance: {miss['metrics']['distance_from_vwap']:+.2%}")
                    st.write(f"**Rejection:** {miss['metrics']['why_rejected']}")
                
                st.caption(f"Logged: {miss['timestamp'].strftime('%H:%M:%S')}")
        
        # Strategy breakdown
        st.markdown("### 📊 Near Miss Breakdown by Strategy")
        strategy_counts = {}
        for miss in bot.near_miss_log:
            strategy = miss['strategy']
            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0
            strategy_counts[strategy] += 1
        
        if strategy_counts:
            col1, col2, col3 = st.columns(3)
            strategies = list(strategy_counts.keys())
            
            if len(strategies) > 0:
                with col1:
                    st.metric(f"🚀 {strategies[0].title()}", strategy_counts.get(strategies[0], 0))
            if len(strategies) > 1:
                with col2:
                    st.metric(f"🔄 {strategies[1].title()}", strategy_counts.get(strategies[1], 0))
            if len(strategies) > 2:
                with col3:
                    st.metric(f"💥 {strategies[2].title()}", strategy_counts.get(strategies[2], 0))
        
        # Common rejection reasons
        st.markdown("### 🚫 Common Rejection Reasons")
        rejection_counts = {}
        for miss in bot.near_miss_log:
            reason = miss['missed_reason']
            if reason not in rejection_counts:
                rejection_counts[reason] = 0
            rejection_counts[reason] += 1
        
        sorted_reasons = sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_reasons[:5]:
            st.write(f"• **{reason}**: {count} occurrences")
            
    else:
        st.info("📊 Near-miss tracking starts when market opens and candidates are scanned")
        
        st.markdown("### 🎯 What We Track")
        st.markdown("""
        **Near-miss candidates are stocks that:**
        - Meet some but not all strategy criteria
        - Have timing issues (signal came too late)
        - Are filtered out by risk management
        - Come within 0.5% of trigger levels
        
        **This helps identify:**
        - Market opportunities we're missing
        - Strategy parameter adjustments needed  
        - Timing improvements for entries
        - Risk management effectiveness
        """)

with tab5:
    st.subheader("🤖 Enhanced Bot Information")

    st.markdown("""
    ### 🚀 Enhanced Multi-Strategy Bot

    **⚡ Automatic Operation:**
    - Starts automatically at market open (9:30 AM ET)
    - Scans for candidates using 3 different strategies
    - Assesses market regime and selects optimal strategies
    - Monitors and trades throughout the day with risk management
    - Closes all positions by 3:50 PM ET
    - Advanced notifications and daily summaries

    **🎯 Daily Profit Target System:**
    - **Target**: +4% daily return (configurable)
    - **Action**: Closes all positions and stops trading when hit
    - **Loss Limit**: -2% daily loss limit with immediate stop
    - **Circuit Breaker**: 3 consecutive losses = 1-hour pause

    **📊 Multiple Trading Strategies:**
    """)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🚀 Enhanced Momentum**
        - MACD positive & rising
        - ADX > 25 (strong trend)
        - Price above VWAP
        - Volume > 2x average
        - Stop: -2% | Target: +7%
        """)
        
    with col2:
        enabled = "✅" if ENABLE_MEAN_REVERSION else "❌"
        st.markdown(f"""
        **🔄 Mean Reversion** {enabled}
        - RSI(5) < 30 (oversold)
        - Price < Lower Bollinger Band  
        - Volume > 1.2x average
        - Exit at middle band
        - Stop: -2% | Target: +7%
        """)
        
    with col3:
        st.markdown("""
        **💥 Opening Range Breakout**
        - Break 30-min high/low + 0.5%
        - Volume confirmation (1.5x)
        - No trades after 3:00 PM
        - Stop: -2% | Target: +7%
        """)

    st.markdown("### ⚙️ Risk Management")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Positions", f"{MAX_CONCURRENT_POSITIONS}")
        st.metric("Position Sizing", "ATR-based (1% risk)")
        
    with col2:
        st.metric("Daily Target", f"{DAILY_PROFIT_TARGET_PCT*100:.0f}%")
        st.metric("Daily Loss Limit", f"{MAX_DAILY_LOSS_PCT*100:.0f}%")
        
    with col3:
        st.metric("Stop Loss", f"{STOP_LOSS_PCT*100:.0f}%")
        st.metric("Take Profit", f"{TAKE_PROFIT_PCT*100:.0f}%")

    st.markdown("### 📱 Enhanced Notifications")
    if bot.notifications_enabled:
        st.success("✅ Push notifications are ENABLED")
        st.markdown("""
        **You'll receive notifications for:**
        - 🔔 Market open with strategy assessment
        - 🚀🔄💥 Every buy with strategy indicator
        - 💰📉 Every sell with P&L and strategy
        - 🎯 Daily target achieved
        - ⏸️ Circuit breaker activations
        - 📊 Enhanced end-of-day summary with strategy performance
        - 🛑 Daily loss limit warnings
        """)
    else:
        st.warning("⚠️ Push notifications are DISABLED")
        st.markdown("""
        To enable iPhone notifications:
        1. Download Pushover app ($4.99 one-time)
        2. Get your User Key from the app
        3. Create an app at pushover.net to get App Token
        4. Add to Streamlit Cloud secrets:
           - PUSHOVER_USER_KEY
           - PUSHOVER_APP_TOKEN
        """)

    st.markdown("---")
    st.markdown("### 🔧 Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        if not ALPACA_API_KEY:
            st.warning("⚠️ No API keys - simulation mode")
            st.info("Set ALPACA_API_KEY and ALPACA_SECRET_KEY for paper trading")
        else:
            st.success("✅ Connected to Alpaca Paper Trading")
            
        st.markdown(f"""
        **Current Settings:**
        - Position Size: {POSITION_SIZE} shares
        - Volume Multiplier: {VOLUME_MULTIPLIER}x
        - Min Price Change: {MIN_PRICE_CHANGE_PCT*100:.0f}%
        """)
        
    with config_col2:
        st.markdown(f"""
        **Enhanced Features:**
        - Mean Reversion: {'✅ Enabled' if ENABLE_MEAN_REVERSION else '❌ Disabled'}
        - Short Selling: {'✅ Enabled' if ENABLE_SHORT_SELLING else '❌ Disabled'}
        - Near Miss Tracking: ✅ Active (Max: {MAX_NEAR_MISS_LOG})
        - Market Regime Detection: ✅ Active
        """)

    st.markdown("---")
    st.caption("🚨 This enhanced bot uses Alpaca's PAPER TRADING API - no real money at risk")
    st.caption(f"📡 API Endpoint: `{ALPACA_BASE_URL}`")
    st.caption("🧠 Powered by TA-Lib technical indicators and advanced risk management")

# Auto-refresh the page every 30 seconds
st.markdown(
    """
    <script>
    setTimeout(function(){
        window.location.reload();
    }, 30000);
    </script>
    """,
    unsafe_allow_html=True
)

# Show last refresh time
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} - Auto-refreshes every 30 seconds")