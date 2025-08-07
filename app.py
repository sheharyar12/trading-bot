#!/usr/bin/env python3
"""
Enhanced Trading Bot - Streamlit Cloud Ready (No TA-Lib dependency)
Production version with robust fallback technical indicators & Alpaca integration
"""

# Core imports
import os
import time
from datetime import datetime, timedelta
import random
import logging
from typing import List, Dict, Optional
import threading
from collections import deque

import pandas as pd
import numpy as np
import pytz
import streamlit as st

# Optional imports with fallbacks
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# --- Use st.secrets directly ---
ALPACA_API_KEY = st.secrets.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = st.secrets.get('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

PUSHOVER_USER_KEY = st.secrets.get('PUSHOVER_USER_KEY', '')
PUSHOVER_APP_TOKEN = st.secrets.get('PUSHOVER_APP_TOKEN', '')

POSITION_SIZE = int(st.secrets.get('POSITION_SIZE', '100'))
STOP_LOSS_PCT = float(st.secrets.get('STOP_LOSS_PCT', '0.02'))
TAKE_PROFIT_PCT = float(st.secrets.get('TAKE_PROFIT_PCT', '0.07'))
VOLUME_MULTIPLIER = float(st.secrets.get('VOLUME_MULTIPLIER', '2.0'))
MIN_PRICE_CHANGE_PCT = float(st.secrets.get('MIN_PRICE_CHANGE_PCT', '0.05'))

DAILY_PROFIT_TARGET_PCT = float(st.secrets.get('DAILY_PROFIT_TARGET_PCT', '0.04'))
MAX_DAILY_LOSS_PCT = float(st.secrets.get('MAX_DAILY_LOSS_PCT', '0.02'))
ENABLE_MEAN_REVERSION = str(st.secrets.get('ENABLE_MEAN_REVERSION', 'true')).lower() == 'true'
ENABLE_SHORT_SELLING = str(st.secrets.get('ENABLE_SHORT_SELLING', 'false')).lower() == 'true'
MAX_NEAR_MISS_LOG = int(st.secrets.get('MAX_NEAR_MISS_LOG', '50'))
MAX_CONCURRENT_POSITIONS = int(st.secrets.get('MAX_CONCURRENT_POSITIONS', '5'))
NO_TRADES_AFTER_HOUR = int(st.secrets.get('NO_TRADES_AFTER_HOUR', '15'))

# Market hours
MARKET_TIMEZONE = pytz.timezone('US/Eastern')
MARKET_OPEN_TIME = (9, 30)
MARKET_CLOSE_TIME = (15, 50)

# Set page config
st.set_page_config(
    page_title="Enhanced Trading Bot Monitor",
    page_icon="🚀",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    # ... All your indicator methods as before (unchanged) ...
    # [NO CHANGES from your original file, so omitted here for brevity]
    # Paste your entire TechnicalIndicators class here

    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema

    # ... (copy all the indicator methods here) ...
    # Rest of TechnicalIndicators unchanged!

class SimulatedMarketData:
    # ... No change, use your original class as is ...
    # Paste your SimulatedMarketData here

    def __init__(self):
        self.stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'CRM']
        self.base_prices = {s: random.uniform(50, 500) for s in self.stocks}
        self.volumes = {s: random.randint(1000000, 50000000) for s in self.stocks}
        self.opening_ranges = {}

    # ... (rest of class unchanged) ...

class EnhancedTradingBot:
    """Enhanced trading bot with live Alpaca & simulation support."""

    def __init__(self):
        self.api = None
        self.positions = {}
        self.trades_log = []
        self.daily_pnl = 0
        self.total_pnl = 0
        self.account_balance = 100000.0
        self.buying_power = 100000.0
        self.initial_balance = 100000.0
        self.daily_starting_balance = 100000.0

        self.market_data = SimulatedMarketData()
        self.near_miss_log = deque(maxlen=MAX_NEAR_MISS_LOG)
        self.all_candidates = {'momentum': [], 'mean_reversion': [], 'breakout': []}
        self.market_regime = "neutral"
        self.trading_enabled_today = True
        self.daily_target_hit = False
        self.daily_loss_limit_hit = False
        self.consecutive_losses = 0
        self.pause_trading_until = None

        # Status
        self.status = "SIMULATION MODE"
        self.is_trading_hours = True
        self.is_simulation = True
        self.notifications_enabled = bool(PUSHOVER_USER_KEY and PUSHOVER_APP_TOKEN)

        # Try to connect to Alpaca for LIVE mode
        if ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_AVAILABLE:
            try:
                self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
                # Test connection and load account
                account = self.api.get_account()
                self.account_balance = float(account.cash)
                self.buying_power = float(account.buying_power)
                self.initial_balance = float(account.equity)
                self.is_simulation = False
                self.status = "LIVE TRADING - CONNECTED TO ALPACA"
                logger.info("Connected to Alpaca successfully. Live trading mode.")
            except Exception as e:
                self.api = None
                self.is_simulation = True
                self.status = "SIMULATION - Alpaca connection failed"
                logger.error(f"Failed to connect to Alpaca: {e}")

        # Load data
        if self.is_simulation:
            self._initialize_sample_data()
        else:
            self._initialize_live_data()

    def _initialize_sample_data(self):
        # ... use your sample data initialization as before ...
        # Paste your original _initialize_sample_data() here (unchanged)

        sample_symbols = ['AAPL', 'MSFT', 'TSLA']
        for i, symbol in enumerate(sample_symbols[:2]):
            strategy = ['momentum', 'mean_reversion'][i]
            entry_price = random.uniform(150, 250)
            if strategy == 'momentum':
                current_price = entry_price * random.uniform(1.01, 1.06)
            else:
                current_price = entry_price * random.uniform(0.96, 1.02)

            self.positions[symbol] = {
                'symbol': symbol,
                'entry_price': entry_price,
                'current_price': current_price,
                'shares': POSITION_SIZE,
                'stop_loss': entry_price * (1 - STOP_LOSS_PCT),
                'take_profit': entry_price * (1 + TAKE_PROFIT_PCT),
                'entry_time': datetime.now() - timedelta(minutes=random.randint(30, 180)),
                'status': 'OPEN',
                'pnl': (current_price - entry_price) * POSITION_SIZE,
                'pnl_pct': (current_price - entry_price) / entry_price * 100,
                'strategy': strategy
            }

        # ... rest of your sample data logic (trades_log, near_miss_log, etc) ...

        for i in range(8):
            pnl = random.randint(-300, 800) if i % 2 == 1 else 0
            self.trades_log.append({
                'date': (datetime.now() - timedelta(days=random.randint(0, 3))).date(),
                'time': f"{9 + (i//2)}:{30 + (i%2)*15:02d}:00",
                'action': 'BUY' if i % 2 == 0 else random.choice(['SELL (TAKE_PROFIT)', 'SELL (STOP_LOSS)', 'SELL (END_OF_DAY)']),
                'symbol': random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMD', 'NVDA']),
                'price': random.uniform(80, 300),
                'shares': POSITION_SIZE,
                'pnl': pnl,
                'balance': self.account_balance + random.randint(-2000, 3000),
                'strategy': random.choice(['momentum', 'mean_reversion', 'breakout'])
            })

        # ... and so on, as in your original sample setup.

        rejection_reasons = [
            "Volume only 1.8x (needed 2.0x)",
            "ADX 23 (needed >25 for trend confirmation)",
            "Price 0.2% below VWAP (momentum requires above)",
            "RSI 32 (needed <30 for oversold)",
            "MACD below signal line",
            "Insufficient breakout momentum (+0.4%, needed +0.5%)"
        ]

        for i in range(15):
            self.near_miss_log.append({
                'timestamp': datetime.now() - timedelta(minutes=random.randint(5, 360)),
                'symbol': random.choice(self.market_data.stocks),
                'strategy': random.choice(['momentum', 'mean_reversion', 'breakout']),
                'missed_reason': random.choice(rejection_reasons),
                'metrics': {
                    'price_change': random.uniform(0.01, 0.12),
                    'volume_ratio': random.uniform(1.3, 2.8),
                    'rsi': random.uniform(25, 75),
                    'distance_from_vwap': random.uniform(-0.03, 0.03),
                    'macd': random.uniform(-0.8, 1.2),
                    'why_rejected': random.choice(rejection_reasons)
                }
            })

        self.all_candidates = {
            'momentum': self.market_data.get_candidates_by_strategy('momentum'),
            'mean_reversion': self.market_data.get_candidates_by_strategy('mean_reversion') if ENABLE_MEAN_REVERSION else [],
            'breakout': self.market_data.get_candidates_by_strategy('breakout')
        }

        regimes = ['trending', 'ranging', 'neutral']
        weights = [0.4, 0.3, 0.3]
        self.market_regime = random.choices(regimes, weights)[0]

        self.daily_pnl = sum(pos['pnl'] for pos in self.positions.values() if pos['status'] == 'OPEN')
        today_trades = [t for t in self.trades_log if t['date'] == datetime.now().date()]
        self.daily_pnl += sum(t.get('pnl', 0) for t in today_trades)
        self.total_pnl = self.daily_pnl + random.randint(-1000, 5000)

    def _initialize_live_data(self):
        """Fetches real positions and trades from Alpaca"""
        try:
            # Get current positions
            positions = self.api.list_positions()
            for pos in positions:
                self.positions[pos.symbol] = {
                    'symbol': pos.symbol,
                    'entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'shares': int(float(pos.qty)),
                    'stop_loss': None,
                    'take_profit': None,
                    'entry_time': None,
                    'status': pos.side.upper(),
                    'pnl': float(pos.unrealized_pl),
                    'pnl_pct': float(pos.unrealized_plpc) * 100,
                    'strategy': 'unknown',
                }

            # Get recent trades (activities)
            activities = self.api.get_activities(activity_types="FILL", limit=50)
            for act in activities:
                # Only include completed orders
                if act.status == 'filled':
                    action = "BUY" if act.side == 'buy' else "SELL"
                    self.trades_log.append({
                        'date': act.transaction_time.date(),
                        'time': act.transaction_time.strftime("%H:%M:%S"),
                        'action': action,
                        'symbol': act.symbol,
                        'price': float(act.price),
                        'shares': int(float(act.qty)),
                        'pnl': 0,  # Realized P&L can be calculated if desired
                        'balance': self.account_balance,
                        'strategy': 'unknown'
                    })
        except Exception as e:
            logger.error(f"Failed to fetch live data from Alpaca: {e}")

    def get_stats(self):
        open_positions = sum(1 for p in self.positions.values() if p['status'] == 'OPEN' or p['status'] == 'LONG' or p['status'] == 'SHORT')
        total_trades = len([t for t in self.trades_log if t.get('action', '').startswith('SELL') or t.get('action', '') == 'SELL'])

        winners = sum(1 for t in self.trades_log
                     if (t.get('action', '').startswith('SELL') or t.get('action', '') == 'SELL') and t.get('pnl', 0) > 0)

        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        # Calculate daily progress toward target (sim only)
        daily_return = (self.daily_pnl / self.daily_starting_balance) if self.daily_starting_balance > 0 else 0
        target_progress = (daily_return / DAILY_PROFIT_TARGET_PCT) * 100

        return {
            'status': self.status,
            'mode': 'LIVE' if not self.is_simulation else 'PROFESSIONAL SIMULATION',
            'is_trading_hours': self.is_trading_hours,
            'open_positions': open_positions,
            'total_trades': total_trades,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'win_rate': win_rate,
            'account_balance': self.account_balance,
            'buying_power': self.buying_power,
            'initial_balance': self.initial_balance,
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

# --- Caching the bot for Streamlit hot reloads ---
@st.cache_resource
def get_bot():
    return EnhancedTradingBot()

# --- MAIN APP LOGIC ---
if __name__ == "__main__":
    bot = get_bot()

    st.title("🚀 Enhanced Multi-Strategy Trading Bot")
    st.caption("Professional-grade bot with Alpaca LIVE mode or simulation fallback")

    # Logging & status info
    if bot.is_simulation:
        st.warning("🔒 Running in simulation mode. No real trades or balances.")
        logger.warning("Simulation mode: Alpaca credentials missing or connection failed.")
    else:
        st.success("✅ Connected to Alpaca – LIVE account and positions loaded.")
        logger.info("LIVE mode: Alpaca connection established.")

    # --- Remainder of your Streamlit UI ---
    # Use your existing Streamlit code for UI. No change needed.

    # ... (Copy and paste your entire Streamlit UI code here as is) ...
    # The logic below will now work for both sim and live automatically!

    # For example:
    stats = bot.get_stats()

    # ... Everything below this is unchanged from your main section ...
    # (All your metric, UI, tab logic, etc.)

    # --- End of file ---
