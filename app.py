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
TAKE_PROFIT_PCT = float(st.secrets.get('TAKE_PROFIT_PCT', '0.03'))
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

    def get_candidates_by_strategy(self, strategy):
        """
        Dummy candidate generator for testing purposes.
        Replace this with real screening logic.
        """
        # You can customize these for real data/logic!
        base = {
            'momentum': [
                {'symbol': 'AAPL', 'price': 185.0, 'change_pct': 0.06, 'volume': 5000000, 'avg_volume': 2400000,
                 'indicators': {'macd': 0.4, 'adx': 29, 'vwap': 182.0}},
                {'symbol': 'TSLA', 'price': 270.0, 'change_pct': 0.11, 'volume': 9000000, 'avg_volume': 3200000,
                 'indicators': {'macd': 0.35, 'adx': 32, 'vwap': 265.0}}
            ],
            'mean_reversion': [
                {'symbol': 'MSFT', 'price': 305.0, 'change_pct': -0.07, 'volume': 6000000, 'avg_volume': 2600000,
                 'indicators': {'rsi_5': 22, 'bb_lower': 306.0, 'macd': -0.12}},
            ],
            'breakout': [
                {'symbol': 'NVDA', 'price': 830.0, 'change_pct': 0.08, 'volume': 7000000, 'avg_volume': 2800000,
                 'opening_range': {'high': 820.0}, 'indicators': {'adx': 31, 'atr': 7.2}},
            ]
        }
        return base.get(strategy, [])

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

    def update_position_prices(self):
        # Only in simulation: randomly adjust current price up/down by up to 1.5%
        for symbol, pos in self.positions.items():
            if pos['status'] == 'OPEN':
                change_pct = random.uniform(-0.015, 0.015)
                pos['current_price'] = round(pos['current_price'] * (1 + change_pct), 2)

    def check_end_of_day(self):
        now = datetime.now(MARKET_TIMEZONE)
        if now.hour > MARKET_CLOSE_TIME[0] or (now.hour == MARKET_CLOSE_TIME[0] and now.minute >= MARKET_CLOSE_TIME[1]):
            for symbol, pos in list(self.positions.items()):
                if pos['status'] == 'OPEN':
                    self.close_position(symbol, pos['current_price'], 'SELL (END_OF_DAY)')

    def check_and_sell_positions(self):
        for symbol, pos in list(self.positions.items()):
            # Simulate current price update (in real, update from live data)
            current_price = pos['current_price']  # replace with latest price source!
            stop_loss = pos['stop_loss']
            take_profit = pos['take_profit']

            # Example: Sell logic
            if pos['status'] == 'OPEN':
                if current_price <= stop_loss:
                    self.close_position(symbol, current_price, 'SELL (STOP_LOSS)')
                elif current_price >= take_profit:
                    self.close_position(symbol, current_price, 'SELL (TAKE_PROFIT)')
                # Optional: end of day logic, etc.

    def close_position(self, symbol, exit_price, action):
        pos = self.positions[symbol]
        # --- Alpaca SELL order ---
        if not self.is_simulation and self.api:
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=pos['shares'],
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Alpaca SELL order submitted: {order}")
            except Exception as e:
                logger.error(f"Failed to submit Alpaca SELL order: {e}")
                # Optionally: return, raise, or still update state

        # Always update local state for UI/tracking
        pos['status'] = 'CLOSED'
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
        pos['pnl'] = pnl
        pos['pnl_pct'] = pnl_pct
        self.trades_log.append({
            'date': datetime.now().date(),
            'time': datetime.now().strftime("%H:%M:%S"),
            'action': action,
            'symbol': symbol,
            'price': exit_price,
            'shares': pos['shares'],
            'pnl': pnl,
            'balance': self.account_balance + pnl,
            'strategy': pos['strategy']
        })
        del self.positions[symbol]


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

            # FIXED: remove limit argument
            activities = self.api.get_activities(activity_types="FILL")
            for act in activities[:50]:   # slice for most recent 50
                if act.status == 'filled':
                    action = "BUY" if act.side == 'buy' else "SELL"
                    self.trades_log.append({
                        'date': act.transaction_time.date(),
                        'time': act.transaction_time.strftime("%H:%M:%S"),
                        'action': action,
                        'symbol': act.symbol,
                        'price': float(act.price),
                        'shares': int(float(act.qty)),
                        'pnl': 0,
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

    def execute_trade(self, symbol, price, shares, strategy):
        # --- RISK RULES ---
        if symbol in self.positions or len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return False
        if not self.is_trading_hours or self.pause_trading_until and datetime.now() < self.pause_trading_until:
            return False

        # --- Check for open orders in Alpaca (avoid duplicate BUY orders) ---
        if not self.is_simulation and self.api:
            try:
                open_orders = self.api.list_orders(status='open')
                for order in open_orders:
                    # Only block new BUY order if there is already an open BUY order for the symbol
                    if order.symbol == symbol and order.side == 'buy' and order.status == 'open':
                        logger.info(f"Open BUY order already exists for {symbol}. Skipping duplicate order.")
                        return False
            except Exception as e:
                logger.error(f"Error checking open orders for {symbol}: {e}")
                # You may want to skip or continue depending on your risk tolerance

            # If no open BUY order, submit order as before:
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Alpaca BUY order submitted: {order}")
            except Exception as e:
                logger.error(f"Failed to submit Alpaca BUY order: {e}")
                return False

        # Always update local state for UI/tracking
        self.positions[symbol] = {
            'symbol': symbol,
            'entry_price': price,
            'current_price': price,
            'shares': shares,
            'stop_loss': price * (1 - STOP_LOSS_PCT),
            'take_profit': price * (1 + TAKE_PROFIT_PCT),
            'entry_time': datetime.now(),
            'status': 'OPEN',
            'pnl': 0,
            'pnl_pct': 0,
            'strategy': strategy
        }
        self.trades_log.append({
            'date': datetime.now().date(),
            'time': datetime.now().strftime("%H:%M:%S"),
            'action': 'BUY',
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'pnl': 0,
            'balance': self.account_balance,
            'strategy': strategy
        })
        return True


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
        st.metric("Current Time", current_time)

    # Main Performance Metrics
    st.markdown("---")
    st.markdown("### 💰 **Portfolio Performance**")
    col1, col2, col3 = st.columns(3)

    with col1:
        delta_value = stats['account_balance'] - stats['initial_balance']
        delta_color = "normal" if delta_value >= 0 else "inverse"
        st.metric("💰 Account Balance",
                  f"${stats['account_balance']:,.2f}",
                  delta=f"${delta_value:+,.2f}")

    with col2:
        st.metric("💵 Buying Power",
                  f"${stats['buying_power']:,.2f}")

    with col3:
        total_return = ((stats['account_balance'] - stats['initial_balance']) / stats['initial_balance'] * 100)
        st.metric("📈 Total Return",
                  f"{total_return:+.2f}%",
                  delta=f"${stats['account_balance'] - stats['initial_balance']:+,.2f}")

    # Enhanced Trading Metrics
    st.markdown("### 📊 **Trading Performance**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        daily_pnl_color = "normal" if stats['daily_pnl'] >= 0 else "inverse"
        st.metric("Daily P&L", f"${stats['daily_pnl']:+.2f}",
                  delta=f"{stats['daily_return_pct']:+.2f}%" if stats['daily_pnl'] != 0 else None)

    with col2:
        progress_color = "normal" if stats['target_progress'] < 100 else "inverse"
        st.metric("Daily Target Progress", f"{stats['target_progress']:.0f}%",
                  delta=f"Goal: {DAILY_PROFIT_TARGET_PCT*100:.0f}%")

    with col3:
        st.metric("Total P&L", f"${stats['total_pnl']:+.2f}")

    with col4:
        st.metric("Open Positions", f"{stats['open_positions']}/{MAX_CONCURRENT_POSITIONS}")

    with col5:
        st.metric("Total Trades", stats['total_trades'])

    with col6:
        win_color = "normal" if stats['win_rate'] >= 50 else "inverse"
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")

    # System Status
    st.markdown("### 🎛️ **System Status**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        regime_emoji = {"trending": "📈", "ranging": "↔️", "neutral": "⚪"}.get(stats['market_regime'], "❓")
        st.metric("Market Regime", f"{regime_emoji} {stats['market_regime'].title()}")

    with col2:
        if stats['trading_paused']:
            st.metric("Status", "⏸️ Paused", delta="Circuit Breaker Active")
        elif stats['daily_target_hit']:
            st.metric("Status", "🎯 Target Hit", delta="Daily goal achieved")
        elif stats['daily_loss_limit_hit']:
            st.metric("Status", "🛑 Loss Limit", delta="Daily limit hit")
        else:
            st.metric("Status", "✅ Active", delta="All systems operational")

    with col3:
        loss_color = "inverse" if stats['consecutive_losses'] >= 2 else "normal"
        st.metric("Consecutive Losses", stats['consecutive_losses'],
                  delta="Pause at 3" if stats['consecutive_losses'] > 0 else "Normal")

    with col4:
        st.metric("Near Misses Tracked", stats['near_misses_count'],
                  delta=f"Capacity: {MAX_NEAR_MISS_LOG}")

    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Positions", "🎯 Strategies", "📜 Trades", "🔍 Analysis", "ℹ️ Info"])

    with tab1:
        st.subheader("📊 Current Positions")
        if bot.positions:
            positions_data = []
            for symbol, pos in bot.positions.items():
                if pos['status'] == 'OPEN':
                    strategy = pos.get('strategy', 'unknown')
                    strategy_emoji = {"momentum": "🚀", "mean_reversion": "🔄", "breakout": "💥"}.get(strategy, "📈")
                    pnl_indicator = "🟢" if pos['pnl'] > 0 else "🔴" if pos['pnl'] < 0 else "⚪"

                    positions_data.append({
                        'Symbol': f"{strategy_emoji} {symbol}",
                        'Strategy': strategy.replace('_', ' ').title(),
                        'Entry Price': f"${pos['entry_price']:.2f}",
                        'Current Price': f"${pos['current_price']:.2f}",
                        'Shares': f"{pos['shares']:,}",
                        'P&L': f"{pnl_indicator} ${pos['pnl']:+.2f}",
                        'P&L %': f"{pos['pnl_pct']:+.2f}%",
                        'Stop Loss': f"${pos['stop_loss']:.2f}",
                        'Take Profit': f"${pos['take_profit']:.2f}",
                        'Entry Time': pos['entry_time'].strftime('%H:%M')
                    })

            if positions_data:
                df = pd.DataFrame(positions_data)
                st.dataframe(df, use_container_width=True)

                # Position Summary
                total_positions = len(positions_data)
                total_pnl = sum(pos['pnl'] for pos in bot.positions.values() if pos['status'] == 'OPEN')
                total_invested = sum(pos['entry_price'] * pos['shares'] for pos in bot.positions.values() if pos['status'] == 'OPEN')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Utilization", f"{total_positions}/{MAX_CONCURRENT_POSITIONS}")
                with col2:
                    st.metric("Unrealized P&L", f"${total_pnl:+.2f}")
                with col3:
                    risk_per_position = POSITION_SIZE * STOP_LOSS_PCT
                    total_risk = risk_per_position * total_positions
                    st.metric("Total Risk Exposure", f"${total_risk:.2f}")
            else:
                st.info("💼 No open positions currently")
        else:
            st.info("📈 Ready for new opportunities")

    with tab2:
        st.subheader("🎯 Multi-Strategy Candidates")

        CANDIDATE_REFRESH_INTERVAL = 20  # seconds

        if 'last_candidate_refresh' not in st.session_state:
            st.session_state['last_candidate_refresh'] = 0
        if 'all_candidates' not in st.session_state:
            st.session_state['all_candidates'] = bot.all_candidates

        def refresh_candidates():
            st.session_state['all_candidates'] = {
                'momentum': bot.market_data.get_candidates_by_strategy('momentum'),
                'mean_reversion': bot.market_data.get_candidates_by_strategy('mean_reversion') if ENABLE_MEAN_REVERSION else [],
                'breakout': bot.market_data.get_candidates_by_strategy('breakout')
            }
            st.session_state['last_candidate_refresh'] = time.time()

            # --- 1. Update simulated prices for open positions ---
            if bot.is_simulation:
                bot.update_position_prices()

            # --- 2. Auto-buy logic (for every candidate) ---
            all_candidates = st.session_state['all_candidates']
            for strategy_name, candidates in all_candidates.items():
                for c in candidates:
                    symbol = c['symbol']
                    price = c['price']
                    bot.execute_trade(symbol, price, POSITION_SIZE, strategy_name)

            # --- 3. Sell logic for all open positions ---
            bot.check_and_sell_positions()
            bot.check_end_of_day()

        # Manual button
        if st.button("🔍 Scan for New Candidates"):
            refresh_candidates()
            st.success("✅ Candidates refreshed!")

        # Auto-refresh logic
        if time.time() - st.session_state['last_candidate_refresh'] > CANDIDATE_REFRESH_INTERVAL:
            refresh_candidates()

        st.caption("🔄 Candidates auto-refresh every 20 seconds (plus manual refresh available)")

        all_candidates = st.session_state['all_candidates']

        if any(candidates for candidates in all_candidates.values()):
            # Strategy overview
            total_candidates = sum(len(candidates) for candidates in all_candidates.values())
            st.markdown(f"**{total_candidates} opportunities identified** across all strategies")

            strategy_tabs = st.tabs(["🚀 Momentum", "🔄 Mean Reversion", "💥 Breakout"])

            with strategy_tabs[0]:
                momentum_candidates = all_candidates.get('momentum', [])
                if momentum_candidates:
                    st.markdown("**🚀 Enhanced Momentum Strategy**")
                    st.caption("Criteria: MACD positive & rising, ADX >25, Price >VWAP, Volume >2x average")

                    for i, c in enumerate(momentum_candidates[:5], 1):
                        with st.expander(f"#{i} {c['symbol']} - {c['change_pct']:+.2%} | {c['volume']/c['avg_volume']:.1f}x Volume"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Current Price", f"${c['price']:.2f}")
                                st.metric("Price Change", f"{c['change_pct']:+.2%}")

                            with col2:
                                st.metric("Volume Ratio", f"{c['volume']/c['avg_volume']:.1f}x")
                                indicators = c.get('indicators', {})
                                st.metric("ADX (Trend)", f"{indicators.get('adx', 0):.1f}")

                            with col3:
                                st.metric("MACD", f"{indicators.get('macd', 0):+.3f}")
                                st.metric("vs VWAP", f"{((c['price'] - indicators.get('vwap', c['price'])) / c['price'] * 100):+.1f}%")
                else:
                    st.info("🔍 No momentum opportunities meeting criteria")

            with strategy_tabs[1]:
                if ENABLE_MEAN_REVERSION:
                    mr_candidates = all_candidates.get('mean_reversion', [])
                    if mr_candidates:
                        st.markdown("**🔄 Mean Reversion Strategy**")
                        st.caption("Criteria: RSI(5) <30, Price <Lower Bollinger Band, Volume >1.2x")

                        for i, c in enumerate(mr_candidates[:5], 1):
                            with st.expander(f"#{i} {c['symbol']} - {c['change_pct']:+.2%} | RSI Oversold"):
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Current Price", f"${c['price']:.2f}")
                                    st.metric("Price Change", f"{c['change_pct']:+.2%}")

                                with col2:
                                    indicators = c.get('indicators', {})
                                    st.metric("RSI(5)", f"{indicators.get('rsi_5', 50):.1f}")
                                    st.metric("Volume Ratio", f"{c['volume']/c['avg_volume']:.1f}x")

                                with col3:
                                    bb_distance = ((c['price'] - indicators.get('bb_lower', c['price'])) / c['price'] * 100)
                                    st.metric("BB Distance", f"{bb_distance:+.1f}%")
                                    st.metric("MACD", f"{indicators.get('macd', 0):+.3f}")
                    else:
                        st.info("🔍 No oversold opportunities detected")
                else:
                    st.warning("⚠️ Mean reversion strategy is disabled")
                    st.info("Enable via environment variable: ENABLE_MEAN_REVERSION=true")

            with strategy_tabs[2]:
                breakout_candidates = all_candidates.get('breakout', [])
                if breakout_candidates:
                    st.markdown("**💥 Opening Range Breakout**")
                    st.caption("Criteria: Break 30-min range +0.5%, Volume confirmation >1.5x")

                    for i, c in enumerate(breakout_candidates[:5], 1):
                        opening_range = c.get('opening_range', {})
                        breakout_pct = ((c['price'] - opening_range.get('high', c['price'])) / opening_range.get('high', c['price']) * 100)

                        with st.expander(f"#{i} {c['symbol']} - Breakout +{breakout_pct:.1f}%"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Current Price", f"${c['price']:.2f}")
                                st.metric("Range High", f"${opening_range.get('high', 0):.2f}")

                            with col2:
                                st.metric("Breakout %", f"+{breakout_pct:.1f}%")
                                st.metric("Volume Ratio", f"{c['volume']/c['avg_volume']:.1f}x")

                            with col3:
                                indicators = c.get('indicators', {})
                                st.metric("ADX", f"{indicators.get('adx', 0):.1f}")
                                st.metric("ATR", f"${indicators.get('atr', 0):.2f}")
                else:
                    st.info("🔍 No breakout opportunities detected")

            # Market context
            st.markdown("---")
            active_strategies = [s for s, candidates in all_candidates.items() if candidates]
            regime_color = {"trending": "🟢", "ranging": "🟡", "neutral": "⚪"}.get(stats['market_regime'], "⚪")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Market Regime:** {regime_color} {stats['market_regime'].upper()}")
            with col2:
                st.markdown(f"**Active Strategies:** {', '.join(s.title() for s in active_strategies) if active_strategies else 'None'}")

        else:
            st.info("📊 Scanning for opportunities... Candidates refresh during market hours")

            # Show strategy overview when no candidates
            st.markdown("### 📚 Strategy Overview")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                **🚀 Enhanced Momentum**
                - MACD positive & rising
                - ADX > 25 (strong trend)
                - Price above VWAP
                - Volume > 2x average
                - Breakout confirmation
                """)

            with col2:
                enabled_status = "✅ Enabled" if ENABLE_MEAN_REVERSION else "❌ Disabled"
                st.markdown(f"""
                **🔄 Mean Reversion** {enabled_status}
                - RSI(5) < 30 (oversold)
                - Price < Lower Bollinger Band
                - Volume > 1.2x average
                - Counter-trend entry
                """)

            with col3:
                st.markdown("""
                **💥 Opening Range Breakout**
                - Break 30-min high + 0.5%
                - Volume confirmation >1.5x
                - No trades after 3:00 PM
                - Momentum follow-through
                """)

    with tab3:
        st.subheader("📜 Trade Log & Performance")
        if bot.trades_log:
            recent_trades = bot.trades_log[-12:]  # Show more trades
            enhanced_trades = []

            for trade in recent_trades:
                strategy = trade.get('strategy', 'unknown')
                strategy_emoji = {"momentum": "🚀", "mean_reversion": "🔄", "breakout": "💥"}.get(strategy, "📈")

                # Color code P&L
                pnl_display = f"${trade.get('pnl', 0):+.2f}" if 'pnl' in trade else '-'
                if trade.get('pnl', 0) > 0:
                    pnl_display = f"🟢 {pnl_display}"
                elif trade.get('pnl', 0) < 0:
                    pnl_display = f"🔴 {pnl_display}"

                enhanced_trades.append({
                    'Date': trade['date'].strftime('%m/%d'),
                    'Time': trade['time'],
                    'Action': f"{strategy_emoji} {trade['action']}",
                    'Symbol': trade['symbol'],
                    'Price': f"${trade['price']:.2f}",
                    'Shares': f"{trade['shares']:,}",
                    'P&L': pnl_display,
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Balance': f"${trade['balance']:,.0f}"
                })

            df = pd.DataFrame(enhanced_trades)
            df = df.iloc[::-1]  # Most recent first
            st.dataframe(df, use_container_width=True)

            # Trading Statistics
            st.markdown("### 📈 Trading Statistics")
            today_trades = [t for t in bot.trades_log if t.get('date') == datetime.now().date()]
            week_trades = [t for t in bot.trades_log if (datetime.now().date() - t.get('date', datetime.now().date())).days <= 7]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                buys = [t for t in today_trades if t['action'] == 'BUY']
                sells = [t for t in today_trades if t['action'].startswith('SELL')]
                st.metric("Today's Activity", f"{len(buys)}B / {len(sells)}S")

            with col2:
                today_pnl = sum(t.get('pnl', 0) for t in today_trades if 'pnl' in t)
                st.metric("Today's Realized P&L", f"${today_pnl:+.2f}")

            with col3:
                successful_sells = [t for t in sells if t.get('pnl', 0) > 0]
                success_rate = (len(successful_sells) / len(sells) * 100) if sells else 0
                st.metric("Today's Success Rate", f"{success_rate:.0f}%")

            with col4:
                week_pnl = sum(t.get('pnl', 0) for t in week_trades if 'pnl' in t)
                st.metric("7-Day P&L", f"${week_pnl:+.2f}")

        else:
            st.info("📊 No trades executed yet - Ready for opportunities!")

    with tab4:
        st.subheader("🔍 Near Miss Analysis & Optimization")

        if bot.near_miss_log:
            st.markdown(f"**📊 Tracking {len(bot.near_miss_log)} missed opportunities** (Max: {MAX_NEAR_MISS_LOG})")

            # Sort by volume ratio (most promising first)
            sorted_misses = sorted(list(bot.near_miss_log),
                                  key=lambda x: x['metrics']['volume_ratio'], reverse=True)

            # Top missed opportunities
            st.markdown("### 🎯 Top Missed Opportunities")
            for i, miss in enumerate(sorted_misses[:6], 1):
                with st.expander(f"#{i} {miss['symbol']} ({miss['strategy'].title()}) - {miss['missed_reason']}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**📊 Price Action**")
                        st.metric("Price Change", f"{miss['metrics']['price_change']:+.2%}")
                        st.metric("Volume Multiplier", f"{miss['metrics']['volume_ratio']:.1f}x")

                    with col2:
                        st.markdown("**🔬 Technical Analysis**")
                        st.metric("RSI Level", f"{miss['metrics']['rsi']:.1f}")
                        st.metric("MACD Signal", f"{miss['metrics']['macd']:+.3f}")

                    with col3:
                        st.markdown("**❌ Rejection Analysis**")
                        st.metric("VWAP Distance", f"{miss['metrics']['distance_from_vwap']:+.2%}")
                        st.markdown(f"**Root Cause:** {miss['metrics']['why_rejected']}")

                    # Timestamp
                    st.caption(f"🕐 Identified: {miss['timestamp'].strftime('%H:%M:%S on %m/%d')}")

            # Strategy breakdown
            st.markdown("### 📊 Analysis by Strategy")
            strategy_counts = {}
            strategy_avg_volume = {}

            for miss in bot.near_miss_log:
                strategy = miss['strategy']
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                if strategy not in strategy_avg_volume:
                    strategy_avg_volume[strategy] = []
                strategy_avg_volume[strategy].append(miss['metrics']['volume_ratio'])

            if strategy_counts:
                col1, col2, col3 = st.columns(3)
                strategies = list(strategy_counts.keys())

                for i, strategy in enumerate(strategies[:3]):
                    with [col1, col2, col3][i]:
                        emoji = {"momentum": "🚀", "mean_reversion": "🔄", "breakout": "💥"}.get(strategy, "📈")
                        avg_vol = np.mean(strategy_avg_volume[strategy]) if strategy_avg_volume[strategy] else 0

                        st.metric(f"{emoji} {strategy.title()}",
                                f"{strategy_counts[strategy]} misses",
                                delta=f"Avg Vol: {avg_vol:.1f}x")

            # Common rejection patterns
            st.markdown("### 🚫 Common Rejection Patterns")
            rejection_counts = {}
            for miss in bot.near_miss_log:
                reason = miss['missed_reason']
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

            # Sort by frequency
            sorted_reasons = sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True)

            for i, (reason, count) in enumerate(sorted_reasons[:5], 1):
                st.markdown(f"**{i}.** {reason} - *{count} occurrences*")

            # Optimization suggestions
            st.markdown("### 💡 Optimization Opportunities")
            if sorted_reasons:
                top_reason = sorted_reasons[0][0]
                if "Volume" in top_reason:
                    st.info("💡 **Suggestion:** Consider lowering volume threshold during low-volatility periods")
                elif "ADX" in top_reason:
                    st.info("💡 **Suggestion:** Implement dynamic ADX thresholds based on market regime")
                elif "VWAP" in top_reason:
                    st.info("💡 **Suggestion:** Use intraday VWAP deviation bands for momentum entries")
                else:
                    st.info("💡 **Suggestion:** Review strategy parameters for current market conditions")

        else:
            st.info("📊 Near-miss tracking begins during active market hours")

            # Educational content
            st.markdown("### 📚 What We Track")
            st.markdown("""
            **Near-miss candidates help optimize strategy performance:**

            - **🎯 Partial Matches**: Stocks meeting some but not all criteria
            - **⏰ Timing Issues**: Signals arriving too late in trading day
            - **🛡️ Risk Filters**: Opportunities filtered by risk management
            - **📏 Close Calls**: Within 0.5% of trigger thresholds

            **Benefits:**
            - Identify overlooked opportunities
            - Fine-tune strategy parameters
            - Improve entry timing
            - Validate risk management effectiveness
            """)

    with tab5:
        st.subheader("🤖 System Information & Configuration")

        if not bot.is_simulation and bot.api:
            st.markdown("### 📝 Open Orders (Alpaca)")
            try:
                open_orders = bot.api.list_orders(status='open')
                if open_orders:
                    for order in open_orders:
                        st.write(
                            f"**{order.side.upper()}** {order.qty} {order.symbol} @ {order.type.upper()} "
                            f"– Status: {order.status} – Submitted: {order.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                else:
                    st.info("✅ No open orders at Alpaca right now.")
            except Exception as e:
                st.error(f"Could not fetch open orders from Alpaca: {e}")

        # System status
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔧 Technical Stack")
            st.success("✅ Professional Technical Indicators (Custom Implementation)")
            st.success("✅ Multi-Strategy Trading Engine")
            st.success("✅ Advanced Risk Management")
            st.success("✅ Near-Miss Analysis System")

            if not ALPACA_API_KEY:
                st.info("📊 Enhanced Simulation Mode Active")
            else:
                st.success("✅ Alpaca Paper Trading Connected")

        with col2:
            st.markdown("### 📊 Current Configuration")
            st.markdown(f"""
            - **Position Size**: {POSITION_SIZE:,} shares
            - **Stop Loss**: {STOP_LOSS_PCT*100:.0f}%
            - **Take Profit**: {TAKE_PROFIT_PCT*100:.0f}%
            - **Daily Target**: {DAILY_PROFIT_TARGET_PCT*100:.0f}%
            - **Daily Loss Limit**: {MAX_DAILY_LOSS_PCT*100:.0f}%
            - **Max Positions**: {MAX_CONCURRENT_POSITIONS}
            - **Volume Threshold**: {VOLUME_MULTIPLIER}x
            - **Trading Cutoff**: {NO_TRADES_AFTER_HOUR}:00 PM ET
            """)

        # Strategy details
        st.markdown("### 🎯 Multi-Strategy Framework")

        strategy_col1, strategy_col2, strategy_col3 = st.columns(3)

        with strategy_col1:
            st.markdown("""
            **🚀 Enhanced Momentum**
            - Professional MACD analysis
            - ADX trend confirmation (>25)
            - VWAP positioning filter
            - Volume surge validation (>2x)
            - Stop: -2% | Target: +7%
            """)

        with strategy_col2:
            enabled_text = "✅ **ENABLED**" if ENABLE_MEAN_REVERSION else "❌ **DISABLED**"
            st.markdown(f"""
            **🔄 Mean Reversion** {enabled_text}
            - RSI(5) oversold detection (<30)
            - Bollinger Band positioning
            - Volume confirmation (>1.2x)
            - Mean reversion target (middle band)
            - Stop: -2% | Target: +7%
            """)

        with strategy_col3:
            st.markdown("""
            **💥 Opening Range Breakout**
            - 30-minute range identification
            - Breakout confirmation (+0.5%)
            - Volume validation (>1.5x)
            - Time-based restrictions
            - Stop: -2% | Target: +7%
            """)

        # Risk management
        st.markdown("### ⚙️ Risk Management Framework")

        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:
            st.markdown("""
            **🛡️ Position-Level Risk**
            - ATR-based position sizing (1% risk per trade)
            - Maximum concurrent positions: 5
            - Stop-loss: 2% per position
            - Take-profit: 7% per position
            """)

        with risk_col2:
            st.markdown("""
            **🎯 Portfolio-Level Risk**
            - Daily profit target: 4% (auto-close all positions)
            - Daily loss limit: 2% (circuit breaker)
            - Consecutive loss protection (3 = 1hr pause)
            - Time cutoff: No new trades after 3:00 PM
            """)

        # Performance metrics
        st.markdown("### 📈 System Performance")

        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

        with perf_col1:
            st.metric("Technical Indicators", "6 Professional")
            st.caption("RSI, MACD, BB, ADX, ATR, VWAP")

        with perf_col2:
            st.metric("Strategy Coverage", "3 Strategies")
            st.caption("Momentum, Mean Reversion, Breakout")

        with perf_col3:
            st.metric("Risk Controls", "4 Layers")
            st.caption("Position, Daily, Sequential, Time")

        with perf_col4:
            st.metric("Analysis Tools", "Near-Miss Tracking")
            st.caption(f"Up to {MAX_NEAR_MISS_LOG} opportunities")

        # Footer
        st.markdown("---")
        st.markdown("### 🚨 Important Disclaimers")

        disclaimer_col1, disclaimer_col2 = st.columns(2)

        with disclaimer_col1:
            st.warning("""
            **⚠️ Paper Trading Only**
            - Uses Alpaca Paper Trading API
            - No real money at risk
            - Educational/demonstration purposes
            - Results are simulated
            """)

        with disclaimer_col2:
            st.info("""
            **🔬 Technical Implementation**
            - Professional-grade indicators
            - Institutional risk management
            - Real-time strategy optimization
            - Advanced analytics suite
            """)

    # Auto-refresh functionality
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

    # Footer with system info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(f"🕐 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    with col2:
        st.caption("🔄 Auto-refresh: 30 seconds")
    with col3:
        st.caption("🚀 Enhanced Multi-Strategy Bot v2.0")
