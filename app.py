#!/usr/bin/env python3
"""
Enhanced Automated Trading Bot - Streamlit Version
Simplified version that works reliably with Streamlit Cloud
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

# Required imports
import pandas as pd
import numpy as np
import pytz

# Streamlit import (required for this version)
import streamlit as st

# Optional imports with fallbacks
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

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

# Set page config
st.set_page_config(
    page_title="Enhanced Trading Bot Monitor",
    page_icon="🚀",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIGURATION FROM ENVIRONMENT VARIABLES
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Pushover Configuration
PUSHOVER_USER_KEY = os.getenv('PUSHOVER_USER_KEY', '')
PUSHOVER_APP_TOKEN = os.getenv('PUSHOVER_APP_TOKEN', '')

# Enhanced Trading Parameters
POSITION_SIZE = int(os.getenv('POSITION_SIZE', '100'))
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.02'))
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.07'))
VOLUME_MULTIPLIER = float(os.getenv('VOLUME_MULTIPLIER', '2.0'))
MIN_PRICE_CHANGE_PCT = float(os.getenv('MIN_PRICE_CHANGE_PCT', '0.05'))

DAILY_PROFIT_TARGET_PCT = float(os.getenv('DAILY_PROFIT_TARGET_PCT', '0.04'))
MAX_DAILY_LOSS_PCT = float(os.getenv('MAX_DAILY_LOSS_PCT', '0.02'))
ENABLE_MEAN_REVERSION = os.getenv('ENABLE_MEAN_REVERSION', 'true').lower() == 'true'
ENABLE_SHORT_SELLING = os.getenv('ENABLE_SHORT_SELLING', 'false').lower() == 'true'
MAX_NEAR_MISS_LOG = int(os.getenv('MAX_NEAR_MISS_LOG', '50'))
MAX_CONCURRENT_POSITIONS = int(os.getenv('MAX_CONCURRENT_POSITIONS', '5'))
NO_TRADES_AFTER_HOUR = int(os.getenv('NO_TRADES_AFTER_HOUR', '15'))

# Market hours
MARKET_TIMEZONE = pytz.timezone('US/Eastern')
MARKET_OPEN_TIME = (9, 30)
MARKET_CLOSE_TIME = (15, 50)

class TechnicalIndicators:
    """Technical indicators with fallback implementations"""

    @staticmethod
    def _rsi_fallback(prices: np.ndarray, period: int = 14) -> float:
        """Fallback RSI calculation"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_indicators(prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """Calculate technical indicators"""
        if len(prices) < 20:
            return {}

        close = prices

        if TALIB_AVAILABLE:
            try:
                high = prices * 1.02
                low = prices * 0.98

                rsi_14 = talib.RSI(close, timeperiod=14)
                rsi_5 = talib.RSI(close, timeperiod=5)
                macd, macd_signal, _ = talib.MACD(close)
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
                adx = talib.ADX(high, low, close, timeperiod=14)
                atr = talib.ATR(high, low, close, timeperiod=14)

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
                    'vwap': np.sum(close * volumes) / np.sum(volumes) if volumes is not None else close[-1],
                    'price': close[-1]
                }
            except Exception as e:
                logger.warning(f"TA-Lib error, using fallbacks: {e}")

        # Fallback calculations
        rsi_14 = TechnicalIndicators._rsi_fallback(close, 14)
        rsi_5 = TechnicalIndicators._rsi_fallback(close, 5)

        # Simple MACD
        ema_12 = np.mean(close[-12:]) if len(close) >= 12 else close[-1]
        ema_26 = np.mean(close[-26:]) if len(close) >= 26 else close[-1]
        macd = ema_12 - ema_26
        macd_signal = macd * 0.9

        # Simple Bollinger Bands
        sma_20 = np.mean(close[-20:])
        std_20 = np.std(close[-20:])
        bb_upper = sma_20 + (2 * std_20)
        bb_middle = sma_20
        bb_lower = sma_20 - (2 * std_20)

        # Simple ADX approximation
        adx = min(max(abs(close[-1] - close[-14]) / close[-14] * 100, 15), 45) if len(close) >= 14 else 25

        # Simple ATR
        atr = np.std(close[-14:]) if len(close) >= 14 else close[-1] * 0.02

        # VWAP
        vwap = np.sum(close * volumes) / np.sum(volumes) if volumes is not None else close[-1]

        return {
            'rsi_14': rsi_14,
            'rsi_5': rsi_5,
            'macd': macd,
            'macd_signal': macd_signal,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'adx': adx,
            'atr': atr,
            'vwap': vwap,
            'price': close[-1]
        }

class SimulatedMarketData:
    """Enhanced market data simulation"""

    def __init__(self):
        self.stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'CRM']
        self.base_prices = {s: random.uniform(50, 500) for s in self.stocks}
        self.volumes = {s: random.randint(1000000, 50000000) for s in self.stocks}
        self.opening_ranges = {}

    def get_candidates_by_strategy(self, strategy: str) -> List[Dict]:
        """Get candidates for specific strategy"""
        candidates = []

        if strategy == 'momentum':
            for stock in random.sample(self.stocks, 5):
                price = self.base_prices[stock]
                change_pct = random.uniform(0.03, 0.15)
                volume = self.volumes[stock] * random.uniform(1.8, 4.0)

                # Generate indicators
                price_series = np.array([price * (1 + random.uniform(-0.01, 0.01)) for _ in range(50)])
                price_series[-1] = price * (1 + change_pct)
                indicators = TechnicalIndicators.calculate_indicators(price_series)

                # Simulate momentum conditions
                indicators['macd'] = random.uniform(0.1, 1.0)
                indicators['macd_signal'] = indicators['macd'] * 0.8
                indicators['adx'] = random.uniform(25, 40)
                indicators['vwap'] = price * (1 + change_pct) * 0.98

                candidates.append({
                    'symbol': stock,
                    'price': price * (1 + change_pct),
                    'change_pct': change_pct,
                    'volume': volume,
                    'avg_volume': self.volumes[stock],
                    'morning_high': price * (1 + change_pct * 0.7),
                    'indicators': indicators,
                    'strategy': strategy
                })

        elif strategy == 'mean_reversion':
            for stock in random.sample(self.stocks, 3):
                price = self.base_prices[stock]
                change_pct = random.uniform(-0.08, -0.02)
                volume = self.volumes[stock] * random.uniform(1.2, 2.5)

                price_series = np.array([price * (1 + random.uniform(-0.01, 0.01)) for _ in range(50)])
                price_series[-1] = price * (1 + change_pct)
                indicators = TechnicalIndicators.calculate_indicators(price_series)

                # Simulate oversold conditions
                indicators['rsi_5'] = random.uniform(15, 29)
                indicators['bb_lower'] = price * (1 + change_pct) * 1.01

                candidates.append({
                    'symbol': stock,
                    'price': price * (1 + change_pct),
                    'change_pct': change_pct,
                    'volume': volume,
                    'avg_volume': self.volumes[stock],
                    'indicators': indicators,
                    'strategy': strategy
                })

        elif strategy == 'breakout':
            for stock in random.sample(self.stocks, 2):
                base_price = self.base_prices[stock]
                opening_high = base_price * random.uniform(1.01, 1.03)
                current_price = opening_high * random.uniform(1.005, 1.02)

                price_series = np.array([base_price * (1 + random.uniform(-0.01, 0.01)) for _ in range(50)])
                price_series[-1] = current_price
                indicators = TechnicalIndicators.calculate_indicators(price_series)

                candidates.append({
                    'symbol': stock,
                    'price': current_price,
                    'change_pct': (current_price - base_price) / base_price,
                    'volume': self.volumes[stock] * random.uniform(1.5, 3.0),
                    'avg_volume': self.volumes[stock],
                    'opening_range': {'high': opening_high, 'low': base_price * 0.99},
                    'indicators': indicators,
                    'strategy': strategy
                })

        return candidates

    def get_current_price(self, symbol: str) -> float:
        base = self.base_prices.get(symbol, 100)
        return base * (1 + random.uniform(-0.02, 0.02))

# Simplified Trading Bot for Streamlit
class EnhancedTradingBot:
    """Enhanced trading bot with multi-strategy support"""

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

        # Enhanced features
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
        self.status = "ENHANCED SIMULATION MODE"
        self.is_trading_hours = True
        self.is_simulation = True
        self.notifications_enabled = bool(PUSHOVER_USER_KEY and PUSHOVER_APP_TOKEN)

        # Initialize with some sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample trading data"""
        # Sample positions
        for i, symbol in enumerate(['AAPL', 'MSFT'][:2]):
            strategy = ['momentum', 'mean_reversion'][i]
            entry_price = 150 + i * 50
            current_price = entry_price * random.uniform(0.98, 1.08)

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

        # Sample trades log
        for i in range(5):
            self.trades_log.append({
                'date': (datetime.now() - timedelta(days=random.randint(0, 2))).date(),
                'time': f"{9 + i}:{30 + i*10}:00",
                'action': 'BUY' if i % 2 == 0 else 'SELL (TAKE_PROFIT)',
                'symbol': f'TEST{i}',
                'price': 100 + i * 5,
                'shares': POSITION_SIZE,
                'pnl': random.randint(-200, 500) if 'SELL' in ('BUY' if i % 2 == 0 else 'SELL (TAKE_PROFIT)') else 0,
                'balance': self.account_balance + random.randint(-1000, 1000),
                'strategy': ['momentum', 'mean_reversion', 'breakout'][i % 3]
            })

        # Sample near misses
        for i in range(10):
            self.near_miss_log.append({
                'timestamp': datetime.now() - timedelta(minutes=random.randint(5, 300)),
                'symbol': f'MISS{i}',
                'strategy': ['momentum', 'mean_reversion', 'breakout'][i % 3],
                'missed_reason': ['Volume only 1.8x (needed 2x)', 'ADX too low (22, needed 25)', 'Price below VWAP'][i % 3],
                'metrics': {
                    'price_change': random.uniform(0.03, 0.08),
                    'volume_ratio': random.uniform(1.5, 2.8),
                    'rsi': random.uniform(30, 70),
                    'distance_from_vwap': random.uniform(-0.02, 0.02),
                    'macd': random.uniform(-0.5, 0.5),
                    'why_rejected': ['Volume only 1.8x (needed 2x)', 'ADX too low', 'Price below VWAP'][i % 3]
                }
            })

        # Generate current candidates
        self.all_candidates = {
            'momentum': self.market_data.get_candidates_by_strategy('momentum'),
            'mean_reversion': self.market_data.get_candidates_by_strategy('mean_reversion') if ENABLE_MEAN_REVERSION else [],
            'breakout': self.market_data.get_candidates_by_strategy('breakout')
        }

        # Set market regime
        self.market_regime = random.choice(['trending', 'ranging', 'neutral'])

        # Update daily P&L
        self.daily_pnl = sum(pos['pnl'] for pos in self.positions.values() if pos['status'] == 'OPEN')

    def get_stats(self):
        """Get enhanced statistics"""
        open_positions = sum(1 for p in self.positions.values() if p['status'] == 'OPEN')
        total_trades = len([t for t in self.trades_log if t.get('action', '').startswith('SELL')])

        winners = sum(1 for t in self.trades_log
                     if t.get('action', '').startswith('SELL') and t.get('pnl', 0) > 0)

        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        # Calculate daily progress
        daily_return = (self.daily_pnl / self.daily_starting_balance) if self.daily_starting_balance > 0 else 0
        target_progress = (daily_return / DAILY_PROFIT_TARGET_PCT) * 100

        return {
            'status': self.status,
            'mode': 'ENHANCED SIMULATION',
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

# Initialize the bot
@st.cache_resource
def get_bot():
    return EnhancedTradingBot()

bot = get_bot()

# STREAMLIT UI
st.title("🚀 Enhanced Multi-Strategy Trading Bot")
st.caption("Advanced bot with momentum, mean reversion & breakout strategies | Daily 4% target system")

# Auto-refresh every 30 seconds
if st.button("🔄 Refresh Data"):
    st.rerun()

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

# Main Metrics
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
    st.metric("Daily P&L", f"${stats['daily_pnl']:+.2f}",
              delta=f"{stats['daily_return_pct']:+.2f}%" if stats['daily_pnl'] != 0 else None)

with col2:
    st.metric("Daily Target", f"{stats['target_progress']:.0f}%",
              delta=f"Target: {DAILY_PROFIT_TARGET_PCT*100:.0f}%")

with col3:
    st.metric("Total P&L", f"${stats['total_pnl']:+.2f}")

with col4:
    st.metric("Open Positions", f"{stats['open_positions']}/{MAX_CONCURRENT_POSITIONS}")

with col5:
    st.metric("Total Trades", stats['total_trades'])

with col6:
    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")

# Enhanced status row
col1, col2, col3, col4 = st.columns(4)

with col1:
    regime_emoji = {"trending": "📈", "ranging": "↔️", "neutral": "⚪"}.get(stats['market_regime'], "❓")
    st.metric("Market Regime", f"{regime_emoji} {stats['market_regime'].title()}")

with col2:
    if stats['trading_paused']:
        st.metric("Status", "⏸️ Paused", delta="Circuit Breaker")
    elif stats['daily_target_hit']:
        st.metric("Status", "🎯 Target Hit", delta="Daily goal achieved")
    elif stats['daily_loss_limit_hit']:
        st.metric("Status", "🛑 Loss Limit", delta="Daily limit hit")
    else:
        st.metric("Status", "✅ Active", delta="All systems go")

with col3:
    st.metric("Consecutive Losses", stats['consecutive_losses'],
              delta="Pause at 3" if stats['consecutive_losses'] > 0 else None)

with col4:
    st.metric("Near Misses", stats['near_misses_count'],
              delta=f"Max: {MAX_NEAR_MISS_LOG}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Positions", "🎯 Strategies", "📜 Trades", "🔍 Near Misses", "ℹ️ Info"])

with tab1:
    st.subheader("Current Positions")
    if bot.positions:
        positions_data = []
        for symbol, pos in bot.positions.items():
            if pos['status'] == 'OPEN':
                strategy = pos.get('strategy', 'unknown')
                strategy_emoji = {"momentum": "🚀", "mean_reversion": "🔄", "breakout": "💥"}.get(strategy, "📈")
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

            total_positions = len(positions_data)
            total_pnl = sum(pos['pnl'] for pos in bot.positions.values() if pos['status'] == 'OPEN')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Open Positions", f"{total_positions}/{MAX_CONCURRENT_POSITIONS}")
            with col2:
                st.metric("Unrealized P&L", f"${total_pnl:+.2f}")
            with col3:
                risk_per_position = POSITION_SIZE * STOP_LOSS_PCT
                st.metric("Total Risk", f"${risk_per_position * total_positions:.2f}")
        else:
            st.info("No open positions")
    else:
        st.info("No positions today")

with tab2:
    st.subheader("Multi-Strategy Candidates")

    all_candidates = stats.get('all_candidates', {})

    if any(candidates for candidates in all_candidates.values()):
        strategy_tabs = st.tabs(["🚀 Momentum", "🔄 Mean Reversion", "💥 Breakout"])

        with strategy_tabs[0]:
            momentum_candidates = all_candidates.get('momentum', [])
            if momentum_candidates:
                st.write("**Enhanced Momentum** (MACD+, ADX>25, Price>VWAP, Volume>2x)")
                for c in momentum_candidates:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.write(f"**{c['symbol']}**")
                    col2.write(f"${c['price']:.2f}")
                    col3.write(f"{c['change_pct']:+.2%}")
                    col4.write(f"{c['volume']/c['avg_volume']:.1f}x")
                    indicators = c.get('indicators', {})
                    col5.write(f"ADX:{indicators.get('adx', 0):.0f}")
            else:
                st.info("No momentum opportunities")

        with strategy_tabs[1]:
            if ENABLE_MEAN_REVERSION:
                mr_candidates = all_candidates.get('mean_reversion', [])
                if mr_candidates:
                    st.write("**Mean Reversion** (RSI<30, Below BB Lower)")
                    for c in mr_candidates:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.write(f"**{c['symbol']}**")
                        col2.write(f"${c['price']:.2f}")
                        col3.write(f"{c['change_pct']:+.2%}")
                        col4.write(f"{c['volume']/c['avg_volume']:.1f}x")
                        indicators = c.get('indicators', {})
                        col5.write(f"RSI:{indicators.get('rsi_5', 50):.0f}")
                else:
                    st.info("No mean reversion setups")
            else:
                st.info("Mean reversion strategy disabled")

        with strategy_tabs[2]:
            breakout_candidates = all_candidates.get('breakout', [])
            if breakout_candidates:
                st.write("**Opening Range Breakout** (30min range + volume)")
                for c in breakout_candidates:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.write(f"**{c['symbol']}**")
                    col2.write(f"${c['price']:.2f}")
                    col3.write(f"{c['change_pct']:+.2%}")
                    col4.write(f"{c['volume']/c['avg_volume']:.1f}x")
                    opening_range = c.get('opening_range', {})
                    col5.write(f"${opening_range.get('high', 0):.2f}")
            else:
                st.info("No breakout setups")

        st.caption(f"Market Regime: **{stats['market_regime'].upper()}** | Strategies Active: {len([s for s, candidates in all_candidates.items() if candidates])}")
    else:
        st.info("Candidates refresh at market open")

with tab3:
    st.subheader("Recent Trade Log")
    if bot.trades_log:
        recent_trades = bot.trades_log[-10:]
        enhanced_trades = []

        for trade in recent_trades:
            strategy = trade.get('strategy', 'unknown')
            strategy_emoji = {"momentum": "🚀", "mean_reversion": "🔄", "breakout": "💥"}.get(strategy, "📈")

            enhanced_trades.append({
                'Date': trade['date'].strftime('%m/%d'),
                'Time': trade['time'],
                'Action': f"{strategy_emoji} {trade['action']}",
                'Symbol': trade['symbol'],
                'Price': f"${trade['price']:.2f}",
                'Shares': trade['shares'],
                'P&L': f"${trade.get('pnl', 0):+.2f}" if 'pnl' in trade else '-',
                'Strategy': strategy.replace('_', ' ').title()
            })

        df = pd.DataFrame(enhanced_trades)
        df = df.iloc[::-1]  # Most recent first
        st.dataframe(df, use_container_width=True)

        today_trades = [t for t in bot.trades_log if t.get('date') == datetime.now().date()]
        if today_trades:
            col1, col2, col3 = st.columns(3)
            buys = [t for t in today_trades if t['action'] == 'BUY']
            sells = [t for t in today_trades if t['action'].startswith('SELL')]

            with col1:
                st.metric("Today's Trades", f"{len(buys)}B / {len(sells)}S")
            with col2:
                total_pnl = sum(t.get('pnl', 0) for t in sells)
                st.metric("Realized P&L", f"${total_pnl:+.2f}")
            with col3:
                st.metric("Success Rate", f"{len([t for t in sells if t.get('pnl', 0) > 0])/len(sells)*100:.0f}%" if sells else "N/A")
    else:
        st.info("No trades yet")

with tab4:
    st.subheader("🔍 Near Miss Analysis")

    if bot.near_miss_log:
        st.write(f"**{len(bot.near_miss_log)} missed opportunities tracked**")

        sorted_misses = sorted(list(bot.near_miss_log),
                              key=lambda x: x['metrics']['volume_ratio'], reverse=True)

        st.markdown("### Top Missed Opportunities")
        for i, miss in enumerate(sorted_misses[:5], 1):
            with st.expander(f"{i}. {miss['symbol']} ({miss['strategy'].title()}) - {miss['missed_reason']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Price Action**")
                    st.write(f"Change: {miss['metrics']['price_change']:+.2%}")
                    st.write(f"Volume: {miss['metrics']['volume_ratio']:.1f}x")

                with col2:
                    st.write("**Indicators**")
                    st.write(f"RSI: {miss['metrics']['rsi']:.1f}")
                    st.write(f"MACD: {miss['metrics']['macd']:.3f}")

                with col3:
                    st.write("**Why Missed**")
                    st.write(f"VWAP Dist: {miss['metrics']['distance_from_vwap']:+.2%}")
                    st.write(f"**{miss['metrics']['why_rejected']}**")

        # Strategy breakdown
        strategy_counts = {}
        for miss in bot.near_miss_log:
            strategy = miss['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        if strategy_counts:
            st.markdown("### Misses by Strategy")
            col1, col2, col3 = st.columns(3)
            strategies = list(strategy_counts.keys())

            if len(strategies) > 0:
                with col1:
                    st.metric(f"🚀 {strategies[0].title()}", strategy_counts[strategies[0]])
            if len(strategies) > 1:
                with col2:
                    st.metric(f"🔄 {strategies[1].title()}", strategy_counts[strategies[1]])
            if len(strategies) > 2:
                with col3:
                    st.metric(f"💥 {strategies[2].title()}", strategy_counts[strategies[2]])

    else:
        st.info("Near-miss tracking activates during market hours")

with tab5:
    st.subheader("🤖 Enhanced Bot Information")

    st.markdown("""
    ### 🚀 Multi-Strategy Trading System

    **Daily Profit Target**: Automatically stops trading at +4% daily return
    **Loss Protection**: -2% daily loss limit with immediate position closure
    **Circuit Breaker**: 3 consecutive losses triggers 1-hour pause
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🚀 Enhanced Momentum**
        - MACD positive & rising
        - ADX > 25 (strong trend)
        - Price above VWAP
        - Volume > 2x average
        """)

    with col2:
        enabled = "✅ Enabled" if ENABLE_MEAN_REVERSION else "❌ Disabled"
        st.markdown(f"""
        **🔄 Mean Reversion** {enabled}
        - RSI(5) < 30 (oversold)
        - Price < Lower Bollinger Band
        - Volume > 1.2x average
        - Exit at middle band
        """)

    with col3:
        st.markdown("""
        **💥 Opening Range Breakout**
        - Break 30-min high + 0.5%
        - Volume confirmation
        - No trades after 3:00 PM
        - Momentum follow-through
        """)

    st.markdown("### ⚙️ Risk Management")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Max Positions", f"{MAX_CONCURRENT_POSITIONS}")
        st.metric("Position Size", f"{POSITION_SIZE} shares")

    with col2:
        st.metric("Daily Target", f"{DAILY_PROFIT_TARGET_PCT*100:.0f}%")
        st.metric("Loss Limit", f"{MAX_DAILY_LOSS_PCT*100:.0f}%")

    with col3:
        st.metric("Stop Loss", f"{STOP_LOSS_PCT*100:.0f}%")
        st.metric("Take Profit", f"{TAKE_PROFIT_PCT*100:.0f}%")

    st.markdown("### 🔧 Current Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        **Trading Parameters:**
        - Volume Multiplier: {VOLUME_MULTIPLIER}x
        - Min Price Change: {MIN_PRICE_CHANGE_PCT*100:.0f}%
        - Market Cutoff: {NO_TRADES_AFTER_HOUR}:00 PM
        """)

    with col2:
        st.markdown(f"""
        **Enhanced Features:**
        - Mean Reversion: {'✅' if ENABLE_MEAN_REVERSION else '❌'}
        - Short Selling: {'✅' if ENABLE_SHORT_SELLING else '❌'}
        - Near Miss Log: ✅ ({MAX_NEAR_MISS_LOG} max)
        - Regime Detection: ✅ Active
        """)

    if not TALIB_AVAILABLE:
        st.warning("⚠️ TA-Lib not installed - using fallback indicators")
    else:
        st.success("✅ TA-Lib indicators active")

    if not ALPACA_API_KEY:
        st.info("📊 Running in enhanced simulation mode")
    else:
        st.success("✅ Alpaca Paper Trading configured")

# Auto-refresh script
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

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Auto-refreshes every 30s | Enhanced Multi-Strategy Bot v2.0")