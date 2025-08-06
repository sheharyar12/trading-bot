import streamlit as st
import os
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import random
from threading import Thread
import logging
from typing import List, Dict, Optional

# Set page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import alpaca_trade_api
try:
    import alpaca_trade_api as tradeapi
    import yfinance as yf
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    st.warning("Alpaca API not installed. Running in simulation mode only.")

# ALPACA PAPER TRADING URL - THIS IS THE KEY!
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Trading Parameters
POSITION_SIZE = 100
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.07
VOLUME_MULTIPLIER = 2.0
MIN_PRICE_CHANGE_PCT = 0.05

class SimulatedMarketData:
    """Simulates market data when real API is not available"""

    def __init__(self):
        self.stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        self.base_prices = {s: random.uniform(50, 500) for s in self.stocks}
        self.volumes = {s: random.randint(1000000, 50000000) for s in self.stocks}

    def get_breakout_candidates(self) -> List[Dict]:
        candidates = []
        for stock in random.sample(self.stocks, 5):
            price = self.base_prices[stock]
            change_pct = random.uniform(0.03, 0.15)
            volume = self.volumes[stock] * random.uniform(1.5, 4.0)

            candidates.append({
                'symbol': stock,
                'price': price * (1 + change_pct),
                'change_pct': change_pct,
                'volume': volume,
                'avg_volume': self.volumes[stock],
                'morning_high': price * (1 + change_pct * 0.7)
            })
        return candidates

    def get_current_price(self, symbol: str) -> float:
        base = self.base_prices.get(symbol, 100)
        return base * (1 + random.uniform(-0.02, 0.02))

class TradingBot:
    """Main trading bot with Alpaca integration"""

    def __init__(self, api_key: str = '', secret_key: str = ''):
        self.api_key = api_key
        self.secret_key = secret_key
        self.api = None
        self.positions = {}
        self.trades_log = []
        self.candidates = []
        self.simulated_data = SimulatedMarketData()
        self.is_simulation = True
        self.is_running = False

        # Initialize Alpaca API connection if credentials provided
        if api_key and secret_key and ALPACA_AVAILABLE:
            try:
                # THIS IS WHERE WE USE THE PAPER TRADING URL!
                self.api = tradeapi.REST(
                    api_key,
                    secret_key,
                    ALPACA_BASE_URL,  # <-- Paper trading URL
                    api_version='v2'
                )
                # Test the connection
                account = self.api.get_account()
                self.is_simulation = False
                st.success(f"✅ Connected to Alpaca Paper Trading! Balance: ${float(account.cash):,.2f}")
                logger.info("Connected to Alpaca Paper Trading API")
            except Exception as e:
                st.error(f"Failed to connect to Alpaca: {e}")
                logger.error(f"Alpaca connection failed: {e}")
                self.is_simulation = True
        else:
            if api_key or secret_key:
                st.info("Missing API credentials. Running in simulation mode.")
            self.is_simulation = True

    def scan_for_breakouts(self) -> List[Dict]:
        """Scan market for breakout candidates"""
        logger.info("Scanning for breakout candidates...")

        if self.is_simulation:
            candidates = self.simulated_data.get_breakout_candidates()
        else:
            candidates = self._scan_real_market()

        # Filter candidates
        filtered = []
        for candidate in candidates:
            if (candidate['change_pct'] >= MIN_PRICE_CHANGE_PCT and
                candidate['volume'] >= candidate['avg_volume'] * VOLUME_MULTIPLIER):
                filtered.append(candidate)

        filtered.sort(key=lambda x: (x['change_pct'], x['volume']/x['avg_volume']), reverse=True)
        self.candidates = filtered[:5]

        return self.candidates

    def _scan_real_market(self) -> List[Dict]:
        """Scan real market using Alpaca API"""
        candidates = []

        try:
            # Get list of active assets from Alpaca
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            symbols = [a.symbol for a in assets if a.tradable][:20]  # Limit for speed

            for symbol in symbols:
                try:
                    # Get latest quote from Alpaca
                    bars = self.api.get_bars(symbol, '1Day', limit=5).df
                    if len(bars) < 2:
                        continue

                    current_price = bars['close'].iloc[-1]
                    prev_close = bars['close'].iloc[-2]
                    change_pct = (current_price - prev_close) / prev_close

                    volume = bars['volume'].iloc[-1]
                    avg_volume = bars['volume'].iloc[:-1].mean()

                    # Get morning high (simplified)
                    morning_high = bars['high'].iloc[-1] * 0.98

                    candidates.append({
                        'symbol': symbol,
                        'price': current_price,
                        'change_pct': change_pct,
                        'volume': volume,
                        'avg_volume': avg_volume,
                        'morning_high': morning_high
                    })

                except Exception as e:
                    continue

        except Exception as e:
            logger.error(f"Error scanning market: {e}")
            return self.simulated_data.get_breakout_candidates()

        return candidates

    def enter_position(self, symbol: str, entry_price: float):
        """Enter a new position"""
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': entry_price,
            'shares': POSITION_SIZE,
            'stop_loss': entry_price * (1 - STOP_LOSS_PCT),
            'take_profit': entry_price * (1 + TAKE_PROFIT_PCT),
            'entry_time': datetime.now(),
            'status': 'OPEN',
            'pnl': 0,
            'pnl_pct': 0
        }

        if self.is_simulation:
            self.positions[symbol] = position
            st.success(f"📈 BUY (Simulated): {symbol} - {POSITION_SIZE} shares @ ${entry_price:.2f}")
        else:
            try:
                # REAL ALPACA PAPER TRADING ORDER
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=POSITION_SIZE,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                self.positions[symbol] = position
                st.success(f"📈 BUY ORDER SUBMITTED (Paper Trading): {symbol} - Order ID: {order.id}")
                logger.info(f"Alpaca order submitted: {order.id}")
            except Exception as e:
                st.error(f"Failed to submit order: {e}")
                logger.error(f"Order failed: {e}")

        self.trades_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': 'BUY',
            'symbol': symbol,
            'price': entry_price,
            'shares': POSITION_SIZE,
            'mode': 'PAPER' if not self.is_simulation else 'SIM'
        })

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if self.is_simulation:
            return self.simulated_data.get_current_price(symbol)
        else:
            try:
                # Get real price from Alpaca
                quote = self.api.get_latest_quote(symbol)
                return float(quote.ap)  # ask price
            except Exception as e:
                logger.error(f"Error getting price: {e}")
                return self.simulated_data.get_current_price(symbol)

    def update_positions(self):
        """Update all position prices and P&L"""
        for symbol, pos in self.positions.items():
            if pos['status'] == 'OPEN':
                current_price = self.get_current_price(symbol)
                pos['current_price'] = current_price
                pos['pnl'] = (current_price - pos['entry_price']) * pos['shares']
                pos['pnl_pct'] = ((current_price - pos['entry_price']) / pos['entry_price']) * 100

                # Check exit conditions
                if pos['pnl_pct'] <= -2:
                    self.exit_position(symbol, current_price, 'STOP_LOSS')
                elif pos['pnl_pct'] >= 7:
                    self.exit_position(symbol, current_price, 'TAKE_PROFIT')

    def exit_position(self, symbol: str, exit_price: float, reason: str):
        """Exit a position"""
        position = self.positions.get(symbol)
        if not position or position['status'] != 'OPEN':
            return

        if self.is_simulation:
            st.info(f"📉 SELL (Simulated): {symbol} @ ${exit_price:.2f} - {reason}")
        else:
            try:
                # REAL ALPACA PAPER TRADING SELL ORDER
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=position['shares'],
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                st.info(f"📉 SELL ORDER SUBMITTED (Paper Trading): {symbol} - {reason}")
                logger.info(f"Sell order submitted: {order.id}")
            except Exception as e:
                st.error(f"Failed to sell: {e}")

        position['status'] = 'CLOSED'
        position['exit_price'] = exit_price
        position['exit_reason'] = reason

        self.trades_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': f'SELL ({reason})',
            'symbol': symbol,
            'price': exit_price,
            'shares': position['shares'],
            'pnl': position['pnl'],
            'mode': 'PAPER' if not self.is_simulation else 'SIM'
        })

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'bot_thread' not in st.session_state:
    st.session_state.bot_thread = None

# Header
st.title("📈 Automated Trading Bot Dashboard")
st.markdown("Control your Alpaca Paper Trading Bot")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Alpaca Configuration")

    st.info(f"**API URL:** `{ALPACA_BASE_URL}`")
    st.caption("☝️ This URL ensures PAPER TRADING only (no real money)")

    api_key = st.text_input(
        "Alpaca API Key",
        type="password",
        value=os.getenv('ALPACA_API_KEY', ''),
        help="Get from alpaca.markets dashboard"
    )
    secret_key = st.text_input(
        "Alpaca Secret Key",
        type="password",
        value=os.getenv('ALPACA_SECRET_KEY', ''),
        help="Get from alpaca.markets dashboard"
    )

    st.markdown("---")

    if st.button("🔌 Connect to Alpaca", type="primary"):
        with st.spinner("Connecting..."):
            bot = TradingBot(api_key, secret_key)
            st.session_state.bot = bot

            if bot.is_simulation:
                st.warning("📊 Running in SIMULATION mode (no API connection)")
            else:
                st.success("✅ Connected to Alpaca PAPER Trading!")

                # Show account info
                try:
                    account = bot.api.get_account()
                    st.metric("Paper Balance", f"${float(account.cash):,.2f}")
                    st.metric("Buying Power", f"${float(account.buying_power):,.2f}")
                except:
                    pass

# Main Controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🚀 Start Bot", disabled=st.session_state.bot is None):
        if st.session_state.bot:
            st.session_state.bot.is_running = True
            st.success("Bot Started!")

with col2:
    if st.button("🛑 Stop Bot", disabled=st.session_state.bot is None):
        if st.session_state.bot:
            st.session_state.bot.is_running = False
            st.warning("Bot Stopped")

with col3:
    if st.button("🔍 Scan Stocks", disabled=st.session_state.bot is None):
        if st.session_state.bot:
            with st.spinner("Scanning..."):
                candidates = st.session_state.bot.scan_for_breakouts()
                st.success(f"Found {len(candidates)} candidates!")

with col4:
    if st.button("🔄 Update Positions", disabled=st.session_state.bot is None):
        if st.session_state.bot:
            st.session_state.bot.update_positions()
            st.rerun()

# Display Status
if st.session_state.bot:
    st.markdown("---")

    # Status indicator
    if st.session_state.bot.is_running:
        st.success("🟢 **Bot Status: RUNNING**")
    else:
        st.warning("🔴 **Bot Status: STOPPED**")

    mode = "PAPER TRADING" if not st.session_state.bot.is_simulation else "SIMULATION"
    st.info(f"**Mode:** {mode}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Positions", "🎯 Candidates", "📜 Trade Log"])

    with tab1:
        st.subheader("Current Positions")
        if st.session_state.bot.positions:
            positions_df = pd.DataFrame.from_dict(st.session_state.bot.positions, orient='index')
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No positions yet")

    with tab2:
        st.subheader("Breakout Candidates")
        if st.session_state.bot.candidates:
            for c in st.session_state.bot.candidates:
                col1, col2, col3, col4 = st.columns(4)
                col1.write(f"**{c['symbol']}**")
                col2.write(f"${c['price']:.2f}")
                col3.write(f"+{c['change_pct']:.2%}")
                col4.button(f"Buy {c['symbol']}", key=c['symbol'],
                           on_click=lambda s=c['symbol']: st.session_state.bot.enter_position(s, c['price']))
        else:
            st.info("Click 'Scan Stocks' to find candidates")

    with tab3:
        st.subheader("Trade Log")
        if st.session_state.bot.trades_log:
            trade_df = pd.DataFrame(st.session_state.bot.trades_log)
            st.dataframe(trade_df, use_container_width=True)
        else:
            st.info("No trades executed yet")
else:
    st.info("👈 Please configure your Alpaca API keys in the sidebar and click 'Connect to Alpaca'")

# Footer
st.markdown("---")
st.caption("⚠️ This bot uses Alpaca's PAPER TRADING API. No real money is at risk.")
st.caption(f"Paper API URL: `{ALPACA_BASE_URL}`")