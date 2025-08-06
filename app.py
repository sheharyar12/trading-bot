import streamlit as st
import os
import json
import time
from datetime import datetime
import pandas as pd
import random
from threading import Thread
import queue

# Set page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'candidates' not in st.session_state:
    st.session_state.candidates = []
if 'total_pnl' not in st.session_state:
    st.session_state.total_pnl = 0

# Simulated trading bot functions
class SimpleTradingBot:
    def __init__(self):
        self.positions = {}
        self.trade_log = []
        self.candidates = []
        self.is_running = False

    def scan_for_breakouts(self):
        """Simulate scanning for breakout stocks"""
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META']
        self.candidates = []

        for symbol in random.sample(stocks, min(5, len(stocks))):
            self.candidates.append({
                'symbol': symbol,
                'price': round(random.uniform(100, 500), 2),
                'change_pct': round(random.uniform(5, 15), 2),
                'volume_ratio': round(random.uniform(2, 5), 1)
            })

        return self.candidates

    def simulate_trade(self, symbol):
        """Simulate a trade execution"""
        entry_price = round(random.uniform(100, 500), 2)

        # Simulate position
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': entry_price,
            'shares': 100,
            'pnl': 0,
            'pnl_pct': 0,
            'status': 'OPEN',
            'entry_time': datetime.now().strftime('%H:%M:%S')
        }

        self.positions[symbol] = position

        # Add to trade log
        self.trade_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': 'BUY',
            'symbol': symbol,
            'price': entry_price,
            'shares': 100
        })

        return position

    def update_positions(self):
        """Simulate price movements and update P&L"""
        for symbol, pos in self.positions.items():
            if pos['status'] == 'OPEN':
                # Simulate price movement
                change = random.uniform(-0.03, 0.08)
                pos['current_price'] = round(pos['entry_price'] * (1 + change), 2)
                pos['pnl'] = round((pos['current_price'] - pos['entry_price']) * pos['shares'], 2)
                pos['pnl_pct'] = round(change * 100, 2)

                # Check for exit conditions
                if pos['pnl_pct'] <= -2:
                    pos['status'] = 'STOPPED'
                    self.close_position(symbol, 'Stop Loss')
                elif pos['pnl_pct'] >= 7:
                    pos['status'] = 'PROFIT'
                    self.close_position(symbol, 'Take Profit')

    def close_position(self, symbol, reason):
        """Log position closure"""
        pos = self.positions[symbol]
        self.trade_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': f'SELL ({reason})',
            'symbol': symbol,
            'price': pos['current_price'],
            'shares': pos['shares'],
            'pnl': pos['pnl']
        })

# Create bot instance
@st.cache_resource
def get_bot():
    return SimpleTradingBot()

bot = get_bot()

# Header
st.title("📈 Automated Trading Bot Dashboard")
st.markdown("Control and monitor your day trading bot from anywhere")

# API Key Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("Alpaca API Key", type="password", value=os.getenv('ALPACA_API_KEY', ''))
    secret_key = st.text_input("Alpaca Secret Key", type="password", value=os.getenv('ALPACA_SECRET_KEY', ''))

    if api_key and secret_key:
        st.success("✅ API Keys Configured")
        mode = "Paper Trading"
    else:
        st.info("📝 Running in Simulation Mode")
        mode = "Simulation"

    st.markdown("---")
    st.header("📊 Settings")

    position_size = st.number_input("Position Size (shares)", min_value=1, value=100)
    stop_loss = st.slider("Stop Loss %", min_value=1, max_value=10, value=2)
    take_profit = st.slider("Take Profit %", min_value=3, max_value=20, value=7)

# Main Controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🚀 Start Bot", type="primary", disabled=st.session_state.bot_running):
        st.session_state.bot_running = True
        bot.is_running = True

        # Simulate scanning for stocks
        candidates = bot.scan_for_breakouts()
        st.session_state.candidates = candidates

        st.success("Bot Started Successfully!")
        st.rerun()

with col2:
    if st.button("🛑 Stop Bot", type="secondary", disabled=not st.session_state.bot_running):
        st.session_state.bot_running = False
        bot.is_running = False
        st.warning("Bot Stopped")
        st.rerun()

with col3:
    if st.button("🔄 Refresh", disabled=not st.session_state.bot_running):
        if st.session_state.bot_running:
            # Simulate some trades
            if len(bot.positions) < 3 and st.session_state.candidates:
                candidate = random.choice(st.session_state.candidates)
                bot.simulate_trade(candidate['symbol'])

            # Update existing positions
            bot.update_positions()

            st.session_state.positions = bot.positions
            st.session_state.trade_log = bot.trade_log
        st.rerun()

with col4:
    st.metric("Mode", mode)

# Status Indicator
if st.session_state.bot_running:
    st.markdown("### 🟢 Bot Status: **RUNNING**")
else:
    st.markdown("### 🔴 Bot Status: **STOPPED**")

# Metrics Row
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

# Calculate metrics
total_pnl = sum(pos.get('pnl', 0) for pos in bot.positions.values())
open_positions = sum(1 for pos in bot.positions.values() if pos.get('status') == 'OPEN')
closed_positions = len(bot.positions) - open_positions
win_rate = 0
if closed_positions > 0:
    wins = sum(1 for pos in bot.positions.values() if pos.get('pnl', 0) > 0 and pos.get('status') != 'OPEN')
    win_rate = (wins / closed_positions * 100) if closed_positions > 0 else 0

with col1:
    st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl/10000*100:.1f}%" if total_pnl != 0 else None)
with col2:
    st.metric("Open Positions", open_positions)
with col3:
    st.metric("Closed Trades", closed_positions)
with col4:
    st.metric("Win Rate", f"{win_rate:.1f}%")
with col5:
    st.metric("Trading Time", datetime.now().strftime("%H:%M:%S"))

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Positions", "🎯 Candidates", "📜 Trade Log", "📈 Performance"])

with tab1:
    st.subheader("Current Positions")
    if bot.positions:
        positions_df = pd.DataFrame.from_dict(bot.positions, orient='index')

        # Format the dataframe for display
        if not positions_df.empty:
            # Style the dataframe
            def color_pnl(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                    return f'color: {color}'
                return ''

            styled_df = positions_df.style.applymap(color_pnl, subset=['pnl', 'pnl_pct'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No positions yet")
    else:
        st.info("No open positions. Start the bot to begin trading!")

with tab2:
    st.subheader("Breakout Candidates")
    if st.session_state.candidates:
        candidates_df = pd.DataFrame(st.session_state.candidates)
        st.dataframe(candidates_df, use_container_width=True)
    else:
        st.info("No candidates scanned yet. Start the bot to scan for breakouts!")

with tab3:
    st.subheader("Trade Log")
    if bot.trade_log:
        trade_df = pd.DataFrame(bot.trade_log)
        # Show most recent trades first
        trade_df = trade_df.iloc[::-1]
        st.dataframe(trade_df, use_container_width=True)
    else:
        st.info("No trades executed yet")

with tab4:
    st.subheader("Performance Chart")

    if bot.trade_log:
        # Create a simple P&L chart
        chart_data = pd.DataFrame({
            'Time': [log['time'] for log in bot.trade_log if 'pnl' in log],
            'Cumulative P&L': [log.get('pnl', 0) for log in bot.trade_log if 'pnl' in log]
        })

        if not chart_data.empty:
            chart_data['Cumulative P&L'] = chart_data['Cumulative P&L'].cumsum()
            st.line_chart(chart_data.set_index('Time'))
        else:
            st.info("No completed trades to display")
    else:
        st.info("No performance data available yet")

# Footer with instructions
st.markdown("---")
with st.expander("📖 How to Use"):
    st.markdown("""
    1. **Configure API Keys** (optional): Add your Alpaca API keys in the sidebar for paper trading
    2. **Start Bot**: Click the Start Bot button to begin scanning for breakout stocks
    3. **Monitor**: Watch your positions update in real-time
    4. **Refresh**: Click Refresh to simulate market updates (in real trading, this would be automatic)
    5. **Stop Bot**: Click Stop Bot to close all positions and end trading

    **Trading Strategy:**
    - Scans for stocks up >5% with 2x volume
    - Buys on breakout above morning high
    - Stop Loss: -2% | Take Profit: +7%
    - Auto-closes all positions at market close
    """)

# Auto-refresh every 5 seconds when bot is running
if st.session_state.bot_running:
    time.sleep(5)
    st.rerun()