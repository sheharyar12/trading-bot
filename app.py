import streamlit as st
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import random
import logging
from typing import List, Dict, Optional
import threading
import pytz

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

# Market hours (Eastern Time)
MARKET_TIMEZONE = pytz.timezone('US/Eastern')
MARKET_OPEN_TIME = (9, 30)  # 9:30 AM ET
MARKET_CLOSE_TIME = (15, 50)  # 3:50 PM ET (close positions 10 min early)

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

        # Initialize API connection
        self._connect_to_alpaca()

        # Start the main trading loop in a separate thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        logger.info("Auto Trading Bot initialized and running")

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
        """Main trading loop that runs continuously"""
        logger.info("Starting automated trading loop")

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
                    # Before market open
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
                    self.status = "TRADING ACTIVE"

                    # Scan for breakouts once per day (in the morning)
                    if current_minutes < market_open + 30 and self.last_scan_date != now.date():
                        logger.info("Market open - scanning for breakout candidates")
                        self.daily_starting_balance = self.account_balance  # Store balance at start of day
                        self.candidates = self._scan_for_breakouts()
                        self.last_scan_date = now.date()
                        self.daily_pnl = 0  # Reset daily P&L

                        # Send market open notification
                        if self.candidates:
                            self._send_notification(
                                "🔔 Market Open - Bot Active",
                                f"Found {len(self.candidates)} breakout candidates\n"
                                f"Balance: ${self.account_balance:,.2f}\n"
                                f"Top pick: {self.candidates[0]['symbol']} +{self.candidates[0]['change_pct']:.1%}",
                                priority=0
                            )

                    # Monitor and trade
                    self._monitor_and_trade()

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

    def _scan_for_breakouts(self) -> List[Dict]:
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

        logger.info(f"Found {len(self.candidates)} breakout candidates")
        return self.candidates

    def _scan_real_market(self) -> List[Dict]:
        """Scan real market using Alpaca API"""
        candidates = []

        try:
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            symbols = [a.symbol for a in assets if a.tradable][:30]

            for symbol in symbols:
                try:
                    bars = self.api.get_bars(symbol, '1Day', limit=5).df
                    if len(bars) < 2:
                        continue

                    current_price = bars['close'].iloc[-1]
                    prev_close = bars['close'].iloc[-2]
                    change_pct = (current_price - prev_close) / prev_close

                    volume = bars['volume'].iloc[-1]
                    avg_volume = bars['volume'].iloc[:-1].mean()
                    morning_high = bars['high'].iloc[-1] * 0.98

                    candidates.append({
                        'symbol': symbol,
                        'price': current_price,
                        'change_pct': change_pct,
                        'volume': volume,
                        'avg_volume': avg_volume,
                        'morning_high': morning_high
                    })
                except:
                    continue

        except Exception as e:
            logger.error(f"Error scanning market: {e}")
            return self.simulated_data.get_breakout_candidates()

        return candidates

    def _monitor_and_trade(self):
        """Monitor candidates and execute trades on breakouts"""
        for candidate in self.candidates:
            symbol = candidate['symbol']

            # Skip if already have position
            if symbol in self.positions and self.positions[symbol]['status'] == 'OPEN':
                continue

            # Skip if already traded today
            if any(t['symbol'] == symbol and t['date'] == datetime.now().date()
                   for t in self.trades_log):
                continue

            current_price = self._get_current_price(symbol)

            # Check for breakout
            if current_price > candidate['morning_high'] * 1.01:
                logger.info(f"BREAKOUT DETECTED: {symbol} at ${current_price:.2f}")
                self._enter_position(symbol, current_price)

    def _enter_position(self, symbol: str, entry_price: float):
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

        cost = entry_price * POSITION_SIZE

        if self.is_simulation:
            self.positions[symbol] = position
            self.account_balance -= cost
            self.buying_power -= cost
            logger.info(f"BUY (Simulated): {symbol} - {POSITION_SIZE} shares @ ${entry_price:.2f}")
            logger.info(f"Account Balance: ${self.account_balance:,.2f}")
        else:
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=POSITION_SIZE,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                self.positions[symbol] = position
                # Update balance from Alpaca
                self._update_account_balance()
                logger.info(f"BUY ORDER: {symbol} - Order ID: {order.id}")
                logger.info(f"Account Balance: ${self.account_balance:,.2f}")
            except Exception as e:
                logger.error(f"Failed to buy {symbol}: {e}")
                return

        # Send buy notification
        self._send_notification(
            f"📈 BUY: {symbol}",
            f"Bought {POSITION_SIZE} shares @ ${entry_price:.2f}\n"
            f"Stop Loss: ${position['stop_loss']:.2f}\n"
            f"Take Profit: ${position['take_profit']:.2f}\n"
            f"Balance: ${self.account_balance:,.2f}",
            priority=1  # High priority for trades
        )

        self.trades_log.append({
            'date': datetime.now().date(),
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': 'BUY',
            'symbol': symbol,
            'price': entry_price,
            'shares': POSITION_SIZE,
            'balance': self.account_balance
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
        """Exit a position"""
        position = self.positions.get(symbol)
        if not position or position['status'] != 'OPEN':
            return

        pnl = (exit_price - position['entry_price']) * position['shares']
        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
        proceeds = exit_price * position['shares']

        if self.is_simulation:
            self.account_balance += proceeds
            self.buying_power += proceeds
            logger.info(f"SELL (Simulated): {symbol} @ ${exit_price:.2f} - {reason} - P&L: ${pnl:.2f}")
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
                logger.info(f"SELL ORDER: {symbol} - {reason} - P&L: ${pnl:.2f}")
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

        # Send sell notification with emoji based on profit/loss
        emoji = "💰" if pnl > 0 else "📉"
        color = "green" if pnl > 0 else "red"

        self._send_notification(
            f"{emoji} SELL: {symbol} - {reason}",
            f"Sold @ ${exit_price:.2f}\n"
            f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n"
            f"Balance: ${self.account_balance:,.2f}\n"
            f"Daily P&L: ${self.daily_pnl:+,.2f}",
            priority=1 if abs(pnl) > 500 else 0  # High priority for big trades
        )

        self.trades_log.append({
            'date': datetime.now().date(),
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': f'SELL ({reason})',
            'symbol': symbol,
            'price': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'balance': self.account_balance
        })

    def _close_all_positions(self):
        """Close all open positions at end of day"""
        logger.info("Closing all positions for end of day...")

        for symbol, position in list(self.positions.items()):
            if position['status'] == 'OPEN':
                current_price = self._get_current_price(symbol)
                self._exit_position(symbol, current_price, 'END_OF_DAY')

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
        """Log daily trading summary and send end-of-day notification"""
        logger.info("="*50)
        logger.info(f"DAILY SUMMARY - {datetime.now().date()}")
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

        logger.info("="*50)

        # Send end-of-day summary notification
        daily_return = ((self.account_balance - self.daily_starting_balance) / self.daily_starting_balance) * 100
        total_return = ((self.account_balance - self.initial_balance) / self.initial_balance) * 100

        # Choose emoji based on performance
        if self.daily_pnl > 1000:
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

        summary_message = (
            f"Today's P&L: ${self.daily_pnl:+,.2f} ({daily_return:+.2f}%)\n"
            f"Trades: {total_today} ({winners}W/{losers}L)\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"─────────────\n"
            f"Balance: ${self.account_balance:,.2f}\n"
            f"Total Return: {total_return:+.2f}%\n"
            f"Total P&L: ${self.total_pnl:+,.2f}"
        )

        self._send_notification(
            f"{emoji} Market Closed - Daily Summary",
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
        """Get current statistics for display"""
        open_positions = sum(1 for p in self.positions.values() if p['status'] == 'OPEN')
        total_trades = len([t for t in self.trades_log if t.get('action', '').startswith('SELL')])

        winners = sum(1 for t in self.trades_log
                     if t.get('action', '').startswith('SELL') and t.get('pnl', 0) > 0)

        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

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
            'initial_balance': self.initial_balance
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

# Second row of metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Daily P&L", f"${stats['daily_pnl']:.2f}",
              delta=f"{stats['daily_pnl']:.2f}" if stats['daily_pnl'] != 0 else None)
with col2:
    st.metric("Total P&L", f"${stats['total_pnl']:.2f}",
              delta=f"{stats['total_pnl']:.2f}" if stats['total_pnl'] != 0 else None)
with col3:
    st.metric("Open Positions", stats['open_positions'])
with col4:
    st.metric("Total Trades", stats['total_trades'])
with col5:
    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Positions", "🎯 Today's Candidates", "📜 Trade Log", "ℹ️ Info"])

with tab1:
    st.subheader("Current Positions")
    if bot.positions:
        positions_data = []
        for symbol, pos in bot.positions.items():
            if pos['status'] == 'OPEN':
                positions_data.append({
                    'Symbol': symbol,
                    'Entry': f"${pos['entry_price']:.2f}",
                    'Current': f"${pos['current_price']:.2f}",
                    'Shares': pos['shares'],
                    'P&L': f"${pos['pnl']:.2f}",
                    'P&L %': f"{pos['pnl_pct']:.2f}%",
                    'Stop': f"${pos['stop_loss']:.2f}",
                    'Target': f"${pos['take_profit']:.2f}",
                    'Status': pos['status']
                })

        if positions_data:
            df = pd.DataFrame(positions_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No open positions")
    else:
        st.info("No positions yet today")

with tab2:
    st.subheader("Today's Breakout Candidates")
    if bot.candidates:
        for c in bot.candidates:
            col1, col2, col3, col4 = st.columns(4)
            col1.write(f"**{c['symbol']}**")
            col2.write(f"Price: ${c['price']:.2f}")
            col3.write(f"Change: +{c['change_pct']:.2%}")
            col4.write(f"Volume: {c['volume']/c['avg_volume']:.1f}x avg")
    else:
        st.info("Candidates are scanned at market open (9:30 AM ET)")

with tab3:
    st.subheader("Recent Trade Log")
    if bot.trades_log:
        # Show last 20 trades
        recent_trades = bot.trades_log[-20:]
        df = pd.DataFrame(recent_trades)
        df = df.iloc[::-1]  # Reverse to show most recent first
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No trades executed yet")

with tab4:
    st.subheader("Bot Information")

    st.markdown("""
    ### 🤖 How This Bot Works:

    **Automatic Operation:**
    - Starts automatically at market open (9:30 AM ET)
    - Scans for breakout stocks in the first 30 minutes
    - Monitors and trades throughout the day
    - Closes all positions by 3:50 PM ET
    - Stops trading on weekends

    **Trading Strategy:**
    - Finds stocks up >5% with 2x normal volume
    - Buys on breakout above morning high
    - Stop Loss: -2% | Take Profit: +7%
    - Maximum 5 positions per day

    **📱 iPhone Notifications:**
    """)

    if bot.notifications_enabled:
        st.success("✅ Push notifications are ENABLED")
        st.markdown("""
        You'll receive notifications for:
        - 🔔 Market open with top picks
        - 📈 Every buy order
        - 💰/📉 Every sell with P&L
        - 📊 End-of-day summary with total gains/losses
        """)
    else:
        st.warning("⚠️ Push notifications are DISABLED")
        st.markdown("""
        To enable iPhone notifications:
        1. Download Pushover app ($4.99 one-time)
        2. Get your User Key from the app
        3. Create an app at pushover.net to get App Token
        4. Add to Streamlit secrets:
           - PUSHOVER_USER_KEY
           - PUSHOVER_APP_TOKEN
        """)

    st.markdown("---")

    st.markdown("**Configuration:**")
    if not ALPACA_API_KEY:
        st.warning("⚠️ No API keys detected - running in simulation mode")
        st.info("To use real paper trading, set ALPACA_API_KEY and ALPACA_SECRET_KEY in Streamlit Cloud secrets")
    else:
        st.success("✅ Connected to Alpaca Paper Trading API")

    st.markdown("---")
    st.caption("This bot uses Alpaca's PAPER TRADING API - no real money at risk")
    st.caption(f"API Endpoint: `{ALPACA_BASE_URL}`")

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