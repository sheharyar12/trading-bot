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
    """Complete technical indicators implementation for trading analysis"""
    
    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average calculation"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    @staticmethod
    def _sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average calculation"""
        sma = np.zeros_like(prices)
        sma[:period-1] = np.nan
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
        return sma
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index calculation"""
        if len(prices) < period + 1:
            return np.full_like(prices, 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        # Initial averages
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # Calculate subsequent averages using EMA formula
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50.0  # Default RSI for initial values
        return rsi
    
    @staticmethod
    def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD with signal line and histogram"""
        if len(prices) < slow:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        ema_fast = TechnicalIndicators._ema(prices, fast)
        ema_slow = TechnicalIndicators._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators._ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2):
        """Bollinger Bands: upper, middle, lower"""
        if len(prices) < period:
            return prices, prices, prices
        
        middle_band = TechnicalIndicators._sma(prices, period)
        
        std_values = np.zeros_like(prices)
        std_values[:period-1] = np.nan
        
        for i in range(period-1, len(prices)):
            std_values[i] = np.std(prices[i-period+1:i+1])
        
        upper_band = middle_band + (std_dev * std_values)
        lower_band = middle_band - (std_dev * std_values)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
        """Average Directional Index for trend strength"""
        if len(high) < period + 1:
            return np.full_like(close, 25.0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # Handle first value
        
        # Calculate Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                          np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR and DM values
        atr = TechnicalIndicators._ema(tr, period)
        dm_plus_smooth = TechnicalIndicators._ema(dm_plus, period)
        dm_minus_smooth = TechnicalIndicators._ema(dm_minus, period)
        
        # Calculate DI+ and DI-
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # Calculate DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = TechnicalIndicators._ema(dx, period)
        
        return adx
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
        """Average True Range"""
        if len(high) < 2:
            return high - low
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # Handle first value
        
        atr = TechnicalIndicators._ema(tr, period)
        return atr
    
    @staticmethod
    def calculate_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray):
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        pv = typical_price * volume
        cumulative_pv = np.cumsum(pv)
        cumulative_volume = np.cumsum(volume)
        
        vwap = np.divide(cumulative_pv, cumulative_volume, 
                        out=np.zeros_like(cumulative_pv), where=cumulative_volume!=0)
        return vwap

class AlpacaMarketData:
    """Real market data fetching and analysis using Alpaca API"""
    
    def __init__(self, api):
        self.api = api
        self.stock_universe = self._get_stock_universe()
        self.available_stocks = set()  # Track which stocks actually work
        self.unavailable_stocks = set()  # Track which stocks don't work
        self.rate_limiter = {'last_request': 0, 'requests_count': 0}
        self.cache = {}  # Simple caching for repeated requests
        
    def _get_stock_universe(self):
        """Return list of stocks available with Alpaca's basic data feed"""
        # Focus on tech stocks and ETFs that are typically available with basic paper trading
        # These are usually included in Alpaca's free data feed
        return [
            # Major Tech (usually available)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX',
            'ADBE', 'ORCL', 'INTC', 'CSCO', 'CRM', 'AVGO',
            
            # Major ETFs (usually free)
            'SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'SLV', 'USO', 'TLT', 'HYG', 'LQD',
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'XLB', 'XLRE', 'XLP', 'XLY',
            
            # Additional tech/growth stocks often available
            'PYPL', 'SHOP', 'SQ', 'ROKU', 'ZOOM', 'DOCU', 'TWLO', 'OKTA', 'CRWD', 'ZM',
            
            # Crypto-related (often available)
            'COIN', 'RIOT', 'MARA', 'MSTR'
        ]
    
    def get_available_universe(self):
        """Get list of stocks that actually work with the current subscription"""
        if not self.available_stocks and not self.unavailable_stocks:
            # First time - return a smaller subset to test
            return self.stock_universe[:10]  # Test first 10 stocks
        
        # Return stocks we know work, or remaining untested stocks
        available = list(self.available_stocks)
        untested = [s for s in self.stock_universe if s not in self.available_stocks and s not in self.unavailable_stocks]
        
        return available + untested[:5]  # Add up to 5 untested stocks per batch
    
    def _rate_limit(self):
        """Simple rate limiting for Alpaca API calls"""
        import time
        current_time = time.time()
        if current_time - self.rate_limiter['last_request'] < 0.1:  # 100ms between requests
            time.sleep(0.1)
        self.rate_limiter['last_request'] = current_time
        self.rate_limiter['requests_count'] += 1
    
    def fetch_market_data(self, symbols, timeframe='1Min', limit=100):
        """Fetch real market data from Alpaca"""
        if not self.api:
            return {}
        
        try:
            # Convert single symbol to list
            if isinstance(symbols, str):
                symbols = [symbols]
            
            market_data = {}
            
            # Fetch data for each symbol individually (Alpaca API requirement)
            for symbol in symbols:
                # Skip symbols we know don't work
                if symbol in self.unavailable_stocks:
                    continue
                
                try:
                    self._rate_limit()
                    
                    end_time = datetime.now(pytz.UTC).replace(microsecond=0)
                    start_time = (end_time - timedelta(days=2)).replace(microsecond=0)
                    
                    # Format times as ISO strings without microseconds for Alpaca API
                    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                    
                    # Use correct parameter name 'symbol' (singular)
                    bars = self.api.get_bars(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start_str,
                        end=end_str,
                        limit=limit
                    )
                    
                    if bars and len(bars) > 0:
                        # Convert to DataFrame if it's a list
                        if hasattr(bars, 'df'):
                            df = bars.df
                        else:
                            # Handle different response formats
                            df = pd.DataFrame([{
                                'high': bar.h,
                                'low': bar.l,
                                'close': bar.c,
                                'volume': bar.v,
                                'timestamp': bar.t
                            } for bar in bars])
                        
                        if not df.empty:
                            market_data[symbol] = {
                                'high': df['high'].values,
                                'low': df['low'].values,
                                'close': df['close'].values,
                                'volume': df['volume'].values,
                                'timestamp': df.index.values if hasattr(df, 'index') else df['timestamp'].values
                            }
                            # Track that this symbol works
                            self.available_stocks.add(symbol)
                    
                except Exception as symbol_error:
                    error_msg = str(symbol_error)
                    if "subscription does not permit" in error_msg or "SIP data" in error_msg:
                        # Mark this symbol as unavailable for future requests
                        self.unavailable_stocks.add(symbol)
                        if symbol in self.available_stocks:
                            self.available_stocks.remove(symbol)
                        logger.debug(f"Skipping {symbol}: Premium data required (not available with free paper trading)")
                    else:
                        logger.warning(f"Error fetching data for {symbol}: {symbol_error}")
                    continue
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def get_current_price(self, symbol):
        """Get current/latest price for a symbol"""
        try:
            self._rate_limit()
            
            # Try to get latest quote first
            try:
                quote = self.api.get_latest_quote(symbol)
                if quote and hasattr(quote, 'bid_price') and hasattr(quote, 'ask_price'):
                    return (quote.bid_price + quote.ask_price) / 2  # Mid price
            except:
                pass
            
            # Fallback: get latest bar data
            try:
                end_time = datetime.now(pytz.UTC).replace(microsecond=0)
                start_time = (end_time - timedelta(hours=1)).replace(microsecond=0)
                
                start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                bars = self.api.get_bars(
                    symbol=symbol,
                    timeframe='1Min',
                    start=start_str,
                    end=end_str,
                    limit=1
                )
                if bars and len(bars) > 0:
                    latest_bar = bars[-1]
                    if hasattr(latest_bar, 'c'):
                        return latest_bar.c  # Close price
                    elif hasattr(latest_bar, 'close'):
                        return latest_bar.close
            except:
                pass
                
            logger.warning(f"Could not get current price for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, symbol_data):
        """Calculate all technical indicators for given symbol data"""
        if not symbol_data or len(symbol_data['close']) < 50:
            return {}
        
        high = np.array(symbol_data['high'])
        low = np.array(symbol_data['low'])
        close = np.array(symbol_data['close'])
        volume = np.array(symbol_data['volume'])
        
        indicators = {}
        
        try:
            # RSI
            indicators['rsi'] = TechnicalIndicators.calculate_rsi(close)[-1]
            indicators['rsi_5'] = TechnicalIndicators.calculate_rsi(close, period=5)[-1]
            
            # MACD
            macd, signal, histogram = TechnicalIndicators.calculate_macd(close)
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = signal[-1]
            indicators['macd_histogram'] = histogram[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(close)
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            
            # ADX
            indicators['adx'] = TechnicalIndicators.calculate_adx(high, low, close)[-1]
            
            # ATR
            indicators['atr'] = TechnicalIndicators.calculate_atr(high, low, close)[-1]
            
            # VWAP
            indicators['vwap'] = TechnicalIndicators.calculate_vwap(high, low, close, volume)[-1]
            
            # Price changes
            indicators['price_change'] = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            indicators['volume_ratio'] = volume[-1] / np.mean(volume[-20:]) if len(volume) > 20 else 1
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            
        return indicators
    
    def screen_momentum_candidates(self):
        """Screen for momentum trading candidates"""
        candidates = []
        
        try:
            # Get available stocks only to avoid subscription errors
            available_universe = self.get_available_universe()
            
            # Get recent data for available stocks (in smaller batches)
            batch_size = 10  # Reduced batch size
            for i in range(0, len(available_universe), batch_size):
                batch = available_universe[i:i+batch_size]
                market_data = self.fetch_market_data(batch, timeframe='1Min', limit=200)
                
                for symbol, data in market_data.items():
                    if len(data['close']) < 50:
                        continue
                    
                    indicators = self.calculate_indicators(data)
                    if not indicators:
                        continue
                    
                    # Momentum criteria
                    if (indicators.get('macd', 0) > 0 and 
                        indicators.get('adx', 0) > 25 and
                        data['close'][-1] > indicators.get('vwap', data['close'][-1]) and
                        indicators.get('volume_ratio', 0) > VOLUME_MULTIPLIER and
                        indicators.get('price_change', 0) > MIN_PRICE_CHANGE_PCT):
                        
                        candidates.append({
                            'symbol': symbol,
                            'price': float(data['close'][-1]),
                            'change_pct': float(indicators['price_change']),
                            'volume': int(data['volume'][-1]),
                            'avg_volume': int(np.mean(data['volume'][-20:])),
                            'indicators': indicators
                        })
                
                # Short delay between batches
                time.sleep(0.2)
                    
        except Exception as e:
            logger.error(f"Error screening momentum candidates: {e}")
        
        return sorted(candidates, key=lambda x: x['change_pct'], reverse=True)[:10]
    
    def screen_mean_reversion_candidates(self):
        """Screen for mean reversion trading candidates"""
        candidates = []
        
        try:
            # Get available stocks only to avoid subscription errors
            available_universe = self.get_available_universe()
            
            batch_size = 10  # Reduced batch size
            for i in range(0, len(available_universe), batch_size):
                batch = available_universe[i:i+batch_size]
                market_data = self.fetch_market_data(batch, timeframe='1Min', limit=200)
                
                for symbol, data in market_data.items():
                    if len(data['close']) < 50:
                        continue
                    
                    indicators = self.calculate_indicators(data)
                    if not indicators:
                        continue
                    
                    # Mean reversion criteria
                    if (indicators.get('rsi_5', 50) < 30 and
                        data['close'][-1] < indicators.get('bb_lower', data['close'][-1]) and
                        indicators.get('volume_ratio', 0) > 1.2 and
                        indicators.get('price_change', 0) < -0.02):  # Oversold condition
                        
                        candidates.append({
                            'symbol': symbol,
                            'price': float(data['close'][-1]),
                            'change_pct': float(indicators['price_change']),
                            'volume': int(data['volume'][-1]),
                            'avg_volume': int(np.mean(data['volume'][-20:])),
                            'indicators': indicators
                        })
                
                time.sleep(0.2)
                    
        except Exception as e:
            logger.error(f"Error screening mean reversion candidates: {e}")
        
        return sorted(candidates, key=lambda x: x['indicators'].get('rsi_5', 50))[:10]
    
    def screen_breakout_candidates(self):
        """Screen for breakout trading candidates"""
        candidates = []
        
        try:
            # Get available stocks only to avoid subscription errors
            available_universe = self.get_available_universe()
            
            batch_size = 10  # Reduced batch size
            for i in range(0, len(available_universe), batch_size):
                batch = available_universe[i:i+batch_size]
                market_data = self.fetch_market_data(batch, timeframe='1Min', limit=200)
                
                for symbol, data in market_data.items():
                    if len(data['close']) < 50:
                        continue
                    
                    indicators = self.calculate_indicators(data)
                    if not indicators:
                        continue
                    
                    # Calculate opening range (first 30 minutes)
                    opening_range_high = np.max(data['high'][:30]) if len(data['high']) > 30 else data['high'][0]
                    current_price = data['close'][-1]
                    
                    # Breakout criteria
                    if (current_price > opening_range_high * 1.005 and  # 0.5% breakout
                        indicators.get('volume_ratio', 0) > 1.5 and
                        indicators.get('adx', 0) > 25):
                        
                        candidates.append({
                            'symbol': symbol,
                            'price': float(current_price),
                            'change_pct': float(indicators['price_change']),
                            'volume': int(data['volume'][-1]),
                            'avg_volume': int(np.mean(data['volume'][-20:])),
                            'opening_range': {'high': float(opening_range_high)},
                            'indicators': indicators
                        })
                
                time.sleep(0.2)
                    
        except Exception as e:
            logger.error(f"Error screening breakout candidates: {e}")
        
        return sorted(candidates, key=lambda x: x['change_pct'], reverse=True)[:10]
    
    def get_candidates_by_strategy(self, strategy):
        """Get candidates for specified strategy"""
        if strategy == 'momentum':
            return self.screen_momentum_candidates()
        elif strategy == 'mean_reversion':
            return self.screen_mean_reversion_candidates()
        elif strategy == 'breakout':
            return self.screen_breakout_candidates()
        else:
            return []

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

        self.market_data = None  # Will be initialized based on mode
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
                self.status = "PAPER TRADING - CONNECTED TO ALPACA"
                self.market_data = AlpacaMarketData(self.api)
                logger.info("📊 PAPER TRADING MODE: Connected to Alpaca successfully. Using real data, paper trades.")
            except Exception as e:
                self.api = None
                self.is_simulation = True
                self.status = "SIMULATION - Alpaca connection failed"
                logger.error(f"Failed to connect to Alpaca: {e}")

        # Initialize market data and check market hours
        if self.is_simulation:
            from collections import namedtuple
            # Create a simple fallback for simulation
            SimData = namedtuple('SimData', ['get_candidates_by_strategy'])
            self.market_data = SimData(get_candidates_by_strategy=lambda x: [])
            self._initialize_sample_data()
        else:
            self._initialize_live_data()
            self.is_trading_hours = self.is_market_open()
            
        # Set daily starting balance (should be done at start of each trading day)
        self.daily_starting_balance = self.account_balance

    def is_market_open(self):
        """Check if market is currently open using Alpaca calendar"""
        if not self.api:
            # Fallback to basic time check
            now = datetime.now(MARKET_TIMEZONE)
            is_weekday = now.weekday() < 5
            is_trading_time = (now.hour > MARKET_OPEN_TIME[0] or 
                             (now.hour == MARKET_OPEN_TIME[0] and now.minute >= MARKET_OPEN_TIME[1])) and \
                            (now.hour < MARKET_CLOSE_TIME[0] or 
                             (now.hour == MARKET_CLOSE_TIME[0] and now.minute < MARKET_CLOSE_TIME[1]))
            return is_weekday and is_trading_time
        
        try:
            # Get today's market calendar
            today = datetime.now(MARKET_TIMEZONE).date()
            calendar = self.api.get_calendar(start=today, end=today)
            
            if not calendar:
                return False
            
            market_day = calendar[0]
            now = datetime.now(MARKET_TIMEZONE)
            
            # Convert market open/close times to datetime objects
            # market_day.open and .close are already time objects
            market_open = datetime.combine(today, market_day.open).replace(tzinfo=MARKET_TIMEZONE)
            market_close = datetime.combine(today, market_day.close).replace(tzinfo=MARKET_TIMEZONE)
            
            return market_open <= now <= market_close
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            # Fallback to basic time check
            now = datetime.now(MARKET_TIMEZONE)
            is_weekday = now.weekday() < 5
            is_trading_time = (now.hour > MARKET_OPEN_TIME[0] or 
                             (now.hour == MARKET_OPEN_TIME[0] and now.minute >= MARKET_OPEN_TIME[1])) and \
                            (now.hour < MARKET_CLOSE_TIME[0] or 
                             (now.hour == MARKET_CLOSE_TIME[0] and now.minute < MARKET_CLOSE_TIME[1]))
            return is_weekday and is_trading_time

    def update_position_prices_live(self):
        """Update position prices with real market data"""
        if not self.api or not self.market_data:
            return
        
        for symbol, pos in list(self.positions.items()):
            if pos['status'] == 'OPEN':
                try:
                    current_price = self.market_data.get_current_price(symbol)
                    if current_price:
                        pos['current_price'] = current_price
                        # Recalculate P&L
                        pos['pnl'] = (current_price - pos['entry_price']) * pos['shares']
                        pos['pnl_pct'] = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                except Exception as e:
                    logger.error(f"Error updating price for {symbol}: {e}")

    def update_position_prices(self):
        """Update position prices based on mode"""
        if self.is_simulation:
            # Simulation: randomly adjust prices
            for symbol, pos in self.positions.items():
                if pos['status'] == 'OPEN':
                    change_pct = random.uniform(-0.015, 0.015)
                    pos['current_price'] = round(pos['current_price'] * (1 + change_pct), 2)
                    pos['pnl'] = (pos['current_price'] - pos['entry_price']) * pos['shares']
                    pos['pnl_pct'] = (pos['current_price'] - pos['entry_price']) / pos['entry_price'] * 100
        else:
            # Live mode: get real prices
            self.update_position_prices_live()

    def place_bracket_order(self, symbol, qty, strategy):
        """Place bracket order with stop-loss and take-profit"""
        if not self.api:
            return self.execute_trade(symbol, 0, qty, strategy)  # Fallback to regular trade
        
        try:
            # Get current price
            current_price = self.market_data.get_current_price(symbol) if self.market_data else None
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return False
            
            stop_price = current_price * (1 - STOP_LOSS_PCT)
            take_profit_price = current_price * (1 + TAKE_PROFIT_PCT)
            
            # Place bracket order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day',
                order_class='bracket',
                stop_loss={'stop_price': str(stop_price)},
                take_profit={'limit_price': str(take_profit_price)}
            )
            
            logger.info(f"Bracket order placed for {symbol}: {order}")
            
            # Update local tracking
            self.positions[symbol] = {
                'symbol': symbol,
                'entry_price': current_price,
                'current_price': current_price,
                'shares': qty,
                'stop_loss': stop_price,
                'take_profit': take_profit_price,
                'entry_time': datetime.now(),
                'status': 'OPEN',
                'pnl': 0,
                'pnl_pct': 0,
                'strategy': strategy,
                'order_id': order.id
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing bracket order for {symbol}: {e}")
            return False

    def monitor_orders(self):
        """Monitor and update order status"""
        if not self.api:
            return
        
        try:
            orders = self.api.list_orders(status='open')
            filled_orders = self.api.list_orders(status='filled', limit=50)
            
            # Update positions based on filled orders
            for order in filled_orders:
                if order.symbol in self.positions:
                    pos = self.positions[order.symbol]
                    if hasattr(pos, 'order_id') and pos.get('order_id') == order.id:
                        if order.side == 'sell':
                            # Position was closed
                            self.close_position(order.symbol, float(order.filled_avg_price), f'SELL ({order.order_type})')
                        
        except Exception as e:
            logger.error(f"Error monitoring orders: {e}")

    def calculate_daily_performance(self):
        """Calculate real daily P&L from Alpaca account"""
        if not self.api:
            # Simulation fallback
            return {
                'daily_pnl': self.daily_pnl,
                'daily_return_pct': (self.daily_pnl / self.daily_starting_balance) * 100,
                'total_return_pct': ((self.account_balance - self.initial_balance) / self.initial_balance) * 100,
                'current_equity': self.account_balance
            }
        
        try:
            account = self.api.get_account()
            current_equity = float(account.equity)
            
            # Calculate total return
            total_return_pct = ((current_equity - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            
            # For daily P&L, use the difference from daily starting balance
            # If we don't have daily starting balance stored, use current vs initial as approximation
            if hasattr(self, 'daily_starting_balance') and self.daily_starting_balance > 0:
                daily_pnl = current_equity - self.daily_starting_balance
                daily_return_pct = (daily_pnl / self.daily_starting_balance) * 100
            else:
                # Fallback: assume starting balance is initial balance
                self.daily_starting_balance = self.initial_balance
                daily_pnl = current_equity - self.initial_balance
                daily_return_pct = (daily_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            
            return {
                'daily_pnl': daily_pnl,
                'daily_return_pct': daily_return_pct,
                'total_return_pct': total_return_pct,
                'current_equity': current_equity
            }
            
        except Exception as e:
            logger.error(f"Error calculating daily performance: {e}")
        
        # Fallback calculation
        current_equity = self.account_balance
        return {
            'daily_pnl': current_equity - self.daily_starting_balance,
            'daily_return_pct': ((current_equity - self.daily_starting_balance) / self.daily_starting_balance * 100) if self.daily_starting_balance > 0 else 0,
            'total_return_pct': ((current_equity - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0,
            'current_equity': current_equity
        }

    def update_daily_pnl(self):
        """Update real-time daily P&L calculation"""
        try:
            if not self.is_simulation:
                # Live mode: get real performance data
                performance = self.calculate_daily_performance()
                self.daily_pnl = performance.get('daily_pnl', 0)
                self.account_balance = performance.get('current_equity', self.account_balance)
            else:
                # Simulation mode: calculate from positions and trades
                unrealized_pnl = sum(pos['pnl'] for pos in self.positions.values() if pos['status'] == 'OPEN')
                today_trades = [t for t in self.trades_log if t['date'] == datetime.now().date()]
                realized_pnl = sum(t.get('pnl', 0) for t in today_trades)
                self.daily_pnl = unrealized_pnl + realized_pnl
                
        except Exception as e:
            logger.error(f"Error updating daily P&L: {e}")

    def check_daily_targets(self):
        """Check profit target (4%) and loss limit (-2%), close all positions if hit"""
        try:
            # Update P&L first
            self.update_daily_pnl()
            
            # Calculate daily return percentage
            if self.daily_starting_balance > 0:
                daily_return_pct = (self.daily_pnl / self.daily_starting_balance)
            else:
                daily_return_pct = 0
            
            # Check profit target (4%)
            if not self.daily_target_hit and daily_return_pct >= DAILY_PROFIT_TARGET_PCT:
                logger.info(f"🎯 DAILY PROFIT TARGET HIT! Return: {daily_return_pct*100:.2f}% (Target: {DAILY_PROFIT_TARGET_PCT*100:.1f}%)")
                self.daily_target_hit = True
                self.close_all_positions_for_target()
                return True
            
            # Check loss limit (-2%) 
            if not self.daily_loss_limit_hit and daily_return_pct <= -MAX_DAILY_LOSS_PCT:
                logger.warning(f"🛑 DAILY LOSS LIMIT HIT! Loss: {daily_return_pct*100:.2f}% (Limit: -{MAX_DAILY_LOSS_PCT*100:.1f}%)")
                self.daily_loss_limit_hit = True
                self.close_all_positions_for_loss_limit()
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking daily targets: {e}")
            return False

    def close_all_positions_for_target(self):
        """Immediately close all positions when daily target achieved"""
        try:
            positions_closed = 0
            for symbol, pos in list(self.positions.items()):
                if pos['status'] == 'OPEN':
                    self.close_position(symbol, pos['current_price'], 'SELL (TARGET_HIT)')
                    positions_closed += 1
            
            if positions_closed > 0:
                logger.info(f"✅ Closed {positions_closed} positions due to daily profit target achievement")
            
        except Exception as e:
            logger.error(f"Error closing positions for target: {e}")

    def close_all_positions_for_loss_limit(self):
        """Immediately close all positions when daily loss limit hit"""
        try:
            positions_closed = 0
            for symbol, pos in list(self.positions.items()):
                if pos['status'] == 'OPEN':
                    self.close_position(symbol, pos['current_price'], 'SELL (LOSS_LIMIT)')
                    positions_closed += 1
            
            if positions_closed > 0:
                logger.warning(f"🛑 Closed {positions_closed} positions due to daily loss limit")
                
            # Set pause until end of day
            market_close = datetime.now(MARKET_TIMEZONE).replace(
                hour=MARKET_CLOSE_TIME[0], 
                minute=MARKET_CLOSE_TIME[1], 
                second=0, 
                microsecond=0
            )
            self.pause_trading_until = market_close
            
        except Exception as e:
            logger.error(f"Error closing positions for loss limit: {e}")

    def check_end_of_day_closure(self):
        """Close positions at 3:45 PM, backup at market close"""
        try:
            now = datetime.now(MARKET_TIMEZONE)
            
            # 3:45 PM closure (15 minutes before market close)
            close_345_time = (15, 45)  # 3:45 PM
            if (now.hour > close_345_time[0] or 
                (now.hour == close_345_time[0] and now.minute >= close_345_time[1])) and \
               (now.hour < MARKET_CLOSE_TIME[0] or 
                (now.hour == MARKET_CLOSE_TIME[0] and now.minute < MARKET_CLOSE_TIME[1])):
                
                # Close all positions at 3:45 PM
                positions_to_close = [symbol for symbol, pos in self.positions.items() 
                                    if pos['status'] == 'OPEN']
                
                if positions_to_close:
                    logger.info(f"⏰ 3:45 PM CLOSURE: Closing {len(positions_to_close)} positions")
                    for symbol in positions_to_close:
                        pos = self.positions[symbol]
                        self.close_position(symbol, pos['current_price'], 'SELL (3:45_PM_CLOSE)')
                    
        except Exception as e:
            logger.error(f"Error in end-of-day closure: {e}")

    def should_allow_new_trades(self):
        """Return (bool, reason) - whether new trades are allowed"""
        try:
            # Check if daily target already hit
            if self.daily_target_hit:
                return False, "Daily profit target achieved - no more trades today"
            
            # Check if daily loss limit hit
            if self.daily_loss_limit_hit:
                return False, "Daily loss limit reached - trading halted"
            
            # Check if trading is paused
            if self.pause_trading_until and datetime.now() < self.pause_trading_until:
                return False, "Trading paused due to circuit breaker"
            
            # Check market hours
            if not self.is_trading_hours:
                return False, "Market is closed"
            
            # Check time-based restrictions (no new trades after 3:00 PM)
            now = datetime.now(MARKET_TIMEZONE)
            if now.hour >= NO_TRADES_AFTER_HOUR:
                return False, f"No new trades after {NO_TRADES_AFTER_HOUR}:00 PM ET"
            
            # Check max positions
            open_positions = sum(1 for pos in self.positions.values() if pos['status'] == 'OPEN')
            if open_positions >= MAX_CONCURRENT_POSITIONS:
                return False, f"Maximum positions reached ({MAX_CONCURRENT_POSITIONS})"
            
            return True, "Trading allowed"
            
        except Exception as e:
            logger.error(f"Error checking trade allowance: {e}")
            return False, "Error checking trade conditions"

    def check_end_of_day(self):
        now = datetime.now(MARKET_TIMEZONE)
        if now.hour > MARKET_CLOSE_TIME[0] or (now.hour == MARKET_CLOSE_TIME[0] and now.minute >= MARKET_CLOSE_TIME[1]):
            for symbol, pos in list(self.positions.items()):
                if pos['status'] in ('OPEN', 'LONG', 'SHORT'):  # Accept all
                    self.close_position(symbol, pos['current_price'], 'SELL (END_OF_DAY)')

    def check_and_sell_positions(self):
        for symbol, pos in list(self.positions.items()):
            current_price = pos['current_price']
            stop_loss = pos.get('stop_loss')
            take_profit = pos.get('take_profit')

            if pos['status'] == 'OPEN':
                # Only do the comparison if stop_loss is not None
                if stop_loss is not None and current_price <= stop_loss:
                    self.close_position(symbol, current_price, 'SELL (STOP_LOSS)')
                # Only do the comparison if take_profit is not None
                elif take_profit is not None and current_price >= take_profit:
                    self.close_position(symbol, current_price, 'SELL (TAKE_PROFIT)')
                # else: skip this position if both are None

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

    def refresh_account_and_positions(self):
        if not self.is_simulation and self.api:
            try:
                account = self.api.get_account()
                self.account_balance = float(account.equity)  # Use equity instead of cash for total value
                self.buying_power = float(account.buying_power)
                # Keep initial_balance as the starting equity (don't overwrite it)

                # Refresh open positions
                positions = self.api.list_positions()
                self.positions = {}
                for pos in positions:
                    self.positions[pos.symbol] = {
                        'symbol': pos.symbol,
                        'entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'shares': int(float(pos.qty)),
                        'stop_loss': None,
                        'take_profit': None,
                        'entry_time': None,
                        'status': 'OPEN',   # <-- set as 'OPEN' for all open positions
                        'pnl': float(pos.unrealized_pl),
                        'pnl_pct': float(pos.unrealized_plpc) * 100,
                        'strategy': 'unknown',
                    }

                # Refresh trades
                self.trades_log = []
                activities = self.api.get_activities(activity_types="FILL")
                for act in activities[:50]:
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
                logger.error(f"Failed to refresh Alpaca account/positions: {e}")



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
            for act in activities[:50]:   # most recent 50 FILL activities
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
            
            # Initialize candidates with real data in live mode
            self.all_candidates = {
                'momentum': self.get_real_market_candidates('momentum'),
                'mean_reversion': self.get_real_market_candidates('mean_reversion') if ENABLE_MEAN_REVERSION else [],
                'breakout': self.get_real_market_candidates('breakout')
            }
        except Exception as e:
            logger.error(f"Failed to fetch live data from Alpaca: {e}")

    def get_real_market_candidates(self, strategy):
        """Get real market data candidates from Alpaca instead of simulated data"""
        if self.is_simulation or not self.market_data:
            return []  # Return empty for simulation
        
        try:
            return self.market_data.get_candidates_by_strategy(strategy)
        except Exception as e:
            logger.error(f"Error fetching real market data for {strategy}: {e}")
            return []

    def get_stats(self):
        open_positions = sum(1 for p in self.positions.values() if p['status'] == 'OPEN' or p['status'] == 'LONG' or p['status'] == 'SHORT')
        total_trades = len([t for t in self.trades_log if t.get('action', '').startswith('SELL') or t.get('action', '') == 'SELL'])

        winners = sum(1 for t in self.trades_log
                     if (t.get('action', '').startswith('SELL') or t.get('action', '') == 'SELL') and t.get('pnl', 0) > 0)

        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        # Get real performance data if connected to Alpaca
        if not self.is_simulation:
            try:
                performance = self.calculate_daily_performance()
                daily_return_pct = performance.get('daily_return_pct', 0)
                total_return_pct = performance.get('total_return_pct', 0)
                current_equity = performance.get('current_equity', self.account_balance)
                
                # Update account balance with real equity
                self.account_balance = current_equity
                self.daily_pnl = performance.get('daily_pnl', self.daily_pnl)
                
            except Exception as e:
                logger.error(f"Error getting real performance data: {e}")
                daily_return_pct = (self.daily_pnl / self.daily_starting_balance * 100) if self.daily_starting_balance > 0 else 0
                total_return_pct = ((self.account_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0
        else:
            # Simulation calculations
            daily_return_pct = (self.daily_pnl / self.daily_starting_balance * 100) if self.daily_starting_balance > 0 else 0
            total_return_pct = ((self.account_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0

        target_progress = (daily_return_pct / (DAILY_PROFIT_TARGET_PCT * 100)) * 100 if DAILY_PROFIT_TARGET_PCT > 0 else 0

        return {
            'status': self.status,
            'mode': 'PAPER TRADING' if not self.is_simulation else 'PROFESSIONAL SIMULATION',
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
            'daily_return_pct': daily_return_pct,
            'total_return_pct': total_return_pct,
            'all_candidates': self.all_candidates
        }

    def execute_trade(self, symbol, price, shares, strategy):
        # --- ENHANCED RISK MANAGEMENT ---
        # Check if new trades are allowed (includes all risk checks)
        allowed, reason = self.should_allow_new_trades()
        if not allowed:
            logger.info(f"Trade blocked for {symbol}: {reason}")
            return False
        
        # Check if symbol already in positions
        if symbol in self.positions:
            logger.info(f"Trade blocked for {symbol}: Position already exists")
            return False

        # Use bracket orders when possible (paper trading mode)
        if not self.is_simulation and self.api:
            try:
                # Check for existing open orders
                open_orders = self.api.list_orders(status='open')
                for order in open_orders:
                    if order.symbol == symbol and order.side == 'buy':
                        logger.info(f"Open BUY order already exists for {symbol}. Skipping duplicate order.")
                        return False
                
                # Try to place bracket order first
                if self.place_bracket_order(symbol, shares, strategy):
                    logger.info(f"Bracket order placed successfully for {symbol}")
                    # Add to trades log
                    self.trades_log.append({
                        'date': datetime.now().date(),
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'action': 'BUY (BRACKET)',
                        'symbol': symbol,
                        'price': price,
                        'shares': shares,
                        'pnl': 0,
                        'balance': self.account_balance,
                        'strategy': strategy
                    })
                    return True
                else:
                    # Fallback to regular market order
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=shares,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    logger.info(f"Regular market order placed for {symbol}: {order}")
                    
            except Exception as e:
                logger.error(f"Failed to submit order for {symbol}: {e}")
                return False

        # Always update local state for UI/tracking (simulation or after successful order)
        actual_price = price
        if not self.is_simulation and self.market_data:
            # Get real current price
            real_price = self.market_data.get_current_price(symbol)
            if real_price:
                actual_price = real_price

        self.positions[symbol] = {
            'symbol': symbol,
            'entry_price': actual_price,
            'current_price': actual_price,
            'shares': shares,
            'stop_loss': actual_price * (1 - STOP_LOSS_PCT),
            'take_profit': actual_price * (1 + TAKE_PROFIT_PCT),
            'entry_time': datetime.now(),
            'status': 'OPEN',
            'pnl': 0,
            'pnl_pct': 0,
            'strategy': strategy
        }
        
        if self.is_simulation:
            self.trades_log.append({
                'date': datetime.now().date(),
                'time': datetime.now().strftime("%H:%M:%S"),
                'action': 'BUY (SIM)',
                'symbol': symbol,
                'price': actual_price,
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

    # 🚨 Always refresh live data before rendering the dashboard!
    if not bot.is_simulation:
        bot.refresh_account_and_positions()
    stats = bot.get_stats()

    st.title("🚀 Enhanced Multi-Strategy Paper Trading Bot")
    st.caption("Professional-grade bot with Alpaca Paper Trading integration")

    # Risk Management Alerts (prominent at top)
    if stats['daily_target_hit']:
        st.success("🎯 **DAILY PROFIT TARGET ACHIEVED!** All trading stopped for today. Congratulations!")
    elif stats['daily_loss_limit_hit']:
        st.error("🛑 **DAILY LOSS LIMIT REACHED!** All positions closed. Trading halted for risk protection.")
    elif stats['trading_paused']:
        st.warning("⏸️ **TRADING PAUSED** Circuit breaker active due to risk management.")
    
    # Logging & status info
    if bot.is_simulation:
        st.warning("🔒 Running in simulation mode. No real trades or balances.")
        logger.warning("Simulation mode: Alpaca credentials missing or connection failed.")
    else:
        st.success("📊 Connected to Alpaca Paper Trading – Real data, paper trades only.")
        logger.info("Paper Trading mode: Using real market data with paper account.")

    # --- Remainder of your Streamlit UI ---
    # Use your existing Streamlit code for UI. No change needed.

    # ... (Copy and paste your entire Streamlit UI code here as is) ...
    # The logic below will now work for both sim and live automatically!

    # For example:
    # Get current stats
    if not bot.is_simulation:
        bot.refresh_account_and_positions()
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
        total_return_pct = stats.get('total_return_pct', 0)
        total_return_dollar = stats['account_balance'] - stats['initial_balance']
        st.metric("📈 Total Return",
                  f"{total_return_pct:+.2f}%",
                  delta=f"${total_return_dollar:+,.2f}")

    # Enhanced Trading Metrics
    st.markdown("### 📊 **Trading Performance**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        daily_pnl_color = "normal" if stats['daily_pnl'] >= 0 else "inverse"
        st.metric("Daily P&L", f"${stats['daily_pnl']:+.2f}",
                  delta=f"{stats['daily_return_pct']:+.2f}%" if stats['daily_pnl'] != 0 else None)

    with col2:
        # Enhanced target progress with risk indicators
        target_progress = stats['target_progress']
        if stats['daily_target_hit']:
            st.metric("🎯 Target Progress", "ACHIEVED! 🎉", 
                     delta=f"Target: {DAILY_PROFIT_TARGET_PCT*100:.0f}% REACHED")
        elif stats['daily_loss_limit_hit']:
            st.metric("🛑 Daily Risk", "LOSS LIMIT HIT", 
                     delta=f"Limit: -{MAX_DAILY_LOSS_PCT*100:.0f}% BREACHED")
        else:
            # Show normal progress
            progress_color = "normal" if target_progress < 100 else "inverse"
            progress_emoji = "🎯" if target_progress >= 75 else "📊"
            st.metric(f"{progress_emoji} Target Progress", f"{target_progress:.0f}%",
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

    # System Status & Risk Management
    st.markdown("### 🎛️ **System Status & Risk Management**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        regime_emoji = {"trending": "📈", "ranging": "↔️", "neutral": "⚪"}.get(stats['market_regime'], "❓")
        st.metric("Market Regime", f"{regime_emoji} {stats['market_regime'].title()}")

    with col2:
        # Enhanced status display with better visual indicators
        if stats['daily_target_hit']:
            st.metric("Status", "🎯 TARGET ACHIEVED", delta="NO MORE TRADES TODAY")
            st.success("Daily profit target reached - trading stopped for today!")
        elif stats['daily_loss_limit_hit']:
            st.metric("Status", "🛑 LOSS LIMIT HIT", delta="TRADING HALTED")
            st.error("Daily loss limit reached - all positions closed!")
        elif stats['trading_paused']:
            st.metric("Status", "⏸️ Paused", delta="Circuit Breaker Active")
            st.warning("Trading paused due to risk management")
        else:
            # Check current time for post-3:45 PM status
            now = datetime.now(MARKET_TIMEZONE)
            if now.hour > 15 or (now.hour == 15 and now.minute >= 45):
                st.metric("Status", "⏰ POST 3:45 PM", delta="POSITIONS CLOSED")
                st.info("Past 3:45 PM - no new trades, positions should be closed")
            elif now.hour >= NO_TRADES_AFTER_HOUR:
                st.metric("Status", "⏰ NO NEW TRADES", delta=f"After {NO_TRADES_AFTER_HOUR}:00 PM")
                st.info("No new trades after 3:00 PM - position management only")
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
                if pos['status'] in ('OPEN', 'LONG', 'SHORT'):
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
                        'Stop Loss': f"${pos['stop_loss']:.2f}" if pos['stop_loss'] is not None else "-",
                        'Take Profit': f"${pos['take_profit']:.2f}" if pos['take_profit'] is not None else "-",
                        'Entry Time': pos['entry_time'].strftime('%H:%M') if pos['entry_time'] is not None else "-",
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
            # --- 1. CRITICAL RISK MANAGEMENT CHECKS FIRST ---
            # Check daily targets (profit/loss limits) - stops all trading if hit
            if bot.check_daily_targets():
                logger.info("Daily target or loss limit hit - stopping all trading activity")
                return  # Stop immediately if target/limit hit
            
            # Check end-of-day closure (3:45 PM)
            bot.check_end_of_day_closure()
            
            # Check if new trades are allowed
            allowed, reason = bot.should_allow_new_trades()
            if not allowed:
                logger.info(f"New trades blocked: {reason}")
                # Still update positions and close existing ones, but no new trades
                if not bot.is_simulation:
                    bot.monitor_orders()
                    bot.update_position_prices_live()
                else:
                    bot.update_position_prices()
                
                bot.check_and_sell_positions()
                bot.check_end_of_day()
                return
            
            # --- 2. Update session state with fresh candidates ---
            st.session_state['all_candidates'] = {
                'momentum': bot.get_real_market_candidates('momentum'),
                'mean_reversion': bot.get_real_market_candidates('mean_reversion') if ENABLE_MEAN_REVERSION else [],
                'breakout': bot.get_real_market_candidates('breakout')
            }
            st.session_state['last_candidate_refresh'] = time.time()

            # --- 3. Monitor orders and update positions ---
            if not bot.is_simulation:
                bot.monitor_orders()
                bot.update_position_prices_live()
            else:
                bot.update_position_prices()

            # --- 4. Auto-buy logic (only if trades are allowed) ---
            all_candidates = st.session_state['all_candidates']
            for strategy_name, candidates in all_candidates.items():
                for c in candidates:
                    symbol = c['symbol']
                    price = c['price']
                    # execute_trade now has built-in risk checks
                    bot.execute_trade(symbol, price, POSITION_SIZE, strategy_name)

            # --- 5. Sell logic for all open positions ---
            bot.check_and_sell_positions()
            bot.check_end_of_day()
            
            # --- 6. Update performance metrics ---
            bot.update_daily_pnl()

        # Manual button
        if st.button("🔍 Scan for New Candidates"):
            refresh_candidates()
            st.success("✅ Candidates refreshed!")
            
            # Show data availability status
            if not bot.is_simulation and bot.market_data:
                available = len(bot.market_data.available_stocks)
                unavailable = len(bot.market_data.unavailable_stocks)
                if available > 0 or unavailable > 0:
                    st.info(f"📊 Data Status: {available} stocks available, {unavailable} require premium subscription")

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
            **✅ Paper Trading Mode**
            - Uses Alpaca Paper Trading API
            - Real market data, simulated trades
            - No real money at risk
            - Perfect for learning and testing strategies
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
