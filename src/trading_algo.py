import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from alpha_vantage.timeseries import TimeSeries
from nsepy import get_history
from nsepy.derivatives import get_expiry_date
import traceback

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)

def fetch_data_alpha_vantage(symbol, api_key, outputsize='full'):
    """
    Fetch historical data from Alpha Vantage API
    
    Args:
        symbol (str): Stock symbol (e.g., 'RELIANCE.BSE' for BSE)
        api_key (str): Alpha Vantage API key
        outputsize (str): Size of the output ('full' or 'compact')
    
    Returns:
        pd.DataFrame: Historical data
    """
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=symbol, outputsize=outputsize)
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data.index = pd.to_datetime(data.index)
        logging.info(f"Successfully fetched data for {symbol} from Alpha Vantage")
        return data
    except Exception as e:
        logging.error(f"Error fetching data from Alpha Vantage: {str(e)}")
        return None

class EMA_Strategy:
    def generate_signals(self, data):
        # Calculate EMAs
        data['EMA_4'] = data['Close'].ewm(span=4, adjust=False).mean()
        data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
        data['EMA_18'] = data['Close'].ewm(span=18, adjust=False).mean()
        data['Signal'] = 0
        # Reset index for easier row-wise assignment
        data = data.reset_index(drop=True)
        in_position = False
        entry_price = 0.0
        for i in range(1, len(data)):
            close = data['Close'].iloc[i]
            ema4 = data['EMA_4'].iloc[i]
            ema9 = data['EMA_9'].iloc[i]
            ema18 = data['EMA_18'].iloc[i]
            # Buy condition: close above all 3 EMAs, not in position
            if not in_position and close > ema4 and close > ema9 and close > ema18:
                data.at[i, 'Signal'] = 1
                in_position = True
                entry_price = close
            # Sell condition: in position and (close below all 3 EMAs or 5% profit or 2.5% stop loss)
            elif in_position:
                profit_target = entry_price * 1.05
                stop_loss = entry_price * 0.975
                if (close < ema4 and close < ema9 and close < ema18) or (close >= profit_target) or (close <= stop_loss):
                    data.at[i, 'Signal'] = -1
                    in_position = False
                    entry_price = 0.0
        return data

class VolTrend:
    def generate_signals(self, data):
        # Strategy 2: Volume + Trend
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Signal'] = 0
        data.loc[(data['Close'] > data['MA_20']) & (data['Volume'] > data['Volume_MA']), 'Signal'] = 1
        data.loc[(data['Close'] < data['MA_20']) & (data['Volume'] > data['Volume_MA']), 'Signal'] = -1
        return data

class TradingAlgorithm:
    def __init__(self, symbol, timeframe='1d', lookback_period=100, strategy=None):
        """
        Initialize the trading algorithm
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NS' for NSE)
            timeframe (str): Data timeframe (e.g., '1d' for daily)
            lookback_period (int): Number of historical periods to analyze
            strategy (object): Strategy object for signal generation
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_period = lookback_period
        self.data = None
        self.position = None
        self.entry_price = None
        self.strategy = strategy
        
    def fetch_data(self):
        """Fetch historical data for the symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_period)
            self.data = yf.download(self.symbol, start=start_date, end=end_date, interval=self.timeframe)
            # Flatten columns if multi-indexed
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in self.data.columns.values]
            self.data = self.data.reset_index(drop=True)
            # Standardize column names if needed
            col_map = {}
            for col in self.data.columns:
                if 'open' in col.lower():
                    col_map[col] = 'Open'
                elif 'high' in col.lower():
                    col_map[col] = 'High'
                elif 'low' in col.lower():
                    col_map[col] = 'Low'
                elif 'close' in col.lower():
                    col_map[col] = 'Close'
                elif 'volume' in col.lower():
                    col_map[col] = 'Volume'
            self.data.rename(columns=col_map, inplace=True)
            logging.info(f"Successfully fetched data for {self.symbol}")
            return True
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            return False
    
    def fetch_data_alpha_vantage(self):
        """Fetch historical data from Alpha Vantage"""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            logging.error("Alpha Vantage API key not found in .env file")
            return False
        self.data = fetch_data_alpha_vantage(self.symbol, api_key)
        return self.data is not None
    
    def calculate_indicators(self):
        """Calculate technical indicators"""
        if self.data is None:
            logging.error("No data available to calculate indicators")
            return False
        try:
            # Ensure 'Close' is a Series, not a DataFrame
            close = self.data['Close']
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            logging.info(f"Type of 'Close': {type(close)}, shape: {close.shape}")

            # Calculate SMA
            self.data['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
            # Calculate RSI
            self.data['RSI'] = RSIIndicator(close=close, window=14).rsi()
            # Calculate Bollinger Bands
            bb = BollingerBands(close=close, window=20, window_dev=2)
            self.data['BB_upper'] = bb.bollinger_hband()
            self.data['BB_lower'] = bb.bollinger_lband()
            self.data['BB_middle'] = bb.bollinger_mavg()
            # Only drop NaNs after all indicators are added
            self.data = self.data.dropna().copy()
            logging.info("Successfully calculated technical indicators")
            return True
        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            return False
    
    def generate_signals(self):
        """Generate trading signals based on technical indicators"""
        if self.data is None:
            logging.error("No data available to generate signals")
            return False
        try:
            if self.strategy:
                self.data = self.strategy.generate_signals(self.data)
            else:
                # Default signal generation logic
                self.data['Signal'] = 0
                buy_condition = (self.data['Close'] > self.data['SMA_20']) & (self.data['RSI'] < 30) & (self.data['Close'] < self.data['BB_lower'])
                sell_condition = (self.data['Close'] < self.data['SMA_20']) | (self.data['RSI'] > 70) | (self.data['Close'] > self.data['BB_upper'])
                self.data.loc[buy_condition, 'Signal'] = 1
                self.data.loc[sell_condition, 'Signal'] = -1
            logging.info("Successfully generated trading signals")
            return True
        except Exception as e:
            logging.error(f"Error generating signals: {str(e)}")
            return False
    
    def backtest(self, initial_capital=100000):
        """
        Backtest the trading strategy
        
        Args:
            initial_capital (float): Initial capital for backtesting
        """
        if self.data is None or 'Signal' not in self.data.columns:
            logging.error("No signals available for backtesting")
            return None
        try:
            # Initialize portfolio metrics
            portfolio = pd.DataFrame(index=self.data.index)
            portfolio['Holdings'] = 0
            portfolio['Cash'] = initial_capital
            portfolio['Total'] = initial_capital
            
            # Simulate trading
            position = 0
            for i in range(len(self.data)):
                if self.data['Signal'].iloc[i] == 1 and position == 0:  # Buy
                    position = 1
                    portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = portfolio.iloc[i-1, portfolio.columns.get_loc('Cash')]
                    portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = 0
                elif self.data['Signal'].iloc[i] == -1 and position == 1:  # Sell
                    position = 0
                    portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio.iloc[i-1, portfolio.columns.get_loc('Holdings')]
                    portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = 0
                else:
                    portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = portfolio.iloc[i-1, portfolio.columns.get_loc('Holdings')]
                    portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio.iloc[i-1, portfolio.columns.get_loc('Cash')]
                
                portfolio.iloc[i, portfolio.columns.get_loc('Total')] = portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] + portfolio.iloc[i, portfolio.columns.get_loc('Cash')]
            
            # Calculate returns
            portfolio['Returns'] = portfolio['Total'].pct_change()
            
            # Calculate performance metrics
            total_return = (portfolio['Total'].iloc[-1] - initial_capital) / initial_capital
            sharpe_ratio = np.sqrt(252) * portfolio['Returns'].mean() / portfolio['Returns'].std()
            
            logging.info(f"Backtest completed. Total Return: {total_return:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")
            return portfolio
            
        except Exception as e:
            logging.error(f"Error during backtesting: {str(e)}\n{traceback.format_exc()}")
            return None

def get_nifty100_symbols():
    """
    Returns the list of Nifty 100 stock symbols
    Returns:
        list: List of Nifty 100 stock symbols with .NS suffix
    """
    nifty100_symbols = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "ULTRACEMCO.NS",
        "TITAN.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "ONGC.NS", "SUNPHARMA.NS",
        "TECHM.NS", "POWERGRID.NS", "NTPC.NS", "HCLTECH.NS", "BRITANNIA.NS",
        "INDUSINDBK.NS", "SHREECEM.NS", "JSWSTEEL.NS", "TATAMOTORS.NS", "BAJAJFINSV.NS",
        "ADANIPORTS.NS", "TATACONSUM.NS", "BPCL.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
        "DRREDDY.NS", "COALINDIA.NS", "GRASIM.NS", "CIPLA.NS", "UPL.NS",
        "M&M.NS", "IOC.NS", "WIPRO.NS", "HDFCLIFE.NS", "TATAPOWER.NS",
        "SBILIFE.NS", "HINDALCO.NS", "DIVISLAB.NS", "TATASTEEL.NS", "APOLLOHOSP.NS",
        "ADANIENT.NS", "BAJAJ-AUTO.NS", "ADANIGREEN.NS", "PIDILITIND.NS", "DABUR.NS",
        "HDFCAMC.NS", "PEL.NS", "BERGEPAINT.NS", "COLPAL.NS", "MARICO.NS",
        "TORNTPHARM.NS", "AMBUJACEM.NS", "ACC.NS", "ICICIGI.NS", "MCDOWELL-N.NS",
        "PFC.NS", "SIEMENS.NS", "BOSCHLTD.NS", "HAVELLS.NS", "INDIGO.NS",
        "TATACOMM.NS", "LUPIN.NS", "NMDC.NS", "DLF.NS", "VEDL.NS",
        "AUROPHARMA.NS", "BANDHANBNK.NS", "PERSISTENT.NS", "MUTHOOTFIN.NS", "GODREJCP.NS",
        "IDEA.NS", "JINDALSTEL.NS", "BANKBARODA.NS", "TATACHEM.NS", "PNB.NS",
        "CANBK.NS", "IOB.NS", "UNIONBANK.NS", "CENTRALBK.NS", "UCOBANK.NS",
        "IDFC.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "RBLBANK.NS", "CSB.NS",
        "KARURVYSYA.NS", "KARNATAKA.NS", "JAMNAAUTO.NS", "JUBLFOOD.NS", "JUBILANT.NS",
        "JUSTDIAL.NS", "KAJARIACER.NS", "KALPATPOWR.NS", "KANSAINER.NS", "KARURVYSYA.NS"
    ]
    
    logging.info(f"Successfully loaded {len(nifty100_symbols)} Nifty 100 symbols")
    return nifty100_symbols

def main():
    # Backtest EMA_Strategy for a few Nifty 100 stocks
    test_symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    lookback_days = 365  # 1 year for quick test
    for symbol in test_symbols:
        print(f"\nBacktesting EMA_Strategy for {symbol}...")
        algo = TradingAlgorithm(symbol, lookback_period=lookback_days, strategy=EMA_Strategy())
        if algo.fetch_data():
            if algo.calculate_indicators():
                if algo.generate_signals():
                    portfolio = algo.backtest()
                    if portfolio is not None:
                        final_value = portfolio['Total'].iloc[-1]
                        total_return = (final_value - 100000) / 100000
                        sharpe_ratio = np.sqrt(252) * portfolio['Returns'].mean() / portfolio['Returns'].std()
                        print(f"Backtest completed for {symbol}.")
                        print(f"Final Portfolio Value: ₹{final_value:,.2f}")
                        print(f"Total Return: {total_return:.2%}")
                        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
                    else:
                        print(f"Backtest failed for {symbol}.")
                else:
                    print(f"Signal generation failed for {symbol}.")
            else:
                print(f"Indicator calculation failed for {symbol}.")
        else:
            print(f"Failed to fetch data for {symbol}.")

if __name__ == "__main__":
    main() 