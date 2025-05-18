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

class TradingAlgorithm:
    def __init__(self, symbol, timeframe='1d', lookback_period=100):
        """
        Initialize the trading algorithm
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NS' for NSE)
            timeframe (str): Data timeframe (e.g., '1d' for daily)
            lookback_period (int): Number of historical periods to analyze
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_period = lookback_period
        self.data = None
        self.position = None
        self.entry_price = None
        
    def fetch_data(self):
        """Fetch historical data for the symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_period)
            
            self.data = yf.download(
                self.symbol,
                start=start_date,
                end=end_date,
                interval=self.timeframe
            )
            
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
            # Initialize signals column
            self.data['Signal'] = 0
            # Ensure all columns are aligned
            cols = ['Close', 'SMA_20', 'RSI', 'BB_lower', 'BB_upper']
            logging.info(f"Columns before dropna: {self.data.columns.tolist()}")
            missing_cols = [col for col in cols if col not in self.data.columns]
            if missing_cols:
                logging.error(f"Missing columns for signal generation: {missing_cols}")
                return False
            self.data = self.data.dropna(subset=cols).copy()
            # Generate buy signals
            buy_condition = (
                (self.data['Close'] > self.data['SMA_20']) &
                (self.data['RSI'] < 30) &
                (self.data['Close'] < self.data['BB_lower'])
            )
            # Generate sell signals
            sell_condition = (
                (self.data['Close'] < self.data['SMA_20']) |
                (self.data['RSI'] > 70) |
                (self.data['Close'] > self.data['BB_upper'])
            )
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

def main():
    # Example usage
    symbol = "RELIANCE.BSE"  # Reliance Industries on BSE
    lookback_days = 1825  # 5 years
    algo = TradingAlgorithm(symbol, lookback_period=lookback_days)
    
    if algo.fetch_data_alpha_vantage():
        # Filter to last 5 years
        five_years_ago = pd.Timestamp(datetime.now() - timedelta(days=lookback_days))
        algo.data = algo.data[algo.data.index >= five_years_ago]
        if algo.calculate_indicators():
            if algo.generate_signals():
                portfolio = algo.backtest()
                if portfolio is not None:
                    print(f"Backtest completed successfully for {symbol}")
                    print(f"Final Portfolio Value: ₹{portfolio['Total'].iloc[-1]:,.2f}")
                    print(f"Total Return: {((portfolio['Total'].iloc[-1] - 100000) / 100000):.2%}")
                    print(f"Sharpe Ratio: {np.sqrt(252) * portfolio['Returns'].mean() / portfolio['Returns'].std():.2f}")

if __name__ == "__main__":
    main() 