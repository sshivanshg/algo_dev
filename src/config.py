"""
Configuration settings for the trading algorithm
"""

# Trading Parameters
TRADING_PARAMS = {
    'initial_capital': 100000,  # Initial capital in INR
    'max_position_size': 0.2,   # Maximum position size as fraction of capital
    'stop_loss_pct': 0.02,      # Stop loss percentage
    'take_profit_pct': 0.05,    # Take profit percentage
    'max_trades_per_day': 3,    # Maximum number of trades per day
}

# Technical Indicators Parameters
INDICATOR_PARAMS = {
    'sma_period': 20,           # Simple Moving Average period
    'rsi_period': 14,           # Relative Strength Index period
    'bb_period': 20,            # Bollinger Bands period
    'bb_std_dev': 2,            # Bollinger Bands standard deviation
}

# Risk Management Parameters
RISK_PARAMS = {
    'max_drawdown': 0.15,       # Maximum allowed drawdown
    'max_daily_loss': 0.05,     # Maximum daily loss percentage
    'position_sizing': 'kelly',  # Position sizing method ('fixed', 'kelly', 'equal')
}

# Trading Schedule
TRADING_SCHEDULE = {
    'market_open': '09:15',     # Market open time (IST)
    'market_close': '15:30',    # Market close time (IST)
    'pre_market': '09:00',      # Pre-market analysis start time
    'post_market': '15:45',     # Post-market analysis end time
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'logs/trading.log',
    'max_log_size': 10485760,   # 10MB
    'backup_count': 5,
}

# API Configuration
API_CONFIG = {
    'data_provider': 'yfinance',  # Data provider ('yfinance', 'alpha_vantage', etc.)
    'update_interval': 60,        # Data update interval in seconds
}

# Watchlist of stocks to monitor
WATCHLIST = [
    'RELIANCE.NS',   # Reliance Industries
    'TCS.NS',        # Tata Consultancy Services
    'HDFCBANK.NS',   # HDFC Bank
    'INFY.NS',       # Infosys
    'ICICIBANK.NS',  # ICICI Bank
] 