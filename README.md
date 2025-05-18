# Indian Stock Market Trading Algorithm

This project implements an automated trading algorithm for the Indian stock market. The algorithm uses technical analysis and machine learning techniques to make trading decisions.

## Features
- Real-time market data fetching
- Technical analysis indicators
- Automated trading signals
- Risk management
- Performance tracking

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API credentials:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

4. Run the trading algorithm:
```bash
python src/trading_algo.py
```

## Project Structure
- `src/` - Source code directory
  - `trading_algo.py` - Main trading algorithm
  - `config.py` - Configuration settings
  - `indicators.py` - Technical indicators
  - `risk_management.py` - Risk management logic
- `tests/` - Test files
- `data/` - Historical data storage
- `logs/` - Trading logs

## Disclaimer
This trading algorithm is for educational purposes only. Always do your own research and never trade with money you cannot afford to lose. 