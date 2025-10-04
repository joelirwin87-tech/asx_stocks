A S X   S T O C K S   B A C K T E S T E R

Offline-first pipeline for fetching, backtesting, and alerting
on ASX-listed stocks using Python, Flask, and yfinance.

Designed for swing-trading strategies and lightweight local use.
Includes a styled dashboard with Bootstrap and Plotly charts.

---------------------------------------------------------------
FEATURES
---------------------------------------------------------------
• Automated daily data fetch from Yahoo Finance
• Hardened data_fetcher with multi-index column flattening
• Multiple built-in strategies:
    - SMA crossovers (10 / 50 day)
    - Pullback in uptrend
    - Donchian channel breakout
    - Gap up with high volume
• Resilient backtester with take-profit targets
• Alerts database auto-initialized, tolerant to empty trades
• Flask dashboard:
    - Bootstrap 5 tables and summary cards
    - Plotly charts for trades and equity curves
    - Graceful fallbacks when data missing
• Integration tests covering config, data, strategies, alerts, and routes

---------------------------------------------------------------
INSTALLATION
---------------------------------------------------------------
1. Clone the repo
   git clone https://github.com/joelirwin87-tech/asx_stocks.git
   cd asx_stocks

2. Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

4. Verify
   PYTHONPATH=. pytest tests/

---------------------------------------------------------------
USAGE
---------------------------------------------------------------
Run the daily pipeline:
   python run_daily.py

Start the dashboard:
   python dashboard.py
Open browser:
   http://127.0.0.1:5000

---------------------------------------------------------------
CONFIGURATION
---------------------------------------------------------------
Edit config.json to set tickers, start date, capital, and profit targets.

Example:
{
  "tickers": ["CBA.AX", "PMT.AX", "OEL.AX"],
  "start_date": "2000-01-01",
  "capital": 10000,
  "tp_percents": [0.03, 0.05]
}

---------------------------------------------------------------
DIRECTORY STRUCTURE
---------------------------------------------------------------
asx_stocks/
│
├── data/              # Price and trades CSVs
├── db/                # SQLite signals.db
├── tests/             # Integration and strategy tests
├── alerts.py          # Alert generation + DB persistence
├── backtester.py      # Trade simulation engine
├── data_fetcher.py    # Yahoo Finance integration
├── dashboard.py       # Flask app with Bootstrap + Plotly
├── run_daily.py       # Orchestrates full daily pipeline
└── config.json        # User configuration

---------------------------------------------------------------
NEXT STEPS
---------------------------------------------------------------
• Extend with stop-loss handling
• Explore portfolio-level backtesting
• Add broker API integration for execution
• Polish dashboard charts with advanced metrics

===============================================================
   BUILT FOR LOCAL, OFFLINE-FIRST, RESILIENT MARKET TESTING
===============================================================
