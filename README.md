# ASX Stocks Trading Pipeline

A production-ready Python project that automates the daily ingestion, analysis, and presentation of Australian Securities Exchange (ASX) stock data. The system downloads OHLCV data, evaluates multiple trading strategies, performs risk-aware backtesting, persists actionable alerts, and renders the results in a modern Flask dashboard.

## Features

- **Robust data ingestion** via Yahoo Finance with schema normalisation and per-ticker CSV storage.
- **Pluggable trading strategies** including SMA crossover, pullback uptrend, Donchian breakout, and gap-up high volume setups.
- **Capital-aware backtesting** with configurable position sizing, take-profit, and stop-loss parameters.
- **Alert management** stored in SQLite for historical reference.
- **Flask dashboard** styled with Bootstrap 5 and Plotly visualisations for strategy summaries, equity curves, and alert tables.
- **Extensive test suite** covering unit, integration, and Flask route behaviour.

## Project Structure

```
.
├── alerts.py
├── backtester.py
├── config.json
├── data_fetcher.py
├── dashboard.py
├── requirements.txt
├── run_daily.py
├── strategies.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── signals.html
│   └── trades.html
└── tests/
    ├── conftest.py
    ├── test_backtester.py
    ├── test_dashboard.py
    ├── test_data_fetcher.py
    ├── test_integration.py
    └── test_strategies.py
```

## Getting Started

1. **Create and activate a virtual environment** (Python 3.12 recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Review `config.json`** and adjust tickers, capital, and strategy parameters to suit your preferences.

## Running the Daily Pipeline

Execute the end-to-end workflow from the command line:
```bash
python run_daily.py --config config.json --log INFO
```
The script will:
- Download OHLCV data to the `data/` directory.
- Run all configured strategies and aggregate signals.
- Backtest the signals, writing trade logs to `reports/`.
- Generate actionable alerts and store them in `signals.db`.

## Dashboard

Launch the Flask dashboard to review performance and alerts:
```bash
export FLASK_APP=dashboard.py
flask run --reload
```
The dashboard provides:
- Overview of strategy metrics and equity curve.
- Table of current alerts.
- Per-ticker trade histories with Plotly charts.

## Testing

Run the full pytest suite:
```bash
pytest
```
All tests are self-contained and rely on mocked data sources to avoid external API calls.

## Extending the System

- Add new strategies by subclassing `BaseStrategy` in `strategies.py` and returning the standard signal DataFrame schema.
- Introduce additional risk rules or reporting formats by extending `Backtester`.
- Modify dashboard views or add new Flask routes as required; templates live in the `templates/` directory.

## License

This project is provided without a specific license. Adapt and extend it to suit your personal trading research needs.
