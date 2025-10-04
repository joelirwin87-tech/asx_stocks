from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    start = date(2023, 1, 1)
    records = []
    price = 100.0
    for i in range(60):
        current = start + timedelta(days=i)
        open_price = price
        high = open_price * 1.02
        low = open_price * 0.98
        close = open_price * (1 + (0.01 if i % 7 == 0 else -0.005))
        volume = 1_000_000 + i * 1000
        records.append(
            {
                "Date": current,
                "Open": round(open_price, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Close": round(close, 2),
                "Volume": volume,
            }
        )
        price = close
    return pd.DataFrame(records)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    cfg = {
        "tickers": ["TEST.AX"],
        "start_date": "2022-01-01",
        "capital": 10000,
        "risk_per_trade": 0.1,
        "take_profit_pct": 0.05,
        "stop_loss_pct": 0.03,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "reports"
    directory.mkdir()
    return directory


@pytest.fixture
def mock_data_fetch(monkeypatch, sample_price_data):
    def fake_download(*args, **kwargs):
        df = sample_price_data.copy()
        df.set_index(pd.to_datetime(df["Date"]), inplace=True)
        df.drop(columns=["Date"], inplace=True)
        return df

    monkeypatch.setattr("yfinance.download", fake_download)


@pytest.fixture
def dummy_signals() -> pd.DataFrame:
    dates = pd.date_range("2023-01-10", periods=4, freq="7D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ["TEST.AX"] * len(dates),
            "Signal": ["BUY", "SELL", "BUY", "SELL"],
            "Price": [101.0, 103.0, 99.0, 104.0],
            "Strategy": ["SMA Crossover"] * len(dates),
        }
    )
