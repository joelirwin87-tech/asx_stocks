from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import dashboard


@pytest.fixture
def dashboard_app(tmp_path: Path, monkeypatch):
    summary = pd.DataFrame(
        {
            "Strategy": ["SMA"],
            "Trades": [2],
            "Wins": [1],
            "Losses": [1],
            "WinRate": [0.5],
            "NetPnL": [150.0],
            "AverageReturnPct": [0.02],
            "TotalReturnPct": [0.04],
            "ExposureDays": [10],
        }
    )
    trades = pd.DataFrame(
        {
            "ticker": ["TEST.AX"],
            "strategy": ["SMA"],
            "entry_date": pd.to_datetime(["2023-01-01"]),
            "exit_date": pd.to_datetime(["2023-01-10"]),
            "entry_price": [100.0],
            "exit_price": [110.0],
            "quantity": [10],
            "pnl": [100.0],
            "return_pct": [0.1],
            "exit_reason": ["Signal Exit"],
            "exposure_days": [10],
        }
    )
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    summary.to_csv(reports_dir / "strategy_summary.csv", index=False)
    trades.to_csv(reports_dir / "trades_sma.csv", index=False)

    class FakeAlertsManager:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_recent(self, limit: int = 50):
            return pd.DataFrame(
                {
                    "ticker": ["TEST.AX"],
                    "strategy": ["SMA"],
                    "signal_date": ["2023-01-10"],
                    "action": ["BUY"],
                    "price": [110.0],
                    "notes": ["Entry"],
                    "created_at": ["2023-01-10"],
                }
            )

    monkeypatch.setattr(dashboard, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(dashboard, "AlertsManager", FakeAlertsManager)
    monkeypatch.setattr(dashboard, "load_config", lambda path: {"capital": 10000})
    return dashboard.app.test_client()


def test_index_route(dashboard_app):
    response = dashboard_app.get("/")
    assert response.status_code == 200
    assert b"Strategies" in response.data


def test_signals_route(dashboard_app):
    response = dashboard_app.get("/signals")
    assert response.status_code == 200
    assert b"Signals" in response.data


def test_trades_route(dashboard_app):
    response = dashboard_app.get("/trades/TEST.AX")
    assert response.status_code == 200
    assert b"TEST.AX" in response.data
