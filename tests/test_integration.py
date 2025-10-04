"""Integration tests covering core workflow components and resilience."""

from __future__ import annotations

import compileall
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

import alerts
import backtester
import dashboard
import data_fetcher
import run_daily
import strategies


def _sample_ohlcv(rows: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {
        "Date": dates,
        "Open": [100 + i for i in range(rows)],
        "High": [101 + i for i in range(rows)],
        "Low": [99 + i for i in range(rows)],
        "Close": [100.5 + i for i in range(rows)],
        "Volume": [1_000_000 + 1000 * i for i in range(rows)],
    }
    return pd.DataFrame(data)


def test_config_json_loads() -> None:
    config = data_fetcher.load_config(data_fetcher.CONFIG_FILE)
    assert "tickers" in config
    assert isinstance(config["tickers"], Iterable)
    assert config["start_date"].year >= 1900


def test_data_fetcher_handles_multi_index_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temp_data_dir = tmp_path / "data"
    monkeypatch.setattr(data_fetcher, "DATA_DIR", temp_data_dir)
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["CBA.AX"]]
    )
    values = [
        [10 + i, 10.5 + i, 9.5 + i, 10.25 + i, 10.2 + i, 1_000_000 + (i * 10_000)]
        for i in range(len(dates))
    ]
    multi_index_df = pd.DataFrame(values, index=dates, columns=columns)

    def _mock_download(*_, **__):  # noqa: ANN001
        return multi_index_df.copy()

    monkeypatch.setattr(data_fetcher.yf, "download", _mock_download)

    success = data_fetcher.update_ticker_data("CBA.AX", datetime(2023, 1, 1))
    assert success

    csv_path = temp_data_dir / "CBA.AX.csv"
    assert csv_path.exists()
    saved = pd.read_csv(csv_path)
    assert list(saved.columns) == data_fetcher.EXPECTED_COLUMNS
    assert not saved.empty


def test_normalize_price_dataframe_consolidates_close_columns() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.25, float("nan")],
            "Adj Close": [10.2, 11.1],
            "Volume": [1_000_000, 1_010_000],
        }
    )

    normalized = data_fetcher.normalize_price_dataframe(frame)

    assert list(normalized.columns) == data_fetcher.EXPECTED_COLUMNS
    assert normalized["Close"].tolist() == [10.25, 11.1]


def test_strategies_produce_boolean_series() -> None:
    frame = _sample_ohlcv(60).set_index("Date")
    for strategy in (
        strategies.sma_cross,
        strategies.pullback_uptrend,
        strategies.donchian_breakout,
        strategies.gapup_highvol,
    ):
        signals = strategy(frame)
        assert isinstance(signals, pd.Series)
        assert signals.dtype == bool
        assert len(signals) == len(frame)


def test_backtester_generates_trades_dataframe() -> None:
    frame = _sample_ohlcv(30).set_index("Date")
    entries = pd.Series([False, True] + [False] * (len(frame) - 2), index=frame.index)
    trades_df, summary = backtester.run_tp_backtest(frame, entries, 0.05, name="integration_test")

    assert isinstance(trades_df, pd.DataFrame)
    assert set(summary.keys()) == {"Strategy", "Trades", "HitRate", "TotalPnL", "CumReturn"}
    assert {"EntryDate", "ExitDate", "EntryPrice", "ExitPrice"}.issubset(trades_df.columns)


def test_run_data_update_logs_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tickers = ["AAA.AX", "BBB.AX", "CCC.AX"]
    start_date = datetime(2020, 1, 1)
    outcomes = {"AAA.AX": True, "BBB.AX": False, "CCC.AX": True}
    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(run_daily.data_fetcher, "DATA_DIR", tmp_path / "data")

    def _fake_load_config(*_, **__):  # noqa: ANN001
        return {"tickers": tickers, "start_date": start_date}

    monkeypatch.setattr(run_daily.data_fetcher, "load_config", _fake_load_config)
    monkeypatch.setattr(run_daily.data_fetcher, "ensure_data_directory", lambda *_: None)

    def _fake_update_ticker(ticker: str, seen_start_date: datetime) -> bool:
        assert seen_start_date == start_date
        result = outcomes[ticker]
        calls.append((ticker, result))
        if len(calls) == len(outcomes):
            success = sum(1 for _, value in calls if value)
            failure = sum(1 for _, value in calls if not value)
            run_daily.LOGGER.info(
                "Fallback data update summary: %s succeeded, %s failed",
                success,
                failure,
            )
        return result

    monkeypatch.setattr(run_daily.data_fetcher, "update_ticker_data", _fake_update_ticker)

    with caplog.at_level(logging.INFO, logger=run_daily.LOGGER.name):
        run_daily._run_data_update()

    assert [ticker for ticker, _ in calls] == tickers
    summary_entries = [record.message for record in caplog.records if "summary" in record.message.lower()]
    assert any("2 succeeded" in message and "1 failed" in message for message in summary_entries)


def test_alerts_skip_empty_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    empty_csv = data_dir / "EMPTY.csv"
    pd.DataFrame(columns=data_fetcher.EXPECTED_COLUMNS).to_csv(empty_csv, index=False)

    frames = alerts.load_latest_ticker_data(data_dir)
    assert isinstance(frames, dict)
    assert not frames


def test_generate_and_store_alerts_preserves_schema_when_no_alerts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db_path = tmp_path / "signals.db"
    data_dir.mkdir()

    alerts.generate_and_store_alerts(data_dir=data_dir, db_path=db_path, strategies=[lambda *_: None])

    assert db_path.exists()
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'"
        )
        row = cursor.fetchone()
    assert row is not None
    assert row[0] == "alerts"


def test_dashboard_routes_respond(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "db"
    db_path = db_dir / "signals.db"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        {
            "Ticker": ["CBA.AX", "BHP.AX"],
            "Trades": [5, 3],
            "HitRate": [0.6, 0.4],
            "TotalPnL": [1500.0, -200.0],
            "CumReturn": [0.12, -0.05],
        }
    )
    summary_df.to_csv(data_dir / "sma_cross_summary.csv", index=False)

    trades_df = _sample_ohlcv(5)
    trades_df.to_csv(data_dir / "CBA.AX.csv", index=False)
    pd.DataFrame(columns=trades_df.columns).to_csv(data_dir / "BHP.AX.csv", index=False)

    with alerts.ensure_alerts_database(db_path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO alerts (date, ticker, strategy, entry_price, target_price, stop_loss)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().date().isoformat(),
                "CBA.AX",
                "sma_cross",
                105.0,
                110.0,
                100.0,
            ),
        )
        connection.commit()

    app = dashboard.app
    original_config = app.config.copy()
    app.config.update(
        TESTING=True,
        DATA_DIRECTORY=data_dir,
        DB_PATH=db_path,
        SECRET_KEY="integration-test",
    )

    dashboard.ensure_runtime_directories(data_dir, db_path)

    try:
        with app.test_client() as client:
            response = client.get("/")
            assert response.status_code == 200

            response = client.get("/signals")
            assert response.status_code == 200
            page = response.get_data(as_text=True)
            assert "table table-striped table-sm align-middle" in page
            assert "CBA.AX" in page
            assert "sma_cross" in page

            response = client.get("/trades/CBA.AX")
            assert response.status_code == 200

            response = client.get("/trades/BHP.AX")
            assert response.status_code == 200

            response = client.get("/trades/MISSING")
            assert response.status_code == 200
    finally:
        app.config.clear()
        app.config.update(original_config)


def test_compileall_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.setenv("PYTHONPATH", str(repo_root))
    success = compileall.compile_dir(str(repo_root), quiet=1)
    assert success
