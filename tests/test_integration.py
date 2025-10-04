"""Integration tests covering core workflow components."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

import alerts
import backtester
import dashboard
import data_fetcher
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


def test_data_fetcher_handles_multi_index_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_alerts_skip_empty_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    empty_csv = data_dir / "EMPTY.csv"
    pd.DataFrame(columns=data_fetcher.EXPECTED_COLUMNS).to_csv(empty_csv, index=False)

    frames = alerts.load_latest_ticker_data(data_dir)
    assert isinstance(frames, dict)
    assert not frames


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
    finally:
        app.config.clear()
        app.config.update(original_config)
