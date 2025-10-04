"""Unit tests for strategy signal generation and alert persistence."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import alerts
import strategies


def _sample_dataframe() -> pd.DataFrame:
    data = {
        "Date": pd.date_range("2020-01-01", periods=60, freq="D"),
        "Open": [100 + i * 0.5 for i in range(60)],
        "High": [101 + i * 0.5 for i in range(60)],
        "Low": [99 + i * 0.5 for i in range(60)],
        "Close": [100 + i * 0.5 for i in range(60)],
        "Volume": [1_000_000 + i * 1_000 for i in range(60)],
    }
    frame = pd.DataFrame(data).set_index("Date")
    return frame


def test_sma_cross_generates_boolean_signals_without_nans():
    frame = _sample_dataframe()
    signals = strategies.sma_cross(frame)

    assert signals.dtype == bool
    assert not signals.isna().any()
    assert signals.sum() >= 0


def test_compute_rsi_with_constant_series_produces_no_nans():
    prices = pd.Series([100.0] * 30)
    rsi = strategies._compute_rsi(prices, period=14)  # pylint: disable=protected-access

    assert len(rsi) == len(prices)
    assert not rsi.isna().any()
    assert rsi.iloc[-1] == pytest.approx(0.0)


def test_donchian_breakout_detects_high_break():
    frame = _sample_dataframe()
    frame.iloc[-1, frame.columns.get_loc("Close")] = frame["High"].max() + 10

    signals = strategies.donchian_breakout(frame)

    assert signals.iloc[-1]
    assert not signals.isna().any()


def test_gapup_highvol_flags_gap_and_volume_spike():
    frame = _sample_dataframe()
    frame.iloc[-1, frame.columns.get_loc("Open")] = frame["High"].iloc[-2] * 1.05
    frame.iloc[-1, frame.columns.get_loc("Volume")] = frame["Volume"].iloc[-2] * 2

    signals = strategies.gapup_highvol(frame)

    assert signals.iloc[-1]
    assert not signals.isna().any()


def test_alert_generation_inserts_rows(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    csv_path = data_dir / "TEST.csv"
    frame = _sample_dataframe().reset_index()
    frame.to_csv(csv_path, index=False)

    db_path = tmp_path / "signals.db"

    class DummyBacktester:
        @staticmethod
        def run_strategy(strategy, data):  # pragma: no cover - simple stub
            _ = strategy  # unused but kept for signature parity
            close = data.iloc[-1]["Close"]
            return {"signal": True, "entry_price": close, "target_price": close * 1.05}

    alerts.generate_and_store_alerts(
        data_dir=data_dir,
        db_path=db_path,
        strategies=[lambda data: data],
        backtester=DummyBacktester,
        run_date=date(2024, 1, 1),
    )

    stored = alerts.get_active_alerts(db_path=db_path, run_date=date(2024, 1, 1))
    assert len(stored) == 1
    assert stored.loc[0, "ticker"] == "TEST"
    assert stored.loc[0, "strategy"]

