from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_fetcher import DataFetcher


def test_fetcher_normalises_and_saves(tmp_path: Path, mock_data_fetch):
    fetcher = DataFetcher(data_dir=tmp_path)
    result = fetcher.fetch("TEST.AX", start_date="2023-01-01")
    assert result.success
    assert (tmp_path / "TEST_AX.csv").exists()
    assert list(result.data.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(result.data["Date"]))


def test_fetcher_handles_empty(monkeypatch, tmp_path: Path):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("yfinance.download", fake_download)
    fetcher = DataFetcher(data_dir=tmp_path)
    result = fetcher.fetch("TEST.AX", start_date="2023-01-01")
    assert not result.success
    assert result.error
