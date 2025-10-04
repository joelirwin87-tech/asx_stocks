"""Tests for the data fetcher utilities."""

from __future__ import annotations

import io
from datetime import datetime
import warnings

import pandas as pd

import data_fetcher


def test_normalize_and_determine_fetch_start_date_without_user_warning() -> None:
    """Mixed date formats should parse without emitting UserWarning."""

    csv_content = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2024-01-01,1,1,1,1,100",
            "20240102,2,2,2,2,200",
            "2024-01-03 00:00:00,3,3,3,3,300",
            "",
        ]
    )
    csv_buffer = io.StringIO(csv_content)
    raw_frame = pd.read_csv(csv_buffer)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        normalized = data_fetcher.normalize_price_dataframe(raw_frame)

    assert normalized["Date"].tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]

    config_start_date = datetime(2023, 12, 31)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fetch_start = data_fetcher.determine_fetch_start_date(normalized, config_start_date)

    assert fetch_start == datetime(2024, 1, 4)
