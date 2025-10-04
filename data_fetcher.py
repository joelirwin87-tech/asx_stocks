"""Data fetching utilities for ASX stocks.

This module is responsible for downloading and normalising daily OHLCV data
from Yahoo Finance via the :mod:`yfinance` package. Fetched data is persisted to
CSV files inside the ``data/`` directory so that the remainder of the pipeline
can operate purely on local files.

The implementation is intentionally defensive: remote errors, empty responses
and schema drift are all handled gracefully with informative log messages. This
ensures the daily automation keeps running even when individual tickers fail.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class FetchResult:
    """Represents the outcome of fetching data for a single ticker."""

    ticker: str
    data: Optional[pd.DataFrame]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.data is not None and self.error is None


class DataFetcher:
    """Fetches and stores OHLCV data for a list of tickers.

    Parameters
    ----------
    data_dir:
        Directory where downloaded CSV files will be stored. The directory is
        created on instantiation if it does not already exist.
    """

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
        auto_adjust: bool = False,
    ) -> FetchResult:
        """Download data for a single ticker.

        Returns a :class:`FetchResult` describing the outcome. On success the
        resulting DataFrame is also persisted to disk.
        """

        end_date = end_date or date.today().isoformat()
        try:
            LOGGER.info("Downloading %s data from %s to %s", ticker, start_date, end_date)
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=auto_adjust,
                threads=False,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            message = f"Failed to download data for {ticker}: {exc}"
            LOGGER.warning(message)
            return FetchResult(ticker=ticker, data=None, error=message)

        if data.empty:
            message = f"No data returned for {ticker}"
            LOGGER.warning(message)
            return FetchResult(ticker=ticker, data=None, error=message)

        data = self._normalise_dataframe(data)
        filepath = self.data_dir / f"{ticker.replace('.', '_')}.csv"
        data.to_csv(filepath, index=False)
        LOGGER.info("Saved %s rows for %s to %s", len(data), ticker, filepath)
        return FetchResult(ticker=ticker, data=data)

    def fetch_many(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: Optional[str] = None,
        auto_adjust: bool = False,
    ) -> Dict[str, FetchResult]:
        """Fetch data for multiple tickers, returning a mapping of results."""

        results: Dict[str, FetchResult] = {}
        for ticker in tickers:
            result = self.fetch(ticker, start_date=start_date, end_date=end_date, auto_adjust=auto_adjust)
            results[ticker] = result
        return results

    def load_local_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load previously downloaded data from disk if it exists."""

        path = self.data_dir / f"{ticker.replace('.', '_')}.csv"
        if not path.exists():
            LOGGER.info("No local CSV found for %s", ticker)
            return None
        data = pd.read_csv(path, parse_dates=["Date"], dayfirst=False)
        data.sort_values("Date", inplace=True)
        return data

    @classmethod
    def _normalise_dataframe(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Normalise downloaded data to the expected schema."""

        data = data.copy()
        if "Adj Close" in data.columns and "Close" not in data.columns:
            data.rename(columns={"Adj Close": "Close"}, inplace=True)

        column_mapping = {col: col.title() for col in data.columns}
        data.rename(columns=column_mapping, inplace=True)

        missing = [col for col in cls.required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Downloaded data missing required columns: {missing}")

        data.reset_index(inplace=True)
        if "Date" not in data.columns:
            # yfinance returns the index as the datetime when reset_index runs.
            data.rename(columns={data.columns[0]: "Date"}, inplace=True)

        data = data[["Date", *cls.required_columns]].copy()
        data["Date"] = pd.to_datetime(data["Date"], utc=True).dt.tz_convert(None)
        data["Date"] = data["Date"].dt.date
        data.sort_values("Date", inplace=True)
        data.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        return data


def load_config(path: Path | str) -> dict:
    """Load a JSON configuration file.

    Parameters
    ----------
    path:
        Path to the JSON file. Raises ``FileNotFoundError`` if the file does
        not exist and ``ValueError`` if the JSON is invalid.
    """

    import json

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - exceptional path
            raise ValueError(f"Invalid JSON configuration: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    logging.basicConfig(level=logging.INFO)
    cfg = load_config("config.json")
    fetcher = DataFetcher()
    fetcher.fetch_many(cfg["tickers"], start_date=cfg["start_date"])
