"""Data fetching script for ASX stocks using yfinance."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import yfinance as yf


LOGGER = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_DIR = Path(__file__).resolve().parent / "db"
DATE_FORMAT = "%Y-%m-%d"
DEFAULT_START_DATE = datetime(1990, 1, 1)
EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
CANONICAL_COLUMN_MAP = {
    "date": "Date",
    "datetime": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adjclose": "Close",
    "adj_close": "Close",
    "adj. close": "Close",
    "closeadj": "Close",
    "price": "Close",
    "last": "Close",
    "volume": "Volume",
    "vol": "Volume",
}

for directory in (DATA_DIR, DB_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path) -> dict:
    """Load and validate the configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {exc}") from exc

    tickers = config.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("Configuration must contain a non-empty 'tickers' list.")

    validated_tickers: List[str] = []
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("All tickers must be non-empty strings.")
        validated_tickers.append(ticker.strip().upper())

    start_date_str = config.get("start_date")
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, DATE_FORMAT)
        except ValueError as exc:
            raise ValueError(
                "'start_date' must be in YYYY-MM-DD format if provided."
            ) from exc
    else:
        start_date = DEFAULT_START_DATE

    return {"tickers": validated_tickers, "start_date": start_date}


def ensure_data_directory(directory: Path) -> None:
    """Ensure the data directory exists."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Unable to create data directory at {directory}: {exc}") from exc


def _canonicalize_column_name(column: object) -> str:
    """Return a canonical column name for price data."""

    if isinstance(column, tuple):
        flattened: List[str] = []
        for part in column:
            if part is None:
                continue
            text = str(part).strip()
            if not text:
                continue
            flattened.append(text)
        if flattened:
            column = flattened[0]
        else:
            column = ""

    column_str = str(column).strip()

    if column_str.startswith("(") and column_str.endswith(")"):
        inner = column_str[1:-1]
        parts = [part.strip() for part in inner.split(",")]
        if parts:
            column_str = parts[0].strip("'\"")

    normalized_key = column_str.lower().replace(" ", "_")
    base_key = normalized_key.replace("__", "_").strip("_")

    if base_key in CANONICAL_COLUMN_MAP:
        return CANONICAL_COLUMN_MAP[base_key]

    for candidate in normalized_key.split("_"):
        if candidate in CANONICAL_COLUMN_MAP:
            return CANONICAL_COLUMN_MAP[candidate]

    return column_str


def normalize_price_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw dataframe from CSV or yfinance into the expected schema."""

    if data_frame.empty:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    normalized = data_frame.copy()

    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [
            entry[0] if isinstance(entry, tuple) else entry
            for entry in normalized.columns.to_flat_index()
        ]

    normalized.columns = [_canonicalize_column_name(col) for col in normalized.columns]

    if "Date" not in normalized.columns and "Datetime" in normalized.columns:
        normalized.rename(columns={"Datetime": "Date"}, inplace=True)

    if "Date" not in normalized.columns and "index" in normalized.columns:
        normalized.rename(columns={"index": "Date"}, inplace=True)

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in normalized.columns]
    if "Date" in missing_columns:
        raise ValueError("Dataframe is missing required Date column")

    for column in missing_columns:
        normalized[column] = pd.NA

    normalized = normalized[EXPECTED_COLUMNS].copy()

    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce", utc=True)
    normalized.dropna(subset=["Date"], inplace=True)
    normalized["Date"] = normalized["Date"].dt.tz_convert(None)

    for column in EXPECTED_COLUMNS[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized.sort_values(by="Date", inplace=True)
    normalized.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    normalized["Date"] = normalized["Date"].dt.strftime(DATE_FORMAT)
    normalized.reset_index(drop=True, inplace=True)

    return normalized


def read_existing_data(csv_path: Path) -> Tuple[pd.DataFrame, bool]:
    """Read existing CSV data if available, returning data and whether it needs resaving."""

    if not csv_path.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS), False

    try:
        raw_data = pd.read_csv(csv_path)
    except (pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Failed to read existing data from {csv_path}: {exc}") from exc

    try:
        normalized = normalize_price_dataframe(raw_data)
    except ValueError as exc:
        raise ValueError(f"Existing data at {csv_path} is invalid: {exc}") from exc

    needs_resave = list(raw_data.columns) != EXPECTED_COLUMNS or not raw_data.equals(normalized)

    return normalized, needs_resave


def determine_fetch_start_date(existing_data: pd.DataFrame, config_start_date: datetime) -> datetime:
    """Determine the start date for fetching new data."""
    if existing_data.empty:
        return config_start_date

    parsed_dates = pd.to_datetime(existing_data["Date"], errors="coerce")
    if parsed_dates.isna().all():
        return config_start_date

    last_date = parsed_dates.max()
    next_day = last_date + timedelta(days=1)
    return max(next_day.to_pydatetime(), config_start_date)


def fetch_new_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch new data for a ticker using yfinance."""
    try:
        data = yf.download(
            ticker,
            start=start_date.strftime(DATE_FORMAT),
            end=(end_date + timedelta(days=1)).strftime(DATE_FORMAT),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Failed to download data for %s: %s", ticker, exc)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    if data.empty:
        LOGGER.info("No data returned for %s between %s and %s", ticker, start_date, end_date)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    data = data.reset_index()

    try:
        normalized = normalize_price_dataframe(data)
    except ValueError as exc:
        LOGGER.warning("Downloaded data for %s is invalid: %s", ticker, exc)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    return normalized


def save_normalized_data(csv_path: Path, data_frame: pd.DataFrame, ticker: str) -> bool:
    """Persist a normalized dataframe to disk, logging warnings on failure."""

    try:
        data_frame.to_csv(csv_path, index=False)
    except OSError as exc:
        LOGGER.warning("Failed to persist normalized data for %s: %s", ticker, exc)
        return False

    return True


def update_ticker_data(ticker: str, config_start_date: datetime) -> bool:
    """Update CSV data for a single ticker.

    Returns ``True`` when the ticker data was processed successfully, even if no
    new rows were added. Returns ``False`` when the update failed.
    """

    csv_path = DATA_DIR / f"{ticker.replace('/', '_')}.csv"

    try:
        existing_data, needs_resave = read_existing_data(csv_path)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Failed to read existing data for %s: %s", ticker, exc)
        return False

    fetch_start = determine_fetch_start_date(existing_data, config_start_date)
    today = datetime.utcnow()

    if fetch_start.date() > today.date():
        if needs_resave and not save_normalized_data(csv_path, existing_data, ticker):
            return False
        LOGGER.info("%s data is already up to date", ticker)
        return True

    new_data = fetch_new_data(ticker, fetch_start, today)

    if new_data.empty:
        if needs_resave and not save_normalized_data(csv_path, existing_data, ticker):
            return False
        LOGGER.info("%s returned no new rows", ticker)
        return True

    if not existing_data.empty:
        existing_data["Date"] = pd.to_datetime(existing_data["Date"])
        existing_data["Date"] = existing_data["Date"].dt.strftime(DATE_FORMAT)

    combined = pd.concat([existing_data, new_data], ignore_index=True)
    try:
        combined.to_csv(csv_path, index=False)
    except OSError as exc:
        LOGGER.warning("Failed to persist updated data for %s: %s", ticker, exc)
        return False

    new_rows = len(combined) - len(existing_data)
    LOGGER.info("%s updated with %s new row(s)", ticker, new_rows)

    return True


def main() -> None:
    """Main entry point for the data fetcher script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(CONFIG_FILE)
        ensure_data_directory(DATA_DIR)
        tickers: Iterable[str] = config["tickers"]
        start_date: datetime = config["start_date"]

        success_count = 0
        failure_count = 0

        for ticker in tickers:
            try:
                if update_ticker_data(ticker, start_date):
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to update %s: %s", ticker, exc)
                failure_count += 1
        LOGGER.info(
            "Data update complete with %s success, %s failed", success_count, failure_count
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Fatal error while updating data: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
