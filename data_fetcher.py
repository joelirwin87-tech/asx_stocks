"""Data fetching script for ASX stocks using yfinance."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf


CONFIG_FILE = Path(__file__).resolve().parent / "config.json"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATE_FORMAT = "%Y-%m-%d"
DEFAULT_START_DATE = datetime(1990, 1, 1)


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


def read_existing_data(csv_path: Path) -> pd.DataFrame:
    """Read existing CSV data if available."""
    if not csv_path.exists():
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    try:
        data_frame = pd.read_csv(csv_path, parse_dates=["Date"], dtype={"Volume": "Int64"})
    except (pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Failed to read existing data from {csv_path}: {exc}") from exc

    expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if list(data_frame.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns in {csv_path}. Expected {expected_columns}, got {list(data_frame.columns)}"
        )

    return data_frame


def determine_fetch_start_date(existing_data: pd.DataFrame, config_start_date: datetime) -> datetime:
    """Determine the start date for fetching new data."""
    if existing_data.empty:
        return config_start_date

    last_date = existing_data["Date"].max()
    if pd.isna(last_date):
        return config_start_date

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
        raise ConnectionError(f"Failed to download data for {ticker}: {exc}") from exc

    if data.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    data = data.reset_index()

    if "Date" not in data.columns:
        raise ValueError(f"Downloaded data for {ticker} does not contain 'Date' column.")

    filtered_columns = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }

    missing_columns = [column for column in filtered_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Downloaded data for {ticker} is missing columns: {missing_columns}")

    result = data[["Date", *filtered_columns.keys()]].copy()
    result.rename(columns=filtered_columns, inplace=True)
    result["Date"] = pd.to_datetime(result["Date"]).dt.tz_localize(None)
    result["Date"] = result["Date"].dt.strftime(DATE_FORMAT)

    return result


def update_ticker_data(ticker: str, config_start_date: datetime) -> None:
    """Update CSV data for a single ticker."""
    csv_path = DATA_DIR / f"{ticker.replace('/', '_')}.csv"

    existing_data = read_existing_data(csv_path)
    fetch_start = determine_fetch_start_date(existing_data, config_start_date)
    today = datetime.utcnow()

    if fetch_start.date() > today.date():
        print(f"{ticker}: data is already up to date.")
        return

    new_data = fetch_new_data(ticker, fetch_start, today)

    if new_data.empty:
        print(f"{ticker}: no new data available.")
        return

    if not existing_data.empty:
        existing_data["Date"] = pd.to_datetime(existing_data["Date"])
        existing_data["Date"] = existing_data["Date"].dt.strftime(DATE_FORMAT)

    combined = pd.concat([existing_data, new_data], ignore_index=True)
    combined.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    combined.sort_values(by="Date", inplace=True)

    try:
        combined.to_csv(csv_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write data to {csv_path}: {exc}") from exc

    new_rows = len(combined) - len(existing_data)
    print(f"{ticker}: data updated with {new_rows} new row(s).")


def main() -> None:
    """Main entry point for the data fetcher script."""
    try:
        config = load_config(CONFIG_FILE)
        ensure_data_directory(DATA_DIR)
        tickers: Iterable[str] = config["tickers"]
        start_date: datetime = config["start_date"]

        for ticker in tickers:
            try:
                update_ticker_data(ticker, start_date)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Error updating {ticker}: {exc}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
