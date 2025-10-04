"""Utilities for generating trading alerts from backtested strategies.

This module provides the plumbing required to load the latest ticker data,
run every registered strategy through the project's backtester on the most
recent bar, and persist any signals that fire today into a SQLite database.

The public surface is intentionally small:

``generate_and_store_alerts``
    Entry point that orchestrates the workflow.

``get_active_alerts``
    Helper that returns a ``pandas.DataFrame`` of today's alerts.

Both functions are designed with testability in mind.  Callers can inject
custom strategy collections, backtester implementations, and data paths when
needed.  In production the module falls back to auto-discovery (attempting to
import ``strategies`` and ``backtester`` modules).

The SQLite schema is simple and debounced via a composite primary key, so
running the workflow multiple times per day will never create duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import sqlite3


LOGGER = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "signals.db"
EXTERNAL_DATA_DIR = Path("/data")
EXTERNAL_DB_DIR = Path("/db")


def _ensure_directories() -> None:
    """Create default and external runtime directories when possible."""

    for directory in {
        DEFAULT_DATA_DIR,
        DEFAULT_DB_PATH.parent,
        EXTERNAL_DATA_DIR,
        EXTERNAL_DB_DIR,
    }:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - diagnostics only
            LOGGER.debug("Unable to create directory %s: %s", directory, exc)


_ensure_directories()


@dataclass(frozen=True)
class Alert:
    """Represents a single actionable alert produced by a strategy."""

    run_date: date
    ticker: str
    strategy: str
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

    def as_row(self) -> Tuple[str, str, str, float, Optional[float], Optional[float]]:
        return (
            self.run_date.isoformat(),
            self.ticker,
            self.strategy,
            float(self.entry_price),
            None if self.target_price is None else float(self.target_price),
            None if self.stop_loss is None else float(self.stop_loss),
        )


class DataLoadError(RuntimeError):
    """Raised when ticker data cannot be loaded."""


def load_latest_ticker_data(data_dir: Path | str = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load the most recent dataset for every ticker found in ``data_dir``.

    The loader supports two directory layouts:

    - ``/data/TICKER.csv`` – a single file per ticker.
    - ``/data/TICKER/*.csv`` – multiple dated files per ticker, the newest is
      used.

    CSV, Parquet, and JSON (records oriented) formats are supported.  All
    columns are left untouched except that a ``Date`` column, if present, is
    converted to ``datetime64`` and sorted ascending.
    """

    data_path = Path(data_dir)
    try:
        data_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - defensive
        raise DataLoadError(f"Unable to access data directory {data_path}: {exc}") from exc

    candidate_paths = [data_path]
    if data_path != EXTERNAL_DATA_DIR and EXTERNAL_DATA_DIR.exists():
        candidate_paths.append(EXTERNAL_DATA_DIR)

    ticker_frames: Dict[str, pd.DataFrame] = {}
    for base_path in candidate_paths:
        for entry in sorted(base_path.iterdir(), key=lambda p: p.name):
            if entry.is_dir():
                latest_file = _find_latest_file(entry)
                if latest_file is None:
                    LOGGER.warning("No data files found for ticker directory %s", entry.name)
                    continue
                ticker = entry.name
                frame = _read_data_file(latest_file)
            elif entry.is_file():
                ticker = entry.stem
                frame = _read_data_file(entry)
            else:
                LOGGER.debug("Skipping unknown filesystem entry: %s", entry)
                continue

            if frame.empty:
                LOGGER.warning("Skipping ticker %s because data frame is empty", ticker)
                continue

            frame = _prepare_dataframe(frame)
            if frame.empty:
                LOGGER.warning("Skipping ticker %s because prepared frame is empty", ticker)
                continue

            ticker_frames.setdefault(ticker.upper(), frame)

    if not ticker_frames:
        LOGGER.info("No ticker data discovered in %s", data_path)

    return ticker_frames


def _find_latest_file(directory: Path) -> Optional[Path]:
    """Return the newest file (by modified time) within ``directory``."""

    candidates = [path for path in directory.iterdir() if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_data_file(path: Path) -> pd.DataFrame:
    """Read a single data file into a DataFrame."""

    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            frame = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix in {".json"}:
            frame = pd.read_json(path, orient="records")
        else:
            raise DataLoadError(f"Unsupported file extension for {path}")
    except pd.errors.EmptyDataError:
        LOGGER.info("Data file %s is empty", path)
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - defensive logging
        raise DataLoadError(f"Failed to load data file {path}: {exc}") from exc

    return frame


def _prepare_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw data by parsing dates and sorting chronologically."""

    frame = frame.copy()

    date_col = None
    for candidate in ("Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"):
        if candidate in frame.columns:
            date_col = candidate
            break

    if date_col is not None:
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col])
        frame = frame.sort_values(by=date_col)
        frame = frame.reset_index(drop=True)
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.sort_index()
    else:
        LOGGER.debug("No explicit date column found in dataframe with columns: %s", frame.columns)

    return frame


def discover_strategies(strategies: Optional[Iterable[Any]] = None) -> List[Any]:
    """Return a list of strategies ready for execution."""

    if strategies is not None:
        return list(strategies)

    try:
        module = importlib.import_module("strategies")
    except ModuleNotFoundError:
        LOGGER.warning("No strategies module found; no alerts will be generated")
        return []

    if hasattr(module, "get_strategies") and callable(module.get_strategies):
        resolved = module.get_strategies()
    elif hasattr(module, "ALL_STRATEGIES"):
        resolved = module.ALL_STRATEGIES
    else:
        resolved = [getattr(module, name) for name in dir(module) if not name.startswith("_")]

    resolved_list = []
    for item in resolved:
        if inspect.isclass(item) or callable(item):
            resolved_list.append(item)
        else:
            LOGGER.debug("Skipping non-callable strategy candidate: %r", item)

    return resolved_list


def _instantiate_strategy(strategy: Any) -> Any:
    if inspect.isclass(strategy):
        try:
            return strategy()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to instantiate strategy {strategy}: {exc}") from exc
    return strategy


def _strategy_name(strategy: Any) -> str:
    name = getattr(strategy, "name", None)
    if isinstance(name, str) and name:
        return name
    return strategy.__class__.__name__ if not inspect.isclass(strategy) else strategy.__name__


def _load_backtester(backtester: Optional[Any] = None) -> Any:
    if backtester is not None:
        return backtester

    try:
        return importlib.import_module("backtester")
    except ModuleNotFoundError as exc:
        raise RuntimeError("A backtester implementation is required to generate alerts") from exc


def _run_strategy_on_latest_bar(
    backtester_module: Any,
    strategy: Any,
    data: pd.DataFrame,
) -> Any:
    """Execute ``strategy`` using ``backtester_module`` over the latest bar only."""

    last_row = data.iloc[[-1]] if len(data.index) else data

    # Try a handful of common backtester interfaces for flexibility.
    if hasattr(backtester_module, "run_strategy") and callable(backtester_module.run_strategy):
        return backtester_module.run_strategy(strategy=strategy, data=last_row)

    if hasattr(backtester_module, "run") and callable(backtester_module.run):
        return backtester_module.run(strategy=strategy, data=last_row)

    if hasattr(backtester_module, "Backtester"):
        bt_cls = backtester_module.Backtester
        try:
            bt_instance = bt_cls(last_row)
        except TypeError:
            bt_instance = bt_cls(data=last_row)

        if hasattr(bt_instance, "run_strategy") and callable(bt_instance.run_strategy):
            return bt_instance.run_strategy(strategy)
        if hasattr(bt_instance, "run") and callable(bt_instance.run):
            return bt_instance.run(strategy)

    raise RuntimeError("Unsupported backtester interface; expected run_strategy or Backtester class")


def _extract_alert_from_result(
    result: Any,
    last_row: pd.Series,
    ticker: str,
    strategy_name: str,
    run_date: date,
) -> Optional[Alert]:
    """Translate a backtester result into an ``Alert`` if a signal fired."""

    parsed = _parse_result_payload(result)
    if not parsed:
        return None

    signal_triggered = bool(parsed.get("signal") or parsed.get("trigger") or parsed.get("alert"))
    if not signal_triggered:
        return None

    entry_price = _coalesce_numeric(
        parsed.get("entry_price"),
        parsed.get("price"),
        parsed.get("entry"),
        parsed.get("close"),
        *(last_row.get(col) for col in ("Close", "close", "Adj Close", "Price", "Last")),
    )
    if entry_price is None:
        LOGGER.warning("Signal triggered for %s/%s but entry price unavailable", ticker, strategy_name)
        return None

    target_price = _coalesce_numeric(parsed.get("target_price"), parsed.get("target"))
    stop_loss = _coalesce_numeric(parsed.get("stop_loss"), parsed.get("stop"))

    return Alert(
        run_date=run_date,
        ticker=ticker,
        strategy=strategy_name,
        entry_price=entry_price,
        target_price=target_price,
        stop_loss=stop_loss,
    )


def _parse_result_payload(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}

    if isinstance(result, dict):
        return result

    if isinstance(result, pd.DataFrame) and not result.empty:
        # Assume signals correspond to the last row of the DataFrame.
        return result.iloc[-1].to_dict()

    if isinstance(result, pd.Series):
        return result.to_dict()

    if isinstance(result, (list, tuple)) and result:
        tail = result[-1]
        if isinstance(tail, dict):
            return tail
        if isinstance(tail, pd.Series):
            return tail.to_dict()

    if hasattr(result, "_asdict"):
        try:
            return result._asdict()
        except Exception:  # pragma: no cover - defensive
            pass

    LOGGER.debug("Unable to parse backtester result of type %s", type(result))
    return {}


def _coalesce_numeric(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)) and pd.notna(value):
            return float(value)
        if isinstance(value, (pd.Series, pd.DataFrame)):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(numeric):
            return float(numeric)
    return None


def _ensure_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            strategy TEXT NOT NULL,
            entry_price REAL NOT NULL,
            target_price REAL,
            stop_loss REAL,
            PRIMARY KEY (date, ticker, strategy)
        )
        """
    )
    connection.commit()
    return connection


def _persist_alerts(alerts: Sequence[Alert], db_path: Path = DEFAULT_DB_PATH) -> None:
    if not alerts:
        return

    with _ensure_database(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO alerts (
                date, ticker, strategy, entry_price, target_price, stop_loss
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [alert.as_row() for alert in alerts],
        )
        conn.commit()


def _print_alerts(alerts: Iterable[Alert]) -> None:
    for alert in alerts:
        LOGGER.info(
            "Alert generated | %s | %s | entry=%s | target=%s | stop=%s",
            alert.run_date.isoformat(),
            f"{alert.ticker} ({alert.strategy})",
            f"{alert.entry_price:.4f}",
            "-" if alert.target_price is None else f"{alert.target_price:.4f}",
            "-" if alert.stop_loss is None else f"{alert.stop_loss:.4f}",
        )
        print(
            {
                "Date": alert.run_date.isoformat(),
                "Ticker": alert.ticker,
                "Strategy": alert.strategy,
                "EntryPrice": alert.entry_price,
                "TargetPrice": alert.target_price,
                "StopLoss": alert.stop_loss,
            }
        )


def generate_and_store_alerts(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    db_path: Path | str = DEFAULT_DB_PATH,
    strategies: Optional[Iterable[Any]] = None,
    backtester: Optional[Any] = None,
    run_date: Optional[date] = None,
) -> List[Alert]:
    """Run every strategy against the latest bar and persist today's alerts."""

    run_date = run_date or datetime.utcnow().date()
    data_frames = load_latest_ticker_data(data_dir)
    strategies_list = discover_strategies(strategies)

    if not strategies_list:
        LOGGER.info("No strategies discovered; aborting alert generation")
        return []

    if not data_frames:
        LOGGER.info("No data available for alert generation")
        _ensure_database(Path(db_path))
        return []

    backtester_module = _load_backtester(backtester)

    alerts: List[Alert] = []

    for ticker, frame in data_frames.items():
        last_row = frame.iloc[-1]
        for strategy_ref in strategies_list:
            strategy_instance = _instantiate_strategy(strategy_ref)
            strategy_name = _strategy_name(strategy_instance)

            try:
                result = _run_strategy_on_latest_bar(backtester_module, strategy_instance, frame)
            except Exception as exc:
                LOGGER.exception(
                    "Failed to run strategy %s for ticker %s: %s", strategy_name, ticker, exc
                )
                continue

            alert = _extract_alert_from_result(result, last_row, ticker, strategy_name, run_date)
            if alert is not None:
                alerts.append(alert)

    _persist_alerts(alerts, Path(db_path))
    _print_alerts(alerts)

    return alerts


def get_active_alerts(
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    run_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return a DataFrame of alerts generated on ``run_date`` (defaults to today)."""

    run_date = run_date or datetime.utcnow().date()
    db_path = Path(db_path)

    if not db_path.exists():
        LOGGER.info("Alert database not found at %s", db_path)
        with _ensure_database(db_path) as connection:
            connection.commit()
        return pd.DataFrame(
            columns=["date", "ticker", "strategy", "entry_price", "target_price", "stop_loss"]
        )

    with sqlite3.connect(db_path) as conn:
        try:
            frame = pd.read_sql_query(
                """
                SELECT date, ticker, strategy, entry_price, target_price, stop_loss
                FROM alerts
                WHERE date = ?
                ORDER BY ticker, strategy
                """,
                conn,
                params=(run_date.isoformat(),),
            )
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                LOGGER.info("Alerts table does not exist in database %s", db_path)
                return pd.DataFrame(
                    columns=[
                        "date",
                        "ticker",
                        "strategy",
                        "entry_price",
                        "target_price",
                        "stop_loss",
                    ]
                )
            raise

    return frame


def main() -> None:  # pragma: no cover - convenience wrapper
    logging.basicConfig(level=logging.INFO)
    try:
        generate_and_store_alerts()
    except DataLoadError as exc:
        LOGGER.error("Unable to generate alerts: %s", exc)


if __name__ == "__main__":  # pragma: no cover - script support
    main()

