"""Daily automation script for updating data, running strategies, and tests.

This module orchestrates the end-to-end workflow expected for the ASX stocks
project.  The responsibilities include:

1. Updating local market data via :func:`data_fetcher.update` (or a compatible
   fallback when ``update`` is not present).
2. Executing every registered strategy through the shared backtester to produce
   trade files and summary CSV outputs under ``/data``.
3. Generating alerts that are stored in ``db/signals.db``.
4. Running the full unit-test suite located beneath ``/tests``.
5. Committing repository changes when, and only when, the tests succeed.

The script is intentionally defensive.  Each high-level stage is isolated to
ensure that a failure does not leave the repository in an inconsistent state.
The workflow is compatible with macOS 10.15.7 and Python 3.9.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

import alerts
import backtester
import data_fetcher
import strategies


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration bundle describing how to execute a strategy."""

    name: str
    signal_func: Callable[[pd.DataFrame], pd.Series]
    take_profit_pct: float


@dataclass(frozen=True)
class DataUpdateStats:
    total_tickers: int
    succeeded: int
    failed: int
    failures: Dict[str, str]


@dataclass(frozen=True)
class BacktestStats:
    strategies_attempted: int
    ticker_count: int
    combinations_attempted: int
    combinations_failed: int
    summaries_written: int
    total_trades: int


@dataclass(frozen=True)
class AlertStats:
    alerts_generated: int
    tickers_with_alerts: int
    strategies_triggered: int


STRATEGY_CONFIGS: List[StrategyConfig] = [
    StrategyConfig("sma_cross", strategies.sma_cross, 0.05),
    StrategyConfig("pullback_uptrend", strategies.pullback_uptrend, 0.04),
    StrategyConfig("donchian_breakout", strategies.donchian_breakout, 0.06),
    StrategyConfig("gapup_highvol", strategies.gapup_highvol, 0.03),
]


def _ensure_runtime_directories(repo_root: Path) -> None:
    for path in {repo_root / "data", repo_root / "db", data_fetcher.DATA_DIR, alerts.DEFAULT_DATA_DIR}:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOGGER.debug("Unable to create directory %s: %s", path, exc)


def _run_data_update() -> DataUpdateStats:
    """Update all configured tickers while capturing aggregate results."""

    LOGGER.info("Starting data updates for configured tickers")

    config = data_fetcher.load_config(data_fetcher.CONFIG_FILE)
    data_fetcher.ensure_data_directory(data_fetcher.DATA_DIR)

    raw_tickers: Iterable[str] = config.get("tickers", [])  # type: ignore[arg-type]
    start_date = config.get("start_date")

    normalized_tickers: List[str] = []
    if isinstance(raw_tickers, str):
        raw_iterable: Iterable[str] = [raw_tickers]
    else:
        raw_iterable = raw_tickers

    for raw in raw_iterable:
        ticker = str(raw).strip().upper()
        if not ticker:
            LOGGER.warning("Encountered empty ticker symbol in configuration; skipping")
            continue
        normalized_tickers.append(ticker)

    seen = list(dict.fromkeys(normalized_tickers))

    total = len(seen)
    successes = 0
    failures: Dict[str, str] = {}

    for ticker in seen:
        try:
            data_fetcher.update_ticker_data(ticker, start_date)
            successes += 1
        except Exception as exc:  # pragma: no cover - logged for diagnostics
            failures[ticker] = str(exc)
            LOGGER.exception("Failed to update data for %s: %s", ticker, exc)

    stats = DataUpdateStats(total_tickers=total, succeeded=successes, failed=len(failures), failures=failures)

    if stats.total_tickers == 0:
        LOGGER.warning("No tickers configured for data update")
    LOGGER.info(
        "Data update summary | total=%s | succeeded=%s | failed=%s",
        stats.total_tickers,
        stats.succeeded,
        stats.failed,
    )

    if stats.failed:
        failed_details = ", ".join(f"{ticker}: {reason}" for ticker, reason in failures.items())
        LOGGER.warning("Tickers with update failures: %s", failed_details)

    return stats


def _prepare_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned DataFrame with a ``DatetimeIndex`` suitable for backtests."""

    prepared = frame.copy()

    date_col = None
    for candidate in ("Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"):
        if candidate in prepared.columns:
            date_col = candidate
            break

    if date_col is not None:
        prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce")
        prepared = prepared.dropna(subset=[date_col])
        prepared = prepared.set_index(date_col)
    elif isinstance(prepared.index, pd.DatetimeIndex):
        prepared = prepared.sort_index()
    else:
        raise ValueError("Price data is missing a date column or datetime index")

    prepared = prepared.sort_index()

    missing_cols = [col for col in backtester.REQUIRED_COLUMNS if col not in prepared.columns]
    if missing_cols:
        raise ValueError(f"Data frame missing required OHLCV columns: {missing_cols}")

    # Ensure deterministic column order and data types.
    for column in backtester.REQUIRED_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=["Open", "High", "Low", "Close"])
    return prepared


def _run_strategies_and_backtests(
    data_frames: Dict[str, pd.DataFrame], output_dir: Path
) -> BacktestStats:
    """Execute each strategy/backtest combination and persist summary CSVs."""

    output_dir.mkdir(parents=True, exist_ok=True)

    ticker_count = len(data_frames)
    strategy_count = len(STRATEGY_CONFIGS)
    combinations_attempted = 0
    combinations_failed = 0
    total_trades = 0
    summaries_written = 0

    for config in STRATEGY_CONFIGS:
        summaries: List[Dict[str, float]] = []

        for ticker, raw_frame in data_frames.items():
            combinations_attempted += 1
            try:
                prepared = _prepare_dataframe(raw_frame)
                signals = config.signal_func(prepared).reindex(prepared.index).fillna(False).astype(bool)
                trades_df, summary = backtester.run_tp_backtest(
                    prepared,
                    signals,
                    config.take_profit_pct,
                    name=f"{config.name}_{ticker}",
                )

                summary_record: Dict[str, float] = {
                    "Ticker": ticker,
                    "Trades": summary["Trades"],
                    "HitRate": summary["HitRate"],
                    "TotalPnL": summary["TotalPnL"],
                    "CumReturn": summary["CumReturn"],
                }
                trades_generated = int(len(trades_df))
                summary_record["TradesGenerated"] = trades_generated
                total_trades += trades_generated
                summaries.append(summary_record)
            except Exception as exc:  # pragma: no cover - logged for diagnostics
                combinations_failed += 1
                LOGGER.exception(
                    "Failed to process strategy %s for ticker %s: %s", config.name, ticker, exc
                )

        summary_df = pd.DataFrame(summaries)
        if summary_df.empty:
            summary_df = pd.DataFrame(
                columns=["Ticker", "Trades", "HitRate", "TotalPnL", "CumReturn", "TradesGenerated"]
            )
        summary_path = output_dir / f"{config.name}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        summaries_written += 1
        LOGGER.info("Saved summary for %s to %s", config.name, summary_path)

    stats = BacktestStats(
        strategies_attempted=strategy_count,
        ticker_count=ticker_count,
        combinations_attempted=combinations_attempted,
        combinations_failed=combinations_failed,
        summaries_written=summaries_written,
        total_trades=total_trades,
    )

    LOGGER.info(
        "Backtest summary | strategies=%s | tickers=%s | attempted=%s | failed=%s | trades=%s",
        stats.strategies_attempted,
        stats.ticker_count,
        stats.combinations_attempted,
        stats.combinations_failed,
        stats.total_trades,
    )

    return stats


def _generate_alerts(data_dir: Path, db_path: Path) -> AlertStats:
    """Trigger alert generation and persistence."""

    LOGGER.info("Generating alerts using data from %s", data_dir)
    generated_alerts = alerts.generate_and_store_alerts(data_dir=data_dir, db_path=db_path)

    alert_count = len(generated_alerts)
    tickers_with_alerts = len({alert.ticker for alert in generated_alerts})
    strategies_triggered = len({alert.strategy for alert in generated_alerts})

    stats = AlertStats(
        alerts_generated=alert_count,
        tickers_with_alerts=tickers_with_alerts,
        strategies_triggered=strategies_triggered,
    )

    LOGGER.info(
        "Alert generation summary | alerts=%s | tickers=%s | strategies=%s",
        stats.alerts_generated,
        stats.tickers_with_alerts,
        stats.strategies_triggered,
    )

    return stats


def _run_tests(tests_path: Path) -> None:
    """Execute the project's pytest suite."""

    LOGGER.info("Running pytest for %s", tests_path)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_path)],
        cwd=tests_path.parent,
        check=False,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError("Pytest suite failed; aborting daily run")


def _commit_changes(repo_root: Path) -> None:
    """Commit repository changes when present."""

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError("Unable to determine git status")

    if not status.stdout.strip():
        LOGGER.info("No changes detected; skipping git commit")
        return

    LOGGER.info("Committing repository changes")
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Daily automated run"],
        cwd=repo_root,
        check=True,
    )


def main() -> None:
    """Entry point for the daily automation workflow."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parent
    data_dir = data_fetcher.DATA_DIR
    output_dir = repo_root / "data"
    db_path = repo_root / "db" / "signals.db"
    tests_dir = repo_root / "tests"

    try:
        _ensure_runtime_directories(repo_root)
        data_stats = _run_data_update()

        LOGGER.info("Loading price data from %s", data_dir)
        data_frames = alerts.load_latest_ticker_data(data_dir)

        if not data_frames:
            LOGGER.warning("No ticker data available for strategy execution")
            backtest_stats = BacktestStats(
                strategies_attempted=len(STRATEGY_CONFIGS),
                ticker_count=0,
                combinations_attempted=0,
                combinations_failed=0,
                summaries_written=0,
                total_trades=0,
            )
        else:
            backtest_stats = _run_strategies_and_backtests(data_frames, output_dir)

        alert_stats = _generate_alerts(data_dir, db_path)
        _run_tests(tests_dir)
        _commit_changes(repo_root)
        LOGGER.info(
            "Daily run metrics | data_total=%s | data_success=%s | data_failed=%s | "
            "backtests_attempted=%s | backtests_failed=%s | trades_generated=%s | "
            "alerts_generated=%s | alert_tickers=%s | alert_strategies=%s",
            data_stats.total_tickers,
            data_stats.succeeded,
            data_stats.failed,
            backtest_stats.combinations_attempted,
            backtest_stats.combinations_failed,
            backtest_stats.total_trades,
            alert_stats.alerts_generated,
            alert_stats.tickers_with_alerts,
            alert_stats.strategies_triggered,
        )
    except Exception as exc:  # pragma: no cover - top-level safety net
        LOGGER.exception("Daily run failed: %s", exc)
        sys.exit(1)

    print("Daily run complete")


if __name__ == "__main__":
    main()

