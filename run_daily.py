"""Daily pipeline runner for the ASX stocks project."""
from __future__ import annotations

import argparse
import logging
from typing import Dict, List

import pandas as pd

from alerts import AlertsManager, generate_alerts
from backtester import Backtester
from data_fetcher import DataFetcher, load_config
from strategies import build_default_strategies

LOGGER = logging.getLogger(__name__)

DEFAULT_TAKE_PROFIT_PCT = 0.08
DEFAULT_STOP_LOSS_PCT = 0.04


def _resolve_percentage_setting(
    config: Dict[str, object],
    key: str,
    default: float,
) -> float:
    """Return a validated percentage value for the given configuration key."""

    if key in config:
        value = config[key]
    else:
        LOGGER.info("%s not provided in config; using default %.2f", key, default)
        value = default

    try:
        percentage = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{key} must be a numeric value") from exc

    if percentage <= 0:
        raise ValueError(f"{key} must be positive; received {percentage}")

    return percentage


def orchestrate(config_path: str = "config.json") -> Dict[str, int]:
    config = load_config(config_path)
    tickers: List[str] = config.get("tickers", [])
    if not tickers:
        raise ValueError("Configuration must include at least one ticker")

    fetcher = DataFetcher()
    fetch_results = fetcher.fetch_many(tickers, start_date=config["start_date"])

    price_data: Dict[str, pd.DataFrame] = {}
    success_count = 0
    for ticker, result in fetch_results.items():
        if result.success and result.data is not None:
            price_data[ticker] = result.data
            success_count += 1
        else:
            local = fetcher.load_local_data(ticker)
            if local is not None:
                price_data[ticker] = local
                LOGGER.info("Loaded cached data for %s", ticker)
            else:
                LOGGER.warning("No data available for %s", ticker)

    strategy_overrides = config.get("strategies")
    strategies = build_default_strategies(strategy_overrides)

    signals_frames: List[pd.DataFrame] = []
    for ticker, data in price_data.items():
        for strategy in strategies:
            signals = strategy.generate_signals(ticker, data)
            if not signals.empty:
                signals_frames.append(signals)

    signals_df = (
        pd.concat(signals_frames, ignore_index=True)
        if signals_frames
        else pd.DataFrame(columns=["Date", "Ticker", "Signal", "Price", "Strategy"])
    )

    take_profit_pct = _resolve_percentage_setting(
        config, "take_profit_pct", DEFAULT_TAKE_PROFIT_PCT
    )
    stop_loss_pct = _resolve_percentage_setting(
        config, "stop_loss_pct", DEFAULT_STOP_LOSS_PCT
    )

    backtester = Backtester(
        capital=float(config["capital"]),
        risk_per_trade=float(config.get("risk_per_trade", 0.02)),
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    trades_df, summary_df = backtester.run(price_data, signals_df)

    latest_trading_day = _latest_trading_day(price_data)
    alerts = generate_alerts(signals_df, as_of=latest_trading_day)
    alerts_manager = AlertsManager(config.get("alerts_db", "signals.db"))
    alerts_saved = alerts_manager.save_alerts(alerts)

    LOGGER.info(
        "Pipeline completed: %s tickers fetched, %s strategies, %s trades, %s alerts",
        success_count,
        len(strategies),
        len(trades_df),
        alerts_saved,
    )

    return {
        "tickers": len(tickers),
        "fetched": success_count,
        "strategies": len(strategies),
        "signals": len(signals_df),
        "trades": len(trades_df),
        "summaries": len(summary_df),
        "alerts": alerts_saved,
    }


def _latest_trading_day(price_data: Dict[str, pd.DataFrame]) -> pd.Timestamp:
    dates = []
    for frame in price_data.values():
        if frame.empty:
            continue
        dates.append(pd.to_datetime(frame["Date"]).max())
    if not dates:
        return pd.Timestamp.utcnow().normalize()
    return max(dates).normalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ASX stocks pipeline")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--log", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))
    try:
        stats = orchestrate(args.config)
    except Exception as exc:  # pragma: no cover - top-level safety
        LOGGER.exception("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("Run completed: %s", stats)


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
