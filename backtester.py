"""Backtesting engine for generated trading signals."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)
REPORTS_DIR = Path("reports")
TRADE_COLUMNS = [
    "ticker",
    "strategy",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "quantity",
    "pnl",
    "return_pct",
    "exit_reason",
    "exposure_days",
]


@dataclass(slots=True)
class Position:
    ticker: str
    strategy: str
    entry_date: pd.Timestamp
    entry_price: float
    quantity: int
    stop_price: float
    target_price: float


@dataclass(slots=True)
class Trade:
    ticker: str
    strategy: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float
    exit_reason: str
    exposure_days: int


class Backtester:
    """Simulate the execution of trading signals."""

    def __init__(
        self,
        capital: float,
        risk_per_trade: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        reports_dir: Path | str = REPORTS_DIR,
    ) -> None:
        if capital <= 0:
            raise ValueError("capital must be positive")
        if not 0 < risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be between 0 and 1")
        if take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")

        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run(self, price_data: Dict[str, pd.DataFrame], signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the backtest returning trade- and strategy-level results."""

        if signals.empty:
            LOGGER.info("No signals provided to backtester")
            summary = pd.DataFrame(
                columns=["Strategy", "Trades", "Wins", "Losses", "WinRate", "NetPnL", "AverageReturnPct", "TotalReturnPct", "ExposureDays"]
            )
            return pd.DataFrame(columns=TRADE_COLUMNS), summary

        required_columns = {"Date", "Ticker", "Signal", "Price", "Strategy"}
        missing = required_columns - set(signals.columns)
        if missing:
            raise ValueError(f"Signals missing required columns: {missing}")

        trades: List[Trade] = []
        signals = signals.copy()
        signals["Date"] = pd.to_datetime(signals["Date"]).dt.tz_localize(None)
        signals.sort_values("Date", inplace=True)

        grouped = signals.groupby(["Strategy", "Ticker"], sort=False)
        for (strategy, ticker), signal_group in grouped:
            price_df = price_data.get(ticker)
            if price_df is None or price_df.empty:
                LOGGER.warning("Skipping %s for %s: no price data", ticker, strategy)
                continue
            price_df = price_df.copy()
            price_df["Date"] = pd.to_datetime(price_df["Date"])
            price_df.sort_values("Date", inplace=True)
            price_df.set_index("Date", inplace=True)

            signal_lookup: Dict[pd.Timestamp, pd.DataFrame] = {
                ts.normalize(): frame for ts, frame in signal_group.groupby("Date")
            }
            position: Optional[Position] = None
            last_date: Optional[pd.Timestamp] = None

            for current_date, row in price_df.iterrows():
                last_date = current_date
                signal_rows = signal_lookup.get(current_date.normalize(), pd.DataFrame(columns=signal_group.columns))

                if position is not None:
                    exit_reason: Optional[str] = None
                    exit_price: Optional[float] = None
                    low = float(row["Low"])
                    high = float(row["High"])
                    if low <= position.stop_price:
                        exit_reason = "Stop Loss"
                        exit_price = position.stop_price
                    elif high >= position.target_price:
                        exit_reason = "Take Profit"
                        exit_price = position.target_price
                    elif not signal_rows.empty and (signal_rows["Signal"] == "SELL").any():
                        exit_reason = "Signal Exit"
                        exit_price = float(row["Close"])

                    if exit_reason and exit_price is not None:
                        trades.append(self._build_trade(position, current_date, exit_price, exit_reason))
                        position = None
                        continue

                if position is None and not signal_rows.empty and (signal_rows["Signal"] == "BUY").any():
                    entry_price = float(row["Close"])
                    stop_price = entry_price * (1 - self.stop_loss_pct)
                    target_price = entry_price * (1 + self.take_profit_pct)
                    risk_per_share = entry_price - stop_price
                    if risk_per_share <= 0:
                        LOGGER.debug("Skipping entry for %s %s due to zero risk per share", ticker, strategy)
                        continue
                    risk_amount = self.capital * self.risk_per_trade
                    quantity = int(risk_amount // risk_per_share)
                    if quantity <= 0:
                        LOGGER.debug("Skipping entry for %s %s due to zero quantity", ticker, strategy)
                        continue
                    position = Position(
                        ticker=ticker,
                        strategy=strategy,
                        entry_date=current_date,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_price=stop_price,
                        target_price=target_price,
                    )

            if position is not None and last_date is not None:
                LOGGER.info("Closing open position for %s %s at end of data", ticker, strategy)
                exit_price = float(price_df.iloc[-1]["Close"])
                trades.append(self._build_trade(position, last_date, exit_price, "End of Data"))

        trades_df = pd.DataFrame([asdict(trade) for trade in trades])
        if trades_df.empty:
            summary_df = pd.DataFrame(
                columns=["Strategy", "Trades", "Wins", "Losses", "WinRate", "NetPnL", "AverageReturnPct", "TotalReturnPct", "ExposureDays"]
            )
        else:
            trades_df.sort_values(["strategy", "entry_date"], inplace=True)
            summary_df = self._summarise(trades_df)
            self._write_reports(trades_df, summary_df)
        return trades_df, summary_df

    def _build_trade(self, position: Position, exit_date: pd.Timestamp, exit_price: float, reason: str) -> Trade:
        pnl = (exit_price - position.entry_price) * position.quantity
        return_pct = (exit_price / position.entry_price) - 1
        exposure_days = max((exit_date.normalize() - position.entry_date.normalize()).days, 0) + 1
        return Trade(
            ticker=position.ticker,
            strategy=position.strategy,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            return_pct=return_pct,
            exit_reason=reason,
            exposure_days=exposure_days,
        )

    def _summarise(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        grouped = trades_df.groupby("strategy")
        summary = grouped.agg(
            Trades=("ticker", "count"),
            Wins=("pnl", lambda s: int((s > 0).sum())),
            Losses=("pnl", lambda s: int((s <= 0).sum())),
            NetPnL=("pnl", "sum"),
            AverageReturnPct=("return_pct", "mean"),
            TotalReturnPct=("return_pct", "sum"),
            ExposureDays=("exposure_days", "sum"),
        ).reset_index().rename(columns={"strategy": "Strategy"})
        summary["WinRate"] = summary.apply(lambda row: row["Wins"] / row["Trades"] if row["Trades"] else 0.0, axis=1)
        return summary[["Strategy", "Trades", "Wins", "Losses", "WinRate", "NetPnL", "AverageReturnPct", "TotalReturnPct", "ExposureDays"]]

    def _write_reports(self, trades_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        for strategy, group in trades_df.groupby("strategy"):
            slug = strategy.lower().replace(" ", "_")
            path = self.reports_dir / f"trades_{slug}.csv"
            group.to_csv(path, index=False)
        summary_path = self.reports_dir / "strategy_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        LOGGER.info("Wrote trade reports to %s", self.reports_dir)
