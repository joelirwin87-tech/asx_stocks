"""Trading strategy implementations.

Each strategy adheres to the :class:`BaseStrategy` interface, producing a
DataFrame of trade signals for an individual ticker. Signals follow a consistent
schema with the columns ``Date``, ``Ticker``, ``Signal``, ``Price`` and
``Strategy``. Downstream components can therefore operate on strategy output
without bespoke handling for each algorithm.

The strategies included are intentionally simple yet illustrative. They balance
clarity, configurability and the ability to compose multiple signals for a
single ticker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd

SignalFrame = pd.DataFrame


@dataclass(slots=True)
class BaseStrategy:
    """Base class for trading strategies."""

    name: str

    def generate_signals(self, ticker: str, data: pd.DataFrame) -> SignalFrame:
        raise NotImplementedError

    def _build_signal_frame(
        self,
        ticker: str,
        data: pd.DataFrame,
        buy_mask: pd.Series,
        sell_mask: pd.Series,
        price_column: str = "Close",
    ) -> SignalFrame:
        """Construct a normalised signal DataFrame from boolean masks."""

        records: List[Dict[str, object]] = []
        for date, price in data.loc[buy_mask, ["Date", price_column]].itertuples(index=False):
            records.append(
                {
                    "Date": pd.to_datetime(date),
                    "Ticker": ticker,
                    "Signal": "BUY",
                    "Price": float(price),
                    "Strategy": self.name,
                }
            )
        for date, price in data.loc[sell_mask, ["Date", price_column]].itertuples(index=False):
            records.append(
                {
                    "Date": pd.to_datetime(date),
                    "Ticker": ticker,
                    "Signal": "SELL",
                    "Price": float(price),
                    "Strategy": self.name,
                }
            )
        if not records:
            return pd.DataFrame(columns=["Date", "Ticker", "Signal", "Price", "Strategy"])
        frame = pd.DataFrame.from_records(records)
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame


@dataclass(slots=True)
class SMACrossoverStrategy(BaseStrategy):
    short_window: int = 10
    long_window: int = 30

    def __post_init__(self) -> None:
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")

    def generate_signals(self, ticker: str, data: pd.DataFrame) -> SignalFrame:
        df = data.copy()
        df["sma_short"] = df["Close"].rolling(self.short_window).mean()
        df["sma_long"] = df["Close"].rolling(self.long_window).mean()
        crossover_up = (df["sma_short"] > df["sma_long"]) & (
            df["sma_short"].shift(1) <= df["sma_long"].shift(1)
        )
        crossover_down = (df["sma_short"] < df["sma_long"]) & (
            df["sma_short"].shift(1) >= df["sma_long"].shift(1)
        )
        return self._build_signal_frame(ticker, df, crossover_up.fillna(False), crossover_down.fillna(False))


@dataclass(slots=True)
class PullbackUptrendStrategy(BaseStrategy):
    fast_window: int = 20
    slow_window: int = 50
    pullback_pct: float = 0.03
    pullback_window: int = 5

    def __post_init__(self) -> None:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be less than slow_window")
        if not (0 < self.pullback_pct < 0.2):
            raise ValueError("pullback_pct should be between 0 and 0.2")

    def generate_signals(self, ticker: str, data: pd.DataFrame) -> SignalFrame:
        df = data.copy()
        df["sma_fast"] = df["Close"].rolling(self.fast_window).mean()
        df["sma_slow"] = df["Close"].rolling(self.slow_window).mean()
        df["pullback_low"] = df["Low"].rolling(self.pullback_window).min()

        uptrend = df["sma_fast"] > df["sma_slow"]
        pullback_trigger = df["pullback_low"] < df["sma_fast"] * (1 - self.pullback_pct)
        cross_above_fast = (df["Close"] > df["sma_fast"]) & (df["Close"].shift(1) <= df["sma_fast"].shift(1))
        entries = uptrend & pullback_trigger.shift(1, fill_value=False) & cross_above_fast
        exits = df["Close"] < df["sma_slow"]
        return self._build_signal_frame(ticker, df, entries.fillna(False), exits.fillna(False))


@dataclass(slots=True)
class DonchianBreakoutStrategy(BaseStrategy):
    breakout_lookback: int = 20
    exit_lookback: int = 10

    def __post_init__(self) -> None:
        if self.breakout_lookback <= 2:
            raise ValueError("breakout_lookback must be greater than 2")
        if self.exit_lookback <= 1:
            raise ValueError("exit_lookback must be greater than 1")

    def generate_signals(self, ticker: str, data: pd.DataFrame) -> SignalFrame:
        df = data.copy()
        df["upper"] = df["High"].rolling(self.breakout_lookback).max().shift(1)
        df["lower"] = df["Low"].rolling(self.exit_lookback).min().shift(1)
        entries = df["Close"] > df["upper"]
        exits = df["Close"] < df["lower"]
        return self._build_signal_frame(ticker, df, entries.fillna(False), exits.fillna(False))


@dataclass(slots=True)
class GapUpHighVolumeStrategy(BaseStrategy):
    gap_pct: float = 0.03
    volume_ratio: float = 1.5
    volume_window: int = 20

    def __post_init__(self) -> None:
        if self.gap_pct <= 0:
            raise ValueError("gap_pct must be positive")
        if self.volume_ratio <= 1:
            raise ValueError("volume_ratio must be greater than 1")

    def generate_signals(self, ticker: str, data: pd.DataFrame) -> SignalFrame:
        df = data.copy()
        df["prev_close"] = df["Close"].shift(1)
        df["gap"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
        df["avg_volume"] = df["Volume"].rolling(self.volume_window).mean()
        entries = (df["gap"] >= self.gap_pct) & (df["Volume"] >= df["avg_volume"] * self.volume_ratio)
        exits = df["Close"] < df["prev_close"]
        return self._build_signal_frame(ticker, df, entries.fillna(False), exits.fillna(False))


def build_default_strategies(config: Dict[str, dict] | None = None) -> Sequence[BaseStrategy]:
    """Create strategy instances using optional configuration overrides."""

    config = config or {}
    strategies: List[BaseStrategy] = [
        SMACrossoverStrategy(name="SMA Crossover", **config.get("sma_crossover", {})),
        PullbackUptrendStrategy(name="Pullback Uptrend", **config.get("pullback_uptrend", {})),
        DonchianBreakoutStrategy(name="Donchian Breakout", **config.get("donchian_breakout", {})),
        GapUpHighVolumeStrategy(name="Gap Up High Volume", **config.get("gap_up_high_volume", {})),
    ]
    return strategies
