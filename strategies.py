"""Trading strategy signal generators.

This module provides several signal-generating functions that operate on
price/volume DataFrames. Each function returns a boolean ``Series`` aligned
with the input DataFrame index, indicating entry opportunities for the
respective strategy.

All functions validate the required columns and handle missing values by
forward-filling rolling calculations. Missing price/volume data will result in
``False`` signals for those rows.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def _validate_dataframe(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Ensure the DataFrame contains the required columns.

    Parameters
    ----------
    df:
        Input OHLCV DataFrame.
    required:
        Iterable of required column names.

    Raises
    ------
    ValueError
        If any required column is missing.
    """

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "DataFrame is missing required columns: " + ", ".join(sorted(missing))
        )


def _safe_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Return a simple moving average while preserving the Series index."""

    return series.rolling(window=window, min_periods=window).mean()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    The implementation uses the classic Wilder smoothing method.
    """

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def sma_cross(df: pd.DataFrame) -> pd.Series:
    """Return entry signals where the 10-period SMA crosses above the 50-period SMA."""

    _validate_dataframe(df, REQUIRED_COLUMNS)
    close = df["Close"].astype(float)

    sma_fast = _safe_rolling_mean(close, 10)
    sma_slow = _safe_rolling_mean(close, 50)

    cross_up = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    cross_up = cross_up & sma_fast.notna() & sma_slow.notna()
    return cross_up.fillna(False)


def pullback_uptrend(df: pd.DataFrame) -> pd.Series:
    """Return entry signals for pullbacks within an uptrend.

    A signal is generated when the close is above the 50-period SMA and the
    14-period RSI is below 40 (oversold in an uptrend scenario).
    """

    _validate_dataframe(df, REQUIRED_COLUMNS)
    close = df["Close"].astype(float)

    sma_50 = _safe_rolling_mean(close, 50)
    rsi_14 = _compute_rsi(close, 14)

    signal = (close > sma_50) & (rsi_14 < 40)
    return signal.fillna(False)


def donchian_breakout(df: pd.DataFrame) -> pd.Series:
    """Return entry signals when price breaks above the prior 20-day high."""

    _validate_dataframe(df, REQUIRED_COLUMNS)
    high = df["High"].astype(float)
    close = df["Close"].astype(float)

    prior_high = high.rolling(window=20, min_periods=20).max().shift(1)
    signal = close > prior_high
    return signal.fillna(False)


def gapup_highvol(df: pd.DataFrame) -> pd.Series:
    """Return entry signals for gap-up openings with elevated volume."""

    _validate_dataframe(df, REQUIRED_COLUMNS)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    volume = df["Volume"].astype(float)

    prev_high = high.shift(1)
    prev_volume = volume.shift(1)

    gap_condition = open_ > (prev_high * 1.01)
    volume_condition = volume > (prev_volume * 1.5)

    signal = gap_condition & volume_condition
    return signal.fillna(False)


__all__ = [
    "sma_cross",
    "pullback_uptrend",
    "donchian_breakout",
    "gapup_highvol",
]
