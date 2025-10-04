"""Backtesting utilities for fixed take-profit strategies."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
TRADE_CAPITAL = 10_000.0
DATA_DIR = Path(__file__).resolve().parent / "data"


def _validate_dataframe(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "DataFrame is missing required columns: " + ", ".join(sorted(missing))
        )


def _ensure_boolean_series(entries: pd.Series, index: pd.Index) -> pd.Series:
    if not isinstance(entries, pd.Series):
        raise TypeError("entries must be a pandas Series")
    if len(entries) != len(index):
        raise ValueError("entries Series must align with the DataFrame index")
    return entries.reindex(index).fillna(False).astype(bool)


def _calculate_fee(amount: float) -> float:
    if amount <= 0:
        return 0.0
    if amount <= 1_000:
        return 5.0
    if amount <= 3_000:
        return 10.0
    if amount <= 10_000:
        return 19.95
    if amount <= 25_000:
        return 29.95
    return amount * 0.0012


def _sanitize_name(name: str) -> str:
    safe_chars = [c if c.isalnum() or c in {"-", "_"} else "_" for c in name.strip()]
    sanitized = "".join(safe_chars) or "strategy"
    return sanitized


def run_tp_backtest(
    df: pd.DataFrame, entries: pd.Series, tp_pct: float, name: str = "strategy"
) -> Tuple[pd.DataFrame, dict]:
    """Run a take-profit backtest.

    Parameters
    ----------
    df:
        OHLCV DataFrame sorted in chronological order.
    entries:
        Boolean Series with entry signals aligned to ``df``.
    tp_pct:
        Target take-profit percentage expressed as a decimal (e.g. ``0.05`` for 5%).
    name:
        Strategy name used for output files.
    """

    if tp_pct <= 0:
        raise ValueError("tp_pct must be positive")

    _validate_dataframe(df, REQUIRED_COLUMNS)
    entries = _ensure_boolean_series(entries, df.index)

    open_prices = df["Open"].astype(float)
    high_prices = df["High"].astype(float)
    close_prices = df["Close"].astype(float)

    trade_records: List[dict] = []

    for idx, signal in enumerate(entries[:-1]):
        if not signal:
            continue

        entry_idx = idx + 1
        entry_date = df.index[entry_idx]
        entry_price = float(open_prices.iat[entry_idx])

        if entry_price <= 0:
            continue

        shares = int(TRADE_CAPITAL // entry_price)
        if shares <= 0:
            continue

        target_price = entry_price * (1 + tp_pct)

        exit_idx = None
        hit_target = False
        for search_idx in range(entry_idx, len(df)):
            if high_prices.iat[search_idx] >= target_price:
                exit_idx = search_idx
                hit_target = True
                break

        if exit_idx is None:
            exit_idx = len(df) - 1

        exit_date = df.index[exit_idx]
        exit_price = target_price if hit_target else float(close_prices.iat[exit_idx])

        entry_value = entry_price * shares
        exit_value = exit_price * shares

        entry_fee = _calculate_fee(entry_value)
        exit_fee = _calculate_fee(exit_value)
        total_fees = entry_fee + exit_fee

        gross_pnl = (exit_price - entry_price) * shares
        net_pnl = gross_pnl - total_fees
        capital_used = entry_price * shares if entry_price > 0 else TRADE_CAPITAL
        return_pct = net_pnl / capital_used if capital_used else 0.0

        trade_records.append(
            {
                "EntryDate": entry_date,
                "SignalDate": df.index[idx],
                "EntryPrice": entry_price,
                "Shares": shares,
                "TargetPrice": target_price,
                "ExitDate": exit_date,
                "ExitPrice": exit_price,
                "HitTarget": hit_target,
                "EntryFee": entry_fee,
                "ExitFee": exit_fee,
                "TotalFees": total_fees,
                "GrossPnL": gross_pnl,
                "NetPnL": net_pnl,
                "ReturnPct": return_pct,
            }
        )

    trades_df = pd.DataFrame(trade_records)

    if not trades_df.empty:
        hit_rate = trades_df["HitTarget"].mean()
        total_pnl = trades_df["NetPnL"].sum()
        cumulative_return = trades_df["NetPnL"].sum() / (TRADE_CAPITAL * len(trades_df))
    else:
        hit_rate = 0.0
        total_pnl = 0.0
        cumulative_return = 0.0

    summary = {
        "Strategy": name,
        "Trades": int(len(trades_df)),
        "HitRate": float(hit_rate),
        "TotalPnL": float(total_pnl),
        "CumReturn": float(cumulative_return),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sanitized_name = _sanitize_name(name)
    csv_path = DATA_DIR / f"{sanitized_name}_trades.csv"
    trades_df.to_csv(csv_path, index=False)

    return trades_df, summary


__all__ = ["run_tp_backtest"]
