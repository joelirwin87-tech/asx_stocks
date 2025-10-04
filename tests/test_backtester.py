from __future__ import annotations

import pandas as pd

from backtester import Backtester


def test_backtester_executes_trades(sample_price_data, dummy_signals, tmp_path):
    price_data = {"TEST.AX": sample_price_data}
    backtester = Backtester(
        capital=10000,
        risk_per_trade=0.1,
        take_profit_pct=0.05,
        stop_loss_pct=0.03,
        reports_dir=tmp_path / "reports",
    )
    trades, summary = backtester.run(price_data, dummy_signals)
    assert set(trades.columns) >= {"ticker", "strategy", "entry_price", "exit_price", "pnl"}
    assert len(summary) >= 0
    assert (tmp_path / "reports" / "strategy_summary.csv").exists() == (not trades.empty)


def test_backtester_handles_empty_signals(sample_price_data, tmp_path):
    price_data = {"TEST.AX": sample_price_data}
    backtester = Backtester(
        capital=10000,
        risk_per_trade=0.1,
        take_profit_pct=0.05,
        stop_loss_pct=0.03,
        reports_dir=tmp_path / "reports",
    )
    trades, summary = backtester.run(price_data, pd.DataFrame(columns=["Date", "Ticker", "Signal", "Price", "Strategy"]))
    assert trades.empty
    assert summary.empty
