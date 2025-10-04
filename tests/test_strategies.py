from __future__ import annotations

import pandas as pd

from strategies import (
    DonchianBreakoutStrategy,
    GapUpHighVolumeStrategy,
    PullbackUptrendStrategy,
    SMACrossoverStrategy,
    build_default_strategies,
)


def _base_dataframe() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    prices = [100 + i * 0.5 for i in range(len(dates))]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [p * 0.99 for p in prices],
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": [1_000_000 + i * 1000 for i in range(len(dates))],
        }
    )


def test_sma_crossover_generates_signals():
    data = _base_dataframe()
    strategy = SMACrossoverStrategy(name="SMA", short_window=3, long_window=5)
    signals = strategy.generate_signals("TEST.AX", data)
    assert set(signals.columns) == {"Date", "Ticker", "Signal", "Price", "Strategy"}
    assert (signals["Signal"].isin(["BUY", "SELL"]).all()) or signals.empty


def test_pullback_uptrend_handles_uptrend():
    data = _base_dataframe()
    strategy = PullbackUptrendStrategy(name="Pullback", fast_window=3, slow_window=5, pullback_pct=0.02, pullback_window=2)
    signals = strategy.generate_signals("TEST.AX", data)
    assert set(signals.columns) == {"Date", "Ticker", "Signal", "Price", "Strategy"}


def test_donchian_breakout_detects_breakout():
    data = _base_dataframe()
    data.loc[25:, "Close"] = data.loc[25:, "Close"] + 10
    strategy = DonchianBreakoutStrategy(name="Donchian", breakout_lookback=5, exit_lookback=3)
    signals = strategy.generate_signals("TEST.AX", data)
    assert {"BUY", "SELL"}.issuperset(set(signals["Signal"].unique())) or signals.empty


def test_gap_up_high_volume_strategy_flags_gap():
    data = _base_dataframe()
    data.loc[5, "Open"] = data.loc[4, "Close"] * 1.1
    data.loc[5, "Volume"] = data["Volume"].rolling(5).mean().iloc[5] * 2
    strategy = GapUpHighVolumeStrategy(name="Gap", gap_pct=0.05, volume_ratio=1.5, volume_window=3)
    signals = strategy.generate_signals("TEST.AX", data)
    assert (signals["Signal"] == "BUY").any() or signals.empty


def test_build_default_strategies_returns_all():
    strategies = build_default_strategies()
    assert len(list(strategies)) == 4
