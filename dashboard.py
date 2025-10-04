"""Flask dashboard for visualising ASX trading results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template
from plotly.utils import PlotlyJSONEncoder

from alerts import AlertsManager
from data_fetcher import load_config

app = Flask(__name__)
REPORTS_DIR = Path("reports")


def _load_strategy_summary() -> pd.DataFrame:
    path = REPORTS_DIR / "strategy_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Strategy", "Trades", "Wins", "Losses", "WinRate", "NetPnL", "AverageReturnPct", "TotalReturnPct", "ExposureDays"])
    return pd.read_csv(path)


def _load_trades() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for file in REPORTS_DIR.glob("trades_*.csv"):
        frame = pd.read_csv(file, parse_dates=["entry_date", "exit_date"])
        frame["source_file"] = file.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[
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
            "source_file",
        ])
    return pd.concat(frames, ignore_index=True)


def _build_equity_curve(trades: pd.DataFrame, starting_capital: float) -> str | None:
    if trades.empty:
        return None
    trades = trades.sort_values("exit_date")
    trades["cumulative_pnl"] = trades["pnl"].cumsum()
    trades["equity"] = starting_capital + trades["cumulative_pnl"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trades["exit_date"],
            y=trades["equity"],
            mode="lines+markers",
            name="Equity",
            line=dict(color="#0d6efd"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Equity Curve",
        xaxis_title="Exit Date",
        yaxis_title="Equity",
        margin=dict(l=30, r=10, t=40, b=30),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _build_strategy_bar_chart(summary: pd.DataFrame) -> str | None:
    if summary.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["Strategy"],
            y=summary["NetPnL"],
            marker_color="#198754",
            name="Net PnL",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Net PnL by Strategy",
        xaxis_title="Strategy",
        yaxis_title="Net PnL",
        margin=dict(l=30, r=10, t=40, b=30),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


@app.route("/")
def index() -> str:
    config = load_config("config.json")
    summary = _load_strategy_summary()
    trades = _load_trades()
    alerts_manager = AlertsManager()
    alerts_df = alerts_manager.fetch_recent()

    equity_chart = _build_equity_curve(trades, starting_capital=float(config.get("capital", 0)))
    strategy_chart = _build_strategy_bar_chart(summary)

    return render_template(
        "index.html",
        summary=summary.to_dict("records"),
        trades_count=len(trades),
        alerts_count=len(alerts_df),
        alerts=alerts_df.to_dict("records"),
        equity_chart=equity_chart,
        strategy_chart=strategy_chart,
    )


@app.route("/signals")
def signals() -> str:
    alerts_manager = AlertsManager()
    alerts = alerts_manager.fetch_recent()
    return render_template("signals.html", alerts=alerts.to_dict("records"), has_alerts=not alerts.empty)


@app.route("/trades/<ticker>")
def trades_view(ticker: str) -> str:
    ticker = ticker.upper()
    trades = _load_trades()
    filtered = trades[trades["ticker"].str.upper() == ticker]
    if filtered.empty:
        return render_template("trades.html", ticker=ticker, has_trades=False, trades=[])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=filtered["exit_date"],
            y=filtered["pnl"],
            marker_color=["#198754" if pnl >= 0 else "#dc3545" for pnl in filtered["pnl"]],
            name="Trade PnL",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Trade PnL for {ticker}",
        xaxis_title="Exit Date",
        yaxis_title="PnL",
        margin=dict(l=30, r=10, t=40, b=30),
    )
    chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return render_template(
        "trades.html",
        ticker=ticker,
        has_trades=True,
        trades=filtered.to_dict("records"),
        chart_json=chart_json,
    )


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    app.run(debug=True)
