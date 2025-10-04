"""ASX Strategies dashboard application."""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, abort, flash, render_template

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
DEFAULT_DB_PATH = APP_ROOT / "signals.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "asx-dashboard-secret")
    app.config["DATA_DIRECTORY"] = DATA_DIR
    app.config["DB_PATH"] = Path(os.environ.get("SIGNALS_DB_PATH", str(DEFAULT_DB_PATH)))
    return app


app = configure_app()


@app.context_processor
def inject_last_refreshed() -> Dict[str, str]:
    return {"last_refreshed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def humanise_number(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1_000_000:
        return f"{number/1_000_000:.2f}M"
    if abs(number) >= 1_000:
        return f"{number/1_000:.2f}K"
    if abs(number) >= 1:
        return f"{number:.2f}"
    return f"{number:.4f}"


def determine_time_column(df: pd.DataFrame) -> Optional[str]:
    time_candidates = ["date", "timestamp", "datetime", "time", "as_of", "created_at"]
    for candidate in time_candidates:
        for column in df.columns:
            if column.lower() == candidate:
                converted = pd.to_datetime(df[column], errors="coerce")
                if converted.notna().any():
                    df[column] = converted
                    return column
    return None


def build_plotly_config(df: pd.DataFrame, title: str) -> Optional[Dict[str, Any]]:
    if df.empty:
        return None
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return None

    x_col = determine_time_column(df)
    if x_col:
        x_values = df[x_col].astype(str).tolist()
    else:
        x_values = list(range(1, len(df) + 1))

    fig = go.Figure()
    max_series = 5
    for col in numeric_df.columns[:max_series]:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=numeric_df[col],
                mode="lines+markers",
                name=col,
                hovertemplate="%{y:.4f}<extra>{}</extra>".format(col),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=30, r=10, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig.to_dict()


def read_csv_file(file_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.warning("CSV file not found: %s", file_path)
        return None
    except pd.errors.EmptyDataError:
        logger.warning("CSV file is empty: %s", file_path)
        return pd.DataFrame()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Unable to read CSV %s: %s", file_path, exc)
        return None

    for column in df.columns:
        if df[column].dtype == object:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().any():
                df[column] = parsed
    return df


def snapshot_numeric_metrics(df: pd.DataFrame) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    if df.empty:
        return metrics

    metric_candidates = {
        "Total Return": ["total_return", "return", "returns"],
        "PnL": ["pnl", "pl", "profit", "net_profit"],
        "Win Rate": ["win_rate", "winrate"],
        "Sharpe": ["sharpe", "sharpe_ratio"],
        "Trades": ["trade_count", "trades", "num_trades"],
    }

    lower_columns = {col.lower(): col for col in df.columns}
    for label, aliases in metric_candidates.items():
        for alias in aliases:
            if alias in lower_columns:
                source_col = lower_columns[alias]
                value = df[source_col].iloc[-1]
                if pd.api.types.is_numeric_dtype(df[source_col]):
                    metrics[label] = humanise_number(value)
                else:
                    metrics[label] = str(value)
                break

    if "Trades" not in metrics:
        metrics["Trades"] = str(len(df))
    return metrics


def load_strategy_summaries(data_dir: Path) -> List[Dict[str, Any]]:
    if not data_dir.exists():
        logger.info("Data directory does not exist: %s", data_dir)
        return []

    summaries: List[Dict[str, Any]] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        df = read_csv_file(csv_path)
        if df is None:
            continue
        display_name = csv_path.stem.replace("_", " ").title()
        table_df = df.copy()
        table_html = table_df.to_html(
            classes="table table-hover table-sm align-middle",
            index=False,
            border=0,
            max_rows=50,
            justify="center",
        )
        summary = {
            "display_name": display_name,
            "source": csv_path,
            "table_html": table_html,
            "numeric_snapshot": snapshot_numeric_metrics(df),
            "chart_id": f"chart-{uuid.uuid4().hex}",
            "chart_json": build_plotly_config(df, f"{display_name} Metrics"),
        }
        summaries.append(summary)
    return summaries


def read_database(db_path: Path) -> Optional[pd.DataFrame]:
    if not db_path.exists():
        logger.warning("Signals database not found at %s", db_path)
        return None
    try:
        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            tables = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            if not tables:
                logger.warning("No tables found in database %s", db_path)
                return pd.DataFrame()
            table_name = "alerts"
            if (table_name,) not in tables:
                table_name = tables[0][0]
            rows = cursor.execute(f"SELECT * FROM {table_name}").fetchall()
    except sqlite3.Error as exc:
        logger.error("Database error reading %s: %s", db_path, exc)
        return None

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rows[0].keys())
    return df


def filter_signals_for_today(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    date_columns = []
    for column in df.columns:
        converted = pd.to_datetime(df[column], errors="coerce")
        if converted.notna().any():
            mask = converted.dt.date == target_date
            if mask.any():
                filtered = df.loc[mask]
                filtered[column] = converted.loc[mask]
                date_columns.append(column)
                break
    if filtered.empty:
        return df.head(0)
    for column in date_columns:
        filtered[column] = pd.to_datetime(filtered[column], errors="coerce")
    filtered = filtered.sort_values(by=date_columns or filtered.columns.tolist())
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def locate_trade_file(ticker: str, data_dir: Path) -> Optional[Path]:
    candidates = [
        data_dir / f"{ticker}.csv",
        data_dir / f"{ticker.lower()}.csv",
        data_dir / f"trades_{ticker}.csv",
        data_dir / f"trades_{ticker.lower()}.csv",
        data_dir / "trades" / f"{ticker}.csv",
        data_dir / "trades" / f"{ticker.lower()}.csv",
        data_dir / "trades" / f"trades_{ticker}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(data_dir.glob(f"**/*{ticker}*.csv"))
    return matches[0] if matches else None


@app.route("/")
def index():
    summaries = load_strategy_summaries(app.config["DATA_DIRECTORY"])
    chart_configs = [
        {"id": summary["chart_id"], "config": summary["chart_json"]} for summary in summaries
    ]
    return render_template(
        "index.html",
        summaries=summaries,
        chart_configs=chart_configs,
        data_directory=app.config["DATA_DIRECTORY"],
    )


@app.route("/signals")
def signals():
    df = read_database(app.config["DB_PATH"])
    if df is None:
        flash("Signals database could not be read. Please verify the connection.")
        df = pd.DataFrame()
    today = datetime.now().date()
    today_signals = filter_signals_for_today(df, today)
    return render_template("signals.html", signals_df=today_signals)


@app.route("/trades/<string:ticker>")
def trades(ticker: str):
    ticker = ticker.upper()
    file_path = locate_trade_file(ticker, app.config["DATA_DIRECTORY"])
    if not file_path:
        abort(404, description=f"No trade history found for ticker {ticker}.")

    df = read_csv_file(file_path)
    if df is None:
        abort(500, description="Unable to read trade data.")

    chart_config = build_plotly_config(df, f"{ticker} Trades")
    return render_template(
        "trades.html",
        ticker=ticker,
        trades_df=df,
        chart_config=chart_config,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
