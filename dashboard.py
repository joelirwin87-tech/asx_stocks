"""ASX Strategies dashboard application."""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, flash, render_template, url_for, current_app

from alerts import ensure_alerts_database

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
DEFAULT_DB_PATH = APP_ROOT / "db" / "signals.db"

logger = logging.getLogger(__name__)

_APP_INSTANCE: Optional[Flask] = None


def ensure_runtime_directories(data_dir: Path, db_path: Path) -> None:
    for directory in {data_dir, db_path.parent}:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - diagnostics
            logger.debug("Unable to create directory %s: %s", directory, exc)


def build_trade_navigation(data_dir: Path) -> List[Tuple[str, str]]:
    if not data_dir.exists():
        return []

    items: List[Tuple[str, str]] = []
    for csv_path in sorted(data_dir.glob("**/*.csv")):
        stem = csv_path.stem
        if "summary" in stem.lower():
            continue
        identifier = stem.upper()
        label = stem.replace("_", " ").replace("-", " ").upper()
        items.append((identifier, label))

    seen = set()
    unique_items: List[Tuple[str, str]] = []
    for identifier, label in items:
        if identifier in seen:
            continue
        seen.add(identifier)
        unique_items.append((identifier, label))

    return unique_items


def _find_equity_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "equity",
        "equity_curve",
        "cumulative_equity",
        "cum_return",
        "cumreturn",
        "cumulative_return",
        "portfolio_value",
        "balance",
        "nav",
    ]
    lower_columns = {col.lower().replace(" ", "_"): col for col in df.columns}
    for candidate in candidates:
        for column_key, original in lower_columns.items():
            if candidate in column_key:
                return original
    return None


def build_equity_curve_chart(summaries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    figure = go.Figure()
    traces_added = 0

    for summary in summaries:
        frame: pd.DataFrame = summary.get("dataframe", pd.DataFrame())
        if frame.empty:
            continue

        working = frame.copy()
        time_col = determine_time_column(working)
        equity_col = _find_equity_column(working)
        if not equity_col:
            continue

        y_values = pd.to_numeric(working[equity_col], errors="coerce").dropna()
        if y_values.empty:
            continue

        if time_col:
            x_values = working.loc[y_values.index, time_col].astype(str).tolist()
        else:
            x_values = list(range(1, len(y_values) + 1))

        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=summary.get("display_name", "Equity"),
                hovertemplate="%{y:.4f}<extra>%{fullData.name}</extra>",
            )
        )
        traces_added += 1

    if traces_added == 0:
        return None

    figure.update_layout(
        title="Equity Curve",
        template="plotly_white",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure.to_dict()


def _extract_trades_count(frame: pd.DataFrame) -> Optional[int]:
    if frame.empty:
        return None

    lower_columns = {col.lower().replace(" ", "_"): col for col in frame.columns}
    for key, column in lower_columns.items():
        if "trade" in key:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().any():
                return int(numeric.fillna(0).sum())

    numeric_rows = frame.select_dtypes(include="number")
    if not numeric_rows.empty:
        return int(len(frame))
    return None


def build_trades_per_strategy_chart(summaries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    bars: List[Tuple[str, int]] = []

    for summary in summaries:
        frame: pd.DataFrame = summary.get("dataframe", pd.DataFrame())
        count = _extract_trades_count(frame)
        if count is None:
            continue
        bars.append((summary.get("display_name", "Strategy"), count))

    if not bars:
        return None

    labels, values = zip(*bars)
    figure = go.Figure(
        data=[
            go.Bar(
                x=list(labels),
                y=list(values),
                marker_color="#0d6efd",
                hovertemplate="%{y} trades<extra>%{x}</extra>",
            )
        ]
    )
    figure.update_layout(
        title="Trades Per Strategy",
        template="plotly_white",
        margin=dict(l=30, r=10, t=40, b=30),
        xaxis_tickangle=-30,
    )
    return figure.to_dict()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "asx-dashboard-secret")
    data_directory = Path(os.environ.get("DATA_DIRECTORY", str(DATA_DIR)))
    db_path = Path(os.environ.get("SIGNALS_DB_PATH", str(DEFAULT_DB_PATH)))
    ensure_runtime_directories(data_directory, db_path)
    app.config["DATA_DIRECTORY"] = data_directory
    app.config["DB_PATH"] = db_path
    register_routes(app)
    return app


def get_app() -> Flask:
    global _APP_INSTANCE
    if _APP_INSTANCE is None:
        _APP_INSTANCE = create_app()
    return _APP_INSTANCE


def register_routes(app: Flask) -> None:
    """Attach context processors and routes to the provided Flask application."""

    if getattr(app, "_dashboard_routes_registered", False):
        return

    @app.context_processor
    def inject_last_refreshed() -> Dict[str, str]:
        links = []
        data_directory: Path = current_app.config.get("DATA_DIRECTORY", DATA_DIR)
        for identifier, label in build_trade_navigation(Path(data_directory)):
            try:
                link_url = url_for("trades", ticker=identifier)
            except Exception:  # pragma: no cover - url build defensive guard
                link_url = "#"
            links.append({"label": label, "url": link_url, "identifier": identifier})

        return {
            "last_refreshed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_nav_links": links,
        }

    @app.route("/")
    def index():
        data_directory: Path = current_app.config.get("DATA_DIRECTORY", DATA_DIR)
        raw_summaries = load_strategy_summaries(data_directory)
        equity_curve_chart = build_equity_curve_chart(raw_summaries)
        trades_chart = build_trades_per_strategy_chart(raw_summaries)
        equity_chart_id = f"equity-chart-{uuid.uuid4().hex}" if equity_curve_chart else None
        trades_chart_id = f"trades-chart-{uuid.uuid4().hex}" if trades_chart else None

        chart_configs = [
            {"id": summary["chart_id"], "config": summary["chart_json"]}
            for summary in raw_summaries
            if summary.get("chart_json")
        ]

        if equity_curve_chart and equity_chart_id:
            chart_configs.append({"id": equity_chart_id, "config": equity_curve_chart})
        if trades_chart and trades_chart_id:
            chart_configs.append({"id": trades_chart_id, "config": trades_chart})

        summaries: List[Dict[str, Any]] = []
        for summary in raw_summaries:
            trimmed = summary.copy()
            trimmed.pop("dataframe", None)
            summaries.append(trimmed)

        return render_template(
            "index.html",
            summaries=summaries,
            chart_configs=chart_configs,
            data_directory=data_directory,
            equity_curve_chart=equity_curve_chart,
            trades_chart=trades_chart,
            equity_chart_id=equity_chart_id,
            trades_chart_id=trades_chart_id,
        )

    @app.route("/signals")
    def signals():
        db_path: Path = current_app.config.get("DB_PATH", DEFAULT_DB_PATH)
        df = read_database(db_path)
        if df is None:
            flash("Signals database could not be read. Please verify the connection.")
            df = pd.DataFrame()
        today = datetime.now().date()
        today_signals = filter_signals_for_today(df, today)
        return render_template("signals.html", signals_df=today_signals)

    @app.route("/trades/<string:ticker>")
    def trades(ticker: str):
        ticker = ticker.upper()
        data_directory: Path = current_app.config.get("DATA_DIRECTORY", DATA_DIR)
        file_path = locate_trade_file(ticker, data_directory)
        if not file_path:
            flash(f"No trade history found for {ticker}.")
            empty_df = pd.DataFrame()
            return (
                render_template(
                    "trades.html",
                    ticker=ticker,
                    trades_df=empty_df,
                    chart_config=None,
                    data_source=None,
                    not_found=True,
                ),
                200,
            )

        df = read_csv_file(file_path)
        if df is None:
            flash("Unable to read trade data. Please verify the CSV contents.")
            return (
                render_template(
                    "trades.html",
                    ticker=ticker,
                    trades_df=pd.DataFrame(),
                    chart_config=None,
                    data_source=file_path,
                    not_found=True,
                ),
                200,
            )

        chart_config = build_plotly_config(df, f"{ticker} Trades")
        return render_template(
            "trades.html",
            ticker=ticker,
            trades_df=df,
            chart_config=chart_config,
            data_source=file_path,
            not_found=False,
        )

    app._dashboard_routes_registered = True  # type: ignore[attr-defined]


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
                hovertemplate=f"%{{y:.4f}}<extra>{col}</extra>",
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
            classes="table table-striped table-hover table-sm align-middle",
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
            "dataframe": df,
        }
        summaries.append(summary)
    return summaries


def read_database(db_path: Path) -> Optional[pd.DataFrame]:
    try:
        with ensure_alerts_database(db_path):
            pass
    except Exception:
        logger.error("Signals database could not be ensured at %s", db_path)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = get_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)


def __getattr__(name: str) -> Any:
    if name == "app":
        return get_app()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if __name__ == "__main__":
    main()
