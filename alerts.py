"""Alert generation and persistence."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DB_PATH = Path("signals.db")


@dataclass(slots=True)
class Alert:
    ticker: str
    strategy: str
    signal_date: pd.Timestamp
    action: str
    price: float
    notes: str = ""


class AlertsManager:
    """Manage SQLite persistence for generated alerts."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, strategy, signal_date, action)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_alerts(self, alerts: Iterable[Alert]) -> int:
        alerts = list(alerts)
        if not alerts:
            return 0
        conn = self._connect()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO alerts (ticker, strategy, signal_date, action, price, notes)
                VALUES (:ticker, :strategy, :signal_date, :action, :price, :notes)
                """,
                [
                    {
                        "ticker": alert.ticker,
                        "strategy": alert.strategy,
                        "signal_date": alert.signal_date.strftime("%Y-%m-%d"),
                        "action": alert.action,
                        "price": alert.price,
                        "notes": alert.notes,
                    }
                    for alert in alerts
                ],
            )
            conn.commit()
            return len(alerts)
        finally:
            conn.close()

    def fetch_recent(self, limit: int = 50) -> pd.DataFrame:
        conn = self._connect()
        try:
            query = "SELECT ticker, strategy, signal_date, action, price, notes, created_at FROM alerts ORDER BY signal_date DESC, created_at DESC LIMIT ?"
            rows = conn.execute(query, (limit,)).fetchall()
        finally:
            conn.close()
        columns = ["ticker", "strategy", "signal_date", "action", "price", "notes", "created_at"]
        return pd.DataFrame(rows, columns=columns)

    def delete_older_than(self, days: int) -> int:
        conn = self._connect()
        try:
            cutoff = datetime.utcnow().timestamp() - days * 86400
            query = "DELETE FROM alerts WHERE strftime('%s', signal_date) < ?"
            cur = conn.execute(query, (cutoff,))
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


def generate_alerts(signals: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> list[Alert]:
    if signals.empty:
        return []
    frame = signals.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    as_of = (as_of or pd.Timestamp.utcnow().normalize())
    frame = frame[frame["Date"] == as_of]
    frame = frame[frame["Signal"].str.upper() == "BUY"]
    alerts: list[Alert] = []
    for _, row in frame.iterrows():
        alerts.append(
            Alert(
                ticker=row["Ticker"],
                strategy=row["Strategy"],
                signal_date=row["Date"],
                action=row["Signal"],
                price=float(row["Price"]),
                notes="Entry signal",
            )
        )
    return alerts
