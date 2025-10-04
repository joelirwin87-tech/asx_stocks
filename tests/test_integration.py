from __future__ import annotations

from pathlib import Path

import alerts
import data_fetcher
import run_daily


def test_full_pipeline(monkeypatch, config_path, sample_price_data, tmp_path, mock_data_fetch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    db_path = tmp_path / "signals.db"

    def fake_fetcher_init(self, data_dir: str | Path = data_dir):  # type: ignore[override]
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    original_backtester_init = run_daily.Backtester.__init__

    def fake_backtester_init(self, *args, **kwargs):  # type: ignore[override]
        kwargs["reports_dir"] = reports_dir
        original_backtester_init(self, *args, **kwargs)

    original_alerts_init = alerts.AlertsManager.__init__

    def fake_alerts_init(self, db_path_param=db_path):  # type: ignore[override]
        original_alerts_init(self, db_path=db_path_param)

    monkeypatch.setattr(data_fetcher.DataFetcher, "__init__", fake_fetcher_init, raising=False)
    monkeypatch.setattr(run_daily.Backtester, "__init__", fake_backtester_init, raising=False)
    monkeypatch.setattr(alerts.AlertsManager, "__init__", fake_alerts_init, raising=False)

    stats = run_daily.orchestrate(str(config_path))
    assert stats["fetched"] == 1
    assert stats["strategies"] == 4
    assert stats["alerts"] >= 0
    assert (reports_dir / "strategy_summary.csv").exists() or stats["trades"] == 0
