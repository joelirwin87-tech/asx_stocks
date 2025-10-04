from __future__ import annotations

import json
from pathlib import Path

import alerts
import data_fetcher
import run_daily


def _prepare_pipeline_environment(monkeypatch, tmp_path):
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

    return reports_dir


def test_full_pipeline(monkeypatch, config_path, sample_price_data, tmp_path, mock_data_fetch):
    reports_dir = _prepare_pipeline_environment(monkeypatch, tmp_path)

    stats = run_daily.orchestrate(str(config_path))
    assert stats["fetched"] == 1
    assert stats["strategies"] == 4
    assert stats["alerts"] >= 0
    assert (reports_dir / "strategy_summary.csv").exists() or stats["trades"] == 0


def test_pipeline_defaults(monkeypatch, config_path, sample_price_data, tmp_path, mock_data_fetch):
    reports_dir = _prepare_pipeline_environment(monkeypatch, tmp_path)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg.pop("take_profit_pct")
    cfg.pop("stop_loss_pct")
    missing_config = tmp_path / "config_missing.json"
    missing_config.write_text(json.dumps(cfg), encoding="utf-8")

    stats = run_daily.orchestrate(str(missing_config))
    assert stats["fetched"] == 1
    assert stats["strategies"] == 4
    assert stats["alerts"] >= 0
    assert (reports_dir / "strategy_summary.csv").exists() or stats["trades"] == 0
