from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import submission_release as release
from release_validation import MarketDataReport


def fake_market_report() -> MarketDataReport:
    return MarketDataReport(
        row_count=180,
        stock_count=2,
        date_min="2026-01-01",
        date_max="2026-03-25",
        latest_day_stock_count=2,
        duplicate_key_count=0,
        invalid_ohlc_count=0,
        critical_missing_count=0,
        suspended_metric_missing_count=0,
        short_history_stock_count=0,
        stock_ids=("000001", "000002"),
        sha256="data-hash",
    )


def config(tmp_path: Path) -> release.ReleaseConfig:
    return release.ReleaseConfig(tmp_path, date(2026, 3, 25), tar_name="bundle.tar")


def test_data_failure_stops_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(release, "_require_clean_git", lambda workspace: None)
    monkeypatch.setattr(release, "stage_market_data", lambda *args: (_ for _ in ()).throw(ValueError("data gate")))
    called = {"local": False}

    def local_should_not_run(*args):
        called["local"] = True
        raise AssertionError("local training was called")

    monkeypatch.setattr(release, "run_local_candidate", local_should_not_run)
    with pytest.raises(ValueError, match="data gate"):
        release.run_release(config(tmp_path))
    assert called["local"] is False


def test_docker_failure_stops_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(release, "_require_clean_git", lambda workspace: None)
    monkeypatch.setattr(release, "stage_market_data", lambda *args: fake_market_report())
    monkeypatch.setattr(release, "run_local_candidate", lambda *args: {"backtest": {}})
    monkeypatch.setattr(release, "run_docker_candidate", lambda *args: (_ for _ in ()).throw(RuntimeError("docker gate")))
    called = {"publish": False}

    def publish_should_not_run(*args):
        called["publish"] = True
        raise AssertionError("publication was called")

    monkeypatch.setattr(release, "_publish", publish_should_not_run)
    with pytest.raises(RuntimeError, match="docker gate"):
        release.run_release(config(tmp_path))
    assert called["publish"] is False


def test_successful_fake_run_returns_published_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(release, "_require_clean_git", lambda workspace: None)
    monkeypatch.setattr(release, "stage_market_data", lambda *args: fake_market_report())
    monkeypatch.setattr(release, "run_local_candidate", lambda *args: {"backtest": {}})
    monkeypatch.setattr(
        release,
        "run_docker_candidate",
        lambda *args: {
            "result": {"stock_ids": ["000001"], "sha256": "result-hash"},
            "tar": {"sha256": "tar-hash"},
            "backtest": {},
        },
    )

    def fake_publish(config, run_dir, manifest, validation_doc):
        (config.workspace / "output").mkdir(exist_ok=True)
        (config.workspace / "output" / "result.csv").write_text("stock_id,weight\n000001,0.2\n", encoding="utf-8")
        (config.workspace / config.tar_name).write_bytes(b"tar")

    monkeypatch.setattr(release, "_publish", fake_publish)
    result = release.run_release(config(tmp_path))
    assert result.result_path.read_text(encoding="utf-8").startswith("stock_id,weight")
    assert result.tar_sha256
