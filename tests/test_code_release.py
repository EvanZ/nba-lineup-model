import subprocess

import pytest

from nba_lineup_model.season import production_site_release as subject
from test_production_site_release import _fixture, _stage


def test_receipts_readable_even_with_restrictive_publisher_umask(tmp_path):
    import os
    from datetime import timedelta
    from test_forecast_publication import candidate, receipt, TIP
    from nba_lineup_model.season.forecast_publication import ForecastRegistry
    registry = ForecastRegistry(tmp_path)
    previous = os.umask(0o077)
    try:
        candidate(registry, "first")
        receipt(registry, "first", TIP - timedelta(minutes=5))
    finally:
        os.umask(previous)
    path = next((tmp_path / "receipts").glob("*.json"))
    assert path.parent.stat().st_mode & 0o777 == 0o755
    assert path.stat().st_mode & 0o777 == 0o644
    before = path.read_bytes()
    path.chmod(0o600)
    assert registry.ensure_receipts_readable() == 1
    assert path.stat().st_mode & 0o777 == 0o644
    assert path.read_bytes() == before
    assert registry.ensure_receipts_readable() == 0


def test_code_release_preserves_data_and_does_not_rebuild(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    base = _stage(tmp_path, fixture)
    (base / "catalog/games.parquet").chmod(0o600)
    original = subject.verify_production_site_release(base)
    source = fixture["code_root"]
    (source / "module.py").write_text("VERSION = 2\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fix"], check=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("A code release must not rebuild serving or forecasts")
    monkeypatch.setattr(subject, "stage_production_site_release", forbidden)
    updated = subject.stage_code_release(base_release=base, code_root=source,
                                        output_root=tmp_path / "code-releases")
    current = subject.verify_production_site_release(updated)
    assert (updated / "catalog/games.parquet").stat().st_mode & 0o777 == 0o644
    assert current["approved_manifest"] == original["approved_manifest"]
    changed = {k for k in current["identity"]
               if current["identity"][k] != original["identity"][k]}
    assert changed == {"code_identity", "code_revision"}
    assert subject.verify_production_site_release(base) == original
    assert (base / "code/module.py").read_text() == "VERSION = 1\n"
    assert (updated / "code/module.py").read_text() == "VERSION = 2\n"
    assert not (tmp_path / "live-release").exists()
    for name in ("schedule-snapshot", "current-season-serving", "current-season-lineups",
                 "catalog", "preseason", "web", "site-releases"):
        assert subject._tree_sha256(base / name) == subject._tree_sha256(updated / name)
    assert subject.stage_code_release(base_release=base, code_root=source,
                    output_root=tmp_path / "code-releases") == updated


def test_code_release_rejects_corrupt_base(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    base = _stage(tmp_path, fixture)
    (base / "catalog/games.parquet").write_text("corrupt")
    with pytest.raises(ValueError, match="Game catalog"):
        subject.stage_code_release(base_release=base, code_root=fixture["code_root"],
                                   output_root=tmp_path / "code-releases")
    assert not (tmp_path / "code-releases").exists()


@pytest.mark.parametrize("served_ids", [["game-1"], ["wrong-game"]])
def test_code_deploy_checks_exact_approved_game_ids(tmp_path, monkeypatch, served_ids):
    import importlib.util
    import json
    import sys
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace
    from test_dev_site_release import _api_validation

    spec = importlib.util.spec_from_file_location("code_deployment", Path(__file__).parents[1] / "scripts/deploy_gestalt_code.py")
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    root = tmp_path / "environment"
    candidate = tmp_path / "candidate"
    (candidate / "selected-schedule/latest").mkdir(parents=True)
    (candidate / "selected-schedule/forecast-selections.json").write_text("{}")
    (candidate / "selected-schedule/latest/2026-10-31.json").write_text(json.dumps({"games": [{"game_id": "game-1"}]}))
    config = SimpleNamespace(environment_root=root, live_pointer=root / "live-release",
        artifact_lake_uri=None, artifact_cache_root=None, validate=lambda: None)
    monkeypatch.setattr(deploy, "Path", lambda value: root if str(value) == "/var/lib/nba-gestalt-dev" else Path(value))
    monkeypatch.setattr(deploy.DailyPipelineConfig, "read", lambda _: config)
    monkeypatch.setattr(sys, "argv", ["deploy", "--config", "config.json", "--code", str(tmp_path)])
    monkeypatch.setattr(deploy, "read_active_activation", lambda _: {"release_path": str(tmp_path / "base"), "generation": 1, "runtime_sha256": "old"})
    monkeypatch.setattr(deploy, "stage_code_release", lambda **kwargs: candidate)
    approved = {"schedule": {"final_game_count": 1}}
    manifest = {"runtime_sha256": "new", "approved_manifest": {"identity": approved},
        "identity": {"code_revision": "revision", "evidence_through_date": "2026-10-31",
            "game_count": 1, "serving_bundle_sha256": "s" * 64, "lineup_bundle_sha256": "l" * 64}}
    monkeypatch.setattr(deploy, "verify_production_site_release", lambda _: manifest)
    monkeypatch.setattr(deploy, "candidate_api", lambda *args: nullcontext("http://candidate"))
    monkeypatch.setattr(deploy, "approved_model_root", lambda *args, **kwargs: tmp_path)
    def expected_ids(identity, *, release):
        assert identity is approved
        assert release == candidate
        return ["game-1"]
    monkeypatch.setattr(deploy, "approved_schedule_game_ids", expected_ids, raising=False)
    evidence = _api_validation()
    evidence["schedule_game_ids"] = served_ids
    monkeypatch.setattr(deploy, "collect_api_validation", lambda *args, **kwargs: evidence)
    monkeypatch.setattr(deploy, "fetch", lambda *args: {"games": [{"game_id": "game-1"}]})
    if served_ids == ["game-1"]:
        deploy.main()
    else:
        with pytest.raises(ValueError, match="schedule"):
            deploy.main()
    report = json.loads((root / "code-release-validation/new/report.json").read_text())
    assert report["status"] == ("validated" if served_ids == ["game-1"] else "failed")


@pytest.mark.parametrize("setting", ["", "PYTHONDONTWRITEBYTECODE=0", "PYTHONDONTWRITEBYTECODE=1"])
def test_code_deploy_requires_service_to_preserve_sealed_code(monkeypatch, setting):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("code_deployment", Path(__file__).parents[1] / "scripts/deploy_gestalt_code.py")
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    monkeypatch.setattr(deploy.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="PYTHONUNBUFFERED=1 " + setting))
    if setting == "PYTHONDONTWRITEBYTECODE=1":
        deploy.verify_dev_service_configuration()
    else:
        with pytest.raises(ValueError, match="preserve the sealed code package"):
            deploy.verify_dev_service_configuration()
