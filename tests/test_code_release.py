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
