from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nba_lineup_model.season import production_site_release as subject
from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256


def _fixture(tmp_path: Path, monkeypatch):
    serving_root = tmp_path / "serving"
    lineups_root = tmp_path / "lineups"
    schedule_root = tmp_path / "schedule"
    site_release_root = tmp_path / "site-releases"
    code_root = tmp_path / "code"
    web_root = tmp_path / "web"
    for root in (
        serving_root,
        lineups_root,
        schedule_root,
        site_release_root,
        code_root,
        web_root,
    ):
        root.mkdir()
    serving_bundle = serving_root / "serving-bundle"
    lineup_bundle = lineups_root / "lineup-bundle"
    serving_bundle.mkdir()
    lineup_bundle.mkdir()
    (serving_bundle / "payload").write_text("serving")
    (lineup_bundle / "payload").write_text("lineups")
    (serving_root / "latest.json").write_text("{}")
    (lineups_root / "latest.json").write_text(
        json.dumps({"bundle_path": "lineup-bundle"})
    )
    (schedule_root / "final").mkdir()
    (schedule_root / "final" / "game.json").write_text("schedule")
    catalog = tmp_path / "games.parquet"
    rankings = tmp_path / "rankings.parquet"
    profiles = tmp_path / "profiles.parquet"
    catalog.write_text("catalog")
    rankings.write_text("rankings")
    profiles.write_text("profiles")
    (web_root / "index.html").write_text("<html>release</html>")
    (code_root / "module.py").write_text("VERSION = 1\n")

    identity = {
        "schema_version": 1,
        "artifact_kind": "dev_site_release",
        "season": "2026-27",
        "evidence_through_date": "2026-10-31",
        "game_count": 88,
        "code_revision": "a" * 40,
        "serving": {"bundle_sha256": "s" * 64},
        "lineups": {"bundle_sha256": "l" * 64},
        "schedule": {"tree_sha256": subject._tree_sha256(schedule_root)},
        "catalog": {"sha256": file_sha256(catalog)},
        "preseason_catalog": {
            "rankings_sha256": file_sha256(rankings),
            "profiles_sha256": file_sha256(profiles),
        },
    }
    release_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
    dev_release = site_release_root / f"release_sha256={release_sha}"
    dev_release.mkdir()
    report = dev_release / "report.html"
    report.write_text("approved")
    dev_manifest = {
        "release_sha256": release_sha,
        "identity": identity,
        "files": {"report.html": file_sha256(report)},
    }
    (dev_release / "manifest.json").write_text(json.dumps(dev_manifest))
    (site_release_root / "latest.json").write_text(
        json.dumps(
            {
                "release_sha256": release_sha,
                "release_path": dev_release.relative_to(site_release_root).as_posix(),
            }
        )
    )

    monkeypatch.setattr(
        subject,
        "resolve_latest_bundle",
        lambda root: (
            Path(root) / "serving-bundle"
            if Path(root).name in {"serving", "current-season-serving"}
            else serving_bundle
        ),
    )
    monkeypatch.setattr(
        subject,
        "verify_serving_bundle",
        lambda path: {"bundle_sha256": "s" * 64},
    )
    monkeypatch.setattr(
        subject,
        "verify_current_season_lineup_bundle",
        lambda path: {"bundle_sha256": "l" * 64},
    )
    monkeypatch.setattr(subject, "_git_revision", lambda path: "a" * 40)
    return {
        "site_release_root": site_release_root,
        "serving_root": serving_root,
        "lineups_root": lineups_root,
        "schedule_root": schedule_root,
        "catalog": catalog,
        "rankings": rankings,
        "profiles": profiles,
        "code_root": code_root,
        "web_root": web_root,
        "release_sha": release_sha,
    }


def _stage(tmp_path: Path, fixture: dict[str, object]) -> Path:
    return subject.stage_production_site_release(
        dev_site_release_root=fixture["site_release_root"],
        serving_root=fixture["serving_root"],
        lineups_root=fixture["lineups_root"],
        schedule_snapshot_root=fixture["schedule_root"],
        schedule_overlay_path=tmp_path / "schedule-overlay",
        catalog_path=fixture["catalog"],
        preseason_rankings_path=fixture["rankings"],
        preseason_profiles_path=fixture["profiles"],
        code_root=fixture["code_root"],
        frontend_root=fixture["web_root"],
        output_root=tmp_path / "production-releases",
        live_pointer=tmp_path / "live-release",
    )


def test_stage_and_switch_one_verified_release(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    candidate = _stage(tmp_path, fixture)

    manifest = subject.verify_production_site_release(candidate)
    assert manifest["identity"]["approved_release_sha256"] == fixture["release_sha"]
    assert (candidate / "schedule").is_symlink()
    assert (candidate / "site-release.env").is_file()

    previous = subject.switch_production_release(
        candidate,
        live_pointer=tmp_path / "live-release",
    )
    assert previous is None
    assert (tmp_path / "live-release").resolve() == candidate


def test_missing_or_stale_candidate_cannot_move_live_pointer(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    old = _stage(tmp_path, fixture)
    live = tmp_path / "live-release"
    subject.switch_production_release(old, live_pointer=live)
    (old / "catalog" / "games.parquet").write_text("changed")

    with pytest.raises(ValueError, match="Game catalog"):
        subject.switch_production_release(old, live_pointer=live)
    assert live.resolve() == old


def test_failed_health_check_restores_previous_release(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    old = _stage(tmp_path, fixture)
    new = tmp_path / "production-releases" / "release_sha256=new"
    new.mkdir()
    live = tmp_path / "live-release"
    subject.switch_production_release(old, live_pointer=live)

    monkeypatch.setattr(subject, "verify_production_site_release", lambda path: {})

    def fail() -> None:
        raise RuntimeError("simulated reload failure")

    with pytest.raises(RuntimeError, match="simulated reload failure"):
        subject.promote_with_rollback(new, live_pointer=live, health_check=fail)
    assert live.resolve() == old
