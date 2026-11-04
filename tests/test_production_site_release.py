from __future__ import annotations

import hashlib
import json
import subprocess
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
    (lineups_root / "latest.json").write_text(json.dumps({"bundle_path": "lineup-bundle"}))
    (schedule_root / "final").mkdir()
    (schedule_root / "final" / "game.json").write_text('{"source":"NBA official CDN"}')
    catalog = tmp_path / "games.parquet"
    rankings = tmp_path / "rankings.parquet"
    profiles = tmp_path / "profiles.parquet"
    catalog.write_text("catalog")
    rankings.write_text("rankings")
    profiles.write_text("profiles")
    (web_root / "index.html").write_text("<html>release</html>")
    (code_root / "module.py").write_text("VERSION = 1\n")
    (code_root / "uv.lock").write_text("version = 1\n")
    subprocess.run(["git", "init", "-q", str(code_root)], check=True)
    subprocess.run(["git", "-C", str(code_root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(code_root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subject._git_revision(code_root)

    identity = {
        "schema_version": 1,
        "artifact_kind": "dev_site_release",
        "season": "2026-27",
        "evidence_through_date": "2026-10-31",
        "game_count": 88,
        "code_revision": revision,
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


def test_stage_one_verified_release_without_moving_pointer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    candidate = _stage(tmp_path, fixture)

    manifest = subject.verify_production_site_release(candidate)
    assert manifest["identity"]["approved_release_sha256"] == fixture["release_sha"]
    assert (candidate / "schedule").is_symlink()
    assert (candidate / "site-release.env").is_file()
    assert (candidate / "code").is_dir()
    assert not (candidate / "code").is_symlink()
    assert manifest["identity"]["code_identity"]["sealed"] is True
    assert not (tmp_path / "live-release").exists()


def test_stale_candidate_fails_verification_without_a_pointer_mutator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    candidate = _stage(tmp_path, fixture)
    (candidate / "catalog" / "games.parquet").write_text("changed")

    with pytest.raises(ValueError, match="Game catalog"):
        subject.verify_production_site_release(candidate)
    assert not hasattr(subject, "switch_production_release")
    assert not hasattr(subject, "promote_with_rollback")


def test_release_cli_does_not_expose_a_switch_command() -> None:
    parser = subject._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["switch", "/tmp/release", "--live-pointer", "/tmp/live"])


@pytest.mark.parametrize("edit", ["working_file", "new_commit"])
def test_checkout_changes_cannot_change_staged_release(tmp_path, monkeypatch, edit):
    fixture = _fixture(tmp_path, monkeypatch)
    candidate = _stage(tmp_path, fixture)
    before = subject.verify_production_site_release(candidate)
    (fixture["code_root"] / "module.py").write_text("VERSION = 2\n")
    if edit == "new_commit":
        subprocess.run(["git", "-C", str(fixture["code_root"]), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(fixture["code_root"]),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "next",
            ],
            check=True,
        )
    assert (candidate / "code" / "module.py").read_text() == "VERSION = 1\n"
    assert subject.verify_production_site_release(candidate) == before
    from nba_lineup_model.season.release_publisher import _validated_release_identity

    assert _validated_release_identity(candidate)["code"] == before["identity"]["code_identity"]


def test_dirty_checkout_cannot_be_staged(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    (fixture["code_root"] / "module.py").write_text("UNCOMMITTED = True\n")
    with pytest.raises(ValueError, match="modified tracked files"):
        _stage(tmp_path, fixture)
    assert not (tmp_path / "production-releases").exists()


@pytest.mark.parametrize(
    "damage", ["changed_file", "missing_seal", "code_symlink", "extra_file", "forged_seal"]
)
def test_publisher_rejects_damaged_code_before_pointer_change(tmp_path, monkeypatch, damage):
    from nba_lineup_model.season import release_publisher as publisher

    fixture = _fixture(tmp_path, monkeypatch)
    candidate = _stage(tmp_path, fixture)
    old = tmp_path / "existing-release"
    old.mkdir()
    live = tmp_path / "live-release"
    live.symlink_to(old, target_is_directory=True)
    if damage == "changed_file":
        path = candidate / "code" / "module.py"
        path.chmod(0o644)
        path.write_text("CHANGED = True\n")
    elif damage == "missing_seal":
        (candidate / "code-package.json").unlink()
    elif damage == "code_symlink":
        (candidate / "code").rename(candidate / "saved-code")
        (candidate / "code").symlink_to(fixture["code_root"], target_is_directory=True)
    elif damage == "extra_file":
        (candidate / "code").chmod(0o755)
        (candidate / "code" / "unexpected.py").write_text("UNTRACKED = True\n")
    else:
        path = candidate / "code-package.json"
        seal = json.loads(path.read_text())
        seal["identity"]["git_revision"] = "b" * 40
        path.write_text(json.dumps(seal))

    with pytest.raises((ValueError, FileNotFoundError)):
        publisher._validated_release_identity(candidate)
    with pytest.raises((ValueError, FileNotFoundError)):
        publisher.activate_release(
            release=candidate,
            live_pointer=live,
            publisher_root=tmp_path / "publisher",
            environment="dev",
            operation_id=f"invalid-{damage}",
            expected_previous_generation=0,
            reason="Regression test",
            reload_api=lambda _: pytest.fail("Must not reload"),
            verify_api=lambda _: pytest.fail("Must not call API"),
        )
    assert live.resolve() == old
    assert not (tmp_path / "publisher" / "generations").exists()


def test_repeated_staging_reuses_the_verified_package(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    first = _stage(tmp_path, fixture)
    assert _stage(tmp_path, fixture) == first


def test_failed_staging_cleans_temporary_read_only_code(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    original = subject.verify_production_site_release

    def reject_staging(path):
        if Path(path).name.startswith(".production-release.tmp-"):
            raise ValueError("Injected staging failure")
        return original(path)

    monkeypatch.setattr(subject, "verify_production_site_release", reject_staging)
    with pytest.raises(ValueError, match="Injected staging failure"):
        _stage(tmp_path, fixture)
    assert list((tmp_path / "production-releases").iterdir()) == []


def test_real_publisher_can_activate_after_source_checkout_advances(tmp_path, monkeypatch):
    from nba_lineup_model.season import release_publisher as publisher

    fixture = _fixture(tmp_path, monkeypatch)
    old = _stage(tmp_path / "old", fixture)
    candidate = _stage(tmp_path, fixture)
    live = tmp_path / "live-release"
    live.symlink_to(old, target_is_directory=True)
    (fixture["code_root"] / "module.py").write_text("VERSION = 99\n")
    reloads = []
    result = publisher.activate_release(
        release=candidate,
        live_pointer=live,
        publisher_root=tmp_path / "publisher",
        environment="dev",
        operation_id="sealed-code-activation",
        expected_previous_generation=0,
        reason="Real staging and publisher regression test",
        reload_api=lambda record: reloads.append(record["code"]),
        verify_api=lambda record: publisher._validated_release_identity(
            Path(record["release_path"])
        ),
    )
    assert result.status == "activated"
    active = publisher.read_active_activation(live)
    assert (
        active["code"]
        == subject.verify_production_site_release(candidate)["identity"]["code_identity"]
    )
    assert reloads == [active["code"]]
    assert (live / "code" / "module.py").read_text() == "VERSION = 1\n"
    publisher._verify_rollback_target(active)


def test_code_import_does_not_pollute_sealed_package(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    candidate = _stage(tmp_path, fixture)
    subprocess.run(
        ["python3", "-c", "import module; assert module.VERSION == 1"],
        cwd=candidate / "code",
        check=True,
    )
    assert not (candidate / "code" / "__pycache__").exists()
    subject.verify_production_site_release(candidate)


def test_source_symlink_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    code = fixture["code_root"]
    (code / "link.py").symlink_to(code / "module.py")
    subprocess.run(["git", "-C", str(code), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(code),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "symlink",
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="symlinks"):
        _stage(tmp_path, fixture)


def test_staged_release_does_not_require_original_checkout(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    candidate = _stage(tmp_path, fixture)
    before = subject.verify_production_site_release(candidate)
    fixture["code_root"].rename(tmp_path / "moved-development-checkout")
    assert subject.verify_production_site_release(candidate) == before


def test_generation_cannot_redirect_code_to_another_package(tmp_path, monkeypatch):
    from nba_lineup_model.season import release_publisher as publisher

    fixture = _fixture(tmp_path, monkeypatch)
    old = _stage(tmp_path / "old", fixture)
    candidate = _stage(tmp_path, fixture)
    live = tmp_path / "live-release"
    live.symlink_to(old, target_is_directory=True)
    result = publisher.activate_release(
        release=candidate,
        live_pointer=live,
        publisher_root=tmp_path / "publisher",
        environment="dev",
        operation_id="test-reference",
        expected_previous_generation=0,
        reason="Verify generation code target",
        reload_api=lambda _: None,
        verify_api=lambda _: None,
    )
    code_link = result.activation_root / "code"
    code_link.unlink()
    code_link.symlink_to(old / "code", target_is_directory=True)
    with pytest.raises(ValueError, match="differs from its sealed runtime"):
        subject.verify_production_site_release(result.activation_root)


def _legacy_fixture(tmp_path, monkeypatch):
    import shutil

    from nba_lineup_model.season import release_publisher as publisher

    fixture = _fixture(tmp_path, monkeypatch)
    old = _stage(tmp_path, fixture)
    candidate = tmp_path / "new-candidate"
    shutil.copytree(old, candidate, symlinks=True)
    root = tmp_path / "publisher"
    generation, active = publisher._stage_generation(
        publisher_root=root,
        environment="dev",
        generation=7,
        operation_id="old",
        release=old,
        previous=None,
        reason="old release",
        approval_reference=None,
    )
    live = tmp_path / "live-release"
    live.symlink_to(generation)
    # Reproduce a genuine pre-seal runtime, not a validator stub.
    for path in (old / "code").rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    (old / "code").chmod(0o755)
    shutil.rmtree(old / "code")
    (old / "code").symlink_to(fixture["code_root"])
    (old / "code-package.json").unlink()
    manifest = json.loads((old / "runtime-manifest.json").read_text())
    manifest["identity"].pop("code_identity")
    manifest["runtime_sha256"] = hashlib.sha256(canonical_json(manifest["identity"])).hexdigest()
    (old / "runtime-manifest.json").write_text(json.dumps(manifest))
    active["runtime_sha256"] = manifest["runtime_sha256"]
    (generation / "activation.json").write_text(json.dumps(active))
    return fixture, old, candidate, root, live, active


@pytest.mark.parametrize("mode", ["success", "rollback", "interrupted"])
def test_explicit_legacy_migration_preserves_data_and_recovers(tmp_path, monkeypatch, mode):
    from nba_lineup_model.season import release_publisher as publisher
    from nba_lineup_model.season.legacy_code_migration import (
        _inventory,
        prepare_legacy_code_migration,
    )

    fixture, old, candidate, root, live, active = _legacy_fixture(tmp_path, monkeypatch)
    original_manifest = (old / "runtime-manifest.json").read_bytes()
    original_inventory = _inventory(old)
    receipt = prepare_legacy_code_migration(
        active=active,
        repository=fixture["code_root"],
        output_root=tmp_path / "migrations",
        approval_reference="user approved dev-only code migration",
    )
    restored = Path(receipt["sealed_runtime"])
    assert live.resolve().name == "generation=00000000000000000007"
    assert subject.verify_production_site_release(restored)
    assert not (restored / "code").is_symlink()
    assert _inventory(restored) == original_inventory
    assert (
        prepare_legacy_code_migration(
            active=active,
            repository=fixture["code_root"],
            output_root=tmp_path / "migrations",
            approval_reference=receipt["approval_reference"],
        )
        == receipt
    )

    reloaded = []

    def verify(record):
        subject.verify_production_site_release(Path(record["release_path"]))
        if mode == "rollback" and record["generation"] == 8:
            raise ValueError("injected API validation failure")

    kwargs = dict(
        release=candidate,
        live_pointer=live,
        publisher_root=root,
        environment="dev",
        operation_id="migrated",
        expected_previous_generation=7,
        reason="daily release",
        reload_api=lambda record: reloaded.append(record["generation"]),
        verify_api=verify,
        legacy_code_migration=receipt,
    )
    if mode == "rollback":
        with pytest.raises(ValueError, match="injected API"):
            publisher.activate_release(**kwargs)
        current = publisher.read_active_activation(live)
        assert current["generation"] == 9
        assert current["release_path"] == str(restored)
        assert reloaded == [8, 9]
    elif mode == "interrupted":

        def crash(record):
            raise RuntimeError("injected interruption")

        with pytest.raises(RuntimeError, match="injected interruption"):
            publisher.activate_release(**kwargs, after_pointer_switch=crash)
        result = publisher.activate_release(**kwargs)
        assert result.status == "recovered"
        assert result.generation == 8
    else:
        result = publisher.activate_release(**kwargs)
        assert result.generation == 8
        assert publisher.activate_release(**kwargs).status == "already_activated"
    current = publisher.read_active_activation(live)
    assert current["rollback_migration"]["migration_sha256"] == receipt["migration_sha256"]
    assert (old / "runtime-manifest.json").read_bytes() == original_manifest
    assert (old / "code").is_symlink()
    assert _inventory(old) == original_inventory


@pytest.mark.parametrize(
    "damage", ["approval", "recorded_code", "data", "generation", "receipt", "extra_file"]
)
def test_legacy_migration_rejects_unsafe_inputs_without_pointer_changes(
    tmp_path, monkeypatch, damage
):
    from nba_lineup_model.season import release_publisher as publisher
    from nba_lineup_model.season.legacy_code_migration import prepare_legacy_code_migration

    fixture, old, candidate, root, live, active = _legacy_fixture(tmp_path, monkeypatch)
    before = live.resolve()
    approval = "approved"
    if damage == "approval":
        approval = ""
    elif damage == "recorded_code":
        active["code"]["package_sha256"] = "f" * 64
    elif damage == "data":
        (old / "schedule-snapshot/final/game.json").write_text("changed")
    if damage in {"approval", "recorded_code", "data"}:
        with pytest.raises(ValueError):
            prepare_legacy_code_migration(
                active=active,
                repository=fixture["code_root"],
                output_root=tmp_path / "migrations",
                approval_reference=approval,
            )
    else:
        receipt = prepare_legacy_code_migration(
            active=active,
            repository=fixture["code_root"],
            output_root=tmp_path / "migrations",
            approval_reference=approval,
        )
        if damage == "generation":
            active["generation"] = 6
            (before / "activation.json").write_text(json.dumps(active))
        elif damage == "receipt":
            receipt["approval_reference"] = "changed"
        else:
            (Path(receipt["sealed_runtime"]) / "extra.txt").write_text("untracked input")
        with pytest.raises(ValueError):
            publisher.activate_release(
                release=candidate,
                live_pointer=live,
                publisher_root=root,
                environment="dev",
                operation_id="bad",
                expected_previous_generation=7,
                reason="test",
                reload_api=lambda _: pytest.fail("must not reload"),
                verify_api=lambda _: None,
                legacy_code_migration=receipt,
            )
    assert live.resolve() == before
