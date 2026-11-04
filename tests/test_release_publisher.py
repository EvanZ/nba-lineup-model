from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from nba_lineup_model.season import release_publisher as subject


def _candidate(root: Path, name: str) -> Path:
    candidate = root / "candidates" / name
    candidate.mkdir(parents=True)
    (candidate / "payload.txt").write_text(name)
    return candidate


def test_sealed_code_ignores_non_runtime_node_modules_but_rejects_runtime_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "code"
    repository.mkdir()
    (repository / "src").mkdir()
    (repository / "src/module.py").write_text("VALUE = 1\n")
    (repository / "uv.lock").write_text("version = 1\n")
    import subprocess

    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "sealed",
        ],
        check=True,
    )
    node_modules = repository / "apps/app/node_modules"
    node_modules.mkdir(parents=True)
    (node_modules / "package.js").write_text("ignored by runtime identity\n")

    identity = subject._sealed_code_identity(repository)
    assert identity["git_revision"]

    (repository / "src/rogue.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="untracked runtime files"):
        subject._sealed_code_identity(repository)


@pytest.fixture
def publisher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    old = _candidate(tmp_path, "old")
    first = _candidate(tmp_path, "first")
    second = _candidate(tmp_path, "second")
    live = tmp_path / "live-release"
    live.symlink_to(old, target_is_directory=True)
    root = tmp_path / "publisher"

    def identity(release: Path) -> dict:
        release = Path(release).resolve()
        return {
            "release_path": release.as_posix(),
            "release_id": f"release-{release.name}",
            "runtime_sha256": f"runtime-{release.name}",
            "binding_map_sha256": f"bindings-{release.name}",
            "edition_id": "original",
            "code": {
                "git_revision": "a" * 40,
                "package_sha256": "b" * 64,
                "dependency_lock_sha256": "c" * 64,
            },
        }

    monkeypatch.setattr(subject, "_validated_release_identity", identity)
    monkeypatch.setattr(
        subject,
        "verify_production_site_release",
        lambda _release: {"identity": {}},
    )
    return old, first, second, live, root


def _activate(
    release: Path,
    live: Path,
    root: Path,
    operation_id: str,
    expected_generation: int,
    *,
    reload_api=lambda _record: None,
    verify_api=lambda _record: None,
    after_pointer_switch=None,
):
    return subject.activate_release(
        release=release,
        live_pointer=live,
        publisher_root=root,
        environment="dev",
        operation_id=operation_id,
        expected_previous_generation=expected_generation,
        reason="test activation",
        reload_api=reload_api,
        verify_api=verify_api,
        after_pointer_switch=after_pointer_switch,
    )


def test_activation_is_generation_fenced_journaled_and_idempotent(publisher):
    _old, first, _second, live, root = publisher

    result = _activate(first, live, root, "op-one", 0)

    assert result.status == "activated"
    assert result.generation == 1
    active = subject.read_active_activation(live)
    assert active is not None
    assert active["generation"] == 1
    assert active["release_path"] == first.as_posix()
    assert [event["state"] for event in subject._operation_events(root, "op-one")] == [
        "requested",
        "validated",
        "pointer_switched",
        "activated",
    ]

    retried = _activate(first, live, root, "op-one", 0)
    assert retried.status == "already_activated"
    assert retried.generation == 1

    with pytest.raises(ValueError, match="different activation inputs"):
        _activate(_second, live, root, "op-one", 0)


def test_unverified_active_release_cannot_become_a_rollback_target(
    publisher,
    monkeypatch,
):
    old, first, _second, live, root = publisher
    original_identity = subject._validated_release_identity

    def reject_old(release):
        if Path(release).resolve() == old:
            raise ValueError("rollback release changed")
        return original_identity(release)

    monkeypatch.setattr(subject, "_validated_release_identity", reject_old)
    with pytest.raises(ValueError, match="rollback release changed"):
        _activate(first, live, root, "unsafe", 0)

    assert live.resolve() == old
    assert subject._last_event(root, "unsafe", "pointer_switched") is None


def test_stale_publisher_cannot_overwrite_active_generation(publisher):
    _old, first, second, live, root = publisher
    _activate(first, live, root, "winner", 0)

    with pytest.raises(subject.StalePublisherError, match="expected generation 0"):
        _activate(second, live, root, "stale", 0)

    assert subject.read_active_activation(live)["operation_id"] == "winner"
    assert subject._last_event(root, "stale", "rejected_stale") is not None


def test_interrupted_activation_finishes_same_release_on_retry(publisher):
    _old, first, _second, live, root = publisher

    def interrupt(_record):
        raise SystemExit("simulated process death")

    with pytest.raises(SystemExit, match="simulated process death"):
        _activate(
            first,
            live,
            root,
            "interrupted",
            0,
            after_pointer_switch=interrupt,
        )

    assert subject.read_active_activation(live)["operation_id"] == "interrupted"
    recovered = _activate(first, live, root, "interrupted", 0)
    assert recovered.status == "recovered"
    assert recovered.generation == 1
    states = [
        event["state"]
        for event in subject._operation_events(root, "interrupted")
    ]
    assert states[-2:] == ["recovery_started", "activated"]


def test_completed_operation_retry_refuses_a_newer_generation(publisher):
    _old, first, second, live, root = publisher
    _activate(first, live, root, "completed", 0)
    _activate(second, live, root, "newer", 1)

    with pytest.raises(subject.ActivationSupersededError):
        _activate(first, live, root, "completed", 0)

    assert subject.read_active_activation(live)["operation_id"] == "newer"


def test_interrupted_activation_never_overwrites_newer_activation(publisher):
    _old, first, second, live, root = publisher

    with pytest.raises(SystemExit):
        _activate(
            first,
            live,
            root,
            "interrupted",
            0,
            after_pointer_switch=lambda _record: (_ for _ in ()).throw(SystemExit()),
        )
    _activate(second, live, root, "newer", 1)

    with pytest.raises(subject.ActivationSupersededError):
        _activate(first, live, root, "interrupted", 0)

    active = subject.read_active_activation(live)
    assert active["operation_id"] == "newer"
    assert active["generation"] == 2


def test_failed_verification_rolls_back_as_new_generation(publisher):
    old, first, _second, live, root = publisher
    reloaded: list[str] = []

    def reload_api(record):
        reloaded.append(record["release_path"])

    def verify_api(record):
        if record["release_path"] == first.as_posix():
            raise RuntimeError("new process has wrong identity")

    with pytest.raises(RuntimeError, match="wrong identity"):
        _activate(
            first,
            live,
            root,
            "bad-release",
            0,
            reload_api=reload_api,
            verify_api=verify_api,
        )

    active = subject.read_active_activation(live)
    assert active["generation"] == 2
    assert active["release_path"] == old.as_posix()
    assert reloaded == [first.as_posix(), old.as_posix()]
    assert subject._last_event(root, "bad-release", "rolled_back") is not None


def test_concurrent_publishers_have_one_winner_and_one_stale_writer(publisher):
    _old, first, second, live, root = publisher
    entered_verification = threading.Event()
    release_verification = threading.Event()
    results: list[tuple[str, object]] = []

    def first_verify(_record):
        entered_verification.set()
        assert release_verification.wait(timeout=5)

    def run(name, candidate, verify):
        try:
            value = _activate(
                candidate,
                live,
                root,
                name,
                0,
                verify_api=verify,
            )
            results.append((name, value))
        except Exception as error:
            results.append((name, error))

    first_thread = threading.Thread(
        target=run,
        args=("first-writer", first, first_verify),
    )
    second_thread = threading.Thread(
        target=run,
        args=("second-writer", second, lambda _record: None),
    )
    first_thread.start()
    assert entered_verification.wait(timeout=5)
    second_thread.start()
    time.sleep(0.1)
    assert second_thread.is_alive()
    release_verification.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    by_name = dict(results)
    assert by_name["first-writer"].status == "activated"
    assert isinstance(by_name["second-writer"], subject.StalePublisherError)
    assert subject.read_active_activation(live)["operation_id"] == "first-writer"


def test_production_activation_requires_explicit_approval(publisher):
    _old, first, _second, live, root = publisher

    with pytest.raises(ValueError, match="approval reference"):
        subject.activate_release(
            release=first,
            live_pointer=live,
            publisher_root=root,
            environment="prod",
            operation_id="prod-op",
            expected_previous_generation=0,
            reason="production release",
            reload_api=lambda _record: None,
            verify_api=lambda _record: None,
        )


def test_loaded_process_identity_is_the_activation_record(publisher):
    _old, first, _second, live, root = publisher
    _activate(first, live, root, "identity", 0)

    loaded = subject.load_process_activation_identity(live)

    assert loaded["operation_id"] == "identity"
    assert loaded["generation"] == 1
    assert loaded["release_id"] == "release-first"

