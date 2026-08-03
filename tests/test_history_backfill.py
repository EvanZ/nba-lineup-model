from __future__ import annotations

import json
from pathlib import Path

import pytest

import nba_lineup_model.flows.backfill_history as backfill


def test_historical_backfill_runs_seasons_and_stages_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, str]] = []

    def execute(season: str, stage: str, _context: dict[str, object]):
        calls.append((season, stage))
        return {"season": season, "stage": stage}

    monkeypatch.setattr(backfill, "_execute_stage", execute)

    manifest, path = backfill.run_historical_backfill(
        ("2020-21", "2021-22"),
        from_stage="bios",
        through_stage="fetch",
        run_manifest_dir=tmp_path,
        run_id="history-test",
    )

    assert calls == [
        ("2020-21", "bios"),
        ("2020-21", "fetch"),
        ("2021-22", "bios"),
        ("2021-22", "fetch"),
    ]
    assert [record.status for record in manifest.records] == ["completed"] * 4
    assert path == tmp_path / "history-test.json"
    persisted = json.loads(path.read_text())
    assert len(persisted["records"]) == 4


def test_historical_backfill_checkpoints_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def execute(_season: str, stage: str, _context: dict[str, object]):
        if stage == "fetch":
            raise RuntimeError("fetch incomplete")
        return {}

    monkeypatch.setattr(backfill, "_execute_stage", execute)

    with pytest.raises(RuntimeError, match="fetch incomplete"):
        backfill.run_historical_backfill(
            ("2020-21",),
            from_stage="bios",
            through_stage="fetch",
            run_manifest_dir=tmp_path,
            run_id="history-failure",
        )

    persisted = json.loads((tmp_path / "history-failure.json").read_text())
    assert [record["status"] for record in persisted["records"]] == [
        "completed",
        "failed",
    ]
    assert persisted["records"][-1]["error_type"] == "RuntimeError"

    resumed_calls: list[str] = []

    def succeed(_season: str, stage: str, _context: dict[str, object]):
        resumed_calls.append(stage)
        return {}

    monkeypatch.setattr(backfill, "_execute_stage", succeed)
    manifest, _ = backfill.run_historical_backfill(
        ("2020-21",),
        from_stage="bios",
        through_stage="fetch",
        run_manifest_dir=tmp_path,
        run_id="history-failure",
    )

    assert resumed_calls == ["fetch"]
    assert [record.status for record in manifest.records] == [
        "completed",
        "completed",
    ]


def test_historical_backfill_passes_explicit_subset_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    received: list[bool] = []

    def execute(_season: str, _stage: str, context: dict[str, object]):
        received.append(bool(context["quality_eligible_only"]))
        return {}

    monkeypatch.setattr(backfill, "_execute_stage", execute)
    backfill.run_historical_backfill(
        ("2018-19",),
        from_stage="compact",
        through_stage="compact",
        quality_eligible_only=True,
        run_manifest_dir=tmp_path,
        run_id="history-subset",
    )

    assert received == [True]
