from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from nba_lineup_model.orchestration.daily_work import (
    DailyWork, STAGES, atomic_json, digest, file_reference, read_object, work_spec,
)
from nba_lineup_model.orchestration.repair_plan import (
    make_impact_plan, require_approval, write_impact_plan,
)
from nba_lineup_model.season.observed_revisions import (
    publish_revision, read_revision, select_revision,
)


def spec_fixture(tmp_path):
    raw = tmp_path / "nba-input.json"
    raw.write_text('{"gameId":"0022600001"}')
    return work_spec(
        season="2026-27", game_date=date(2026, 11, 2),
        source_revisions={"0022600001": {"box": file_reference(raw)}},
        model_configuration={"version": "unchanged"}, parent_model="a"*64,
        edition_id="original", code_identity="tested-code",
        inputs={"box": file_reference(raw)}, expected_game_ids=["0022600001"],
    )


def callbacks(tmp_path, seen, fail=None):
    result = {}
    for stage in STAGES:
        def run(name=stage):
            seen.append(name)
            if name == fail:
                raise InterruptedError("injected worker death")
            output = tmp_path / f"{name}.json"
            atomic_json(output, {"stage": name}, immutable=True)
            return {"artifacts": [file_reference(output)], "result": {"stage": name}}
        result[stage] = run
    return result


@pytest.mark.parametrize("stage", STAGES)
def test_resume_each_interruption_without_repeating_verified_work(tmp_path, stage):
    spec = spec_fixture(tmp_path)
    work = DailyWork(tmp_path / "work", spec)
    seen = []
    with pytest.raises(InterruptedError):
        work.run(callbacks(tmp_path, seen, stage))
    first = list(seen)
    result = work.run(callbacks(tmp_path, seen))
    prefix = list(STAGES[:STAGES.index(stage)])
    assert result["reused_stages"] == prefix
    assert seen == first + list(STAGES[STAGES.index(stage):])
    before = list(seen)
    repeated = work.run(callbacks(tmp_path, seen))
    assert seen == before
    assert repeated["reused_stages"] == list(STAGES)
    assert (work.root / "report.html").exists()
    assert len(list((work.root / "failures").glob("*.json"))) == 1


def test_corrupt_checkpoint_is_visible_and_never_reused(tmp_path):
    work = DailyWork(tmp_path / "work", spec_fixture(tmp_path))
    seen = []
    work.run(callbacks(tmp_path, seen))
    (tmp_path / "model.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="checksum"):
        work.run(callbacks(tmp_path, seen))
    assert seen == list(STAGES)
    assert read_object(work.root / "status.json")["status"] == "failed"


@pytest.mark.parametrize("field,value", [
    ("parent_model","b"*64), ("edition_id","approved-replay"),
    ("code_identity","different-code"), ("model_configuration",{"version":"new"}),
    ("source_revisions",{"0022600001":{"box":"new-revision"}}),
])
def test_causal_input_changes_receive_new_identity(tmp_path, field, value):
    first = spec_fixture(tmp_path)
    changed = deepcopy(first["identity"])
    changed[field] = value
    assert digest(changed) != first["work_id"]


def test_mutated_source_cannot_be_silently_adopted(tmp_path):
    spec = spec_fixture(tmp_path)
    (tmp_path / "nba-input.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="checksum"):
        DailyWork(tmp_path / "work", spec)


def test_no_partial_slate_or_duplicate_games(tmp_path):
    kw = dict(season="2026-27", game_date=date(2026,11,2), source_revisions={},
              model_configuration={"v":1}, parent_model=None, edition_id="original",
              code_identity="code", inputs={}, expected_game_ids=["0022600001"])
    with pytest.raises(ValueError, match="exact"):
        work_spec(**kw)
    kw["expected_game_ids"] = ["a", "a"]
    with pytest.raises(ValueError, match="Duplicate"):
        work_spec(**kw)
    kw["expected_game_ids"] = []
    spec = work_spec(**kw)
    assert spec["identity"]["expected_game_ids"] == []


def test_competing_worker_cannot_execute_same_partition(tmp_path):
    work = DailyWork(tmp_path / "work", spec_fixture(tmp_path))
    seen = []
    with work.lock(), pytest.raises(RuntimeError, match="already running"):
        work.run(callbacks(tmp_path, seen))
    assert seen == []


def test_immutable_record_rejects_replacement(tmp_path):
    path = tmp_path / "record.json"
    atomic_json(path, {"a":1}, immutable=True)
    atomic_json(path, {"a":1}, immutable=True)
    with pytest.raises(ValueError, match="Immutable"):
        atomic_json(path, {"a":2}, immutable=True)
    assert read_object(path) == {"a":1}


def impact_fixture(tmp_path, classification="model_input", **extra):
    raw = tmp_path / "change.json"
    raw.write_text('{"corrected":true}')
    args = dict(season="2026-27", classification=classification,
                changed_inputs=[{"before":None,"after":file_reference(raw),"reason":"official correction"}],
                affected_games={"2026-11-02":["0022600001"]},
                affected_artifacts=["daily-data","current-player-state"],
                model_suffix=["2026-11-02","2026-11-03"] if classification=="model_input" else [],
                forecast_replacement=False, base_release_id="base",edition_id="original",
                explanation="Scope presented for explicit user review.")
    args.update(extra)
    return make_impact_plan(**args)


def test_plan_can_be_written_before_approval(tmp_path):
    plan = impact_fixture(tmp_path)
    path = write_impact_plan(tmp_path / "plans", plan)
    assert "not executed" in path.read_text()
    with pytest.raises(PermissionError):
        require_approval(plan, None, "causal_replay")


def test_approval_cannot_expand_scope_or_replace_forecasts(tmp_path):
    plan = impact_fixture(tmp_path)
    approval = {"plan_id":plan["plan_id"],"approved_by":"user",
                "approval_reference":"explicit user message", "actions":["causal_replay"]}
    require_approval(plan, approval, "causal_replay")
    with pytest.raises(PermissionError):
        require_approval(plan, approval, "select_historical_data")
    other = impact_fixture(tmp_path, model_suffix=["2026-11-02","2026-11-03","2026-11-04"])
    with pytest.raises(PermissionError):
        require_approval(other, approval, "causal_replay")
    approval["actions"].append("replace_forecast_edition")
    with pytest.raises(ValueError, match="preserves"):
        require_approval(plan, approval, "replace_forecast_edition")


@pytest.mark.parametrize("kind", ["presentation_only","grade_only"])
def test_independent_repair_cannot_trigger_causal_replay(tmp_path, kind):
    plan = impact_fixture(tmp_path, classification=kind)
    assert plan["identity"]["model_suffix"] == []
    with pytest.raises(ValueError, match="independent"):
        impact_fixture(tmp_path, classification=kind, model_suffix=["2026-11-02"])


def test_corrections_preserve_raw_captures_and_require_exact_approval(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text('{"gameId":"0022600001","score":99}')
    first = publish_revision(root=tmp_path / "revisions", feed="nba_box_score",
        game_id="0022600001", raw_path=raw, normalized={"score":99},
        source_url="https://cdn.nba.com/box.json", parser_identity="parser-v1")
    assert publish_revision(root=tmp_path / "revisions", feed="nba_box_score",
        game_id="0022600001", raw_path=raw, normalized={"score":99},
        source_url="https://cdn.nba.com/box.json", parser_identity="parser-v1") == first
    corrected = tmp_path / "corrected.json"
    corrected.write_text('{"gameId":"0022600001","score":100}')
    second = publish_revision(root=tmp_path / "revisions", feed="nba_box_score",
        game_id="0022600001", raw_path=corrected, normalized={"score":100},
        source_url="https://cdn.nba.com/box.json", parser_identity="parser-v1",
        supersedes=first, reason="official scoring correction")
    with pytest.raises(PermissionError):
        select_revision(root=tmp_path/"selections",revision_reference=second,previous_reference=first)
    plan = impact_fixture(tmp_path,changed_inputs=[{"before":first,"after":second,"reason":"score correction"}])
    approval={"plan_id":plan["plan_id"],"approval_reference":"user decision",
              "approved_by":"user","actions":["select_historical_data"]}
    selected=select_revision(root=tmp_path/"selections",revision_reference=second,
                            previous_reference=first,impact_plan=plan,approval=approval)
    assert Path(selected["path"]).exists()
    assert read_revision(first)["identity"]["normalized"]["score"] == 99
    assert read_revision(second)["identity"]["normalized"]["score"] == 100
    assert raw.read_text() == '{"gameId":"0022600001","score":99}'


def test_correction_cannot_masquerade_as_first_capture(tmp_path):
    raw=tmp_path/"raw.json"; raw.write_text("{}")
    kwargs=dict(root=tmp_path/"revisions",feed="nba_box_score",game_id="0022600001",
                raw_path=raw,normalized={},source_url="https://cdn.nba.com/a",parser_identity="v1")
    first=publish_revision(**kwargs)
    second=publish_revision(**kwargs,supersedes=first,reason="reparse")
    with pytest.raises(PermissionError):
        select_revision(root=tmp_path/"selections",revision_reference=second,previous_reference=None)


def test_only_official_nba_basketball_revisions(tmp_path):
    raw=tmp_path/"raw.json";raw.write_text("{}")
    with pytest.raises(ValueError,match="official NBA"):
        publish_revision(root=tmp_path/"revisions",feed="nba_box_score",game_id="0022600001",
            raw_path=raw,normalized={},source_url="https://example.com/box",parser_identity="v1")
