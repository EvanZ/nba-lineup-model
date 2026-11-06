import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from nba_lineup_model.orchestration import forecast_publication_workflow as subject


WARNING = "retrospective_unavailable:Official minutes player 123 has no rating-before state"


def test_market_only_publication_retains_existing_missing_rating_warning():
    before = {"projection": {"home_margin": 2}, "market": None}
    after = {**before, "market": {"home_spread": -3}}
    assert subject.retained_market_warnings(before, after, [WARNING], [WARNING]) == [WARNING]


@pytest.mark.parametrize("case", ["new", "unrelated", "forecast", "unchanged", "polymarket"])
def test_market_warning_policy_rejects_other_changes(case):
    before = {"projection": {"home_margin": 2}, "market": None}
    after = {**before, "market": {"home_spread": -3}}
    existing, errors = [WARNING], [WARNING]
    if case == "new":
        existing = []
    elif case == "unrelated":
        existing = errors = ["invalid odds"]
    elif case == "forecast":
        after["projection"] = {"home_margin": 3}
    elif case == "unchanged":
        after = before
    elif case == "polymarket":
        after["prediction_market"] = {"home_probability": 0.9}
    with pytest.raises(ValueError, match="publication-dependent"):
        subject.retained_market_warnings(before, after, existing, errors)


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    schedule = tmp_path / "schedule"
    schedule.mkdir()
    candidate = tmp_path / "candidate"
    (candidate / "selected-schedule/latest").mkdir(parents=True)
    game = {"game_id": "one", "projection": {"home_margin": 2},
            "injury_report": {"source_url": "official"}, "projected_minutes": {"home": []}}
    (candidate / "selected-schedule/latest/day.json").write_text(json.dumps({"games": [game]}))
    config = SimpleNamespace(environment_root=tmp_path / "nba-gestalt-dev",
        pipeline_root=tmp_path / "pipeline", live_pointer=tmp_path / "live",
        observation_root=tmp_path / "observations", model_root=tmp_path / "model",
        data_root=tmp_path / "data", artifact_lake_uri=None)
    manifest = {"runtime_sha256": "old",
        "identity": {"evidence_through_date": "2026-11-02", "game_count": 107,
        "serving_bundle_sha256": "serving", "lineup_bundle_sha256": "lineup"},
        "approved_manifest": {"identity": {"schedule": {"final_game_count": 107}}}}
    calls = []
    monkeypatch.setattr(subject, "read_active_activation", lambda _: {
        "release_path": str(base), "runtime_sha256": "old", "generation": 10})
    monkeypatch.setattr(subject, "validate_forecast_handoff", lambda *a: ["one"])
    monkeypatch.setattr(subject, "stage_schedule_release", lambda **kw: (calls.append("stage") or candidate))
    monkeypatch.setattr(subject, "verify_production_site_release", lambda _: manifest)
    monkeypatch.setattr(subject, "verify_schedule_selection", lambda _: {
        "dates": {"day": {"path": "latest/day.json"}}})
    @contextmanager
    def api(*a, **kw):
        calls.append("preflight")
        yield "test://candidate"
        calls.append("preflight-stopped")
    monkeypatch.setattr(subject, "candidate_api", api)
    monkeypatch.setattr(subject, "collect_api_validation", lambda *a, **kw: {"status": "clean"})
    monkeypatch.setattr(subject, "_normalize_api_validation", lambda *a, **kw: None)
    monkeypatch.setattr(subject, "approved_schedule_game_ids", lambda _, **kwargs: [])
    monkeypatch.setattr(subject, "approved_model_root", lambda _, **kwargs: tmp_path / "bound-model")
    monkeypatch.setattr(subject, "fetch", lambda *a: {"games": [game]})
    def activate(**kw):
        assert kw["expected_previous_generation"] == 10
        assert kw["release"] == candidate
        calls.append("activate")
        return SimpleNamespace(generation=11)
    monkeypatch.setattr(subject, "activate_release", activate)
    return config, base, schedule, calls


def test_market_publication_records_warning_without_hiding_it(workflow, monkeypatch):
    config, base, schedule, calls = workflow
    candidate_day = base.parent / "candidate/selected-schedule/latest/day.json"
    board = json.loads(candidate_day.read_text())
    original = json.loads(candidate_day.read_text())
    base_day = base / "selected-schedule/latest/day.json"
    base_day.parent.mkdir(parents=True)
    base_day.write_text(json.dumps(original))
    board["games"][0]["market"] = {"home_spread": -3}
    candidate_day.write_text(json.dumps(board))
    board["games"][0]["data_quality_errors"] = [WARNING]
    original["games"][0]["data_quality_errors"] = [WARNING]
    monkeypatch.setattr(subject, "fetch", lambda api, path: board if api == "test://candidate" else original)
    result = subject.publish_forecast_schedule(config, schedule=schedule, base=base,
                                               work_id="markets", activate=True)
    assert result["status"] == "activated"
    assert result["retained_game_warnings"] == {"one": [WARNING]}
    assert board["games"][0]["data_quality_errors"] == [WARNING]


def test_job_publishes_only_after_full_preflight(workflow):
    config, base, schedule, calls = workflow
    result = subject.publish_forecast_schedule(config, schedule=schedule, base=base,
                                               work_id="request", activate=True)
    assert calls == ["stage", "preflight", "preflight-stopped", "activate"]
    assert result["status"] == "activated"
    assert result["code_changed"] is False and result["model_changed"] is False


def test_failed_api_check_is_a_failed_job_and_does_not_publish(workflow, monkeypatch):
    config, base, schedule, calls = workflow
    monkeypatch.setattr(subject, "fetch", lambda *a: {"games": []})
    with pytest.raises(ValueError, match="coverage differs"):
        subject.publish_forecast_schedule(config, schedule=schedule, base=base,
                                           work_id="request", activate=True)
    assert "activate" not in calls
    report = json.loads((config.pipeline_root / "forecast-publication/request/report.json").read_text())
    assert report["status"] == "failed"


def test_validation_only_does_not_activate(workflow):
    config, base, schedule, calls = workflow
    result = subject.publish_forecast_schedule(config, schedule=schedule, base=base,
                                               work_id="request", activate=False)
    assert result["status"] == "validated"
    assert "activate" not in calls


def test_changed_base_is_rejected_before_staging(workflow, monkeypatch):
    config, base, schedule, calls = workflow
    monkeypatch.setattr(subject, "read_active_activation", lambda _: None)
    with pytest.raises(ValueError, match="Active release changed"):
        subject.publish_forecast_schedule(config, schedule=schedule, base=base,
                                           work_id="request", activate=True)
    assert not calls


def test_active_generation_wrapper_and_release_use_same_verified_identity(workflow, monkeypatch):
    config, base, schedule, calls = workflow
    monkeypatch.setattr(subject, "read_active_activation", lambda _: {
        "release_path": str(base.parent / "different-release-path"),
        "runtime_sha256": "old", "generation": 10})
    result = subject.publish_forecast_schedule(config, schedule=schedule, base=base,
                                               work_id="wrapper", activate=True)
    assert result["status"] == "activated"
