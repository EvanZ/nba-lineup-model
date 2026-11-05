import copy
import json
from datetime import date, datetime, timedelta

import pytest
from test_official_schedule_runtime import GAME_DATE, START, _board
from test_upcoming_forecasts import fixture_model

from nba_lineup_model.season.injury_forecast_replay import replay_injury_selection
from nba_lineup_model.season.official_schedule_runtime import _publish_pregame_candidate
from nba_lineup_model.season.serving_selection import build_schedule_selection, verify_schedule_selection


def test_approved_replay_preserves_originals_and_other_dates(tmp_path, monkeypatch):
    bundle, response = fixture_model(tmp_path, monkeypatch)
    cutoff = datetime.fromisoformat(START.replace("Z", "+00:00"))
    captured = cutoff - timedelta(minutes=20)
    raw = tmp_path / "raw"
    (raw / "latest").mkdir(parents=True)
    board = _board(collected_at=captured)
    (raw / "latest" / f"{GAME_DATE}.json").write_text(json.dumps(board))
    other = copy.deepcopy(board)
    previous_day = (date.fromisoformat(GAME_DATE) - timedelta(days=1)).isoformat()
    other["game_date"] = previous_day
    other["games"][0]["game_id"] = "0022600999"
    (raw / "latest" / f"{previous_day}.json").write_text(json.dumps(other))
    source = tmp_path / "source"
    build_schedule_selection(raw, source, edition_id="accepted-hca")
    original_files = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    fresh = copy.deepcopy(board["games"][0])
    fresh["injury_report"].update(source_url="https://nba.example/new-report.pdf",
        published_at=captured.isoformat(), artifact_sha256="b" * 64)
    observations = tmp_path / "observations"
    _publish_pregame_candidate(fresh, state_root=observations, captured_at=captured)
    args = dict(source=source, target=tmp_path / "replay", observation_root=observations,
        model_bundle=bundle, forecast_response=response, game_date=date.fromisoformat(GAME_DATE),
        approval_reference="test-explicit-approval", now=cutoff + timedelta(hours=3),
        code={"sealed": True, "package_sha256": "d" * 64,
              "dependency_lock_sha256": "e" * 64, "git_revision": "f" * 40})
    result = replay_injury_selection(**args)
    assert result["computed_at"] == args["now"].isoformat()
    index = verify_schedule_selection(args["target"])
    actual = json.loads((args["target"] / "latest" / f"{GAME_DATE}.json").read_text())["games"][0]
    assert actual["injury_report"]["source_url"].endswith("new-report.pdf")
    assert actual["pregame"]["injury_report"] == actual["injury_report"]
    assert actual["projection"]["ledger"]["home_court_adjustment"] == 1.6
    assert actual["forecast_selection"]["provenance"] == "retrospective_replay"
    assert actual["forecast_selection"]["original_publication_receipt"] is None
    assert index["edition_id"] == "accepted-hca"
    assert (args["target"] / "latest" / f"{previous_day}.json").read_bytes() == original_files[f"latest/{previous_day}.json"]
    for relative, content in original_files.items():
        assert (source / relative).read_bytes() == content
    assert replay_injury_selection(**args) == result
    with pytest.raises(ValueError, match="requires explicit approval"):
        replay_injury_selection(**{**args, "approval_reference": ""})
    with pytest.raises(ValueError, match="different inputs"):
        replay_injury_selection(**{**args, "approval_reference": "another request"})
