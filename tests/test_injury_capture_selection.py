"""Forecasts use collector observations, never stale inputs or future reports."""
import copy
import json
from datetime import datetime, timedelta

import pytest

from nba_lineup_model.season.official_schedule_runtime import (
    _object_sha256, _publish_pregame_candidate, select_captured_injuries,
)


def fixture(tmp_path):
    cutoff = datetime.fromisoformat("2026-11-04T19:00:00-05:00")
    game = {"game_id": "0022600173", "start_time": cutoff.isoformat(),
            "away_team": "PHI", "home_team": "DET", "status": {"state": "pre"},
            "injury_report": {"artifact_sha256": "a" * 64,
                "source_url": "https://ak-static.cms.nba.com/referee/injury/nov2.pdf",
                "published_at": "2026-11-02T19:30:00-05:00",
                "team_reports": {"PHI": {"status": "not_in_report"},
                                 "DET": {"status": "not_in_report"}}, "assumptions": []}}
    source, observations = tmp_path / "schedule", tmp_path / "collector"
    path = source / "latest/2026-11-04.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"games": [game]}))
    fresh = copy.deepcopy(game)
    fresh["injury_report"].update(source_url="https://ak-static.cms.nba.com/referee/injury/nov4.pdf",
        published_at=(cutoff - timedelta(minutes=45)).isoformat(), artifact_sha256="b" * 64,
        team_reports={"PHI": {"status": "submitted"}, "DET": {"status": "submitted"}})
    capture, _ = _publish_pregame_candidate(fresh, state_root=observations,
                                          captured_at=cutoff - timedelta(minutes=30))
    return source, observations, cutoff, capture


def select(source, observations, now, **kwargs):
    return select_captured_injuries(schedule_root=source, observation_root=observations,
        start_date="2026-11-04", days_ahead=0, now=now, **kwargs)


def test_uses_collector_not_old_selected_report_and_preserves_inputs(tmp_path):
    source, observations, cutoff, capture = fixture(tmp_path)
    before = (source / "latest/2026-11-04.json").read_bytes(), capture.read_bytes()
    selected = select(source, observations, cutoff - timedelta(minutes=5))
    injury = selected["0022600173"]["injury_report"]
    assert injury["source_url"].endswith("nov4.pdf")
    assert injury["artifact_sha256"] == "b" * 64
    assert all(v["status"] == "submitted" for v in injury["team_reports"].values())
    assert before == ((source / "latest/2026-11-04.json").read_bytes(), capture.read_bytes())


def test_replay_never_uses_capture_after_tipoff(tmp_path):
    source, observations, cutoff, capture = fixture(tmp_path)
    late = json.loads(capture.read_text())
    late["captured_at"] = (cutoff + timedelta(minutes=1)).isoformat()
    late["game"]["injury_report"]["source_url"] = "https://nba.example/after-tipoff.pdf"
    late["candidate_sha256"] = _object_sha256({k: v for k, v in late.items() if k != "candidate_sha256"})
    capture.with_name("late.json").write_text(json.dumps(late))
    assert select(source, observations, cutoff + timedelta(hours=3)) == {}
    selected = select(source, observations, cutoff + timedelta(hours=3), include_started=True)
    assert selected["0022600173"]["injury_report"]["source_url"].endswith("nov4.pdf")


def test_future_capture_is_not_used_for_open_forecast(tmp_path):
    source, observations, cutoff, capture = fixture(tmp_path)
    with pytest.raises(ValueError, match="No verified pre-tip"):
        select(source, observations, cutoff - timedelta(hours=1))


def test_missing_capture_fails_instead_of_reusing_stale_forecast(tmp_path):
    source, observations, cutoff, capture = fixture(tmp_path)
    capture.unlink()
    with pytest.raises(ValueError, match="No verified pre-tip"):
        select(source, observations, cutoff - timedelta(minutes=5))


@pytest.mark.parametrize("kind", ["checksum", "published_after_capture", "wrong_team"])
def test_rejects_invalid_capture(tmp_path, kind):
    source, observations, cutoff, capture = fixture(tmp_path)
    x = json.loads(capture.read_text())
    if kind == "checksum":
        x["candidate_sha256"] = "0" * 64
    else:
        if kind == "published_after_capture":
            x["game"]["injury_report"]["published_at"] = cutoff.isoformat()
        else:
            x["game"]["away_team"] = "BOS"
        x["candidate_sha256"] = _object_sha256({k: v for k, v in x.items() if k != "candidate_sha256"})
    capture.write_text(json.dumps(x))
    with pytest.raises(ValueError):
        select(source, observations, cutoff - timedelta(minutes=5))


def test_open_forecast_uses_collector_board_before_pregame_archiving(tmp_path):
    source, observations, cutoff, capture = fixture(tmp_path)
    fresh = json.loads(capture.read_text())["game"]
    capture.unlink()
    stamp = cutoff - timedelta(hours=5)
    fresh["injury_report"]["published_at"] = stamp.isoformat()
    path = observations / "latest/2026-11-04.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"collected_at": stamp.isoformat(), "games": [fresh]}))
    result = select(source, observations, cutoff - timedelta(hours=4))
    assert result["0022600173"]["capture"]["kind"] == "collector_snapshot"
    assert result["0022600173"]["injury_report"]["source_url"].endswith("nov4.pdf")
    # The latest board must never masquerade as a historical pre-tip archive.
    with pytest.raises(ValueError, match="No verified pre-tip"):
        select(source, observations, cutoff + timedelta(hours=3), include_started=True)
