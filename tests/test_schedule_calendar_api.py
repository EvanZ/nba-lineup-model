"""Calendar grades belong to the hourly report, never neighboring Schedule reads."""

from datetime import UTC, date, datetime

import boto3
import pytest
from fastapi.testclient import TestClient
from test_market_performance_api import SavedS3 as MarketS3
from test_market_performance_api import totals
from test_saved_schedule_api import DAY, GID, SEASON, fixtures
from test_schedule_pinned_inputs import fixture, publish, serve

from nba_lineup_model.web_api import market_performance as market_api
from nba_lineup_model.web_api.gateway import MarketGateway
from nba_lineup_model.web_api.market_calendar import calendar_summaries, summarize_model_grades
from nba_lineup_model.web_api.schedule import ScheduleReader, _cached_view


def model_row(gid, winner, ats):
    return {
        "game_id": gid,
        "comparisons": {"model": {"winner_correct": winner, "ats": {"result": ats}}},
    }


def report(day, rows):
    return {
        "game_date_pacific": day,
        "games": rows,
        "selection": {"uri": "s3://nba-gestalt-data/saved-report.json", "sha256": "a" * 64},
    }


def test_count_saved_per_game_grades_without_regrading_or_shared_cohort_filter():
    rows = {
        str(i): model_row(str(i), winner, ats)
        for i, (winner, ats) in enumerate(
            [(True, "win"), (False, "push"), (True, "no_pick"), (None, "unavailable")]
        )
    }
    record, coverage, status = summarize_model_grades(rows)
    assert record == {"wins": 2, "losses": 1, "atsWins": 1, "atsLosses": 0, "atsPushes": 1}
    assert status == "partial"
    assert coverage["model_graded_game_count"] == 3 and coverage["ats_graded_game_count"] == 2
    assert coverage["ats_no_pick_game_count"] == 1


def test_hourly_calendar_contains_missing_and_no_game_states_and_stops_at_through():
    days = [report("2026-11-12", [model_row(GID, True, "loss")])]
    scheduled = {"2026-11-12": {GID}, "2026-11-13": {"0022600999"}, "2026-11-15": {"0022600998"}}
    cal = calendar_summaries(days, scheduled, date(2026, 11, 15))
    assert len(cal) == 4
    assert cal["2026-11-12"]["record_status"] == "available"
    assert cal["2026-11-12"]["record"] == {
        "wins": 1,
        "losses": 0,
        "atsWins": 0,
        "atsLosses": 1,
        "atsPushes": 0,
    }
    assert cal["2026-11-13"]["record_status"] == "missing" and cal["2026-11-13"]["record"] is None
    assert cal["2026-11-14"]["record_status"] == "no_games" and cal["2026-11-14"]["record"] is None
    assert "2026-11-16" not in cal


@pytest.mark.parametrize("winner,ats", [("win", "win"), (True, "bad_status"), ([], "win")])
def test_bad_saved_grades_have_an_explicit_calendar_state(winner, ats):
    row = model_row(GID, winner, ats)
    if winner == []:
        row["comparisons"]["model"] = []
    cal = calendar_summaries([report(DAY, [row])], {DAY: {GID}}, date.fromisoformat(DAY))
    assert cal[DAY]["record_status"] == "invalid" and cal[DAY]["record"] is None


def test_zero_published_grades_are_unavailable_not_zero_zero():
    cal = calendar_summaries(
        [report(DAY, [model_row(GID, None, "unavailable")])], {DAY: {GID}}, date.fromisoformat(DAY)
    )
    assert cal[DAY]["record_status"] == "unavailable" and cal[DAY]["record"] is None


def test_schedule_loads_selected_date_only_and_returns_only_calendar_counts():
    s3, _ = fixtures()
    for other in [
        "2026-11-12",
        "2026-11-13",
        "2026-11-14",
        "2026-11-16",
        "2026-11-17",
        "2026-11-18",
    ]:
        key = (
            f"lake/artifacts/market_performance/season={SEASON}/model_version=v1/"
            f"game_date_pacific={other}/latest.json"
        )
        s3.put_fixture(key, {})
    view = ScheduleReader(s3).view(SEASON, DAY, now=datetime(2026, 11, 17, tzinfo=UTC))
    assert len(view["calendar"]) == 7
    assert view["calendar"][DAY] == {"game_date": DAY, "game_count": 1}
    for _op, key in s3.calls:
        if "game_date_pacific=" in key:
            assert ("game_date_pacific=" + DAY + "/") in key
    assert _cached_view.cache_parameters()["maxsize"] == 1


def test_market_calendar_reuses_already_loaded_reports_and_hourly_cache(monkeypatch):
    s3 = MarketS3({"2026-11-14": totals(1, 0.2, 1, 1), DAY: totals(2, 0.3, 2, 2)})
    reads = []
    original = s3.get_object

    def get(**kwargs):
        reads.append(kwargs["Key"])
        return original(**kwargs)

    s3.get_object = get
    monkeypatch.setattr(boto3, "client", lambda service: s3)
    clock = [1800000001.0]
    monkeypatch.setattr(market_api, "time", lambda: clock[0])
    market_api._cached_report.cache_clear()
    try:
        with TestClient(MarketGateway(None)) as client:
            url = "/api/market-performance?season=2026-27&through=" + DAY
            first = client.get(url)
            assert first.status_code == 200
            cal = first.json()["calendar"]
            assert cal[DAY]["record"]["wins"] == 2 and cal[DAY]["record"]["atsWins"] == 2
            assert sorted(reads) == sorted(s3.objects)
            count = len(reads)
            assert client.get(url).json() == first.json() and len(reads) == count
            clock[0] += 3600
            assert client.get(url).status_code == 200 and len(reads) == 2 * count
    finally:
        market_api._cached_report.cache_clear()


@pytest.mark.parametrize("ats", ["unavailable", "no_pick", "not_applicable"])
def test_ungraded_ats_keeps_saved_model_winner_grade(ats):
    s3, r, g, actual, published, ref = fixture()
    published["comparisons"]["model"]["ats"]["result"] = ats
    publish(s3, published, actual, ref)
    view = serve(r, g)
    assert view["availability"]["grading"] == "available"
    assert view["final_audit"]["grading"] == {"model_favorite_won": True, "market": None}
