"""Pipeline-selected per-game odds stay visible throughout a game's lifecycle."""

import hashlib
from copy import deepcopy
from datetime import UTC, datetime

import pytest
from test_saved_schedule_api import DAY, NOW, SEASON, START
from test_schedule_details_api import latest, pointer, setup_game

from nba_lineup_model.web_api.schedule_details import DetailsReader

AFTER = datetime(2026, 11, 16, 2, tzinfo=UTC)


@pytest.mark.parametrize("state", ["pre", "in", "post"])
def test_frozen_pointer_is_independent_of_report_and_latest_capture(state):
    s3, r, g, baseline = setup_game()
    key, selected = pointer(s3, g, frozen=True)
    original = deepcopy(g["projection"])
    # Today's selected event can disappear from the newest league capture.
    latest(s3, [], "2026-11-16T01:00:00Z")
    g["status"]["state"] = state
    s3.calls.clear()
    DetailsReader(r, SEASON, DAY, AFTER).enrich([g])
    assert g["availability"]["market"] == "available"
    assert g["market"]["home_spread"] == -3
    assert g["prediction_market"]["home_win_probability"] == 0.8
    assert g["projection"] == original
    assert not g.get("final_audit") and not g.get("retrospective")
    assert g["sources"]["market"]["sha256"] == selected["snapshot"]["sha256"]
    observed = g["sources"]["market"]["selection"]
    assert observed["document"] == selected
    assert observed["observed_sha256"] == hashlib.sha256(s3.objects[key][0]).hexdigest()
    assert not any(
        op == "list" or k.endswith("odds-api/season=2026-27/latest.json") for op, k in s3.calls
    )


def test_unfrozen_pointer_advances_only_when_producer_publishes_it():
    s3, r, g, _ = setup_game()
    key, selected = pointer(s3, g)
    d = DetailsReader(r, SEASON, DAY, datetime(2026, 11, 15, 23, tzinfo=UTC))
    first = d.game_odds(g)
    # A newer raw capture cannot alter the accepted selection.
    latest(s3, [], "2026-11-15T21:00:00Z")
    assert d.game_odds(g) == first
    doc = latest(s3, captured="2026-11-15T22:00:00Z")
    pointer(s3, g, doc)
    assert d.game_odds(g)[2]["captured_at"] == doc["collected_at"]


@pytest.mark.parametrize(
    "damage",
    [
        "game",
        "team",
        "date",
        "season",
        "rule",
        "schema",
        "purpose",
        "tipoff",
        "cutoff",
        "future",
        "unfrozen",
        "freeze_time",
        "snapshot_uri",
        "snapshot_sha",
        "capture_time",
        "provider_event",
        "provider_update",
        "moneyline",
        "no_quotes",
        "raw_corrupt",
    ],
)
def test_invalid_pointer_or_pinned_event_is_visible_not_replaced(damage):
    s3, r, g, _ = setup_game()
    doc = latest(s3)
    if damage == "provider_update":
        doc["events"][0]["bookmakers"][0]["markets"][0]["last_update"] = START
    if damage == "moneyline":
        doc["events"][0]["bookmakers"][0]["markets"][1]["outcomes"][0]["price"] = 0
    if damage == "no_quotes":
        doc["events"][0]["bookmakers"] = []
    doc = latest(s3, doc["events"])
    key, p = pointer(s3, g, doc, frozen=True)
    changes = {
        "game": ("game_id", "0022600999"),
        "team": ("home_team_id", 999),
        "date": ("game_date_pacific", "2026-11-14"),
        "season": ("season", "2025-26"),
        "rule": ("selection_rule", "other"),
        "schema": ("schema_version", True),
        "purpose": ("purpose", "other"),
        "tipoff": ("scheduled_start_utc", "2026-11-16T01:00:00Z"),
        "cutoff": ("cutoff_utc", "2026-11-16T01:00:00Z"),
        "future": ("selected_at", "2026-11-17T00:00:00Z"),
        "unfrozen": ("frozen", False),
        "freeze_time": ("frozen_at", "2026-11-15T23:00:00Z"),
        "provider_event": ("provider_event_id", "wrong"),
    }
    if damage in changes:
        field, value = changes[damage]
        p[field] = value
    if damage == "snapshot_uri":
        p["snapshot"]["uri"] = p["snapshot"]["uri"].replace(SEASON, "2025-26")
    if damage == "snapshot_sha":
        p["snapshot"]["sha256"] = "0" * 64
    if damage == "capture_time":
        p["snapshot"]["collected_at"] = START
    if damage == "raw_corrupt":
        s3.put_fixture(p["snapshot"]["uri"].split("/", 3)[3], {})
    s3.put_fixture(key, p)
    d = DetailsReader(r, SEASON, DAY, AFTER)
    # Even an available historical report must not hide a corrupt pointer.
    d.historical_game = lambda *args: pytest.fail("Invalid pointer used historical quotes")
    d.enrich_game(g, ({}, {}))
    assert g["availability"]["market"] == "invalid"
    assert g["detail_errors"]["market"] == "Saved data failed validation."
    assert g["market"] is None and g["projection"] is not None


def test_absent_polymarket_is_valid_and_approved_bookmaker_is_preserved():
    s3, r, g, _ = setup_game()
    doc = latest(s3)
    books = doc["events"][0]["bookmakers"]
    books[:] = [b for b in books if b["key"] != "polymarket"]
    books[0]["key"] = "espnbet"  # Odds API bookmaker, not ESPN basketball data.
    doc = latest(s3, doc["events"])
    pointer(s3, g, doc)
    market, poly, _ = DetailsReader(r, SEASON, DAY, NOW).game_odds(g)
    assert poly is None and "espnbet" in market["bookmaker_keys"]
    assert market["home_spread"] == -3


def test_missing_pointer_has_no_raw_archive_fallback():
    s3, r, g, _ = setup_game()
    latest(s3)
    s3.calls.clear()
    DetailsReader(r, SEASON, DAY, NOW).enrich([g])
    assert g["availability"]["market"] == "missing" and g["market"] is None
    assert not any("odds-api" in k or op == "list" for op, k in s3.calls)


@pytest.mark.parametrize("late_update", [False, True])
def test_earlier_provider_tipoff_and_pacific_date_are_enforced(late_update):
    s3, r, g, _ = setup_game()
    doc = latest(s3)
    earlier = "2026-11-15T23:00:00Z"
    doc["events"][0]["commence_time"] = earlier
    if late_update:
        doc["events"][0]["bookmakers"][0]["last_update"] = earlier
    doc = latest(s3, doc["events"])
    key, p = pointer(s3, g, doc, frozen=True)
    p["cutoff_utc"] = earlier
    s3.put_fixture(key, p)
    d = DetailsReader(r, SEASON, DAY, AFTER)
    if late_update:
        with pytest.raises(ValueError, match="not pregame"):
            d.game_odds(g)
    else:
        market, poly, ref = d.game_odds(g)
        assert market["home_spread"] == -3
        assert ref["selection"]["document"]["game_date_pacific"] == DAY
