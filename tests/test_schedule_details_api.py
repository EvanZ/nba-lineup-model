"""Read-only Schedule market and postgame contracts, including adverse inputs."""

import hashlib
from copy import deepcopy
from datetime import UTC, datetime

import pytest
from test_saved_schedule_api import DAY, GID, NOW, SEASON, START, encoded, fixtures, prediction

from nba_lineup_model.web_api.schedule import ScheduleReader, _prediction_view
from nba_lineup_model.web_api.schedule_details import DetailsReader, StaleSelection, market_view


def setup_game():
    s3, _ = fixtures()
    r = ScheduleReader(s3)
    record = prediction()
    record["away"]["players"][0]["player_id"] = 200
    game = r.view(SEASON, DAY, now=NOW)["day"]["games"][0]
    game.update(_prediction_view(record))
    game.update(home_team_name="Home Team", away_team_name="Away Team")
    return s3, r, game, record


def stub_pinned_box(expected):
    def read(_game, reference):
        if reference != expected:
            raise ValueError("Pinned box reference failed verification")
        return {"homeTeam": {"score": 100}, "awayTeam": {"score": 90}}, reference

    return read


def latest(s3, events=None, captured="2026-11-15T20:00:00Z"):
    def book(key, spread, price):
        return {
            "key": key,
            "title": key,
            "last_update": captured,
            "markets": [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Home Team", "point": spread, "price": -110},
                        {"name": "Away Team", "point": -spread, "price": -110},
                    ],
                },
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Home Team", "price": price},
                        {"name": "Away Team", "price": -price},
                    ],
                },
            ],
        }

    if events is None:
        events = [
            {
                "id": "provider-123",
                "sport_key": "basketball_nba",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "commence_time": START,
                "bookmakers": [
                    book("a", -2, -150),
                    book("b", -4, -200),
                    book("polymarket", -10, -400),
                ],
            }
        ]
    doc = {
        "schema_version": 1,
        "provider": "The Odds API",
        "season": SEASON,
        "sport_key": "basketball_nba",
        "collected_at": captured,
        "events": events,
    }
    root = f"lake/raw/odds-api/season={SEASON}/"
    s3.put_fixture(root + "latest.json", doc)
    stamp = datetime.fromisoformat(captured).strftime("%Y%m%dT%H%M%SZ")
    s3.put_fixture(root + "snapshots/" + stamp + ".json", doc)
    return doc


def test_latest_market_medians_exclude_polymarket_and_preserve_quotes():
    s3, r, g, _ = setup_game()
    latest(s3)
    s3.calls.clear()
    d = DetailsReader(r, SEASON, DAY, NOW)
    market, poly, ref = d.upcoming(g, d.latest_odds())
    assert market["home_spread"] == -3 and market["away_spread"] == 3
    assert market["home_win_probability"] == pytest.approx((0.6 + 2 / 3) / 2)
    assert market["home_moneyline"] is None
    assert market["spread_bookmaker_count"] == 2
    assert market["bookmakers"][0]["home_moneyline"] == -150
    assert poly["home_spread"] == -10 and poly["home_win_probability"] == 0.8
    assert ref["provider_event_id"] == "provider-123"
    assert (
        len([key for op, key in s3.calls if key.endswith("odds-api/season=2026-27/latest.json")])
        == 1
    )


@pytest.mark.parametrize(
    "damage", ["ambiguous", "checksum", "posttip", "wrongdate", "duplicatebook", "wrongteams"]
)
def test_upcoming_input_safety(damage):
    s3, r, g, _ = setup_game()
    doc = latest(s3)
    events = deepcopy(doc["events"])
    if damage == "ambiguous":
        events.append({**deepcopy(events[0]), "id": "other-event"})
    if damage == "duplicatebook":
        events[0]["bookmakers"].append(deepcopy(events[0]["bookmakers"][0]))
    if damage == "wrongteams":
        events[0]["bookmakers"][0]["markets"][0]["outcomes"][0]["name"] = "Wrong Team"
    if damage == "wrongdate":
        events[0]["commence_time"] = "2026-11-17T00:00:00Z"
    captured = START if damage == "posttip" else "2026-11-15T20:00:00Z"
    latest(s3, events, captured)
    if damage == "checksum":
        key = "lake/raw/odds-api/season=2026-27/snapshots/20261115T200000Z.json"
        s3.put_fixture(key, {})
    d = DetailsReader(r, SEASON, DAY, datetime(2026, 11, 16, 1, tzinfo=UTC))
    if damage in ("posttip", "wrongdate"):
        assert d.upcoming(g, d.latest_odds()) is None
    else:
        with pytest.raises(ValueError):
            d.upcoming(g, d.latest_odds())


def actual_fixture(s3, g, baseline):
    boxref = {
        "uri": "s3://nba-gestalt-data/box.json",
        "sha256": "a" * 64,
        "metadata_uri": "s3://nba-gestalt-data/box.meta.json",
        "metadata_sha256": "b" * 64,
    }
    teams = {}
    for side in ("home", "away"):
        p = baseline[side]["players"][0]
        teams[side] = {
            "team_id": g[side + "_team_id"],
            "team": g[side + "_team"],
            "total_actual_minutes": 265.0,
            "total_projected_minutes": 240,
            "actual_player_total": 0.123456789,
            "projected_player_total": baseline["components"][side + "_player_total"],
            "players": [
                {
                    "player_id": p["player_id"],
                    "name": p["name"],
                    "in_baseline_prediction": True,
                    "projected_minutes": p["projected_minutes"],
                    "actual_minutes": 0.0,
                    "nail": p["nail"],
                    "rating_source": "daily_state",
                    "projected_rating_contribution": p["rating_contribution"],
                    "actual_rating_contribution": 0.0,
                    "contribution_delta": -p["rating_contribution"],
                }
            ],
        }
    teams["home"]["players"].append(
        {
            "player_id": 300,
            "name": "Extra Player",
            "in_baseline_prediction": False,
            "projected_minutes": 0.0,
            "actual_minutes": 8.123456789,
            "nail": -2.0,
            "rating_source": "replacement",
            "projected_rating_contribution": 0.0,
            "actual_rating_contribution": -0.321,
            "contribution_delta": -0.321,
        }
    )
    doc = {
        "schema_version": 1,
        "artifact_kind": "game_prediction_actual_minutes",
        "purpose": "postgame_actual_minutes_counterfactual",
        "season": SEASON,
        "model_version": "v1",
        "game_date_pacific": DAY,
        "game_id": GID,
        "scheduled_start_utc": START,
        "baseline_prediction_purpose": g["prediction_purpose"],
        "generated_at_utc": "2026-11-16T05:00:00Z",
        "original_predicted_home_margin": g["projection"]["home_margin"],
        "original_home_win_probability": g["projection"]["home_win_probability"],
        "counterfactual_home_margin": -2.123456789,
        "counterfactual_home_win_probability": 0.4123456789,
        "actual_home_score": 100,
        "actual_away_score": 90,
        "components": deepcopy(baseline["components"]),
        "inputs": {"prediction": deepcopy(g["sources"]["prediction"]), "box_score": boxref},
        **teams,
    }
    return doc, boxref


def select_actual(s3, doc):
    root = (
        f"lake/artifacts/game_predictions_actual_minutes/season={SEASON}/model_version=v1/"
        f"game_date_pacific={DAY}/game_id={GID}/"
    )
    sha = hashlib.sha256(encoded(doc)).hexdigest()
    key = root + sha + ".json"
    s3.put_fixture(key, doc)
    s3.put_fixture(
        root + "latest.json",
        {
            k: doc[k]
            for k in ["schema_version", "season", "model_version", "game_date_pacific", "game_id"]
        }
        | {"sha256": sha, "uri": "s3://nba-gestalt-data/" + key},
    )


def test_actual_saved_precision_overtime_dnp_extra_player_and_original_unchanged():
    s3, r, g, baseline = setup_game()
    before = deepcopy(g["projection"])
    doc, boxref = actual_fixture(s3, g, baseline)
    select_actual(s3, doc)
    d = DetailsReader(r, SEASON, DAY, NOW)
    d.pinned_box = stub_pinned_box(boxref)
    view, _ = d.actual_minutes(g)
    assert view["home_margin"] == -2.123456789 and view["home_win_probability"] == 0.4123456789
    assert view["ledger"]["home"]["actual_minutes"] == 265
    assert view["actual_minutes"]["home"][0]["actual_minutes"] == 0
    extra = view["actual_minutes"]["home"][1]
    assert extra["actual_minutes"] == 8.123456789 and extra["contribution_delta"] == -0.321
    assert extra["in_baseline_prediction"] is False and extra["rating_source"] == "replacement"
    assert g["projection"] == before


@pytest.mark.parametrize(
    "damage", ["baseline", "box", "purpose", "checksum", "identity", "duplicateplayer"]
)
def test_actual_stale_or_corrupt_inputs_remain_unavailable_without_hiding_game(damage):
    s3, r, g, baseline = setup_game()
    doc, boxref = actual_fixture(s3, g, baseline)
    if damage == "baseline":
        doc["inputs"]["prediction"]["sha256"] = "0" * 64
    if damage == "box":
        doc["inputs"]["box_score"] = {**boxref, "sha256": "0" * 64}
    if damage == "purpose":
        doc["purpose"] = "pregame_projection"
    if damage == "identity":
        doc["home"]["team_id"] = 999
    if damage == "duplicateplayer":
        doc["away"]["players"][0]["player_id"] = 100
    select_actual(s3, doc)
    if damage == "checksum":
        key = next(
            k
            for k in s3.objects
            if "/game_predictions_actual_minutes/" in k and not k.endswith("latest.json")
        )
        s3.put_fixture(key, {})
    d = DetailsReader(r, SEASON, DAY, NOW)
    d.pinned_box = stub_pinned_box(boxref)
    g["status"]["state"] = "post"
    d.enrich_game(g, None, None)
    assert g["game_id"] == GID and g["projection"] is not None
    assert g.get("retrospective") is None
    assert g["availability"]["postgame_details"] == ("stale" if damage == "baseline" else "invalid")
    assert "postgame_details" in g["detail_errors"]


def saved_market(g):
    snapshot = {
        "uri": "s3://nba-gestalt-data/lake/raw/odds-api/season=2026-27/snapshots/20261115T200000Z.json",
        "sha256": "a" * 64,
    }

    def q(kind, key, value):
        outcomes = [{"name": "Home Team", "price": -150}, {"name": "Away Team", "price": 150}]
        if kind == "spreads":
            outcomes[0]["point"] = -2
            outcomes[1]["point"] = 2
        return {
            "bookmaker_key": key,
            "title": key,
            "market": kind,
            "value": value,
            "outcomes": outcomes,
            "observed_at": "2026-11-15T20:00:00Z",
            "last_updated_at": "2026-11-15T19:59:00Z",
            "provider_commence_time": START,
            "snapshot": snapshot,
        }

    return {
        "provider": "The Odds API",
        "selection_rule": "latest_pregame_polymarket_and_same_snapshot_books",
        "spreads": {
            "polymarket": q("spreads", "polymarket", 2),
            "books": [q("spreads", "a", 2)],
            "consensus": 2.1256789,
        },
        "h2h": {
            "polymarket": q("h2h", "polymarket", 0.6),
            "books": [q("h2h", "a", 0.6)],
            "consensus": 0.6123456789,
        },
    }


def test_historical_saved_consensus_and_grades_are_not_recomputed():
    s3, r, g, baseline = setup_game()
    _, boxref = actual_fixture(s3, g, baseline)
    row = {
        "game_id": GID,
        "scheduled_start_utc": START,
        "home": {"team_id": 1, "team_abbreviation": "HOM", "final_score": 100},
        "away": {"team_id": 2, "team_abbreviation": "AWY", "final_score": 90},
        "inputs": {"prediction": g["sources"]["prediction"], "final_box_score": boxref},
        "markets": saved_market(g),
        "comparisons": {
            "model": {
                "predicted_home_margin": g["projection"]["home_margin"],
                "home_win_probability": g["projection"]["home_win_probability"],
                "winner_correct": False,
                "ats": {"result": "push"},
            }
        },
    }
    d = DetailsReader(r, SEASON, DAY, NOW)
    d.pinned_box = stub_pinned_box(boxref)
    historical = ({GID: row}, {"uri": "saved report", "sha256": "a" * 64})
    market, _, _ = d.historical_game(g, historical)
    audit, _ = d.historical_grading(g, historical)
    assert market["home_spread"] == -2.1256789 and market["home_win_probability"] == 0.6123456789
    assert market["selection"] == "Saved pregame"
    assert audit["grading"] == {"model_favorite_won": False, "market": {"model_ats_result": "push"}}
    row["inputs"]["prediction"] = {}
    with pytest.raises(StaleSelection):
        d.historical_grading(g, ({GID: row}, {}))


def test_clippers_explicit_provider_alias():
    _, _, g, _ = setup_game()
    g.update(home_team="LAC", home_team_name="LA Clippers")
    markets = saved_market(g)
    for kind in ("spreads", "h2h"):
        for quote in [*markets[kind]["books"], markets[kind]["polymarket"]]:
            quote["outcomes"][0]["name"] = "Los Angeles Clippers"
    assert market_view(markets, g, "Saved pregame")[0] is not None


def test_started_game_never_uses_latest_as_pregame_fallback():
    s3, r, g, _ = setup_game()
    latest(s3)
    g["status"]["state"] = "in"
    before = len(s3.calls)
    DetailsReader(r, SEASON, DAY, datetime(2026, 11, 16, 2, tzinfo=UTC)).enrich([g])
    assert g["market"] is None and g["availability"]["market"] == "missing"
    assert not any("odds-api" in key for op, key in s3.calls[before:])


def test_invalid_latest_is_component_error_and_game_stays_visible():
    s3, r, g, _ = setup_game()
    latest(s3)
    s3.put_fixture("lake/raw/odds-api/season=2026-27/latest.json", {})
    DetailsReader(r, SEASON, DAY, NOW).enrich([g])
    assert g["availability"]["market"] == "invalid" and g["game_id"] == GID


def test_saved_quotes_can_exist_without_a_model_grade():
    s3, r, g, baseline = setup_game()
    _, boxref = actual_fixture(s3, g, baseline)
    row = {
        "game_id": GID,
        "scheduled_start_utc": START,
        "home": {"team_id": 1, "team_abbreviation": "HOM", "final_score": 100},
        "away": {"team_id": 2, "team_abbreviation": "AWY", "final_score": 90},
        "inputs": {"prediction": None, "final_box_score": boxref},
        "markets": saved_market(g),
        "comparisons": {"model": None},
    }
    g["sources"]["prediction"] = None
    g["projection"] = None
    d = DetailsReader(r, SEASON, DAY, NOW)
    d.pinned_box = stub_pinned_box(boxref)
    market, _, _ = d.historical_game(g, ({GID: row}, {}))
    assert market["home_spread"] == -2.1256789
    assert d.historical_grading(g, ({GID: row}, {})) is None
