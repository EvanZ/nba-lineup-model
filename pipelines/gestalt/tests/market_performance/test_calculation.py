from copy import deepcopy
from datetime import timedelta

import pytest
from gestalt_pipelines.market_performance.utils import aggregate_totals, comparison, daily_totals, game_result, select_markets
from .conftest import START, snapshot


def selected(record=None):
    game = {"game_id": "g", "game_time_utc": START}
    value = record if record is not None else snapshot()
    return select_markets([game], {"g": {"home": "Celtics", "away": "76ers"}}, [(value, {"uri": "source", "sha256": "a"*64})])["g"]


def test_same_snapshot_book_medians_preserve_espnbet():
    result = selected()
    assert result["h2h"]["polymarket"]["value"] == .625
    assert result["h2h"]["consensus"] == .6
    assert result["spreads"]["consensus"] == 5
    assert [q["bookmaker_key"] for q in result["h2h"]["books"]] == ["fanduel", "espnbet"]


@pytest.mark.parametrize("late", ["capture", "update", "provider"])
def test_strict_scheduled_and_provider_cutoffs(late):
    value = snapshot()
    if late == "capture": value["collected_at"] = START.isoformat()
    elif late == "update": value["events"][0]["bookmakers"][0]["last_update"] = START.isoformat()
    else: value["events"][0]["commence_time"] = value["collected_at"]
    assert selected(value) == {}


def test_separate_moneyline_and_spread_anchors_do_not_take_newer_unrelated_books():
    older = snapshot("2026-10-21T23:00:00Z")
    newer = snapshot()
    newer["events"][0]["bookmakers"][0]["markets"].pop(1)
    newer["events"][0]["bookmakers"][1]["markets"][1]["outcomes"][0]["point"] = -40
    game = {"game_id": "g", "game_time_utc": START}
    result = select_markets([game], {"g": {"home": "Celtics", "away": "76ers"}},
        [(newer, {"uri": "new"}), (older, {"uri": "old"})])["g"]
    assert result["h2h"]["polymarket"]["snapshot"]["uri"] == "new"
    assert result["spreads"]["polymarket"]["snapshot"]["uri"] == "old"
    assert result["spreads"]["consensus"] == 5


@pytest.mark.parametrize("margin,actual,benchmark,pick,outcome", [
    (6, 8, 5, "home", "win"), (4, 8, 5, "away", "loss"), (-7, -8, -5, "away", "win"),
    (6, 5, 5, "home", "push"), (5, 5, 5, None, "no_pick"), (None, 8, 5, None, "unavailable"),
])
def test_ats_signs_pushes_and_no_pick(margin, actual, benchmark, pick, outcome):
    value = comparison(margin, .5, actual, benchmark, "pregame_projection")
    assert value["ats"]["pick"] == pick and value["ats"]["result"] == outcome
    assert value["winner_pick"] == "home"


def test_totals_aggregate_by_game_count_instead_of_daily_averages():
    def row(error):
        comparisons = {name: comparison(8+error, .7, 8, 5, "pregame_projection")
                       for name in ("model", "actual_minutes_counterfactual", "polymarket", "bookmaker_consensus")}
        return {"in_shared_comparison": True, "comparisons": comparisons}
    days = [daily_totals([row(1)]), daily_totals([row(4), row(4), row(4)])]
    combined = aggregate_totals(days)["model"]
    assert combined["margin_rmse"] == 3.5
    assert combined["margin_mae"] == 3.25
    assert combined["brier"] == pytest.approx(.09)
    assert combined["games"] == 4
    assert aggregate_totals([])["model"]["brier"] is None
