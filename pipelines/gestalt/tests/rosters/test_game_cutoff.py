import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta

import pytest
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.rosters.resources import NBARosterArchive
from gestalt_pipelines.rosters.utils import game_roster, RosterError
from .test_roster_capture import roster_job, FIRST, SECOND, SEASON, GAME

START = datetime(2026, 11, 9, 3, tzinfo=UTC)
GAME_NOW = {**GAME, "game_time_utc": START}


def evidence(archive):
    return archive.game_evidence(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON))


@pytest.mark.parametrize("delta,included", [(-1, True), (0, False), (1, False)])
def test_observations_strictly_before_tipoff(roster_job, delta, included):
    _, objects, _, _, _ = roster_job
    archive = NBARosterArchive()
    source = evidence(archive)
    source["transactions"][0]["first_observed_at"] = (START+timedelta(seconds=delta)).isoformat()
    record = game_roster(season=SEASON, game=GAME_NOW, **source)
    assert (1 in record["teams"][str(SECOND)]) == included
    assert (1 in record["teams"][str(FIRST)]) != included
    assert bool(record["cutoff_selection"]["assumptions"]) != included


def test_date_only_transaction_is_deferred_to_next_pacific_day(roster_job):
    archive = NBARosterArchive();source = evidence(archive)
    earlier = game_roster(season=SEASON, game=GAME_NOW, **source)
    later = game_roster(season=SEASON, game={**GAME_NOW, "game_time_utc": START+timedelta(days=1)}, **source)
    assert 1 in earlier["teams"][str(FIRST)] and 1 in later["teams"][str(SECOND)]
    warning = earlier["cutoff_selection"]["assumptions"][0]
    assert warning["reported_date_pacific"] == "2026-11-08"
    assert warning["assumed_application_date_pacific"] == "2026-11-09"
    assert warning["included_for_game"] is False
    assert warning["recorded_observation_time"] == source["transactions"][0]["first_observed_at"]


@pytest.mark.parametrize("action", ["add", "remove"])
def test_operator_repairs_support_confirmed_time_without_fabricating_one(roster_job, action):
    archive = NBARosterArchive();source = evidence(archive)
    pid = 1 if action == "remove" else 99999
    source["repairs"] = [{"effective_date": "2026-11-08", "action": action, "team_id": FIRST, "player_id": pid}]
    same_day = game_roster(season=SEASON, game=GAME_NOW, **source)
    assert (pid in same_day["teams"][str(FIRST)]) == (action == "remove")
    assert any(w["source"] == "repair_transactions" for w in same_day["cutoff_selection"]["assumptions"])
    source["repairs"][0]["effective_at"] = (START-timedelta(minutes=1)).isoformat()
    confirmed = game_roster(season=SEASON, game=GAME_NOW, **source)
    assert (pid in confirmed["teams"][str(FIRST)]) == (action == "add")
    source["repairs"][0]["effective_at"] = START.isoformat()
    assert (pid in game_roster(season=SEASON, game=GAME_NOW, **source)["teams"][str(FIRST)]) == (action == "remove")


def test_pregame_membership_is_reused_without_reading_later_transactions(roster_job, monkeypatch):
    archive = NBARosterArchive()
    def fail(*args, **kwargs):raise AssertionError("A genuine pregame roster must not be replaced by later data")
    monkeypatch.setattr(NBARosterArchive, "game_evidence", fail)
    projection = {"purpose": "pregame_projection", "season": SEASON, "game_id": GAME_NOW["game_id"],
                  "scheduled_start_utc": START.isoformat(),
                  "teams": {str(FIRST): {"players": [{"player_id": 1}]}, str(SECOND): {"players": [{"player_id": 2}]}},
                  "inputs": {"rosters": {"uri": "original", "sha256": "a"*64}}}
    selected = archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON),
        pregame={GAME_NOW["game_id"]: {"projection": projection, "reference": {"uri": "pregame", "sha256": "b"*64}}})
    assert selected[GAME_NOW["game_id"]]["teams"] == {str(FIRST): [1], str(SECOND): [2]}
    assert selected[GAME_NOW["game_id"]]["cutoff_selection"]["selection_kind"] == "saved_pregame"


def test_d_plus_one_warning_is_explicit_and_uses_existing_rosters_error_path(roster_job):
    _, objects, _, writes, _ = roster_job
    archive = NBARosterArchive()
    selections = archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON))
    ref = archive.save_cutoff_warning(season=SEASON, day="2026-11-08", selections=selections,
        run_id="cutoff-run", consumer="update_player_minutes", occurred_at=START+timedelta(days=2))
    key = ref["uri"].removeprefix("s3://nba-gestalt-data/")
    assert key.startswith("lake/artifacts/errors/pipeline=rosters/season=2026-27/") and key.endswith(".warning.json")
    warning = json.loads(objects[key])
    assert warning["warning_type"] == "date_only_transaction_d_plus_1" and warning["status"] == "warning"
    assert warning["assumptions"][0]["game_id"] == GAME_NOW["game_id"]
    assert warning["consumer"] == "update_player_minutes" and warning["dagster_run_id"] == "cutoff-run"
    assert len(writes) == 1


@pytest.mark.parametrize("action", ["Signing", "Waive", "Trade"])
def test_late_moves_do_not_change_same_day_rosters(roster_job, action):
    archive = NBARosterArchive();source = evidence(archive)
    row = source["transactions"][0]
    row["Transaction_Type"] = action
    if action == "Signing": row["PLAYER_ID"] = 99999;row["TEAM_ID"] = FIRST
    if action == "Waive": row["TEAM_ID"] = FIRST
    today = game_roster(season=SEASON, game=GAME_NOW, **source)
    tomorrow = game_roster(season=SEASON, game={**GAME_NOW, "game_time_utc": START+timedelta(days=1)}, **source)
    pid, team = row["PLAYER_ID"], str(row["TEAM_ID"])
    assert (pid in today["teams"][team]) == (action == "Waive")
    assert (pid in tomorrow["teams"][team]) == (action != "Waive")


def test_d_plus_one_does_not_extend_a_ten_day_contract(roster_job):
    archive = NBARosterArchive();source = evidence(archive)
    source["opening"]["date"] = date(2026, 10, 19)
    source["transactions"] = [{"Transaction_Type": "Signing", "TRANSACTION_DATE": "2026-10-29T00:00:00",
        "PLAYER_ID": 99999, "TEAM_ID": FIRST, "first_observed_at": "2026-11-11T00:00:00Z", "TRANSACTION_DESCRIPTION": "Signed a 10-day contract"}]
    source["contract_games"] = [{**GAME_NOW, "game_time_utc": datetime(2026, 10, 29, 23, tzinfo=UTC)+timedelta(days=i)} for i in [0, 3, 6]]
    last_active = game_roster(season=SEASON, game={**GAME_NOW, "game_time_utc": START-timedelta(days=1)}, **source)
    expired = game_roster(season=SEASON, game=GAME_NOW, **source)
    assert 99999 in last_active["teams"][str(FIRST)]
    assert 99999 not in expired["teams"][str(FIRST)]
    assert expired["ten_day_contracts"][0]["removal_date"] == "2026-11-08"


def test_source_guard_detects_a_changed_transaction_file(roster_job):
    _, objects, _, _, _ = roster_job
    archive = NBARosterArchive()
    selections = archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON))
    key = "lake/raw/nba/transactions/season=2026-27/game_date_pacific=2026-11-08/latest.json"
    changed = json.loads(objects[key]);changed["transactions"][0]["TEAM_ID"] = FIRST
    objects[key] = json.dumps(changed).encode()
    with pytest.raises(ValueError, match="Transaction evidence changed"):
        archive.assert_game_selections_current(SEASON, selections)


class FinalBoxes:
    bucket = "nba-gestalt-data"
    def __init__(self, *, listed=True, available=True):
        self.calls = [];self.available = available
        self.box = {"gameId": GAME_NOW["game_id"], "gameStatus": 3,
                    "homeTeam": {"teamId": FIRST, "players": [{"personId": 2}]},
                    "awayTeam": {"teamId": SECOND, "players": [{"personId": 3}]}}
        if listed:self.box["awayTeam"]["players"].append({"personId": 1, "statistics": {"minutes": "PT00M00.00S"}})
        self.reference = {"uri": "s3://nba-gestalt-data/fixture/final-box.json", "sha256": "b"*64}
    def read_saved(self, season, game, *, allow_missing=False):
        assert allow_missing
        self.calls.append(game["game_id"])
        return (json.dumps({"game": self.box}).encode(), self.reference) if self.available else None


@pytest.mark.parametrize("action", ["Signing", "Waive", "Trade", "AwardOnWaivers", "ContractConverted"])
@pytest.mark.parametrize("listed", [True, False])
def test_historical_same_day_transactions_use_box_membership_only(roster_job, action, listed):
    archive = NBARosterArchive();source = evidence(archive)
    source["transactions"][0]["Transaction_Type"] = action
    boxes = FinalBoxes(listed=listed)
    result = archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON), evidence=source, box_archive=boxes)[GAME_NOW["game_id"]]
    assert boxes.calls == [GAME_NOW["game_id"]]
    assert (1 in result["teams"][str(SECOND)]) == listed
    assert 1 not in result["teams"][str(FIRST)]
    # Player 4 has no transaction and remains rostered despite absence from this box.
    assert 4 in result["teams"][str(SECOND)]
    proof = result["cutoff_selection"]["historical_box_confirmation"]
    assert proof["status"] == "confirmed" and proof["decisions"][0]["included_for_game"] == listed
    assert result["cutoff_selection"]["inputs"]["final_box_score"] == boxes.reference
    assert not result["cutoff_selection"]["assumptions"]


def test_box_is_not_read_without_a_same_day_transaction(roster_job):
    archive = NBARosterArchive();source = evidence(archive)
    source["transactions"][0]["TRANSACTION_DATE"] = "2026-11-07T00:00:00"
    boxes = FinalBoxes()
    archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON), evidence=source, box_archive=boxes)
    assert boxes.calls == []


def test_box_is_not_read_for_live_roster_selection(roster_job, monkeypatch):
    from gestalt_pipelines.rosters.resources import archive as module
    archive = NBARosterArchive();source = evidence(archive)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):return (START-timedelta(hours=1)).astimezone(tz or UTC)
    monkeypatch.setattr(module, "datetime", Clock)
    boxes = FinalBoxes()
    archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON), evidence=source, box_archive=boxes)
    assert boxes.calls == []


def test_no_final_box_preserves_d_plus_one_warning_and_guards_appearance(roster_job):
    archive = NBARosterArchive();boxes = FinalBoxes(available=False)
    result = archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON), box_archive=boxes)
    selected = result[GAME_NOW["game_id"]]
    assert 1 in selected["teams"][str(FIRST)]
    assert selected["cutoff_selection"]["assumptions"]
    archive.assert_game_selections_current(SEASON, result, boxes, [GAME_NOW])
    boxes.available = True
    with pytest.raises(ValueError, match="Final box evidence changed"):
        archive.assert_game_selections_current(SEASON, result, boxes, [GAME_NOW])


@pytest.mark.parametrize("invalid", ["not_final", "wrong_game", "wrong_team", "missing_players"])
def test_invalid_existing_boxes_do_not_fall_back_to_an_assumption(roster_job, invalid):
    archive = NBARosterArchive();boxes = FinalBoxes()
    if invalid == "not_final":boxes.box["gameStatus"] = 2
    elif invalid == "wrong_game":boxes.box["gameId"] = "different"
    elif invalid == "wrong_team":boxes.box["homeTeam"]["teamId"] = SECOND
    else:boxes.box["homeTeam"]["players"] = []
    with pytest.raises(RosterError):
        archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON), box_archive=boxes)


def test_box_is_not_read_for_an_unrelated_team_transaction(roster_job):
    archive = NBARosterArchive(); source = evidence(archive)
    source["transactions"][0].update(PLAYER_ID=99999, TEAM_ID=FIRST+2)
    boxes = FinalBoxes()
    archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON), evidence=source, box_archive=boxes)
    assert boxes.calls == []


def test_trade_away_checks_the_prior_team_even_when_current_owner_is_elsewhere(roster_job):
    archive = NBARosterArchive(); source = evidence(archive)
    source["transactions"][0].update(TEAM_ID=FIRST+2, first_observed_at=(START-timedelta(minutes=1)).isoformat())
    boxes = FinalBoxes(listed=False)
    boxes.box["homeTeam"]["players"].append({"personId": 1})
    result = archive.for_games(SEASON, [GAME_NOW], NBAScheduleResource(season=SEASON), evidence=source, box_archive=boxes)[GAME_NOW["game_id"]]
    assert boxes.calls == [GAME_NOW["game_id"]]
    assert 1 in result["teams"][str(FIRST)]
    assert 1 not in result["teams"][str(FIRST+2)]
