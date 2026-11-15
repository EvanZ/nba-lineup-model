import json
from copy import deepcopy
from datetime import UTC, datetime

from gestalt_pipelines.minutes.utils import update_state
from gestalt_pipelines.rosters.resources import NBARosterArchive
from .helpers import FIRST, FIRST_STATE, HOME, priors, slate
from .test_asset import minutes_job


def test_zero_observations_use_the_game_roster_not_the_end_of_day_catalog():
    data = slate();gid = data["games"][0]["game_id"]
    selected = deepcopy(data["game_rosters"][gid])
    selected["teams"][str(HOME)].remove(15)
    data["game_rosters"][gid] = selected
    state = update_state(priors=priors(), parent=None, **data)
    assert 15 in data["rosters"]["teams"][str(HOME)]
    assert all(row["player_id"] != 15 for row in state["observations"])
    assert state["players"]["15"]["history"] == []


def test_d_plus_one_warning_is_linked_in_a_successful_minutes_run(minutes_job, monkeypatch):
    job, archive, objects, _, _ = minutes_job
    original = NBARosterArchive.for_games
    def select(self, *args, **kwargs):
        values = original(self, *args, **kwargs)
        for gid, value in values.items():
            value["cutoff_selection"]["assumptions"] = [{"game_id": gid, "player_id": 15, "team_id": HOME,
                "reported_date_pacific": str(FIRST), "assumed_application_date_pacific": "2026-10-21",
                "included_for_game": False, "reason": "Date-only transaction; D+1 rule"}]
        return values
    monkeypatch.setattr(NBARosterArchive, "for_games", select)
    result = job.execute_in_process(partition_key=str(FIRST_STATE))
    assert result.success
    meta = result.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    ref = meta["roster_cutoff_warning"].value
    assert "/pipeline=rosters/" in ref["uri"] and ref["uri"].endswith(".warning.json")
    event = json.loads(objects[ref["uri"].removeprefix("s3://nba-gestalt-data/")])
    assert event["consumer"] == "update_player_minutes" and event["status"] == "warning"
    assert event["assumptions"][0]["assumed_application_date_pacific"] == "2026-10-21"
