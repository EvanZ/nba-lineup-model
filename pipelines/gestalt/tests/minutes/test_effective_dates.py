"""Effective-date boundaries, old selections, and pregame consumer safety."""

from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
import json

import pytest
from dagster import AssetKey

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.minutes.assets import player_minutes
from gestalt_pipelines.minutes.partitions import minutes_dates
from gestalt_pipelines.minutes.utils import MinutesError, json_bytes, update_state, validate_state
from gestalt_pipelines.utils import PACIFIC

from .helpers import FIRST, FIRST_STATE, SEASON, priors, slate
from .test_asset import minutes_job
from .test_projection import example


def test_first_updated_file_and_today_are_available():
    keys = minutes_dates.get_partition_keys(current_time=datetime(2026, 11, 14, 6, tzinfo=PACIFIC))
    assert keys[0] == "2026-10-21"
    assert keys[-1] == "2026-11-14"
    assert len(keys) == 25


@pytest.mark.parametrize("source", ["nba_stints", "rosters_by_date", "injury_status_by_game_date"])
def test_november_14_maps_to_november_13_inputs(source):
    upstream = defs.get_assets_def(AssetKey(source)).partitions_def
    mapping = player_minutes.get_partition_mapping(AssetKey(source))
    result = mapping.get_upstream_mapped_partitions_result_for_partitions(
        downstream_partitions_subset=minutes_dates.empty_subset().with_partition_keys(["2026-11-14"]),
        downstream_partitions_def=minutes_dates, upstream_partitions_def=upstream,
        current_time=datetime(2026, 11, 14, 6, tzinfo=PACIFIC),
    )
    assert set(result.partitions_subset.get_partition_keys()) == {"2026-11-13"}
    assert not result.required_but_nonexistent_subset.get_partition_keys()


def test_first_updated_state_uses_opening_day_games_and_preserves_introduction_date():
    result = update_state(priors=priors(), parent=None, **slate())
    assert result["game_date_pacific"] == "2026-10-21"
    assert result["observed_through_game_date_pacific"] == "2026-10-20"
    assert result["players"]["1"]["introduced_on"] == "2026-10-20"
    assert {row["game_date_pacific"] for row in result["observations"]} == {"2026-10-20"}


@pytest.mark.parametrize("source", ["rosters", "stints", "injuries", "game"])
def test_updater_rejects_evidence_from_its_effective_date(source):
    data = slate()
    if source == "game":
        data["games"][0]["game_time_utc"] += timedelta(days=1)
    else:
        data[source]["game_date_pacific"] = str(FIRST_STATE)
    with pytest.raises(MinutesError, match="season/date|slate|different Pacific date"):
        update_state(priors=priors(), parent=None, **data)


def test_old_observation_date_state_and_future_history_are_rejected():
    state = update_state(priors=priors(), parent=None, **slate())
    old = deepcopy(state)
    del old["observed_through_game_date_pacific"]
    with pytest.raises(MinutesError, match="previous-day cutoff"):
        validate_state(old, SEASON)
    state["players"]["1"]["history"][0]["game_date_pacific"] = str(FIRST_STATE)
    with pytest.raises(MinutesError, match="Future or unprocessed"):
        validate_state(state, SEASON)


def test_normal_publication_replaces_old_current_date_without_deleting_it(minutes_job):
    job, archive, objects, _, _ = minutes_job
    assert job.execute_in_process(partition_key=str(FIRST_STATE)).success
    state, _, _ = archive.state(str(FIRST_STATE))
    del state["observed_through_game_date_pacific"]
    raw = json_bytes(state)
    digest = sha256(raw).hexdigest()
    prefix = archive.prefix+f"game_date_pacific={FIRST_STATE}/"
    objects[prefix+digest+".json"] = raw
    objects[prefix+"latest.json"] = json_bytes({
        "schema_version":1, "season":SEASON, "model_version":"v1", "game_date_pacific":str(FIRST_STATE),
        "uri":f"s3://nba-gestalt-data/{prefix}{digest}.json", "sha256":digest,
    })
    with pytest.raises(MinutesError, match="previous-day cutoff"):
        archive.state(str(FIRST_STATE))
    assert job.execute_in_process(partition_key=str(FIRST_STATE)).success
    corrected, _, _ = archive.state(str(FIRST_STATE))
    assert corrected["observed_through_game_date_pacific"] == str(FIRST)
    assert objects[prefix+digest+".json"] == raw


def test_projection_accepts_same_effective_date_but_not_future_or_old_format():
    from gestalt_pipelines.minutes.utils import build_game_projection
    data = example()
    record = build_game_projection(**data)
    assert data["baseline"]["game_date_pacific"] == data["day"]
    assert record["baseline_through_date"] == str(FIRST)
    old = deepcopy(data)
    del old["baseline"]["observed_through_game_date_pacific"]
    with pytest.raises(MinutesError, match="mismatched season/date"):
        build_game_projection(**old)
    data["baseline"]["game_date_pacific"] = str(FIRST_STATE+timedelta(days=1))
    data["baseline"]["observed_through_game_date_pacific"] = str(FIRST_STATE)
    with pytest.raises(MinutesError, match="mismatched season/date"):
        build_game_projection(**data)
