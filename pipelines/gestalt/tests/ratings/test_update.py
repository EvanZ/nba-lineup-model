"""Joint evidence accumulation, roster membership and date lineage."""

from copy import deepcopy
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from gestalt_pipelines.ratings.utils import json_bytes, update_state

FIRST = date(2026, 10, 20)
DAY = FIRST
OBSERVED_DAY = DAY


@pytest.fixture
def daily(record):
    # Small numerical fixture; integration tests below use the full 524-row artifact.
    priors = deepcopy(record)
    priors["players"] = priors["players"][:12]
    game = {"game_id": "0022600001", "game_time_utc": datetime(2026,10,21,0,tzinfo=UTC),
            "home_team_id": 1610612737, "away_team_id": 1610612738}
    rows = [
        {"game_id": game["game_id"], "home_player_ids": [1,2,3,4,5], "away_player_ids": [6,7,8,9,10],
         "possessions": 12.5, "target_home_net_rating": 16.0, "period": 1},
        {"game_id": game["game_id"], "home_player_ids": [1,2,3,4,11], "away_player_ids": [6,7,8,9,10],
         "possessions": 3.25, "target_home_net_rating": -20.0, "period": 5},
        {"game_id": game["game_id"], "home_player_ids": [1,2,3,4,5], "away_player_ids": [6,7,8,9,10],
         "possessions": 0.0, "target_home_net_rating": None, "period": 5},
    ]
    data = {
        "day": DAY, "first_date": FIRST, "season": "2026-27", "priors": priors, "parent": None,
        "games": [game], "previous_games": [],
        "rosters": {"season": "2026-27", "game_date_pacific": str(OBSERVED_DAY),
                    "teams": {str(game["home_team_id"]): [1,2,3,4,5,11,12],
                              str(game["away_team_id"]): [6,7,8,9,10]}},
        "stints": {"season": "2026-27", "game_date_pacific": str(OBSERVED_DAY), "rows": rows,
                   "validation": {"complete": True, "expected_game_ids": [game["game_id"]],
                                  "processed_game_ids": [game["game_id"]]}},
        "inputs": {"priors": {"uri": "s3://nba-gestalt-data/test/priors.json", "sha256": "a"*64},
                   "parent": None, "stints": {"sha256": "b"*64}, "rosters": {"sha256": "c"*64},
                   "schedule": {"sha256": "d"*64}},
    }
    return data



def next_date(daily, parent, *, no_games=False):
    data = deepcopy(daily); data["day"] += timedelta(days=1); data["parent"] = parent
    data["inputs"]["parent"] = {"sha256": "e"*64}
    for key in ("stints", "rosters"):
        data[key]["game_date_pacific"] = str(data["day"])
    if no_games:
        data["games"] = []; data["stints"]["rows"] = []
        data["stints"]["validation"].update(expected_game_ids=[], processed_game_ids=[])
    else:
        data["games"][0]["game_id"] = "0022600002"
        data["games"][0]["game_time_utc"] += timedelta(days=1)
        for row in data["stints"]["rows"]: row["game_id"] = "0022600002"
        data["stints"]["validation"].update(expected_game_ids=["0022600002"], processed_game_ids=["0022600002"])
    return data


def test_joint_solve_matches_direct_normal_equations(daily):
    result = update_state(**daily); stats = result["statistics"]
    gram, score = np.array(stats["gram"]), np.array(stats["score"])
    n = len(result["players"]); settings = daily["priors"]["settings"]
    penalty = np.array([settings["player_prior_equivalent_possessions"]]*n+[settings["hca"]["prior_equivalent_possessions"]])
    score[-1] += penalty[-1]*settings["hca"]["prior_mean"]
    expected = np.linalg.solve(gram+np.diag(penalty), score)
    actual = [p["residual"] for p in result["players"]]+[result["hca"]["estimate"]]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert stats["exposure"][-1] == 15.75
    assert result["validation"]["games"]["0022600001"]["positive_stint_count"] == 2
    assert result["validation"]["games"]["0022600001"]["zero_possession_stint_count"] == 1
    assert any(gram[i,-1] != 0 for i in range(n))


def test_second_date_adds_evidence_without_changing_starting_priors(daily):
    first = update_state(**daily); second = update_state(**next_date(daily, first))
    for key in ("gram", "score", "exposure"):
        np.testing.assert_allclose(second["statistics"][key], 2*np.array(first["statistics"][key]), atol=1e-12)
    assert [p["starting_nail"] for p in second["players"]] == [p["starting_nail"] for p in first["players"]]
    assert len(second["included_game_ids"]) == 2


def test_identical_rerun_does_not_double_apply_games(daily):
    first = update_state(**daily)
    assert json_bytes(update_state(**daily)) == json_bytes(first)


def test_new_catalog_member_preserves_hca_cross_terms(daily):
    first = update_state(**daily); data = next_date(daily, first, no_games=True)
    data["rosters"]["teams"]["1610612737"].append(9999)
    second = update_state(**data)
    remap = list(range(12))+[13]
    np.testing.assert_array_equal(np.array(second["statistics"]["gram"])[np.ix_(remap, remap)], first["statistics"]["gram"])
    assert second["hca"]["estimate"] == pytest.approx(first["hca"]["estimate"], abs=1e-12)
    assert second["players"][-1]["player_id"] == 9999
    assert second["players"][-1]["residual"] == 0.0
    assert second["included_game_ids"] == first["included_game_ids"]


def test_valid_played_evidence_is_not_rejected_by_end_of_day_roster(daily):
    daily["rosters"]["teams"]["1610612737"].remove(5)
    result = update_state(**daily)
    player = next(p for p in result["players"] if p["player_id"] == 5)
    assert player["on_court_possessions"] == 12.5
    assert player["team_id"] is None


def test_new_id_gets_replacement_without_stat_profile(daily):
    daily["rosters"]["teams"]["1610612737"].append(9999)
    result = update_state(**daily)
    player = next(p for p in result["players"] if p["player_id"] == 9999)
    assert player["starting_nail"] == daily["priors"]["settings"]["replacement_prior"]
    assert player["forecast_source"] == "in_season_replacement_cold_start_prior"
    assert player["initialization"]["kind"] == "replacement"
    assert "context_profile" not in player
    assert "profile_source" not in player["initialization"]
    assert result["new_replacement_player_ids"] == [9999]


@pytest.mark.parametrize("missing", ["coverage", "positive_evidence", "lineup"])
def test_missing_required_evidence_is_not_silently_removed(daily, missing):
    if missing == "coverage":
        daily["stints"]["validation"]["processed_game_ids"] = []
    elif missing == "positive_evidence":
        daily["stints"]["rows"] = daily["stints"]["rows"][-1:]
    else:
        daily["stints"]["rows"][0]["home_player_ids"] = [1, 2, 3, 4, 4]
    with pytest.raises(ValueError, match="complete expected slate|no positive-possession|five unique players"):
        update_state(**daily)


def test_missing_or_incompatible_parent_fails(daily):
    first = update_state(**daily); data = next_date(daily, first)
    data["parent"] = None
    with pytest.raises(ValueError, match="preceding ratings state"):
        update_state(**data)
    data["parent"] = first
    data["inputs"]["priors"] = {"sha256": "f"*64}
    with pytest.raises(ValueError, match="wrong date or frozen initialization"):
        update_state(**data)


def test_fixed_b2b_is_separate_from_joint_hca(daily):
    baseline = update_state(**daily)
    daily["previous_games"] = [{"home_team_id": 1610612737, "away_team_id": 1610612739}]
    changed = update_state(**daily)
    np.testing.assert_array_equal(changed["statistics"]["gram"], baseline["statistics"]["gram"])
    shift = -daily["priors"]["settings"]["fixed_back_to_back"]*baseline["statistics"]["exposure"][-1]
    assert changed["statistics"]["score"][-1]-baseline["statistics"]["score"][-1] == pytest.approx(shift)
    assert changed["settings"] == baseline["settings"]


def test_daily_calculation_ignores_unused_saved_profiles(daily, monkeypatch):
    import builtins
    expected = update_state(**daily)
    daily["priors"].update(profiles="unused", context="unused", aging="unused")
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.split(".")[0] in {"joblib", "sklearn", "nba_lineup_model"}:
            raise AssertionError("Daily ratings must use numerical priors and stints only")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)
    assert json_bytes(update_state(**daily)) == json_bytes(expected)


@pytest.fixture
def manual_job(daily, archive, monkeypatch, record):
    from hashlib import sha256
    from dagster import Definitions
    from gestalt_pipelines.ratings.assets import player_ratings
    from gestalt_pipelines.ratings.jobs import update_player_ratings
    from gestalt_pipelines.resources import NBAPlayerNamesResource, NBAScheduleResource
    resource, s3 = archive
    saved = resource.save_priors(record, prior=None, expected_etag=None)
    reference = {key: saved[key] for key in ("uri", "sha256")}
    for name in ("stints", "rosters"):
        prefix = f"lake/artifacts/{name}/season=2026-27/game_date_pacific={OBSERVED_DAY}/"
        raw = json_bytes(daily[name]); digest = sha256(raw).hexdigest()
        s3.objects[prefix+digest+".json"] = raw
        s3.objects[prefix+"latest.json"] = json_bytes({
            "schema_version": 1, "season": "2026-27", "game_date_pacific": str(OBSERVED_DAY),
            "uri": f"s3://nba-gestalt-data/{prefix}{digest}.json", "sha256": digest,
        })
    ref = {"uri": "s3://nba-gestalt-data/test/schedule.json", "sha256": "d"*64}
    monkeypatch.setattr(NBAScheduleResource, "games_between", lambda self,*args,**kwargs: (daily["games"],ref))
    monkeypatch.setattr(NBAScheduleResource, "load", lambda self,*args,**kwargs: (b"{}",ref))
    monkeypatch.setattr(NBAPlayerNamesResource, "load", lambda self: (
        {9999: "Test Newcomer"}, {"uri": "s3://nba-gestalt-data/test/player_names.json", "sha256": "f"*64},
    ))
    defs = Definitions(assets=[player_ratings], jobs=[update_player_ratings],
                       resources={"ratings_archive": resource, "nba_schedule": NBAScheduleResource(season="2026-27"),
                                  "nba_player_names": NBAPlayerNamesResource(season="2026-27")})
    return defs.resolve_job_def("update_player_ratings"), resource, s3


def test_manual_job_publishes_and_identical_rerun_reuses(manual_job):
    from dagster import DagsterInstance
    job, resource, s3 = manual_job
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(instance=instance, partition_key=str(DAY))
        assert result.success
        before = dict(s3.objects)
        result = job.execute_in_process(instance=instance, partition_key=str(DAY))
        assert result.success and s3.objects == before
    record, reference, _ = resource.state(str(DAY))
    assert record["validation"]["complete"] and len(record["included_game_ids"]) == 1
    assert reference["uri"].startswith(f"s3://nba-gestalt-data/{resource.state_prefix}game_date_pacific={DAY}/")


def test_manual_job_failure_logs_and_real_retry_records_recovery(manual_job):
    import json
    from dagster import DagsterInstance
    from dagster._core.definitions.job_base import InMemoryJob
    from dagster._core.execution.api import execute_run
    job, resource, s3 = manual_job
    key = f"lake/artifacts/stints/season=2026-27/game_date_pacific={OBSERVED_DAY}/latest.json"
    saved = s3.objects.pop(key)
    with DagsterInstance.ephemeral() as instance:
        failed = job.execute_in_process(instance=instance, partition_key=str(DAY), raise_on_error=False)
        assert not failed.success
        assert resource.state_selection(str(DAY))[0] is None
        [failure] = [json.loads(raw) for name,raw in s3.objects.items() if name.endswith(".failure.json")]
        assert failure["stage"] == "player_ratings" and failure["game_date_pacific"] == str(DAY)
        s3.objects[key] = saved
        parent = instance.get_run_by_id(failed.run_id)
        child = instance.create_run_for_job(job, run_config=parent.run_config, tags=parent.tags,
                                           parent_run_id=parent.run_id, root_run_id=parent.run_id)
        recovered = execute_run(InMemoryJob(job), child, instance)
        assert recovered.success
    [recovery] = [json.loads(raw) for name,raw in s3.objects.items() if name.endswith(".recovery.json")]
    assert recovery["stage"] == "player_ratings" and recovery["dagster_run_id"] == recovered.run_id


def test_failed_rerun_preserves_accepted_daily_selection(manual_job):
    from dagster import DagsterInstance
    job, resource, s3 = manual_job
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(instance=instance, partition_key=str(DAY)).success
        selected = resource.state_selection(str(DAY))
        del s3.objects[f"lake/artifacts/rosters/season=2026-27/game_date_pacific={OBSERVED_DAY}/latest.json"]
        failed = job.execute_in_process(instance=instance, partition_key=str(DAY), raise_on_error=False)
        assert not failed.success
        assert resource.state_selection(str(DAY)) == selected


def test_competing_daily_publication_cannot_replace_selection(manual_job):
    from dagster import DagsterInstance
    job, resource, s3 = manual_job
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(instance=instance, partition_key=str(DAY)).success
    record, reference, etag = resource.state(str(DAY))
    before = s3.objects[resource.state_prefix+f"game_date_pacific={DAY}/latest.json"]
    record["implementation"]["test_correction"] = True
    s3.race = True
    with pytest.raises(ValueError, match="competing ratings run"):
        resource.save_state(record, prior=reference, expected_etag=etag)
    assert s3.objects[resource.state_prefix+f"game_date_pacific={DAY}/latest.json"] == before


def test_delayed_catalog_arrival_uses_saved_prior(daily):
    daily["rosters"]["teams"]["1610612737"].remove(12)
    first = update_state(**daily)
    data = next_date(daily, first, no_games=True)
    data["rosters"]["teams"]["1610612737"].append(12)
    result = update_state(**data)
    player = next(p for p in result["players"] if p["player_id"] == 12)
    expected = next(p for p in daily["priors"]["players"] if p["player_id"] == 12)
    assert player["starting_nail"] == expected["rapm"]
    assert player["forecast_source"] == expected["forecast_source"]
    assert player["initialization"]["kind"] == "frozen_catalog"
    assert result["new_replacement_player_ids"] == []


def test_replacement_state_survives_absence_and_return(daily):
    daily["rosters"]["teams"]["1610612737"].append(9999)
    for row in daily["stints"]["rows"]:
        row["home_player_ids"] = [9999 if i == 5 else i for i in row["home_player_ids"]]
    first = update_state(**daily)
    original = next(p for p in first["players"] if p["player_id"] == 9999)
    assert original["residual"] != 0
    absent = next_date(daily, first, no_games=True)
    absent["rosters"]["teams"]["1610612737"].remove(9999)
    second = update_state(**absent)
    returning = next_date(absent, second, no_games=True)
    returning["rosters"]["teams"]["1610612737"].append(9999)
    third = update_state(**returning)
    player = next(p for p in third["players"] if p["player_id"] == 9999)
    for key in ("starting_nail", "forecast_source", "initialized_on", "initialization"):
        assert player[key] == original[key]
    assert player["residual"] == pytest.approx(original["residual"], abs=1e-12)
    assert second["new_replacement_player_ids"] == third["new_replacement_player_ids"] == []


def add_new_player_to_saved_test_roster(s3):
    import json
    from hashlib import sha256
    prefix = f"lake/artifacts/rosters/season=2026-27/game_date_pacific={OBSERVED_DAY}/"
    pointer = json.loads(s3.objects[prefix+"latest.json"])
    record = json.loads(s3.objects[prefix+pointer["sha256"]+".json"])
    record["teams"]["1610612737"].append(9999)
    raw = json_bytes(record); digest = sha256(raw).hexdigest()
    s3.objects[prefix+digest+".json"] = raw
    s3.objects[prefix+"latest.json"] = json_bytes({
        "schema_version": 1, "season": "2026-27", "game_date_pacific": str(OBSERVED_DAY),
        "uri": f"s3://nba-gestalt-data/{prefix}{digest}.json", "sha256": digest,
    })


def test_new_replacement_warning_is_visible_and_nonfatal(manual_job):
    import json
    from dagster import DagsterInstance
    job, resource, s3 = manual_job
    add_new_player_to_saved_test_roster(s3)
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(instance=instance, partition_key=str(DAY))
        assert result.success
        messages = [entry.message for entry in instance.all_logs(result.run_id)]
        assert any("Test Newcomer" in message and "replacement starting NAIL" in message for message in messages)
    warnings = [(key,json.loads(raw)) for key,raw in s3.objects.items() if key.endswith(".warning.json")]
    assert len(warnings) == 1
    key, warning = warnings[0]
    assert key.startswith("lake/artifacts/errors/pipeline=ratings/season=2026-27/occurred_date_pacific=")
    assert warning["status"] == "warning" and warning["stage"] == "player_ratings"
    assert warning["player_id"] == 9999 and warning["player_name"] == "Test Newcomer"
    assert warning["dagster_run_id"] == result.run_id and warning["game_date_pacific"] == str(DAY)
    record,_,_ = resource.state(str(DAY))
    player = next(p for p in record["players"] if p["player_id"] == 9999)
    assert warning["starting_nail"] == player["starting_nail"] == record["settings"]["replacement_prior"]
    assert record["inputs"]["player_names"]["sha256"] == "f"*64


def test_warning_write_failure_does_not_publish_unreviewable_assignment(manual_job, monkeypatch):
    from botocore.exceptions import ClientError
    from dagster import DagsterInstance
    job, resource, s3 = manual_job
    add_new_player_to_saved_test_roster(s3)
    original = s3.put_object
    def put(**kwargs):
        if kwargs["Key"].endswith(".warning.json"):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
        return original(**kwargs)
    monkeypatch.setattr(s3, "put_object", put)
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(instance=instance, partition_key=str(DAY), raise_on_error=False)
        assert not result.success
    assert resource.state_selection(str(DAY))[0] is None
    assert any(key.endswith(".failure.json") for key in s3.objects)


def test_first_completed_date_includes_games_and_retains_the_entire_catalog(daily, record):
    daily["priors"] = record
    result = update_state(**daily)
    assert len(result["players"]) == 524
    assert result["included_game_ids"] == ["0022600001"]
    assert result["statistics"]["exposure"][-1] == 15.75
    assert not result["validation"]["initialization_only"]
    unused = next(p for p in result["players"] if p["player_id"] == 12)
    assert unused["nail"] == unused["starting_nail"]


def test_completed_date_uses_its_own_observations(daily):
    result = update_state(**daily)
    assert result["game_date_pacific"] == result["observed_through_game_date_pacific"] == "2026-10-20"
    assert result["effective_from_game_date_pacific"] == "2026-10-21"


@pytest.mark.parametrize("wrong", ["stints", "rosters", "game"])
def test_other_day_evidence_is_rejected(daily, wrong):
    if wrong == "game":
        daily["games"][0]["game_time_utc"] += timedelta(days=1)
    else:
        daily[wrong]["game_date_pacific"] = str(DAY+timedelta(days=1))
    with pytest.raises(ValueError, match="selected completed game date|another game date"):
        update_state(**daily)


def test_first_completed_day_requires_observations(daily):
    daily["stints"] = None
    with pytest.raises(ValueError, match="selected completed game date"):
        update_state(**daily)


def test_partition_includes_yesterday_but_not_today():
    from gestalt_pipelines.ratings.partitions import ratings_dates
    from gestalt_pipelines.utils import PACIFIC
    keys = ratings_dates.get_partition_keys(current_time=datetime(2026,11,14,4,15,tzinfo=PACIFIC))
    assert keys[-1] == "2026-11-13" and keys[0] == "2026-10-20"


def test_first_completed_job_reads_same_day_game_inputs(manual_job):
    from dagster import DagsterInstance
    job, resource, _ = manual_job
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(instance=instance, partition_key=str(FIRST)).success
    state,_,_ = resource.state(str(FIRST))
    assert not state["validation"]["initialization_only"] and len(state["players"]) == 524
    assert state["included_game_ids"] == ["0022600001"]


def install_old_date_selection(resource, s3, day, template):
    import json
    from hashlib import sha256
    record = deepcopy(template)
    record["game_date_pacific"] = str(day)
    del record["observed_through_game_date_pacific"]
    raw = json_bytes(record); digest = sha256(raw).hexdigest()
    prefix = resource.state_prefix+f"game_date_pacific={day}/"
    s3.objects[prefix+digest+".json"] = raw
    s3.objects[prefix+"latest.json"] = json_bytes({
        "schema_version": 1, "season": "2026-27", "model_version": "v1", "game_date_pacific": str(day),
        "uri": f"s3://nba-gestalt-data/{prefix}{digest}.json", "sha256": digest,
    })


def test_old_observation_date_parent_is_rejected(manual_job, daily):
    from dagster import DagsterInstance
    job, resource, s3 = manual_job
    install_old_date_selection(resource, s3, FIRST, update_state(**daily))
    with DagsterInstance.ephemeral() as instance:
        failed = job.execute_in_process(instance=instance, partition_key=str(DAY+timedelta(days=1)), raise_on_error=False)
        assert not failed.success
    assert resource.state_selection(str(DAY+timedelta(days=1)))[0] is None


def test_normal_publication_can_replace_old_current_date_after_parent_is_corrected(manual_job, daily):
    from dagster import DagsterInstance
    job, resource, s3 = manual_job
    install_old_date_selection(resource, s3, DAY, update_state(**daily))
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(instance=instance, partition_key=str(DAY)).success
    state,_,_ = resource.state(str(DAY))
    assert state["observed_through_game_date_pacific"] == str(OBSERVED_DAY)


def test_asset_backfill_failure_is_scoped_to_ratings(manual_job):
    from types import SimpleNamespace
    from dagster import AssetKey, DagsterInstance
    job, resource, s3 = manual_job
    del s3.objects[f"lake/artifacts/stints/season=2026-27/game_date_pacific={OBSERVED_DAY}/latest.json"]
    with DagsterInstance.ephemeral() as instance:
        failed = job.execute_in_process(instance=instance, partition_key=str(DAY), raise_on_error=False)
        parent = instance.get_run_by_id(failed.run_id)._replace(
            job_name="__ASSET_JOB", asset_selection=frozenset({AssetKey("player_ratings")}),
        )
        context = SimpleNamespace(run=SimpleNamespace(parent_run_id=parent.run_id),
            instance=SimpleNamespace(get_run_by_id=lambda _:parent,get_records_for_run=instance.get_records_for_run))
        assert resource.parent_failure(context,stage="player_ratings",day=str(DAY))


def test_single_stint_uses_only_frozen_nail_b2b_and_joint_hca(daily):
    from gestalt_pipelines.ratings.utils import game_contribution, solve_joint
    ids = list(range(1, 11))
    starting = {i: 2.0 if i <= 5 else 0.5 for i in ids}
    row = {"home_player_ids": ids[:5], "away_player_ids": ids[5:],
           "possessions": 10.0, "target_home_net_rating": 20.0}
    contribution = game_contribution([row], player_ids=ids, starting=starting, b2b=-1.5)
    # Signed frozen NAIL sum = 7.5; scoring residual = 20 - (-1.5) - 7.5 = 14.
    expected_score = [140.0]*5+[-140.0]*5+[140.0]
    np.testing.assert_array_equal(contribution["score"], expected_score)
    np.testing.assert_array_equal(np.diag(contribution["gram"]), [10.0]*11)
    assert contribution["gram"][0, 5] == -10.0
    assert contribution["gram"][0, -1] == 10.0
    assert contribution["gram"][5, -1] == -10.0
    settings = daily["priors"]["settings"]
    solution = solve_joint(contribution["gram"], contribution["score"], settings)
    # Closed-form single-observation update, independently of the matrix solver.
    player_strength = settings["player_prior_equivalent_possessions"]
    hca_strength = settings["hca"]["prior_equivalent_possessions"]
    hca_prior = settings["hca"]["prior_mean"]
    gain = 10*(14-hca_prior)/(1+10*(10/player_strength+1/hca_strength))
    expected = [gain/player_strength]*5+[-gain/player_strength]*5+[hca_prior+gain/hca_strength]
    np.testing.assert_allclose(solution["estimates"], expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("side,sign", [("home", 1), ("away", -1)])
def test_fixed_b2b_sign_and_unchanged_coefficient(daily, side, sign):
    team = daily["games"][0][f"{side}_team_id"]
    daily["previous_games"] = [{"home_team_id": team, "away_team_id": 1610612739}]
    result = update_state(**daily)
    diagnostic = result["validation"]["games"]["0022600001"]
    assert diagnostic["fixed_b2b_offset"] == sign*daily["priors"]["settings"]["fixed_back_to_back"]
    assert result["settings"]["fixed_back_to_back"] == daily["priors"]["settings"]["fixed_back_to_back"]


@pytest.mark.parametrize("marker", [None, "old_profile_correction"])
def test_previous_calculation_cannot_supply_cumulative_evidence(daily, marker):
    daily = next_date(daily, update_state(**daily))
    if marker is None:
        del daily["parent"]["calculation_contract"]
    else:
        daily["parent"]["calculation_contract"] = marker
    with pytest.raises(ValueError, match="another calculation"):
        update_state(**daily)


def test_saved_forecast_can_read_exact_old_state_after_date_migration(daily, archive):
    from hashlib import sha256
    from gestalt_pipelines.ratings.utils import validate_state
    resource, s3 = archive
    old = update_state(**daily)
    old['game_date_pacific'] = str(DAY+timedelta(days=1))
    del old['date_contract']
    del old['effective_from_game_date_pacific']
    raw = json_bytes(old)
    digest = sha256(raw).hexdigest()
    key = resource.state_prefix+f"game_date_pacific={old['game_date_pacific']}/{digest}.json"
    s3.objects[key] = raw
    reference = {'uri': f's3://{resource.bucket}/{key}', 'sha256': digest}
    pointer_key=resource.state_prefix+f"game_date_pacific={old['game_date_pacific']}/latest.json"
    s3.objects[pointer_key]=json_bytes({'schema_version':1,'season':resource.season,'model_version':'v1',
        'game_date_pacific':old['game_date_pacific'],**reference})
    saved_objects=dict(s3.objects)
    assert resource.state(old['game_date_pacific'])==(None,None,None)
    # New daily predecessors reject it; a frozen forecast can read its pinned source.
    with pytest.raises(ValueError, match='date contract'):
        validate_state(old, resource.season)
    restored = resource.read_state(reference, effective_date=str(DAY+timedelta(days=1)), observed_through=str(DAY))
    assert restored == old
    assert s3.objects == saved_objects
    with pytest.raises(ValueError, match='saved forecast'):
        resource.read_state(reference, effective_date=str(DAY+timedelta(days=2)), observed_through=str(DAY+timedelta(days=1)))
    s3.objects[key] = raw+b' '
    with pytest.raises(ValueError, match='checksum'):
        resource.read_state(reference, effective_date=str(DAY+timedelta(days=1)), observed_through=str(DAY))
