from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nba_lineup_model.orchestration.daily_work import file_reference, work_spec
from nba_lineup_model.orchestration.resumable_pipeline import (
    PartitionSteps, artifact_files, bind_execution, validate_execution,
)
from nba_lineup_model.season.daily_input_selection import build_selected_daily_data
from test_daily_artifacts import _publish_data, _publish_model
from test_daily_pipeline import config_fixture


def test_pinned_zero_game_day_builds_without_network_or_forecasts(tmp_path, monkeypatch):
    catalog=tmp_path/"catalog.parquet"
    pd.DataFrame(columns=["season","season_type","game_id","game_time_utc",
                          "home_team_id","home_team_tricode","away_team_id","away_team_tricode","source_url"]).to_parquet(catalog)
    monkeypatch.setattr("nba_lineup_model.season.daily_data_replay._fetch_verified_s3_json",
                        lambda *a:pytest.fail("Unexpected source fetch"))
    artifact=build_selected_daily_data(selection={},catalog_reference=file_reference(catalog),
        season="2026-27",game_date=date(2026,11,3),output_root=tmp_path/"data")
    assert artifact.manifest["identity"]["game_count"] == 0
    assert not (tmp_path/"frozen").exists()


def test_pinned_data_selection_cannot_drop_game(tmp_path):
    catalog=tmp_path/"catalog.parquet"
    pd.DataFrame([dict(season="2026-27",season_type="regular",game_id="0022600001",
        game_time_utc="2026-10-20T23:00:00Z",home_team_id=100,home_team_tricode="HOM",away_team_id=200,away_team_tricode="AWY",source_url="https://cdn.nba.com/schedule.json")]).to_parquet(catalog)
    with pytest.raises(ValueError,match="exact official slate"):
        build_selected_daily_data(selection={},catalog_reference=file_reference(catalog),
            season="2026-27",game_date=date(2026,10,20),output_root=tmp_path/"data")


def execution_fixture(tmp_path, target=date(2026,10,21)):
    cfg=config_fixture(tmp_path)
    d=_publish_data(tmp_path/"seed",game_date=date(2026,10,20),game_ids=["0022600001"])
    m=_publish_model(tmp_path/"seed",data_bundle=d.path)
    base=cfg.live_pointer/"runtime-manifest.json"
    base.write_text(json.dumps({"runtime_sha256":"base-release","approved_manifest":{"identity":{
        "evidence_through_date":"2026-10-20","model":{"bundle_sha256":m.manifest["bundle_sha256"]}}}}))
    refs=artifact_files(d.path)+artifact_files(m.path)+[file_reference(base)]
    spec=work_spec(season="2026-27",game_date=target,source_revisions={},
        model_configuration=m.manifest["identity"]["model_provenance"],
        parent_model=m.manifest["bundle_sha256"],edition_id="original",code_identity="test",
        inputs={str(i):ref for i,ref in enumerate(refs)},expected_game_ids=[])
    execution={"base_release_manifest":file_reference(base),"data_manifests":[file_reference(d.path/"manifest.json")],
               "model_manifests":[file_reference(m.path/"manifest.json")]}
    return cfg,bind_execution(spec,cfg,execution),d,m


def test_valid_forward_parent_and_data_are_compatible(tmp_path):
    cfg,spec,_,_=execution_fixture(tmp_path)
    validate_execution(cfg,spec,None)


def test_historical_request_cannot_bypass_approval_with_false_flags(tmp_path):
    cfg,spec,_,_=execution_fixture(tmp_path,target=date(2026,10,20))
    with pytest.raises(PermissionError,match="exact selected"):
        validate_execution(cfg,spec,None)


def test_gap_cannot_skip_a_daily_model_parent(tmp_path):
    cfg,spec,_,_=execution_fixture(tmp_path,target=date(2026,10,22))
    with pytest.raises(ValueError,match="immediately follow"):
        validate_execution(cfg,spec,None)


def test_revised_parent_data_requires_earlier_causal_repair(tmp_path):
    cfg,spec,_,_=execution_fixture(tmp_path)
    changed=_publish_data(tmp_path/"changed",game_date=date(2026,10,20),game_ids=["0022600001"],suffix="new")
    ref=file_reference(changed.path/"manifest.json")
    spec["identity"]["inputs"]["changed"]=ref
    spec["identity"]["execution"]["data_manifests"]=[ref]
    with pytest.raises(ValueError,match="invalidates this parent"):
        validate_execution(cfg,spec,None)


def test_new_dagster_boundary_does_not_import_legacy_definitions():
    code="""import sys
sys.modules['nba_lineup_model.orchestration.definitions']=None
from dagster import Definitions
from nba_lineup_model.orchestration.partition_definitions import defs
Definitions.validate_loadable(defs)
assert {j.name for j in defs.jobs} == {'prepare_daily_partition'}
assert len(defs.sensors) == 1
assert not defs.schedules
"""
    import sys
    subprocess.run([sys.executable,"-c",code],check=True)
