"""Readiness follows successful source assets, including native backfills/corrections."""
from datetime import UTC, datetime
from dagster import AssetMaterialization, DagsterEventType, DagsterInstance, RunRequest, SkipReason, build_run_status_sensor_context, job, op
import pytest
from gestalt_pipelines.ratings.sensors import publish_player_game_history_sensor


@op(config_schema={'day':str,'digest':str,'emit':bool,'fail':bool,'asset':str,'season':str,'tag_day':str,'empty':bool})
def source(context):
    c=context.op_config;asset=c['asset'];day=c['day'];digest=c['digest']
    metadata={'season':c['season'],'model_version':'v1'}
    if asset=='player_ratings':
        metadata.update(ratings_uri=f"s3://nba-gestalt-data/lake/artifacts/player_ratings/season={c['season']}/model_version=v1/game_date_pacific={day}/{digest}.json",sha256=digest)
    else:
        refs={} if c['empty'] else {'0022600001':{'uri':f"s3://nba-gestalt-data/lake/artifacts/game_predictions_actual_minutes/season={c['season']}/model_version=v1/game_date_pacific={day}/game_id=0022600001/{digest}.json",'sha256':digest}}
        from dagster import MetadataValue
        metadata.update(counterfactuals=MetadataValue.json(refs),game_count=len(refs),counterfactual_count=len(refs))
    if c['emit']:context.log_event(AssetMaterialization(asset_key=asset,partition=day,metadata=metadata))
    if c['fail']:raise RuntimeError('Expected isolated source failure')


@job
def synthetic_asset_backfill_source():source()


def request(instance,**changes):
    c={'day':'2026-10-21','digest':'a'*64,'emit':True,'fail':False,'asset':'player_ratings','season':'2026-27','tag_day':'','empty':False};c.update(changes)
    tags={'dagster/partition':c['tag_day']} if c['tag_day'] else {'dagster/asset_partition_range_start':c['day'],'dagster/asset_partition_range_end':c['day']}
    result=synthetic_asset_backfill_source.execute_in_process(instance=instance,run_config={'ops':{'source':{'config':c}}},tags=tags,raise_on_error=False)
    kind=DagsterEventType.RUN_FAILURE if c['fail'] else DagsterEventType.RUN_SUCCESS
    event=next(e for e in result.all_events if e.event_type==kind)
    with build_run_status_sensor_context(sensor_name='publish_player_game_history_sensor',dagster_event=event,dagster_instance=instance,dagster_run=instance.get_run_by_id(result.run_id)) as context:
        return publish_player_game_history_sensor(context)


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return datetime(2026,10,24,18,tzinfo=UTC)
    monkeypatch.setattr('gestalt_pipelines.ratings.sensors.datetime',Clock)


@pytest.mark.parametrize('first',['player_ratings','game_predictions_actual_minutes'])
def test_both_sources_must_succeed_and_same_inputs_deduplicate(monkeypatch,first):
    monkeypatch.setattr('boto3.client',lambda *a,**kw:pytest.fail('Readiness must not call S3/providers'))
    second='game_predictions_actual_minutes' if first=='player_ratings' else 'player_ratings'
    with DagsterInstance.ephemeral() as i:
        assert isinstance(request(i,asset=first),SkipReason)
        ready=request(i,asset=second)
        repeat=request(i,asset=first,tag_day='2026-10-22')
        assert isinstance(ready,RunRequest) and ready.partition_key=='2026-10-21'
        assert repeat.run_key==ready.run_key and ready.tags['dagster/priority']=='-2'
        assert request(i,asset=second,digest='b'*64).run_key!=ready.run_key


def test_older_postgame_correction_refreshes_latest_head():
    with DagsterInstance.ephemeral() as i:
        request(i);request(i,asset='game_predictions_actual_minutes')
        request(i,day='2026-10-22');latest=request(i,day='2026-10-22',asset='game_predictions_actual_minutes')
        correction=request(i,asset='game_predictions_actual_minutes',digest='c'*64)
        repeat=request(i,asset='game_predictions_actual_minutes',digest='c'*64)
        assert correction.partition_key=='2026-10-22' and correction.run_key!=latest.run_key
        assert correction.run_key==repeat.run_key


@pytest.mark.parametrize('changes',[{'fail':True},{'emit':False},{'asset':'player_game_history'},{'season':'2025-26'},{'day':'2026-10-24'}])
def test_failed_unrelated_foreign_and_future_runs_do_not_launch(changes):
    with DagsterInstance.ephemeral() as i:
        assert isinstance(request(i,**changes),SkipReason)


def test_no_game_date_can_update_current_without_game_points():
    with DagsterInstance.ephemeral() as i:
        request(i)
        assert isinstance(request(i,asset='game_predictions_actual_minutes',empty=True),RunRequest)


def test_invalid_checksum_is_visible():
    with DagsterInstance.ephemeral() as i,pytest.raises(ValueError,match='checksum'):
        request(i,digest='bad')
