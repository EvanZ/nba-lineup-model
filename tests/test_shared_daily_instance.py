"""Real shared Dagster storage, migration, execution and environment isolation."""
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from dagster import DagsterInstance, RunsFilter, build_sensor_context

from nba_lineup_model.orchestration.daily_instance import DailyInstance, execute_manual, binding_for_instance
from nba_lineup_model.orchestration.daily_history_migration import verify_migration, rows, record_digest
from nba_lineup_model.orchestration.daily_run_status import read_daily_runs
from nba_lineup_model.orchestration.shared_daily_history import migrate_shared
from nba_lineup_model.orchestration.daily_work import atomic_json
from nba_lineup_model.orchestration.partition_definitions import resumable_daily_partition_sensor
from test_daily_instance_wiring import pipeline, environment, request

TEMPLATE = Path(__file__).resolve().parents[1] / 'deploy/dagster-daily.yaml'


def shared(tmp_path):
    dev, prod = environment(tmp_path, 'dev'), environment(tmp_path, 'prod')
    home = tmp_path / 'shared'
    roots = {'dev': dev.root, 'prod': prod.root}
    return dev, prod, home, roots


def migrate(dev, prod, home, roots):
    return migrate_shared(home, roots, {'dev': dev.home, 'prod': prod.home}, TEMPLATE, 'test-only approval')


def test_migration_preserves_records_and_cross_environment_success_cannot_clear_failure(tmp_path, monkeypatch, pipeline):
    dev, prod, home, roots = shared(tmp_path)
    dr, pr = request(dev), request(prod)
    failed = execute_manual(dev.home, 'prepare_daily_partition', dr)
    pipeline['fail'] = False
    success = execute_manual(prod.home, 'prepare_daily_partition', pr)
    originals = {str(b.home): {r['run_id']: record_digest(r) for r in rows(b.home / 'history/runs.db')} for b in (dev, prod)}
    result = migrate(dev, prod, home, roots)
    assert result['migrated_runs'] == 2
    assert migrate(dev, prod, home, roots) == result
    assert [r['run_id'] for r in read_daily_runs(home, 'dev', dev.root)['unresolved_failures']] == [failed.run_id]
    assert read_daily_runs(home, 'prod', prod.root)['last_success_run_id'] == success.run_id
    monkeypatch.setenv('GESTALT_DAILY_ENVIRONMENT', 'dev')
    repaired = execute_manual(home, 'prepare_daily_partition', dr)
    assert repaired.success
    assert not read_daily_runs(home, 'dev', dev.root)['unresolved_failures']
    assert read_daily_runs(home, 'prod', prod.root)['total_run_count'] == 1
    for b in (dev, prod):
        assert {r['run_id']: record_digest(r) for r in rows(b.home / 'history/runs.db')} == originals[str(b.home)]
    assert verify_migration(home)['verified']


def test_shared_binding_requires_environment_and_rejects_wrong_root(tmp_path, monkeypatch, pipeline):
    dev, prod, home, roots = shared(tmp_path)
    migrate(dev, prod, home, roots)
    monkeypatch.delenv('GESTALT_DAILY_ENVIRONMENT', raising=False)
    with pytest.raises(ValueError, match='explicit registered environment'):
        DailyInstance.read(home)
    monkeypatch.setenv('GESTALT_DAILY_ENVIRONMENT', 'dev')
    binding = DailyInstance.read(home)
    assert binding.home == home and binding.root == dev.root
    with pytest.raises(ValueError, match='different environment'):
        binding.tags('prepare_daily_partition', request(prod))
    monkeypatch.setenv('DAGSTER_HOME', str(prod.home))
    with DagsterInstance.from_config(str(home)) as instance:
        with pytest.raises(ValueError, match='DAGSTER_HOME'):
            binding_for_instance(instance)


def test_manual_execution_cannot_bypass_global_lock(tmp_path, monkeypatch, pipeline):
    dev, prod, home, roots = shared(tmp_path)
    migrate(dev, prod, home, roots)
    monkeypatch.setenv('GESTALT_DAILY_ENVIRONMENT', 'prod')
    pipeline['fail'] = False
    with (home / '.daily-execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute_manual(home, 'prepare_daily_partition', request(prod))
    assert not result.success
    assert pipeline['calls'] == 0


def test_production_success_does_not_suppress_dev_sensor_work(tmp_path, monkeypatch, pipeline):
    dev, prod, home, roots = shared(tmp_path)
    dr, pr = request(dev), request(prod)
    pipeline['fail'] = False
    assert execute_manual(prod.home, 'prepare_daily_partition', pr).success
    migrate(dev, prod, home, roots)
    atomic_json(dev.requests / 'pending.json', dr)
    from nba_lineup_model.orchestration.shared_daily_control import set_dev_automation
    set_dev_automation(home, True, 'test window')
    monkeypatch.setenv('GESTALT_DAILY_ENVIRONMENT', 'dev')
    with DagsterInstance.from_config(str(home)) as instance:
        with build_sensor_context(instance=instance) as context:
            result = resumable_daily_partition_sensor(context)
        assert result.tags['gestalt/environment'] == 'dev'
        assert result.run_key.startswith('dev:daily:')
        # Indexed migrated tags also scope Dagster's native queries.
        assert len(instance.get_runs(filters=RunsFilter(tags={'gestalt/environment': 'prod'}))) == 1
        assert not instance.get_runs(filters=RunsFilter(tags={'gestalt/environment': 'dev'}))


def test_migration_rejects_active_source_without_installing_destination(tmp_path, pipeline):
    dev, prod, home, roots = shared(tmp_path)
    from nba_lineup_model.orchestration.partition_definitions import prepare_daily_partition_job
    with DagsterInstance.from_config(str(dev.home)) as instance:
        instance.create_run_for_job(prepare_daily_partition_job, run_config={"ops": {"prepare_daily_partition_op": {"config": request(dev)}}})
    with pytest.raises(ValueError, match='unfinished'):
        migrate(dev, prod, home, roots)
    assert not home.exists()


def test_shared_history_rejects_overlapping_environment_roots(tmp_path):
    root = tmp_path / 'root'
    nested = root / 'nested'
    nested.mkdir(parents=True)
    home = tmp_path / 'history'
    home.mkdir()
    atomic_json(home / 'environment.json', {'schema_version': 2, 'environments': {'dev': str(root), 'prod': str(nested)}})
    with pytest.raises(ValueError, match='disjoint'):
        DailyInstance.read(home, 'dev')


def test_dev_window_defaults_off_and_cancels_only_queued_dev_sensor_work(tmp_path, monkeypatch, pipeline):
    from dagster import DagsterRunStatus
    from nba_lineup_model.orchestration.partition_definitions import prepare_daily_partition_job
    from nba_lineup_model.orchestration.shared_daily_control import automation_enabled, set_dev_automation
    dev, prod, home, roots = shared(tmp_path)
    migrate(dev, prod, home, roots)
    assert not automation_enabled(DailyInstance.read(home, 'dev'))
    assert automation_enabled(DailyInstance.read(home, 'prod'))
    set_dev_automation(home, True, 'validate new code')
    assert automation_enabled(DailyInstance.read(home, 'dev'))
    from dagster._core.remote_origin import RemoteJobOrigin, RemoteRepositoryOrigin, RegisteredCodeLocationOrigin
    runs = {}
    with DagsterInstance.from_config(str(home)) as instance:
        for env, trigger in [('dev', 'sensor'), ('dev', 'manual'), ('prod', 'sensor')]:
            config = request(dev if env == 'dev' else prod)
            runs[env, trigger] = instance.create_run_for_job(prepare_daily_partition_job,
                run_config={'ops': {'prepare_daily_partition_op': {'config': config}}},
                tags={'gestalt/environment': env, 'gestalt/trigger': trigger},
                status=DagsterRunStatus.QUEUED,
                remote_job_origin=RemoteJobOrigin(RemoteRepositoryOrigin(
                    RegisteredCodeLocationOrigin(f'nba_gestalt_{env}_daily_pipeline'),
                    '__repository__'), 'prepare_daily_partition'))
    result = set_dev_automation(home, False, 'validation finished')
    assert result['canceled_queued_runs'] == [runs['dev', 'sensor'].run_id]
    with DagsterInstance.from_config(str(home)) as instance:
        assert instance.get_run_by_id(runs['dev', 'sensor'].run_id).status == DagsterRunStatus.CANCELED
        for key in [('dev', 'manual'), ('prod', 'sensor')]:
            assert instance.get_run_by_id(runs[key].run_id).status == DagsterRunStatus.QUEUED


def test_window_refuses_active_dev_work(tmp_path, pipeline):
    from dagster import DagsterRunStatus
    from nba_lineup_model.orchestration.partition_definitions import prepare_daily_partition_job
    from nba_lineup_model.orchestration.shared_daily_control import set_dev_automation, automation_enabled
    dev, prod, home, roots = shared(tmp_path)
    migrate(dev, prod, home, roots)
    set_dev_automation(home, True, 'test')
    with DagsterInstance.from_config(str(home)) as instance:
        instance.create_run_for_job(prepare_daily_partition_job,
            run_config={'ops': {'prepare_daily_partition_op': {'config': request(dev)}}},
            tags={'gestalt/environment': 'dev', 'gestalt/trigger': 'sensor'},
            status=DagsterRunStatus.STARTED)
    with pytest.raises(RuntimeError, match='active dev work'):
        set_dev_automation(home, False, 'close test')
    assert automation_enabled(DailyInstance.read(home, 'dev'))


def test_closed_window_blocks_a_sensor_run_already_in_flight(tmp_path, monkeypatch, pipeline):
    from nba_lineup_model.orchestration.partition_definitions import prepare_daily_partition_job
    dev, prod, home, roots = shared(tmp_path)
    migrate(dev, prod, home, roots)
    monkeypatch.setenv('GESTALT_DAILY_ENVIRONMENT', 'dev')
    pipeline['fail'] = False
    config = request(dev)
    tags = DailyInstance.read(home, 'dev').tags('prepare_daily_partition', config)
    with DagsterInstance.from_config(str(home)) as instance:
        result = prepare_daily_partition_job.execute_in_process(instance=instance,
            tags={**tags, 'gestalt/trigger': 'sensor'},
            run_config={'ops': {'prepare_daily_partition_op': {'config': config}}}, raise_on_error=False)
    assert not result.success
    assert pipeline['calls'] == 0


def test_monitoring_distinguishes_identical_sensor_names_by_location(tmp_path, monkeypatch):
    import io
    from nba_lineup_model.orchestration.data_quality_probe import effective_automation_states
    nodes = [{'location': {'name': f'nba_gestalt_{name}_daily_pipeline'}, 'schedules': [],
              'sensors': [{'name': 'automatic_daily_lifecycle', 'sensorState': {'status': state}}]}
             for name, state in [('dev', 'STOPPED'), ('prod', 'RUNNING')]]
    payload = json.dumps({'data': {'repositoriesOrError': {'__typename': 'RepositoryConnection', 'nodes': nodes}}}).encode()
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(payload))
    for env, status in [('dev', 'STOPPED'), ('prod', 'RUNNING')]:
        result = effective_automation_states(tmp_path / 'absent.db', 'http://unused', f'nba_gestalt_{env}_daily_pipeline')
        assert result['automatic_daily_lifecycle']['automation_status'] == status


def test_shared_prod_reload_uses_narrow_polkit_permission(monkeypatch):
    from nba_lineup_model.orchestration.daily_activation import reload_environment_api
    calls = []
    monkeypatch.setenv('GESTALT_DAILY_ENVIRONMENT', 'prod')
    monkeypatch.setattr('subprocess.run', lambda args, **kwargs: calls.append(args))
    reload_environment_api(SimpleNamespace(environment_root=Path('/var/lib/nba-gestalt')), {})
    assert calls == [['systemctl', '--no-ask-password', 'restart', 'gestalt-web']]
    monkeypatch.delenv('GESTALT_DAILY_ENVIRONMENT')
    reload_environment_api(SimpleNamespace(environment_root=Path('/var/lib/nba-gestalt')), {})
    assert calls[-1][:2] == ['sudo', '-n']


def test_shared_service_plan_isolated_code_locations_and_shared_controller(tmp_path):
    from nba_lineup_model.orchestration.shared_daily_deployment import prepare
    package = tmp_path / 'package'
    (package / 'code/deploy').mkdir(parents=True)
    (package / 'code/deploy/dagster-daily.yaml').write_bytes(TEMPLATE.read_bytes())
    locations = {env: {'environment': env, 'package': str(package), 'python': '/usr/bin/python3',
        'package_sha256': 'test', 'port': port, 'config': f'/var/lib/nba-gestalt-{env}/config.json',
        'location_name': f'nba_gestalt_{env}_daily_pipeline', 'aws_cli_bin': '/opt/test/bin'}
        for env, port in [('dev', 4014), ('prod', 4016)]}
    output = tmp_path / 'plan'
    plan = prepare(output, locations)
    import yaml
    assert len(yaml.safe_load((output / 'workspace.yaml').read_text())['load_from']) == 2
    assert yaml.safe_load((output / 'dagster.yaml').read_text())['run_coordinator']['config']['max_concurrent_runs'] == 1
    dev = (output / 'gestalt-daily-shared-dev-code.service').read_text()
    prod = (output / 'gestalt-daily-shared-prod-code.service').read_text()
    assert 'ReadOnlyPaths=/var/lib/nba-gestalt\n' in dev
    assert 'ReadOnlyPaths=/var/lib/nba-gestalt-dev\n' in prod
    assert 'User=gestalt-daily-prod\n' in prod
    assert 'NoNewPrivileges=true' in prod
    assert plan['identity']['requires_production_cutover_approval']
    assert not any('gestalt-collection' in name for name in plan['identity']['retire_after_verified_cutover'])
    with pytest.raises(FileExistsError):
        prepare(output, locations)


def test_migration_rejects_a_stale_prepared_copy(tmp_path, pipeline):
    dev, prod, home, roots = shared(tmp_path)
    migrate(dev, prod, home, roots)
    execute_manual(dev.home, 'prepare_daily_partition', request(dev))
    with pytest.raises(ValueError, match='Source history changed since'):
        migrate(dev, prod, home, roots)


@pytest.mark.parametrize('mismatch', [None, 'season', 'model_track', 'model_code_revision', 'model_configuration_sha256', 'prior'])
def test_explicit_legacy_verification_never_changes_accepted_inputs(tmp_path, monkeypatch, mismatch):
    from nba_lineup_model.orchestration.lifecycle import verify_legacy_accepted_configuration
    live = tmp_path / 'release'
    live.mkdir()
    atomic_json(live / 'runtime-manifest.json', {'approved_manifest': {'identity': {'model': {'bundle_path': str(tmp_path / 'model'), 'bundle_sha256': 'selected'}}}})
    model = {'bundle_sha256': 'selected', 'identity': {'season': '2026-27', 'model_provenance': {
        'model_version': 'track', 'code_revision': 'revision', 'configuration_sha256': 'configuration', 'prior_sha256': 'prior'}}}
    monkeypatch.setattr('nba_lineup_model.season.daily_model_artifact.verify_daily_model_artifact', lambda path: model)
    config_path = tmp_path / 'config.json'
    config_path.write_text('{}')
    config = SimpleNamespace(live_pointer=live, season='2026-27', model_track='track',
        model_code_revision='revision', model_configuration_sha256='configuration',
        model_configuration_path=config_path, validate=lambda: None)
    monkeypatch.setattr('nba_lineup_model.orchestration.daily_pipeline.causal_hca_replay_config', lambda c: None)
    def prior(c, provenance):
        if mismatch == 'prior':
            raise ValueError('coupled prior mismatch')
        return {'rankings.parquet': {'sha256': 'exact'}}
    monkeypatch.setattr('nba_lineup_model.orchestration.lifecycle.coupled_prior_files', prior)
    if mismatch and mismatch != 'prior':
        setattr(config, mismatch, 'different')
    if mismatch:
        with pytest.raises(ValueError):
            verify_legacy_accepted_configuration(config)
    else:
        evidence = verify_legacy_accepted_configuration(config)
        assert evidence['model_sha256'] == 'selected'
        assert evidence['original_producing_job_record'] == 'unavailable'
        assert evidence['historical_model_recomputed'] is False
