import json

import pytest

from nba_lineup_model.orchestration.daily_history_migration import migrated_tags
from nba_lineup_model.orchestration.daily_instance import DailyInstance
from nba_lineup_model.orchestration.daily_work import atomic_json


@pytest.mark.parametrize('scope', ['live', 'rehearsal', 'foreign', 'symlink'])
def test_migration_preserves_rehearsal_without_live_recovery(tmp_path, monkeypatch, scope):
    root = tmp_path / 'dev'
    root.mkdir()
    foreign = tmp_path / 'prod'
    foreign.mkdir()
    rehearsal = root / 'rehearsal'
    rehearsal.mkdir()
    link = root / 'linked-prod'
    link.symlink_to(foreign, target_is_directory=True)
    pointer = {'live': root, 'rehearsal': rehearsal,
               'foreign': foreign, 'symlink': link}[scope] / 'live-release'
    config = root / 'config.json'
    atomic_json(config, {'environment_root': str(root), 'live_pointer': str(pointer)})
    record = {'run_id': 'original-id', 'pipeline_name': 'prepare_daily_partition',
              'run_body': json.dumps({'run_config': {'ops': {'prepare': {
                  'config': {'config_path': str(config)}}}}})}
    monkeypatch.setattr(
        'nba_lineup_model.orchestration.daily_history_migration.request_identity',
        lambda *_: ('work', '2026-10-30', 'prepare_daily_partition:season:date:edition'))
    binding = DailyInstance('dev', root)
    if scope in {'foreign', 'symlink'}:
        with pytest.raises(ValueError, match='conflicting live environment'):
            migrated_tags(binding, record, root / 'archive')
    else:
        tags = migrated_tags(binding, record, root / 'archive')
        assert (tags.get('gestalt/history_only') == 'true') == (scope == 'rehearsal')
        assert (':rehearsal:' in tags['gestalt/recovery_key']) == (scope == 'rehearsal')
        assert json.loads((root / 'archive/request-evidence/original-id/config_path.json').read_text()) == json.loads(config.read_text())
