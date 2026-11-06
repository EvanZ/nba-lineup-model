"""Explicit dev automation windows for a shared daily instance."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from nba_lineup_model.orchestration.daily_instance import DailyInstance
from nba_lineup_model.orchestration.daily_work import atomic_json, read_object


def automation_enabled(binding: DailyInstance) -> bool:
    if not binding.instance_home or binding.environment != 'dev':
        return True
    path = binding.home / 'dev-automation.json'
    if not path.exists():
        return False
    value = read_object(path)
    if value.get('schema_version') != 1 or type(value.get('enabled')) is not bool:
        raise ValueError('Invalid dev automation window')
    return value['enabled']


@contextmanager
def control_lock(home: Path):
    with (home / '.automation-control.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def set_dev_automation(home: Path, enabled: bool, reason: str) -> dict:
    """Close admission first, cancel queued sensor runs, never kill active work."""
    from dagster import DagsterInstance, DagsterRunStatus, RunsFilter
    binding = DailyInstance.read(home, 'dev')
    if not binding.instance_home or not reason.strip():
        raise ValueError('A shared instance and an explicit reason are required')
    with control_lock(home):
        with DagsterInstance.from_config(str(home)) as instance:
            tags = {'gestalt/environment': 'dev', 'gestalt/trigger': 'sensor'}
            active = instance.get_runs(filters=RunsFilter(tags=tags, statuses=[
                DagsterRunStatus.STARTING, DagsterRunStatus.STARTED, DagsterRunStatus.CANCELING]))
            if active:
                raise RuntimeError('Wait for active dev work to finish before changing its window')
            value = {'schema_version': 1, 'enabled': enabled, 'reason': reason,
                     'changed_at': datetime.now(timezone.utc).isoformat()}
            # This is a new control record, not an edit to any run or alert history.
            atomic_json(home / 'dev-automation.json', value)
            canceled = []
            if not enabled:
                for run in instance.get_runs(filters=RunsFilter(tags=tags, statuses=[DagsterRunStatus.QUEUED])):
                    instance.report_run_canceled(run, message='Dev automation window closed by operator')
                    canceled.append(run.run_id)
            value['canceled_queued_runs'] = canceled
            return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--mode', choices=('enable', 'disable'), required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    print(json.dumps(set_dev_automation(args.home, args.mode == 'enable', args.reason), indent=2))


if __name__ == '__main__':
    main()
