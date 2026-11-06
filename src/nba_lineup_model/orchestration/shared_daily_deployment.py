"""Prepare reviewable shared daily service files from independently sealed code.

Preparation does not stop services, enable automation, migrate live history, or
publish either site. Installing the production replacement is a separate gate.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import os
from pathlib import Path
import shlex

from nba_lineup_model.orchestration.daily_work import atomic_json, digest, read_object
from nba_lineup_model.season.sealed_code_package import verify_sealed_code_package

ROOTS = {'dev': Path('/var/lib/nba-gestalt-dev'), 'prod': Path('/var/lib/nba-gestalt')}
PORTS = {'dev': 4014, 'prod': 4016}
HOME = Path('/var/lib/nba-gestalt-daily/dagster')


def prepare_configuration(environment: str, package: Path, source: Path, *, verify_legacy: bool = False) -> Path:
    from dataclasses import replace
    from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
    from nba_lineup_model.orchestration.lifecycle import (
        bind_accepted_configuration, MissingProducingConfigurationError,
        verify_legacy_accepted_configuration,
    )
    from nba_lineup_model.orchestration.resumable_pipeline import config_identity
    from nba_lineup_model.storage.duckdb_lake import DEFAULT_LAKE_URI
    template = DailyPipelineConfig.read(source)
    if template.environment_root != ROOTS[environment]:
        raise ValueError('Source configuration belongs to another environment')
    try:
        config = bind_accepted_configuration(template)
    except MissingProducingConfigurationError:
        if not verify_legacy:
            raise
        evidence = verify_legacy_accepted_configuration(template)
        evidence['source_configuration'] = str(source)
        evidence['source_configuration_sha256'] = digest(read_object(source))
        atomic_json(package / f'{environment}-legacy-configuration-verification.json', evidence, immutable=True)
        config = template
    config = replace(config, code_root=package / 'code', activate_after_validation=True,
        observation_root=Path('/var/lib/nba-gestalt/schedule-serving/current'),
        forecast_injury_inputs_path=None, artifact_lake_uri=DEFAULT_LAKE_URI,
        artifact_cache_root=config.environment_root / 'artifact-cache')
    target = package / f'{environment}-shared-config.json'
    atomic_json(target, config_identity(config), immutable=True)
    return target


def validated_location(environment: str, package: Path, python: Path, config: Path) -> dict:
    if environment not in ROOTS:
        raise ValueError('Unknown daily environment')
    package, python, config = package.absolute(), python.absolute(), config.absolute()
    # Do not resolve bin/python symlinks: that would discard the selected venv.
    for path in (package, python, config):
        if any(c.isspace() for c in str(path)) or any(c in str(path) for c in '%"\\'):
            raise ValueError('Service paths must not contain systemd substitutions or quoting characters')
    manifest = verify_sealed_code_package(package)
    if not python.is_file() or not (package / 'code/src/nba_lineup_model/orchestration/shared_daily_control.py').is_file():
        raise ValueError('Selected runtime/package does not support shared daily services')
    settings = read_object(config)
    root = ROOTS[environment]
    if Path(settings['environment_root']).resolve() != root or Path(settings['live_pointer']).absolute() != root / 'live-release':
        raise ValueError('Code location configuration targets another environment')
    if Path(settings['code_root']).resolve() != package.resolve() / 'code':
        raise ValueError('Code location configuration uses a different code package')
    match = re.search(r'^version=([0-9.]+)$', (package / 'code/scripts/install_gestalt_worker_aws.sh').read_text(), re.MULTILINE)
    if not match:
        raise ValueError('Worker AWS CLI version must be pinned')
    aws_version = match.group(1)
    subprocess.run([str(python), str(package / 'code/scripts/validate_gestalt_runtime_environment.py'),
        '--profile', 'orchestration', '--requirements', str(package / 'code/deploy/web-requirements.txt'),
        str(package / 'code/deploy/orchestration-requirements.txt')], check=True,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)
    return {'aws_cli_bin': f'/opt/nba-gestalt-tools/aws-cli/{aws_version}/bin', 'environment': environment, 'package': str(package), 'python': str(python),
            'config': str(config), 'config_sha256': digest(settings),
            'package_sha256': manifest['identity']['package_sha256'],
            'code_revision': manifest['identity']['git_revision'],
            'root': str(root), 'port': PORTS[environment],
            'location_name': f'nba_gestalt_{environment}_daily_pipeline'}


def unit(location: dict, component: str, home: Path = HOME) -> str:
    environment = location['environment']
    package, python = location['package'], location['python']
    arguments = [python, '-m', 'nba_lineup_model.orchestration.daily_service',
        '--package', package, '--sha256', location['package_sha256'], '--home', str(home),
        '--grpc-port', str(location['port']), '--web-port', '3003', '--component', component]
    code = component == 'code'
    if code:
        arguments += ['--environment', environment]
    # systemd's ExecStart uses its own tokenizer; the selected paths are strictly
    # constrained above and shlex quoting handles plain command tokens.
    lines = ['[Unit]', f'Description=NBA GESTALT shared daily {environment if code else "control"} {component}',
        'After=network-online.target', 'Wants=network-online.target', '', '[Service]',
        'Type=simple', f'User={"gestalt-daily-prod" if code and environment == "prod" else "ubuntu"}', 'Group=ubuntu', f'WorkingDirectory={package}/code',
        'Environment=PYTHONDONTWRITEBYTECODE=1', 'Environment=PYTHONUNBUFFERED=1',
        f'Environment=PYTHONPATH={package}/code/src', f'Environment=DAGSTER_HOME={home}',
        'Environment=AWS_REGION=us-west-1',
        f'Environment=PATH={location["aws_cli_bin"]}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin']
    if code:
        lines += [f'Environment=GESTALT_DAILY_ENVIRONMENT={environment}',
                  f'Environment=GESTALT_AUTOMATION_CONFIG={location["config"]}',
                  f'ReadOnlyPaths={ROOTS["prod" if environment == "dev" else "dev"]}',
                  'Nice=10']
        if environment == 'dev':
            lines += ['MemoryMax=1050M', 'MemorySwapMax=0', 'OOMScoreAdjust=1000']
    else:
        lines += [f'ReadOnlyPaths={ROOTS["dev"]} {ROOTS["prod"]}']
    lines += ['ExecStart=' + shlex.join(arguments), 'Restart=on-failure', 'RestartSec=10',
              'TimeoutStopSec=60', 'NoNewPrivileges=true', 'PrivateTmp=true',
              '', '[Install]', 'WantedBy=multi-user.target', '']
    return '\n'.join(lines)


def prepare(output: Path, locations: dict[str, dict], home: Path = HOME) -> dict:
    if set(locations) != {'dev', 'prod'}:
        raise ValueError('Exactly two explicit code locations are required')
    if output.exists():
        raise FileExistsError('Do not overwrite an existing deployment plan')
    output.mkdir(parents=True)
    files = {}
    for name, location in locations.items():
        files[f'gestalt-daily-shared-{name}-code.service'] = unit(location, 'code', home)
    # The controller is pinned independently from future code-location updates.
    for component in ('daemon', 'web'):
        files[f'gestalt-daily-shared-{component}.service'] = unit(locations['dev'], component, home)
    files['workspace.yaml'] = 'load_from:\n' + ''.join(
        f'  - grpc_server:\n      host: 127.0.0.1\n      port: {location["port"]}\n      location_name: {location["location_name"]}\n'
        for location in locations.values())
    template = Path(locations['dev']['package']) / 'code/deploy/dagster-daily.yaml'
    files['dagster.yaml'] = template.read_text()
    for name, content in files.items():
        (output / name).write_text(content)
    identity = {'schema_version': 1, 'home': str(home), 'locations': locations,
        'files': {name: digest(content) for name, content in files.items()},
        'dev_automation_default': 'disabled', 'max_concurrent_daily_runs': 1,
        'requires_production_cutover_approval': True,
        'retire_after_verified_cutover': ['gestalt-daily-dev@code', 'gestalt-daily-dev@daemon',
            'gestalt-daily-dev@web', 'gestalt-daily-code-server', 'gestalt-daily-daemon', 'gestalt-daily-webserver']}
    plan = {'identity': identity, 'sha256': digest(identity)}
    atomic_json(output / 'plan.json', plan, immutable=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-legacy-production-config', action='store_true')
    for name in ('dev', 'prod'):
        for option in ('package', 'python', 'config'):
            parser.add_argument(f'--{name}-{option}', type=Path, required=True)
    args = parser.parse_args()
    locations = {}
    for name in ('dev', 'prod'):
        package = getattr(args, name + '_package')
        config = prepare_configuration(name, package, getattr(args, name + '_config'),
            verify_legacy=name == 'prod' and args.verify_legacy_production_config)
        locations[name] = validated_location(name, package, getattr(args, name + '_python'), config)
    plan = prepare(args.output, locations)
    print(json.dumps({'plan': str(args.output / 'plan.json'), 'sha256': plan['sha256'],
                      'installed': False}, indent=2))


if __name__ == '__main__':
    main()
