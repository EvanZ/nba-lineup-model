# Daily job history and recovery

Each environment owns exactly one durable Dagster home:
`<environment_root>/daily-dagster`. Its immutable `environment.json` records
the environment name and root. Do not create a Dagster home inside a release,
rehearsal or temporary directory for real dev or production work.

The sensor, manual entry point and watchdog use this same home. Both launch
paths use `partition_definitions` and the same environment, code and work
identity checks. A request for another environment or another code package
is rejected before calculations. The environment-level execution lock also
prevents a manual run from overlapping a scheduled run.

`scripts/deploy_gestalt_daily_dev.sh` installs the dev instance from committed,
sealed code using the existing versioned orchestration environment. It enables
the explicit-request sensor, but creates no work requests, changes no site
pointer, imports no history, and does not start a model rebuild. The sensor
does not invent nightly input selections: approved requests still need to be
produced by the canonical workflow. This is not production promotion.

For a manual rerun, use `nba_lineup_model.orchestration.daily_service` with
`--component manual`, the installed service's exact `--package`, `--sha256`
and `--home`, and `--job` plus the pinned `--request`. Run it with the same
versioned Python interpreter and sealed-code `PYTHONPATH` as the service.
Use Python's `-B` flag: cache writes inside sealed code invalidate its inventory.
Do not run a new isolated Dagster instance or execute an old work request
against different code. Code/input changes require a newly pinned request.

Recovery uses the environment, job, target date and model edition. Each attempt
retains its complete code/input work identity, but a code fix does not prevent
its successful rerun from resolving the failed attempt. A later SUCCESS
for the same output clears its failure on the next watchdog check (including a
manual rerun). An unrelated date, another job, another environment, a failed
retry, or a queued/running/cancelled retry cannot clear it. All unresolved
work failures are checked; a newer successful job cannot hide an older one.
The independent hourly watchdog remains responsible for reports and alerts.

An empty new history is explicitly unverified, not healthy or recovered.
Existing histories are not searched or silently imported. Any initial
migration requires approval, preserves original run IDs/statuses and source
history, and must be verified before the monitor switches to it.

`daily_history_migration` accepts an explicit list of source homes and an approval
reference. It refuses unfinished runs, conflicting identities and foreign
environments. It preserves original run records, event logs and run outputs,
keeps checksummed copies of the source histories, and checks every copied run
before atomically installing the permanent home. It never rewrites statuses or
imports scheduler settings. Archived request evidence supplies environment/date/
edition labels for old runs without changing their original records. The monitor
checks those records against the migration receipt before using them.

The tests in `test_daily_instance_wiring.py` execute real Dagster runs and
read their persisted SQLite history. Basketball computations are test doubles;
these tests do not claim to have rerun the actual season. They cover sensor
failure, repeated manual failure, successful manual recovery, another work
item, another environment, forecasts, restart persistence and rejected
environment/code mismatches. Production remains unchanged until approved.
