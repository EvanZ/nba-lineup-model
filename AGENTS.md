# NBA GESTALT operating contract

These instructions apply to every task in this repository.

## Execution environment

- Treat `gestalt-web-server:~/nba-lineup-model` as the development checkout.
- Implement, test, materialize data, and build releases on EC2. Do not use the
  laptop checkout as the working implementation environment.
- Keep large datasets and model artifacts on EC2/S3. Never copy them to the
  laptop without the user's explicit approval.
- Keep remote inspection remote. Do not transfer or stream substantial source
  files, logs, reports, directory listings, artifacts, or other remote context
  from EC2 into the conversation or laptop without the user's explicit approval.
- Search, filter, edit, test, and summarize on EC2. Return only the smallest
  relevant excerpts, diffs, test summaries, and validation evidence needed for
  the user to understand and verify the result.
- For implementation work, use one persistent SSH shell rooted in the EC2
  checkout instead of mixing standalone remote and local shell commands. Before
  any mutating batch, verify the remote hostname, working directory, and Git
  branch inside that shell.
- Treat that verified shell as an execution-context lock for the task. Keep its
  session open and run all repository inspection, editing, testing, Git, build,
  materialization, and deployment work through that same session.
- If the persistent shell closes, resets, changes host/path/branch, or otherwise
  becomes ambiguous, stop before making another change. Reconnect to EC2 and
  re-verify hostname, checkout path, branch, and worktree status; never fall
  back to the laptop checkout.
- Keep every edit, Git operation, test, build, and deployment command inside the
  verified EC2 shell. Do not run local Git or file-edit commands for repository
  implementation, and never place local commands after a remote heredoc or
  closing SSH boundary.

## Release workflow

- Deploy every site change to `gestalt-dev.toplines.app` first.
- Inspect the rendered dev page and validate the relevant API/data coverage
  before saying a change is ready.
- Do not promote or deploy to production without the user's explicit approval.
- Dev and production must use the same canonical artifact schemas and coverage
  checks; environment-specific pointers may differ.
- Use the canonical daily release workflow for data publication. Code-only dev
  API deployments use scripts/deploy_gestalt_code.py, which preserves all data
  selections, tests the packaged API, and activates through the same fenced
  publisher. Code deployment must not require running a forecasting/model job.
  Forecast-only data publication is the final step of `prepare_forecast_date`:
  it reuses deployed code and accepted model artifacts, validates all schedule
  coverage and the served forecasts, and uses the fenced publisher. An approved
  prepared correction may enter that same job without recomputation.
- Do not manually assemble artifacts, invoke lower-level publisher commands, or
  edit release pointers to bypass either workflow.
- A small UI/reporting change does not justify a release shortcut. If the
  canonical workflow cannot express the requested change, report the limitation
  and obtain explicit approval for a scoped workflow repair before proceeding.
- A staged release must own a sealed copy of its exact committed code, with a
  verified file inventory, package checksum, dependency-lock checksum, and Git
  revision. Never link a release's code to an editable checkout, including a
  detached checkout: a commit label alone does not freeze its files.
- Enforce this contract in staging and activation. Tests must prove that later
  checkout edits cannot change a staged release and that code symlinks, changed
  package files, and missing code seals are rejected before pointer movement.
- Never modify an already activated release in place to make a failed check pass.
  A legacy-release repair requires explicit approval for its scope and verified
  identity; never weaken or bypass validation to perform the repair.
- When a release check fails, report the failure and repair the canonical path.
  Never patch manifests, invent substitute inputs, or silently omit a failed gate.

## Daily job history and alert recovery

- Real dev/production daily runs use the permanent bound Dagster instance. Under
  the approved #59 shared deployment, each code location must explicitly bind its
  registered environment, code, configuration, outputs, and tagged run history.
  Manual reruns and sensors use that same instance; never create a release-local
  or temporary instance for real work. Temporary instances are for tests only.
- Watchdogs scope shared history and automation state to the exact environment
  and code location. A success in another environment, another job, or different
  work must not clear its failure.
- Dev automation is off outside explicit validation windows. Production remains
  scheduled. Enforce one heavy daily run across the shared instance, including
  manual runs; keep protected collection in its independent service.
- Recovery comes from actual successful runs, never edited alert state or fabricated
  run records. Any initial history migration requires approval and preserves originals.

## Change classification and historical immutability

- Classify every change before implementation as UI-only, observed-data,
  model/pipeline, or a combination. State the classification to the user.
- A UI-only change must not rebuild causal artifacts, alter pipeline behavior,
  rematerialize observed data, or change model inputs. If a requested UI change
  depends on any of those actions, it is not UI-only: announce the underlying
  data/model/pipeline dependency and its consequences before proceeding.
- Frozen pregame projections for games in the past are immutable. Never
  recompute, replace, relabel, or silently change them without the user's
  explicit approval for that exact historical scope.
- Never allow a repair, release, backfill, UI deployment, or pointer update to
  rewrite historical frozen projections as an incidental side effect.
- A correction to historical observed data or historical model inputs normally
  requires a causal model rebuild from the earliest affected Pacific date
  through the current evidence date. Do not start that rebuild automatically.
- Before any causal rebuild or backfill, report the affected dates, artifacts,
  downstream model state, and whether frozen forecasts would change. Obtain the
  user's explicit approval for the stated scope.
- Data-artifact repair and causal model replay are separate operations. Publish
  and validate the repaired data selection first; run the approved model suffix
  replay second. Historical frozen forecasts remain unchanged unless separately
  and explicitly approved.

## Correctness gates

- Prefer the existing canonical artifacts over reconstructing, approximating,
  or inventing substitute inputs.
- Validate complete game/date coverage, not a sample. A missing or quarantined
  game must be reported and repaired without silently removing other games.
- Do not claim success from unit tests alone. Verify the materialized EC2
  artifacts, the API response, and the rendered dev UI.
- Preserve immutable pregame projections and source provenance. Postgame
  retrospective artifacts must be separately labeled.
- NBA GESTALT basketball data must come from the official NBA sources selected
  by the repository architecture.
- When validation exposes a bug, schema mismatch, or broken contract, stop the
  affected workflow and report the root cause and scope to the user.
- Do not hide a bug with a one-off manifest edit, pointer coercion, fallback,
  compatibility exception, or partial repair. Fix the canonical producer or
  contract and add regression coverage that reproduces the failure.
- If that root-cause repair expands the approved scope or changes pipeline,
  artifact, model, or product behavior, obtain the user's approval first.

## Protected raw-data collection

- Run these as three separate, independently controlled jobs:
  - official NBA injury-report PDF collection;
  - The Odds API market-odds polling;
  - official NBA CDN scoreboard and live-score polling.
- Keep all three jobs continuously enabled whenever their own source and
  collector are healthy. A failure in one collector must not stop either of the
  other two collectors.
- These collectors are not model training, causal replay, release assembly, or
  production publication. Stopping any downstream system does not authorize
  stopping a protected collector.
- Never stop a protected collector merely because Dagster, another
  orchestrator, a model job, or a publishing pipeline must be stopped. If a
  shared orchestrator cannot isolate the three jobs safely, move them to a
  collection-only service before stopping the shared process.
- Stop a protected collector only when that collector has an error that
  prevents safe operation, or when the user explicitly approves stopping that
  specific feed. A general instruction to stop Dagster or a broken pipeline
  does not include these feeds.
- Dev and production consume the same canonical immutable capture streams; do
  not duplicate paid odds polling merely to populate dev.
- After any operational or deployment change, verify and report each job's
  enabled/running state, last successful capture, next expected capture, and
  data coverage. Do not declare the system healthy while any feed is stale.
- Monitor the three jobs independently and alert the user immediately when a
  job or its scheduler stops, a run fails, or a capture misses its expected
  freshness window. Alerting must not depend on the collector being monitored.
- If a protected collector is interrupted, restore it immediately, preserve
  the outage interval in the audit record, and backfill what its official
  source still makes available. Report any irrecoverable gap to the user.

## Deprecated shared Dagster pipeline

- The shared legacy services `gestalt-dagster-code-server`,
  `gestalt-dagster-webserver`, and `gestalt-dagster-daemon` are deprecated
  and must remain disabled. Never start, enable, unmask, install, or restore
  them.
- `scripts/deploy_gestalt_orchestration.sh` is a fail-closed tombstone. Do
  not bypass it or use an older revision of that installer.
- Protected collection runs only in the isolated `gestalt-collection-*`
  services. Stopping or changing the deprecated shared services must not
  interrupt those collectors.
- A replacement nightly data/model publisher must use a separately named,
  least-privilege service and expose only the atomic daily pipeline jobs. It
  must not load the legacy all-in-one Dagster definitions.
- Production promotion must fail if any deprecated shared Dagster service is
  active. Promotion must never restart a deprecated service, even if an old
  operator state or unit file says it was previously active.

## ESPN prohibition

- NBA GESTALT must never request, ingest, copy, reconcile, repair, serve, or
  retain ESPN basketball data.
- Do not use ESPN as a primary source, fallback, crosswalk, score source, box
  score source, schedule source, player/team identity source, or recovery path.
- Do not add or preserve ESPN endpoints, source adapters, files, filenames,
  payload fields, game IDs, player IDs, team IDs, provenance labels, or cached
  responses anywhere in NBA GESTALT artifacts or serving roots.
- A bookmaker market delivered by an independent odds provider is market data,
  not permission to use ESPN basketball data. Normalize or exclude legacy
  bookmaker keys containing `espn`; never expose them as source provenance.
- If the official NBA source is unavailable or incomplete, fail the affected
  artifact explicitly and report it. Never silently substitute ESPN or any
  other unapproved source.
- Before materialization, release, promotion, or repair, scan the exact NBA
  GESTALT artifact scope for ESPN-named files and ESPN-specific fields. Treat
  any match as a hard failure. Never copy contaminated artifacts forward.
- Removal or quarantine must be restricted to the NBA GESTALT roots named in
  the task. Never delete ESPN data belonging to another project or site.

## Questions are not authorization

- Treat questions, requests for explanation, corrections, criticism, and
  statements of concern as discussion, not permission to implement a fix or
  expand previously approved work.
- When the user asks a question during implementation, pause changes, answer
  the question, and wait for an explicit instruction to resume implementation.
  Do not automatically resume earlier work after answering.
- Read-only inspection needed to answer the question is allowed. A question
  does not authorize editing files, reverting or deleting changes, changing
  configuration, running tests or operational jobs, materializing data,
  publishing artifacts, deploying, or updating tickets.
- Explain findings and proposed next steps without executing them. A correction
  identifying your mistake does not authorize repairing it or cleaning it up.
- After a stop instruction, do not perform cleanup, rollback, or other actions
  unless the user explicitly requests them. A later instruction authorizes
  only the action it names; it does not implicitly restart the stopped task.
- Explicit action requests such as "implement this," "update AGENTS.md," or
  "resume the work" authorize their stated scope. Polite action requests such
  as "can you update this file?" also authorize that specific action; questions
  such as "why is this broken?" or "what job does this?" do not.

## Collaboration

- Surface actual findings and failures to the user; do not bury them in files.
- Ask for input before making a new modeling assumption, changing a defined
  metric, or introducing a fallback that alters product behavior.
- If a task starts becoming materially more complex than the requested outcome
  requires, stop before adding a workaround, new architecture, extra tooling,
  or speculative optimization. Explain the concrete source of complexity,
  present the simplest viable choices, and ask the user to decide before
  proceeding.
- After completing any material work, promptly report the outcome to the user:
  what changed, the validation evidence, any remaining failures or limitations,
  and the exact next state. Do not silently stop after tool use, leave completed
  work unreported. Follow the Questions are not authorization rules above:
  after a question, correction, or stop, wait for explicit authorization before
  resuming changes.
- When the user stops or redirects work, stop immediately.
