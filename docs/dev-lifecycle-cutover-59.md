# Issue 59: automatic dev lifecycle and production cutover

Status: implementation and dev verification in progress. Production activation
requires separate user approval. Do not infer it from installation of dev jobs.

## Boundaries

- Shared official NBA/market captures remain running independently.
- The automatic lifecycle sensor uses the environment's permanent Dagster home.
- At most one daily/forecast operation may be queued or running in an environment.
- An incomplete earlier slate blocks advancement to a later slate.
- Empty dates carry identical model components forward only when needed to reach
  the next completed slate. No new fit occurs on an empty date.
- Forward work pins the accepted model prefix, exact observed inputs, worker
  revision and configuration. Checkpoint validation and the fenced publisher
  remain mandatory.
- Nightly data publication retains deployed application code and frontend.
- Forecast jobs reuse the accepted model and preserve locked historical forecasts.
- Actual minutes and retrospective grading use official final observations through
  the existing serving path, not a competing postgame forecast writer.
- Three failures stop retries and raise an error; a queue backlog is not recovery.
- The dev worker keeps NoNewPrivileges. A narrow Polkit rule permits restarting
  only gestalt-web-dev.service; production permission is not granted.

## Dev validation before approval

1. Run focused trigger, checkpoint, forecast, publisher and alert tests on EC2.
2. Install the sealed watchdog and worker through their deployment scripts.
3. Observe sensor-created runs in the permanent dev Dagster instance.
4. Verify every selected date/game and every affected player; compare frozen
   forecasts and market selections before/after, and explain every difference.
5. Verify new model lineage, HCA and coverage; verify application code is unchanged.
6. Inspect dev pages at wide/narrow sizes and present the Data Quality report.
7. Record exact release/edition/data identities and publisher generation for
   approved cutover and rollback. Do not fill these with guessed or future values.

## Production checkpoint

Production remains unchanged until the user reviews dev evidence and approves
the exact cutover. Promotion must reuse the validated data and code selections,
not regenerate the season or run a different model. Verify live APIs/pages and
collector status after promotion, and present results before closing #59/#52.

The initial regression run includes optional artifact-backed tests that are
skipped without their explicit EC2 inputs. Those skips are not end-to-end proof;
real dev execution and artifact-backed verification are still required.

## Verification checkpoint (2026-11-04 Pacific)

- Initial suites: 188 passed, 9 artifact-backed skips. Follow-up prior and forecast
  suite: 38 passed (overlaps the initial suites; do not add these totals).
- Full served-data audit: 528 players, 3,676 player-game rows, 20 dates, 154 games,
  119 finals and 119 actual ledgers, no mismatches. Generation 14 stayed unchanged.
- Automatic pregame execution failed before publication: the old template omitted
  model settings and historical runtime. Resolve those from verified accepted
  producer configuration / serving bindings, not legacy defaults.
- A display-only rankings copy lacks the metadata included in the original prior
  identity. The original coupled bundle reproduces the accepted fingerprint.
  Pin and copy all prior inputs (including rankings.json) into private work.
  Never change the accepted prior fingerprint to accept an incomplete copy.
- Protected collection services remain active; production has not changed.

Outstanding: real automatic publication, recovery/alert clearing, browser review,
and complete nightly lifecycle acceptance. The existing model-compatibility check
rejects any modeling-directory change and assumes Git metadata in the deployed
worker. User approval is pending for a precise calculation/input check with
equivalence tests for loading/storage changes. Do not bypass the existing gate.
The detailed GitHub progress comment was blocked by the permission reviewer;
approval to disclose that summary publicly is pending.

### Automatic activation result

Run 7c5c0eb6-a93c-4deb-8b69-1f195f88e0a8 passed artifact and temporary-API
validation, then failed live activation because forecast-registry/receipts was
root-owned and unwritable by the normal ubuntu worker. The publisher restored
the previous release automatically (generation 16); this run is a failure, not
acceptance. Commit b08179a adds that exact directory to repeatable deployment
permission setup. Installation verified ubuntu write access. The retry has not
yet been verified successful. Dev health returned HTTP 200 after rollback.
Do not remove the failed-run history or manually clear its alert.

### Approved compatibility repair (2026-11-05)

The user approved precise calculation/input verification and equivalence tests.
The portable gate follows internal imports from daily_model_replay and inference,
including package initializers and dependency definitions. It checks the exact
reviewed fingerprints stored in committed code. Deployment does not regenerate
or relax that policy. Unknown revisions and later dependency/calculation edits
fail closed. The worker records the verified policy checksum in its work spec.

Read-only EC2 equivalence test: old and current loaders produced identical values
for every loaded evaluator field; all 14 accepted data dates / 107 games verified
identically. Updater, minutes, HCA, context and signed-matrix implementation files
were byte-identical to accepted code. Plot import changes were AST-equivalent
after removing only the relocated matplotlib import. Suite: 47 passed, including
the real-artifact comparison. Expanded regression: 103 passed, 1 optional artifact
test skipped in that second invocation (it passed explicitly in the first).
Nightly automatic publication and all affected served outputs still need proof.


### Restricted-worker and empty-date publication fixes (Nov 5)

The native AWS CLI installer verifies the official signature and installs a
version-pinned, unprivileged binary. The worker retains NoNewPrivileges. Its
restricted read-only AWS authentication check passed. Protected collectors were
not changed.

Automatic run d49ad66e-ecc5-415e-aebf-1f1cbd01edd7 then completed data, model,
serving and lineup stages for the zero-game carry-forward date, but publication
failed because the API gate incorrectly required a positive daily schedule count.
The canonical gate now compares exact API game IDs and their count against the
verified daily NBA data artifact in initial staging, forecast publication and
post-activation verification. No date exception, fabricated count or historical
forecast change is used. Empty/missing/substituted/duplicate-game regression
cases and empty-date publication are covered: 95 passed, 10 optional tests
skipped. Real automatic publication remains to be verified; this is not a claim
that ticket 59 is complete or that production has been changed.


The automatic retry 89442a88-68a5-4e1d-98ce-657c1abace70 passed API coverage
validation, then failed because runtime staging inspected deployed sealed code
as a Git checkout. The canonical code inspector now verifies and reuses an
existing package identity/inventory; source checkouts still require clean Git.
Staging tests exercise a Git-free deployed package and reject tampered content.
Regression suite: 126 passed, 1 optional artifact comparison skipped (previously
run explicitly). No published release or historical forecast was modified.

Explicit recorded-source lifecycle tests against the current dev runtime found
four failures (5 passed): three assertions expect a retrospective field even
when a damaged/missing box is correctly reported, and the suspended-game path
fails on missing scheduling_instance_id. These remain unresolved; ticket 59 is
not complete. Real nightly publication and browser review also remain required.


The recorded-source lifecycle failures are corrected in the checkout: missing
retrospectives are asserted as absent/empty; locked selection metadata is read
from the canonical forecast member rather than the open-candidate envelope.
An added post-cutoff reassembly test requires exact original forecast, market,
scheduling ID and cutoff preservation. Explicit recorded-dev suite: 10 passed.
Related publication/compatibility regression: 28 passed, 1 optional artifact
comparison skipped. These changes do not affect a worker already executing its
sealed package; they require the next normal worker deployment.

Automatic run 90b024d8-f62e-4ba8-baa0-b70067df8a5c passed staging and moved dev
to the verified zero-game carry-forward, with post-activation checks still in
progress at this checkpoint. Independent comparison: all 121 historical
forecasts and market payloads equal acceptance-59/before.json. Application code
remains 4a344334236e5708efab8eda653538c64eafcc15. No production pointer moved.


## Issue 62 implementation checkpoint (2026-11-05 Pacific)

Classification: pipeline/storage reliability. Work is on EC2 in
`codex/fix-s3-daily-recovery-62`; no service or release has been changed.

Implemented in the checkout:
- Release archival re-verifies and registers both exact selected daily chains
  at staging, including zero-game, existing-model and checkpoint-resume paths.
  It does not depend on an in-memory list populated by prior callbacks.
- Forecast refresh outputs use their complete request identity as a directory
  name. Unchanged retries reuse their output; changed inputs get a distinct
  directory. Conflicting explicit targets remain errors. The complete request
  identity is now stored with the selection to make differences auditable.
- Regression coverage exercises a real zero-game carry-forward and durable
  data/model publication with an in-memory S3 implementation, then repeats
  archival after clearing process state and verifies no additional uploads.

Verification: 93 tests passed across daily pipeline, forecast refresh,
 automatic lifecycle, durable release, daily artifact and active migration
 suites. The two reproductions run against the original code fail with the
 original missing-dependency and conflicting-forecast-input errors. These
 are regression results, not dev acceptance or evidence of live recovery.

Remaining gate: complete S3-backed serving. Existing releases depend on bound
historical inputs (80 files, approximately 111 MB at inspection), the canonical
forecast registry (approximately 5.7 MB), and original daily-artifact paths.
The registry is an active authority: resolving a game after cutoff can write
its immutable lock selection, and successful publisher activation writes
publication receipts. Copying it into a frozen release without addressing
that authority is not a correct recovery implementation.

A contract decision was presented to the user before adding a new dependency
mapping/registry durability schema: include those missing dependencies and
resolve legacy references into verified EC2 cache while preserving original
bytes and the existing cutoff authority. That work is pending approval, not
implemented. Do not deploy these partial changes or mark #62 complete from
these test results. No production activation, historical replay, data rewrite,
collector change, or incident-state edit has occurred.

### #62 durable recovery implementation checkpoint

The S3 audit supersedes the earlier assumption that historical dependencies were
missing: all 80 pinned historical files already exist in the versioned
`gestalt-web-d739c28` S3 release and match the exact EC2 bytes. No historical
re-upload or model replay is needed. The agreed scope is recorded in issue #62.

The canonical producers now pin a durable serving dependency inventory, reuse
those historical S3 objects, archive immutable forecast registry records with
write-through publication, and restore release/data/model/history/registry
components into verified EC2 caches. Original manifests and forecast contents
remain unchanged. Code-only, daily, and forecast publication use the same
recovery contract; production remains separately gated.

Validation at this checkpoint: 160 focused tests passed, including interrupted
registry publication, unchanged/changed identities, cold restore without source
directories, corrupt/missing inputs, cache reuse, and existing publication/retry
paths. The source-policy audit passed for all 80 historical files, the registry,
and the active selected schedule. Real dev activation and daily recovery are
still pending; this checkpoint is not a production-readiness claim.

Real preflight exposed validator memory growth: it retained every full player
profile while checking only current-season trajectories. This exhausted the
700 MiB test cgroup on the 3.7 GiB EC2 host. The validator now retains only the
trajectories needed by the same complete coverage checks. The regression fails
against the preceding commit; 93 relevant tests pass after the repair. This is
validation-memory behavior only, with no model, forecast, or coverage change.
All further real validation runs are serial, memory-capped, and assigned a high
OOM kill priority to protect the running services. The earlier overlapping test
attempt caused a dev API OOM/restart; issue #62 records that incident explicitly.


### #62 live recovery checkpoint: resource blocker

The candidate at `f8e765c` passed complete canonical API validation and a second
fresh-cache S3 recovery with the old historical release, daily-pipeline root,
forecast registry, and previous cache hidden in a private mount namespace.
Both runs checked all 20 selected dates / 154 games and 3,676 player opportunities.
All 236 selected-schedule JSON files match the pre-change baseline.

Live activation at generation 26 failed because the installed dev service lacked
the canonical unit's `PYTHONDONTWRITEBYTECODE=1` setting. Python added 46 `.pyc`
files, correctly violating the code seal. The publisher restored and verified
the prior runtime at generation 27. The rejected release was left untouched.
The canonical service unit is now installed/reloaded; `5f3a482` adds a deployment
guard for that setting (8 targeted tests passed).

The next candidate, runtime
`378d7a942587014b123473f606160d7998aabe72ddea2c68b444c710633659af`,
was archived/restored but its preflight API was killed by a host-wide OOM.
The test's elevated OOM score protected the live services; no pointer moved.
The host is a t4g.medium (4 GiB, about 3.7 GiB usable, no swap), with roughly
3 GiB already occupied by services. Further deployment attempts are paused
pending a capacity decision. Dev remains on the prior runtime at generation 27;
production pointers are unchanged. Collector services have zero restarts and
latest capture jobs succeeded. No new daily worker or catch-up run was installed.
Rendered dev verification, real daily recovery, and ticket completion remain
pending. This checkpoint is not a deployment-readiness claim.


### Issue 62: verified serving recovery and approved compatibility review

On November 6 UTC, code `0f0fd7e` activated dev generation 30 through the canonical publisher, runtime `042869e115b05d0895f14db3f85568dabb72b0788ae97425e7a1f5df21688e3f`. Full API coverage passed for all 20 selected dates / 154 games, with all 236 selected-schedule JSON hashes unchanged. The live API uses the S3 recovery cache. Restart retained the same loaded identity and the sizes/mtimes of all 40,173 existing cache files. Model evidence remains November 2 / 107 games; this is not completed catch-up.

EC2 Chromium rendered six main pages at desktop and phone widths without JS/API errors, using the deployed static files and live API through a temporary local streaming test server. Authenticated public-proxy browser acceptance is not established. Wins overflowed horizontally at both widths, and the narrow team page also overflowed; layout fixes are outside #62.

The new automatic worker submitted permanent dev run `119a41e3-67ec-427b-9068-54b65779b015`; it failed before data processing because the reviewed model-compatibility inventory includes the changed web requirements file. Automatic daily lifecycle is paused to avoid repeating this known failure. All dev services and independent collectors remain running. A runtime-only 900 MiB/no-swap cap bounds the dev code service and job descendants. Production is unchanged.

The first rebuilt orchestration environment unintentionally resolved botocore 1.43.125 instead of 1.43.124 and threadpoolctl 3.8.0 instead of 3.7.0. The proposed requirements pins retain the accepted versions. A freshly built candidate environment (`requirements_sha256=f766104c7fd0dc21423d0cc088900f31c605d93d33c2452de675a3f3e94c962a`) has the same Python version and all 89 package versions as the accepted environment (`requirements_sha256=be07cd04e2ee64a9589a66d4e7ade0f6885535aa2a2e89b0e69f4dfb4d83e5a5`). Of 75 calculation-inventory entries, only the two requirements files differ; calculation source files match the reviewed policy. Twenty-three focused runtime/deployment/storage tests pass. This environment is prepared but not activated.

The user explicitly approved a narrowly scoped equivalence review of these two requirements changes for accepted model revision `1459c83f82b1fa50d901048e65bb39c9fd678204`, including updating its reviewed compatibility inventory, deployment, and normal forward catch-up. The policy update changes only the two approved requirements hashes; all calculation-source checks remain enforced. Do not bypass the guard, replay accepted model history, or rewrite frozen forecasts. Evidence is retained on EC2 at `/tmp/gestalt-62-compatibility-review.json` and `/tmp/gestalt-62-pinned-runtime-tests.log`.

The user approved deletion of exactly two unused serving recovery caches (`10f656c6...` and `378d7a94...`). Their S3 storage manifests were verified and their archives preserved; only those two cache directories were removed. The audit is `/tmp/gestalt-62-approved-cache-cleanup.json`. No other NBA artifact/cache deletion is authorized by that approval.


### Memory review and shared daily orchestration (#59)

The dashboard-only installer and 1,050 MiB worker proposal were prepared but never installed. The user superseded that workaround with one shared daily Dagster deployment, separate dev/production code locations, and dev automation enabled only for deliberate test windows. Implementation is tracked in #59, separate from #62's S3 recovery work. Production cutover still requires explicit approval after dev evidence. The proposed changes were preserved on EC2 in `/tmp/gestalt-59-superseded-dashboard.patch` before restoring the canonical installer.

Measured idle service memory before this change: production API 524 MiB; dev API 555 MiB; production daily-job services 623 MiB; dev daily-job services 525 MiB (dashboard about 123 MiB); isolated collection services 433 MiB. The failed catch-up also launched a candidate API using about 440 MiB, plus its run process. This is multiple Python services and copies of serving data; boto3 is not the dominant consumer. Current dev remains generation 30; the next catch-up attempt must still demonstrate success within the revised budget.

The user additionally approved removing exactly the two unused serving caches for runtimes `4719fd66...` and `ee96b046...`. Their S3 manifests were verified and S3 archives/release packages retained. Both removals are recorded in `/tmp/gestalt-62-second-cache-cleanup-review.json`. The briefly frozen dev worker was thawed; no protected collector was paused.

### Shared daily Dagster implementation checkpoint (2026-11-06)

The implementation is on `codex/shared-daily-dagster-59`, stacked on the #62
recovery branch. It supports a shared, registered environment binding; scoped
retry queries and watchdog automation lookups; a global execution lock; dev
validation windows off by default; and isolated service plan preparation from
independently sealed code packages. The production code service uses the dedicated
`gestalt-daily-prod` account and a policy permitting only its API restart. The
controller cannot write either environment's data roots, and each code location
mounts the other environment's root read-only. Live deployment is not yet performed.

An isolated rehearsal at `/tmp/gestalt-59-actual-history-rehearsal` preserved all
61 actual dev/production run records, their event records, and archived source
files. Manifest checksum: `d2cd61fd745c9c5f8d21c3137e3307907a157c66d5ea84083e78325bdbdd9841`.
The original histories were unchanged. This is migration evidence, not a real
job run or a substitute permanent instance. Legacy production backfill failures
remain failures and cannot be cleared by a different job's success.

The focused suite passed 80 tests, with one opt-in model loader comparison
skipped; actual shared service validation remains required. The user expanded
root EBS volume `vol-057082397c4de7463` (gp3) from 30 to 60 GiB. After verifying
the AWS modification was optimizing and Linux saw the new disk size, this task
ran a growpart dry run, expanded partition 1, and grew its ext4 filesystem online.
The filesystem now reports 58 GiB total, 30 GiB available, and 49% usage.
Both websites and all collection services remained active with unchanged restart counts. Shared orchestration, data catch-up, full coverage/UI
acceptance, and the separately approved production cutover are still outstanding.

Production bootstrap provenance: the current production model is the accepted
November 1 artifact `2fb91ebfcb125162bb35fad27111eaba4b20c8eb95d0505cacf10a2ec231dba0`,
created before the canonical producing-work records. Its existing pipeline
configuration verifies against model revision `1459c83f82b1fa50d901048e65bb39c9fd678204`,
settings checksum `558385560bbb2b2d518bceb847468e2f87900a33964191f54a4ecc2bd07534c3`,
and complete coupled prior `2d92682a63b485d1e47111a31e42b2e2437d866cb97ba73318822feb057c693e`.
The shared preparer has an explicit `--verify-legacy-production-config` mode that
checks those exact accepted inputs and records a new verification document. It
never fabricates the absent original job record or recomputes the model. Ordinary
worker deployment still requires verified producing work; corrupt/mismatched
producing work cannot take the legacy verification path. The focused follow-up
suite passed 42 tests, including mismatched legacy-input rejection. EC2 evidence:
`/tmp/gestalt-59-production-provenance-review.json`.

### S3 candidate activation boundary (2026-11-06)

Automatic run `12d113c5-9225-4e1c-be8c-64b5871a5a26` completed candidate
validation without an OOM (code-service peak 1,064,415,232 bytes with its
1,101,004,800-byte cap). Publication correctly stopped before pointer movement:
the restored S3 release was under the configured artifact cache, while activation
still accepted only the pre-S3 pipeline releases directory. This is a #62 storage
integration defect within the #59 prerequisite, not a changed model assumption.
The canonical activation boundary now accepts only the exact environment cache
path named by the fully verified durable runtime identity. Other environment
roots, digest mismatches, missing durable inventories, escaped symlinks, and wrong
live pointers are rejected. Tests covering activation, rollback, storage, and the
new rejection cases passed: 54 tests. The original failure remains in permanent
Dagster history; only a real successful retry may recover its incident.

### Shared control-service rehearsal and monitoring plan

The real systemd rehearsal used the sealed `68cd5a6` package and a test-only
copy of the 61 original runs at `/var/lib/nba-gestalt-daily-preview/68cd5a6`.
Both code locations loaded; all four sensors were STOPPED; no real work ran in
the copied history. The two code servers, shared dashboard, and shared daemon
were active with zero restarts, and `/proc` mount checks proved each code
location saw the opposite data root read-only. Their combined memory was
515.7 MiB with automatic work stopped; this is not a steady-state production
measurement. The original dev stack was restored after the rehearsal.
Production and protected collectors were not switched.

The service plan now includes independent dev/production watchdog services and
hourly timers using the shared home and dashboard, with environment-scoped run
and sensor lookup. It retains the existing dev watchdog cadence. Production
monitor installation is included in the separate production approval gate.
`verify_plan(path, expected_sha256)` rechecks the reviewed plan hash, every service
file, each sealed code package, its installed runtime, and configuration identity
before installation. The focused shared/history/probe suite passed 34 tests.

The apparently different accepted dev and production preseason-prior identities
were traced to metadata: ratings parquet bytes match at
`569b36fbcc4dfca51a96620c2c1d1feb2ada0db01186c5e76fed820f9bba9a73`, and minutes
JSON bytes match at `579d7b5d7852a9b5339e08dd27dccf5cd6b224d43308021317f1bf0cdb194226`.
Dev additionally retains the ratings metadata sidecar, while the original
production release does not. Preserve both original provenance identities;
this observation does not authorize changing either historical model artifact.

### Shared orchestration cutover procedure (pending production approval)

This procedure is prepared for review; none of these production steps has run.
The final approval record must bind the final plan checksum, code/configuration
identities, validated site release/edition/data revisions, publisher generation,
rollback site identity, and the collector verification report. Current rehearsal
hashes are evidence, not authorization to install a later plan.

1. Verify the reviewed plan with its recorded checksum. Verify both site
   selections still match the approval record. Record the exact installed unit
   files, enabled states, code/configuration pointers, and original daily homes.
   Recheck all deprecated `gestalt-dagster-*` units are disabled and inactive.
2. Disable only the two old daily stacks' sensors, drain queued/running work,
   and stop their six daily services. Leave `gestalt-collection-*`, both APIs,
   and raw collectors running. Reject migration if any run remains active.
3. Create a **fresh** audited history migration from the two permanent original
   homes into `/var/lib/nba-gestalt-daily/dagster`, using the recorded approval
   reference. Do not reuse rehearsal histories after source histories advance.
   Verify every original run body, event history, recovery identity, and archived
   source file. Keep both original homes and migration evidence intact.
4. Install only the reviewed shared service files, workspace, and Dagster config.
   Use the existing dedicated `gestalt-daily-prod` user for its code location and
   install the narrowly scoped `49-gestalt-shared-prod-api.rules` policy. Grant
   it write access only to required production publication/working/cache
   directories and the shared Dagster history; preserve immutable artifact files.
   Record permission changes and verify dev cannot write production output.
5. Start the two code locations, shared dashboard, and shared daemon with all
   sensors initially stopped. Verify location/config identities, API health,
   mount isolation, shared history totals, and the global one-run limit. Install
   the scoped watchdog units/timers and verify real environment reports.
6. Disable the old daily units only after verification. Explicitly start only
   the production sensors in the shared home; leave the dev window closed.
   Verify sensor state by code-location name, not the shared sensor name alone.
   Run any separately approved site promotion through the canonical publisher;
   no pointer edits or lower-level publication shortcuts.
7. Verify actual prod APIs and rendered pages against the approved identity and
   full coverage, audit both environments' histories, and record each protected
   collector's state, latest/next capture, and complete applicable coverage.
   Measure idle/active memory with the actual production schedule enabled.
8. If startup verification fails **before any real shared run**, stop the new
   stack and restore the recorded old daily unit/sensor states and watchdog
   configuration. Keep the failed migration copy for audit. Once a real shared
   run exists, keep that shared history authoritative: stop affected automation
   and repair/redeploy compatible sealed code. Never fall back to stale original
   histories and lose new runs or replay already completed work. Site activation
   failure uses the publisher's verified API-and-pointer rollback; later site
   rollback must likewise use the approved canonical workflow and recorded target.

Remaining acceptance work: successful dev catch-up, complete frozen forecast and
market comparison, automatic lifecycle evidence, rendered dev review, exact final
promotion/rollback release identities, and explicit user production approval.

### Approved creation-timestamp retry repair

The user approved reusing an existing artifact after reviewing its content-derived
identity and the complete file comparison. The narrow implementation is in the
durable daily publisher: when the only manifest metadata difference is
`generated_at`, canonical identity/provenance, dependencies, and every other
inventory entry must match. The original S3 objects are checked for presence,
size, and checksum; their payload bytes are read through the verified cache.
The result identifies the original storage manifest and original creation time.
Neither the existing S3 artifact nor the new local candidate is overwritten.
Other artifact kinds retain strict byte-level inventory matching. A changed
payload/provenance is a new identity or a validation failure, never an overwrite.

Tests: 37 passed; one pre-existing opt-in model-loader test skipped. They exercise
independently created data and model artifacts, original-manifest preservation,
changed source/provenance and payload rejection, and missing/corrupt S3 objects
with both fresh and populated caches. The real November 3 artifacts were checked
with an S3 client that forbids writes: 11 data files and 13 model files verified,
zero uploads, and local manifests unchanged. Evidence is on EC2 at
`/tmp/gestalt-62-creation-reuse-verification.json`.

The compatibility review verified all 75 reviewed calculation files unchanged
against accepted model revision `1459c83f82b1fa50d901048e65bb39c9fd678204`.
The existing compatibility policy remains unchanged; no new exception, dependency
version, schema, model calculation, or historical forecast is introduced.


### Final shared-service rehearsal and memory incident (2026-11-06 UTC)

The approved S3 creation-time retry repair succeeded in real automatic run
`2bf7e7b7-20f6-4c3d-bfb2-19fa5470369b`; dev generation 31 serves runtime
`d115ef1e9ab7d34aa7af1c3b07dd4483e3921b32c0437295e3ba0b2838805baf`,
evidence through November 3. The full served comparison covered 20 dates, 154
games, and all 123 completed games: zero frozen-field differences, no comparison
errors, and unchanged production code/data pointers. Evidence remains on EC2 at
`/tmp/gestalt-59-served-comparison-d115ef1e9ab7.json`.

The November 4 run `6c595587-d894-469b-9778-a71474a39edf` failed during model
preparation because its private context API exceeded the startup timeout. This
coincided with the hourly data-quality process (approximately 503 MiB) and the
existing duplicate daily stacks. At 16:02:25 UTC the kernel recorded a **global
out-of-memory kill of the production API**, PID 1477727. Systemd restarted the
service at 16:02:30; application startup completed at 16:03:02. Production and
dev health endpoints subsequently returned HTTP 200. Do not characterize this
as an uneventful retry or claim there was no production impact. Production's
release selection did not change. Protected collection services stayed active
with zero restarts; the 16:00 schedule/injury collection completed successfully.

Both dev sensors were stopped through Dagster's supported CLI after the incident;
no active/queued dev runs remained. The dev dashboard was restored for monitoring.
Do not resume heavy catch-up in the duplicate-stack configuration merely because
the retry cooldown expires. Neither #62 nor #59 is complete.

The final prepared shared package is:
`/var/lib/nba-gestalt-dev/shared-daily-releases/revision=5a8154c930fbc040440c86a8a58093a86402e922`.
Package checksum: `c1a13a2c89da02f413604cc72cd75bc374f18211ccb48741fc357ed877da4d66`.
Plan: `shared-plan/plan.json`, checksum
`91dcdb5859fcd17dc35dc07a5f72d5cc09f66595cc6aefa2068accd2ef28ed2b`.
Dev config: `a1ea01a8a298bb9a2f9b0c785ecc8a8e533c79d768042b99e34527b716478f71`.
Prod config: `f5d265c9e59c86223c2928aa9e8324d05d8d06bf311a818818b3a6aa7e1a8e13`.
Both use the existing pinned orchestration runtime and reviewed model revision;
no new model/dependency version is proposed.

Actual systemd rehearsal evidence is under
`/var/lib/nba-gestalt-daily-preview/5a8154c`. It copied and verified all 66 original
runs; migration checksum
`2ef49de1e101893fd1c01e62801363dd77384c968308b579d5d748885297a155`.
Both locations loaded, all four sensors stayed stopped, opposite-environment
mounts were read-only, all services had zero restarts, and total idle memory was
515.3 MiB. No real jobs used that copied instance. This is a rehearsal measurement,
not a promise about warmed production memory or active-job peaks. The original
dev services were restored afterward. A live cutover still requires a fresh
migration after stopping/draining the original daily services.

Proposed revised approval boundary: install only the reviewed shared orchestration
and migrate its real histories first, with both locations' sensors initially
stopped. Open a dev-only validation window in the permanent shared instance to
finish November 4/5 catch-up and final site checks; temporarily leave production
daily automation paused during that window. Keep both sites on their existing
selected releases, and leave protected collection running. This temporary pause
and production orchestration replacement require explicit user approval. Site
promotion and enabling production publication remain a later approval checkpoint.
The shared dev worker has the reviewed 1050 MiB limit and OOMScoreAdjust=1000;
measure active memory and stop on resource failure rather than promising that
consolidation alone proves adequate capacity.

Production permission preparation must be restricted to mutable containers:
`/var/lib/nba-gestalt`, `daily-pipeline-v2`, its new `forward` and `partitions`
containers/runner lock, `daily-requests`, `artifact-cache`, `forecast-registry`
and `receipts`, and `release-publisher` with `generations`, `journal`, and its
publisher lock. Record current ownership/modes before changing them, create
missing containers for the dedicated production user, and grant only the shared
history and these publication/work containers the required group write access.
Do not recursively rewrite existing artifact files or grant write access to the
captured source/prior inputs. Verify these exact paths against the selected
configuration before installation. The live plan remains uninstalled.
