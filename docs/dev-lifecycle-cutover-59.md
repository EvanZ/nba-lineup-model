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
