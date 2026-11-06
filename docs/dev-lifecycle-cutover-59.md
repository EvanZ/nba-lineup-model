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
