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
