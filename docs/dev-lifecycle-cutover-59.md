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
