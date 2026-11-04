# Integrated lifecycle verification (#57)

Status: implemented; acceptance is determined by the clean-commit EC2 command
below. This is verification only, not a model change or authorization to
activate dev/prod, enable a new sensor, or replay the real season.

## Recorded-source game lifecycle

`tests/test_release_lifecycle_integration.py` reads an explicitly pinned dev
release, verifies its selection map and release bindings, and uses its entire
earliest completed slate and archived official NBA boxes. All mutations are in
pytest's private EC2 directory. Network requests are blocked. Source hashes
are checked again after each test.

The fixtures simulate publication times; they do not assert that the replay
edition was historically published at those times. The injury/rescheduling
events and bookmaker snapshots are controlled test variations, not new
production data or a model-performance study.

The integration uses the real forecast registry, selection builder, release
binding loader, API middleware/endpoints, official box parser, ledger builder,
grading, metrics, and records. The model evaluator is supplied as an unused
placeholder: these endpoints do not require it. No tested endpoint response
or schedule/ledger/metric reader is mocked.

Recorded-source cases cover:

- Full slate pregame publication, delayed cutoff processing, final scores
  before boxes, official box arrival, actual contributions, metrics and W-L.
- Injury-driven minute/spread revisions before cutoff and observation-only
  injury updates after cutoff.
- Missing original forecasts, including a late publication attempt.
- Missing, wrong-game and malformed boxes; explicit errors without filtering
  other games, and late repair without forecast changes.
- Stale API release pins and tampered selected forecast files.
- Ordinary late starts versus official rescheduling.
- Suspended/resumed games retaining original forecast and market selections.

Actual contributions are checked using the existing normalized formula
`5 * actual minutes / team actual minutes * frozen NAIL`, including the NBA's
rounding differences in summed player minutes. No formula was changed.

## Reproducible EC2 command

Use the already installed project runtime and test dependencies:

```sh
PYTHONPATH=src:/var/lib/nba-gestalt-dev/test-tools/pytest-8.4.2 \
  OPENBLAS_NUM_THREADS=1 \
  /opt/nba-gestalt/releases/gestalt-web-d739c28/.venv/bin/python \
  scripts/verify_release_lifecycle.py \
  --source-release /var/lib/nba-gestalt-dev/live-release \
  --source-work-root \
    /var/lib/nba-gestalt-dev/release-architecture-56/approved-nov2-bound \
  --output \
    /var/lib/nba-gestalt-dev/release-architecture-57/acceptance-COMMIT
```

The output directory must not exist. The runner pins the resolved source
release and work root at startup. It emits JSON, HTML, JUnit, individual test
results, runtime manifests, activation evidence, browser evidence, logs and
screenshots. A failure or skipped integration case makes the command fail.

## Connected nightly lifecycle

`tests/test_nightly_release_lifecycle_integration.py` runs the real isolated
Dagster job from the approved #56 Nov. 2 inputs. It builds observed data,
coupled ratings/minutes, all six serving surfaces and one runtime without
activating a live environment. Its materialized API assertions require:

- all 15 Nov. 2 games and all 107 completed-season games;
- 528 current rankings players;
- schedule badges, projected and retrospective ledgers and aggregate metrics;
- Curry's Nov. 2 Phoenix row and trajectory point;
- coherent players, rankings, teams, lineups, wins and schedule endpoints;
- unchanged source artifacts and no retraining.

The publisher test uses private pointers only. It proves successful activation,
stale-writer rejection, verification-failure rollback and interrupted-pointer
recovery. No dev or production pointer is used.

## Rendered browser and release transition

The runner starts the assembled API privately and renders schedule, player,
rankings, team, lineups and wins pages at 1440, 768 and 390 pixels. It requires
the same counts as the API, no application alerts, complete visible official
NBA-logo rendering and zero ESPN requests.

EC2 Chromium's direct HTTP/2 requests to the official NBA logo CDN are
unreliable. The test-only route fetches the exact requested `cdn.nba.com`
asset through Node and fulfills it unchanged; this does not alter the site,
the artifact or its provenance. The headless browser process is restarted
after every view so the 18-view matrix fits the production-sized EC2 host.

During the final browser check, the proxy deliberately switches from the
pinned old API to the private new API after serving the old release identity.
The client must observe a fenced 409, reload the new release identity and then
succeed. A mixed-release 200 response fails acceptance.

## Layout assessment

Functional assertions and screenshot capture are separate from layout
findings. The current site retains its existing desktop table minimum widths:
all six pages horizontally scroll at 768 pixels. At 390 pixels, player,
rankings, team and lineups retain a 1280-pixel width and wins retains a
1342-pixel width; schedule is responsive. These are recorded in
`browser-evidence.json` and are not data loss, mixed-release rendering, or an
alert. Any UI change to remove those minimum widths is separate from #57 and
must still follow dev-first review.

## Safety boundary

The verifier creates only private output roots and private API/publisher
processes. It does not activate dev or production, change collectors, enable a
sensor, replay a live season, alter frozen forecasts or introduce ESPN data.
The production and dev release pointers must be unchanged before and after the
run.
