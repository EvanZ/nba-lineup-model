# Integrated lifecycle verification (#57)

Status: implementation in progress. This is test/pipeline verification, not a
model change or an authorization to deploy production, enable the new nightly
sensor, or replay the real season. The minutes clipping study stays in #60.

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

Nine cases currently cover:
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
PYTHONPATH=src:/tmp/nba-gestalt-test-deps OPENBLAS_NUM_THREADS=1 \
  /opt/nba-gestalt/releases/gestalt-web-d739c28/.venv/bin/python \
  scripts/verify_release_lifecycle.py \
  --source-release /var/lib/nba-gestalt-dev/live-release \
  --output /tmp/gestalt-lifecycle-57-NEW-RUN
```

The output directory must not exist. The runner pins the resolved source
release at startup and emits JSON, HTML, JUnit, individual test results, source
hashes and explicit remaining work. A skipped integration suite is not success.

## Still required before accepting #57

- Connect the isolated Dagster job, data builder, coupled updater, all serving
  surfaces and publisher in the same nightly lifecycle fixture.
- Fault-inject the real assembled runtime/API activation, not just publisher
  unit-test callbacks; test multi-request browser coherence during activation.
- Repeat full materialized dev API and rendered coverage checks, including
  missed-game rows, points, badges, ledgers, metrics and Wins advancement.
- Resolve or explicitly assess the existing 768px header overflow and verify
  NBA logo rendering; capture viewport evidence.

The earlier focused tests remain regression coverage, not a substitute for
these outstanding integrated acceptance checks.

## Initial verification result

On EC2, the first tranche passed 155 tests with no failures or skips, including
all nine new recorded-source integration cases. The complete fixture slate was
Oct 20 (three games), with 68 positive-minute NBA player rows checked against
the retrospective ledger. Forecast/source artifact hashes remained unchanged.

Source runtime:
`d3e989748736ea256913de49c5a7fd813770aa9ce05273c036900181b23813b2`.

Evidence on EC2:
`/tmp/gestalt-lifecycle-57-verified/report.html` and adjacent JSON/JUnit/logs.
The report was rendered at 1440, 768, and 390 pixels without horizontal
overflow. This is validation of the report layout, not a resolution of the
separate site-header overflow listed above.

No dev/prod activation, collector change, nightly sensor enablement or real
season replay occurred. #57 remains open until the outstanding lifecycle and
rendered-site checks are complete.
