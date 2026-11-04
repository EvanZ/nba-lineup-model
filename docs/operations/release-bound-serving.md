# Release-bound serving (#55)

This slice changes serving/publication infrastructure, not model algorithms or
historical evidence. Production migration remains gated by #59 and user approval.

## One selected view

A staged release declares `serving-bindings.json` with explicit immutable paths,
checksums, selected edition, observed-data revision, and per-surface dependencies.
The API resolves the activation once at startup. Ratings, profiles, lineups,
catalogs and historical caches cannot change underneath a loaded process.
The existing historical runtime is checksum-pinned, not copied to the laptop.

`selected-schedule/forecast-selections.json` identifies each game's forecast,
minutes/injuries/ledger inputs, and separate market selections. Cards, grading,
actual-minute attribution, team records and performance use the same reader.
Existing approved replay selections are imported unchanged and labeled
`preserved_existing_selection`; no live publication receipts are fabricated.

## Future original-live forecasts

The forecast job stages a validated `ForecastVersionManifest` plus its exact
payload in `ForecastRegistry`. Assembly supplies the explicit version references
and registry root to `stage_production_site_release`. This does not poll providers
or train a model. The publisher records eligibility only after the API verifies
the loaded activation. Recovery is never backdated.

At/after official scheduled tip, the registry authority serializes an immutable
selection: the latest successfully published **strictly pre-cutoff** forecast.
A late worker uses the same rule. Open candidates remain replaceable before tip;
unpublished, failed and in-game publications cannot become the original.
The common reader consults this authority for original-live entries, so waiting
for a periodic worker cannot expose a late candidate.

Bookmaker consensus and Polymarket are selected independently from their valid
pre-tip capture records. They remain separate comparisons. An official pre-play
reschedule uses a new scheduling instance. Ordinary delay cannot move the cutoff.
Suspension retains the original forecast and market choice.

Missing original forecasts produce a durable registry failure and visible
per-game error. They are excluded from forecast metrics, not from the schedule
or observed team records. This must not be inferred from missing legacy receipts.

## Independent live observations

A request captures one observation-only snapshot: NBA scores/box scores, official
injury reports, and market observations. Collector forecasts, final grades and
retrospective predictions are not accepted as model inputs. The snapshot supplies
every related schedule read in that response.

New final grades use the selected pregame. Actual-minute ledgers use that game's
frozen player ratings and context; missing box scores are reported and do not
block grades or other games. Existing selected historical result revisions are
flagged, not silently regraded. No injury/odds/score collector is stopped by this
serving change.

## Browser consistency

Each mounted page obtains a release pin. Related API requests send it; a stale pin
gets HTTP 409. The page discards pending responses and remounts together. A route
change starts a fresh pin rather than retaining a browser-session-wide version.
The Schedule view gets games, calendar records and aggregate metrics in one API
response and one observation snapshot. It does not invent a replacement forecast
from current team strength when an original is absent.

## EC2 verification / migration

- `scripts/validate_release_bound_serving.py` captures all current player pages,
  all team pages, full lineups, wins/records/performance and every served date.
  Its comparison reports any changed data, rather than checking a few players.
- `scripts/stage_release_bound_serving.py` binds an existing reviewed dev state;
  the existing data, model, serving and lineup content IDs must remain unchanged.
  It does not activate an environment.
- Activation still uses the generation-fenced publisher from #54.
- `test_forecast_publication.py` tests cutoff, receipt, reschedule and suspension
  rules. `test_release_bound_serving.py` tests selection integrity, observation
  isolation, actual-minute arrival, missing forecasts and stale page pins.
- `node apps/lineup-explorer/scripts/test-release-consistency.mjs` tests concurrent
  requests and delayed responses across page/release switches.

The dev migration intentionally preserves the currently approved model evidence
date. It does not perform the missing nightly data/model work assigned to #56;
new serving machinery is not evidence that missing player updates were rebuilt.
