# Release and artifact contracts

Approved on 2026-11-03 for GitHub issue 53. This slice adds schemas and
contract tests only. It does not migrate jobs, activate artifacts, rebuild
history, change the site, or deploy a release.

## Boundary

Collectors record observations. Model jobs calculate from explicitly selected
data. Forecast jobs produce auditable game forecasts. A publisher selects the
complete view the website may serve. Readers consume that selection and may not
invent fallback forecasts or follow another workspace's mutable pointer.

The observation clock and model clock remain separate. Injuries, odds, scores,
and finals can advance independently. A pregame injury update can create a new
forecast against the same daily model state; it cannot refit the model or alter
an already selected historical forecast. A malformed feed retains its last
valid explicit selection and cannot stop unrelated collectors.

## Immutable identities

Every artifact reference includes kind, schema version, content SHA-256, and a
retrievable URI. `latest` and `current` aliases are invalid as artifact
identity. Promotable artifacts require a sealed code-package hash, dependency
lock hash, and Git revision. Publication and activation are append-only
receipts, separate from reusable artifact manifests.

The executable catalog in
`nba_lineup_model.season.release_artifact_contracts` covers:

- raw captures and normalized observed revisions;
- independent selected observation generations;
- exact daily data partitions, including zero-game days;
- coupled ratings, minutes, and context model state;
- official scheduling instances, forecast versions, and publication receipts;
- authoritative forecast and independent evaluation-market selections;
- original-live and retrospective season editions;
- complete surface bindings and coordinated releases;
- fenced, monotonic activation receipts.

Existing daily data, daily model, schedule, and serving artifacts remain the
implementation foundations. These contracts constrain their future
publication and selection; they do not create a parallel data platform.

## Selection rules

The original forecast is the latest successful publication strictly before the
official scheduled tipoff for that scheduling instance. Computation time and
file modification time do not establish eligibility. A forecast first
published at tipoff or later is ineligible.

The evaluation market is selected independently: the latest valid archived
bookmaker-consensus snapshot captured strictly before that same cutoff.
Polymarket remains a separate source. Missing forecast coverage removes the
game from winner, Brier, ATS, and model-versus-market denominators while
retaining the game and its observations. Missing market coverage retains winner
and Brier eligibility but removes ATS and model-versus-market eligibility.
Both failures must be raised and reported; neither may suppress the game.

An ordinary delay does not move the cutoff. An official pre-play reschedule
creates a new scheduling instance and retains the old one for audit. A suspended
game retains its original selection.

## Replay and correction rules

A forecast can use only model evidence through an earlier complete Pacific-date
slate. Same-slate and target-game leakage are rejected. Retrospective editions
must be labeled, disclosed, and explicitly approved; original receipts remain
intact.

Observed corrections create new revisions and an impact assessment. A
presentation-only correction may rebuild presentation artifacts independently.
A historical model-input correction requires explicit approval and invalidates
the dependent causal suffix. No correction or replay silently changes a past
forecast or metric edition.

## Coordinated release and activation

A release contains complete bindings for Schedule, Player pages, Rankings,
Teams, Lineups, and Wins. Each surface declares the identities it needs. The
current-model surfaces advance together; live observations may be newer only
where the surface contract allows it. A missing or incompatible binding rejects
the release.

The later filesystem publisher will use one exclusive lock, verify an expected
current generation, write and flush a complete immutable generation directory,
and atomically switch one pointer. A stale writer is rejected. Rollback creates
a new monotonic generation that reuses earlier approved artifacts; it never
rewrites activation history.

### Executable code seal

Runtime staging exports the exact committed Git tree into a release-owned
`code/` directory. The source checkout must be clean. A symlink to any checkout,
including a detached checkout, is not a code package.

`code-package.json` contains the existing `SealedCodeIdentity` and the complete
file inventory. The runtime identity pins the same code identity, so its
content hash also binds the package digest, lock digest, and Git revision.
The staging verifier and publisher both recompute the file and package digests.
The code tree is read-only; Python bytecode cannot be written into the package.
Subsequent edits or commits in the source checkout cannot change the release.
Publisher generation directories may reference the sealed runtime, as before.
Their code and manifest references must resolve to that exact runtime's owned
package, and its digest must match the activation receipt. The runtime itself
may never reference a checkout.

Legacy releases without a pinned code package are rejected explicitly. This
change does not rewrite old manifests, move pointers, or migrate existing
activations. Repairing an active legacy release requires an explicitly approved
migration; a missing seal must never trigger a permissive fallback.

Tests exercise real Git checkouts, staging, and the publisher together, including
checkout advancement, altered or missing package files, and a forged code seal.

## Acceptance examples

Focused tests prove these contract cases:

- 19:59 publication is eligible for a 20:00 tip; 20:00 is not;
- a 19:59 computation first published at 20:01 is not eligible;
- a newer pre-tip market snapshot is selected independently of the forecast;
- zero-game day is a clean no-op and partial slate cannot drop a game;
- same-slate replay evidence is rejected;
- mutable artifact aliases and unsealed code are rejected;
- incomplete or split current-model surface bindings are rejected;
- a stale expected generation is rejected;
- official reschedules require an explicit superseded scheduling instance;
- forecast ledgers must reconcile to the reported spread.

### Explicit migration of a pre-seal runtime

The daily activation command accepts `--legacy-migration-approval` only for an
explicitly approved migration. It constructs a separate sealed rollback runtime
from the existing activation's exact Git revision and recorded package checksum.
The old runtime and generation are never edited. Every non-code file and link,
the approved artifacts, and all existing runtime identity fields must match.
Missing recorded identities, different code bytes, changed data, an altered
receipt, or a different active generation fail before pointer movement.

The migration receipt is persisted in the daily report, publisher journal, and
new generation. Normal activation and rollback use fully verified sealed
packages. Retrying an interrupted activation retains that same rollback target;
it does not return to the mutable checkout. This mechanism does not authorize
historical data corrections, model replay, forecast changes, or production rollout.

Resumable preparation is handed to the same daily activation command with
`--work <canonical-work-directory>`. The command verifies the pinned inputs,
configuration, every ordered checkpoint and parent receipt, the sealed runtime,
unchanged schedule, and expected active release before creating its activation
report. Operators must not translate `serving_validated` to `validated_candidate`
by editing JSON or invoke the lower-level publisher instead of this handoff.
