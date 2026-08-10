---
last_updated: "2026-08-09"
---

# Train Aging Portable Contextual RAPM

This command trains the bounded hierarchical portable-matchup contextual RAPM
candidate with an aging-model prior for returning players:

```bash
uv run nba-train-forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm \
  --through-season 2025-26
```

The resulting artifact is written beneath:

```text
artifacts/models/forward_aging_bounded_hierarchical_portable_matchup_contextual_rapm/2025-26/
```

## Prior Boundary

For each season (t), returning-player priors come from an aging ridge model
fit to the candidate's own completed RAPM transitions through (t-1). The
aging model predicts a target-season RAPM level from the immediately prior
RAPM, its possession exposure, known target-season age and experience, and
known draft and physical-profile inputs. Its ridge regularization is selected
using expanding validation folds drawn only from those earlier transitions.

First-NBA-season players continue to use the forward exposure-gated
draft/replacement prior. No target-season possessions, box-score rates, or
outcomes enter either branch of the target-season player prior.

The contextual state remains unchanged from the bounded portable candidate:
each completed season fits bounded P-spline functions with curvature and
prior-season temporal penalties, then contributes an offset only to the next
season.

## Outputs

In addition to the standard recursive contextual artifacts, each run writes:

| File | Contents |
| --- | --- |
| `season_player_prior_metadata.parquet` | Per-season aging availability, selected regularization, transition support, and cold-start metadata. |
| `season_player_priors.parquet` | The actual prior vector entering every historical RAPM fit. |
| `frozen_2025_26_player_priors.parquet` | The pre-2025-26 player state used by the frozen evaluation. |

The Frozen Preseason Leaderboard is updated only after the completed 2025-26
evaluation has been validated.
