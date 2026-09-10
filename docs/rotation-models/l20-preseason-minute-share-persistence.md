---
last_updated: "2026-09-08"
---

# L20-MSP v0.0

**First-20-Game Preseason Minute-Share Persistence** is a static preseason
allocation model. Before game one, it predicts each player's cumulative share
of team minutes over a team's first 20 regular-season games. It does not update
after any current-season game.

## Target

For player \(i\) on team \(T\), the target is cumulative team-minute share:

\[
y_{i,T,s}^{1:20} =
\frac{\sum_{g=1}^{20}m_{i,T,s,g}}
{\sum_j\sum_{g=1}^{20}m_{j,T,s,g}}.
\]

This gives each team one stable target allocation rather than treating 30
opening games as the whole evaluation set.

## Forecast

Let \(C_{T,s}\) be the transaction-derived opening candidate roster and
\(M_{i,s-1}\) a player's total prior regular-season minutes, regardless of
their prior team. L20-MSP restricts and normalizes those minutes within the
new team's candidate roster, then applies uniform cold-start smoothing:

\[
\hat y_{i,T,s}^{1:20} =
(1-\epsilon)
\frac{M_{i,s-1}\mathbf{1}[i\in C_{T,s}]}
{\sum_{j\in C_{T,s}}M_{j,s-1}}
+\frac{\epsilon}{|C_{T,s}|}.
\]

The source season selects \(\epsilon\) by mean team allocation total
variation; ties choose the smaller value. That value is frozen for the next
season. Opening-game reconciliation rows are excluded from the input roster,
so the model does not use target-season box-score membership as a preseason
feature.

## Run

```bash
uv run nba-evaluate-l20-preseason-minute-share-persistence \
  --tune-season 2024-25 \
  --holdout-season 2025-26 \
  --team-games 20
```

The command writes source and frozen team allocations, the epsilon grid, and
metadata under `artifacts/rotation/l20_preseason_minute_share_persistence/`.

## Initial Frozen Result

Tuning on 2024-25 selected \(\epsilon = 0.32\), an interior grid point. Frozen
evaluation on 2025-26 produced mean team allocation total variation of
\(0.252\), player-share MAE of \(0.028\), and mean Brier score of \(0.024\).

For reference, a uniform allocation over the same transaction-derived opening
candidates had frozen total variation of \(0.347\) and Brier score of
\(0.036\). Prior-season playing time therefore provides useful preseason
information for the first-20-game allocation. This is still a baseline: it
does not yet model injuries, availability, coaches, or game-specific rotation
changes.
