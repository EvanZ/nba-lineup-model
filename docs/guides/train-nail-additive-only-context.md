---
title: Train NAIL-RAPM Additive-Only Context Ablation
---

# Train NAIL-RAPM Additive-Only Context Ablation

This controlled ablation keeps NAIL-RAPM v1.0's recursive player-prior
pipeline unchanged, but retains only its eight lineup-additive basketball
profile coordinates. It removes the six non-additive lineup coordinates.

```bash
uv run nba-train-nail-additive-only-context --through-season 2025-26
uv run nba-evaluate-nail-additive-only-context
```

The second command compares the newly trained artifact directly against
NAIL-RAPM v1.0 across the fixed 2023-24 through 2025-26 regular-season and
pooled-playoff cohorts. It never refits on a target season.
