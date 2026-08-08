---
last_updated: "2026-08-08"
---

# Train Student-t Talent-Prior Contextual RAPM

Train the combined Student-t player-prior and lagged contextual-offset model:

```bash
uv run nba-train-student-t-talent-contextual-rapm --through-season 2025-26
```

The command fixes the Student-t prior at \(\nu=3\) and a three-point scale,
keeps Gaussian stint errors, and uses the completed Gaussian forward
exposure-gated lambda schedule. Each season subtracts its prior-season context
function before fitting the Student-t player state, then fits the new context
function from the completed player residuals.

The trainer checkpoints after each completed season at
`artifacts/models/student_t_talent_contextual_rapm/<season>/.checkpoint.joblib`.
Re-run the same command after an interruption to resume. Use `--max-seasons 3`
for a deliberate short batch.

See [Student-t Talent-Prior Contextual RAPM](../models/student-t-talent-contextual-rapm.md)
for the transition equations and frozen evaluation.
