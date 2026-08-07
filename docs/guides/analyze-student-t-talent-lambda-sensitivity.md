# Analyze Student-t Talent Lambda Sensitivity

Refit the completed 2025-26 Student-t talent-prior model using a fixed final
season lambda of 0.10 while holding every entering player prior fixed:

```bash
uv run nba-analyze-student-t-talent-lambda-sensitivity \
  --season 2025-26 \
  --alternative-lambda 0.10
```

This is a post-season ranking-sensitivity analysis, not a frozen-preseason
evaluation. The source run supplies the pre-2025-26 player-prior vector and
Student-t hyperparameters; only the final-season ridge penalty changes. The
immutable artifact records both ranking vectors, player-level rating and rank
movement, and summary correlations.

See [Student-t Talent-Prior Lambda Sensitivity](../models/student-t-talent-lambda-sensitivity.md).
