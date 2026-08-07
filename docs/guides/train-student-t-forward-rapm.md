# Train Student-t Forward RAPM

Train the robust-likelihood counterpart to the recursive forward
exposure-gated RAPM state through the latest completed season:

```bash
uv run nba-train-student-t-forward-rapm --through-season 2025-26
```

The default has five Student-t degrees of freedom. It uses iteratively
reweighted least squares (IRLS), preserving the Gaussian player-coefficient
prior and the exact per-season ridge regularization chosen by the completed
Gaussian forward run. This isolates the effect of robust stint errors.

The trainer checkpoints after every completed season at
`artifacts/models/student_t_forward_rapm/<season>/.checkpoint.joblib`. Re-run
the same command after interruption; it resumes from the next season. The
checkpoint is removed only after the immutable artifact run is published.

For an intentional short batch, add `--max-seasons 3`. To test a different
fixed tail thickness, pass `--degrees-of-freedom 8`; that produces a distinct
artifact but is not directly comparable to the default until evaluated under
the same contract.

See [Student-t Forward RAPM](../models/student-t-forward-rapm.md) for the
statistical contract and current result.
