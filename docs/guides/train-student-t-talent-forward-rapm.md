# Train Student-t Talent-Prior RAPM

Train the Gaussian-error, heavy-tailed-talent version of the recursive forward
RAPM state through the latest completed season:

```bash
uv run nba-train-student-t-talent-forward-rapm --through-season 2025-26
```

The first specification fixes the coefficient prior at a Student-t with three
degrees of freedom and a three-point RAPM tail scale. Stint errors remain
Gaussian. The trainer keeps the exact completed Gaussian forward run's
per-season ridge lambda schedule, so this is a coefficient-prior ablation.

The trainer checkpoints after every completed season at
`artifacts/models/student_t_talent_forward_rapm/<season>/.checkpoint.joblib`.
Re-run the same command after interruption to resume. Use `--max-seasons 3`
for a deliberate short batch.

See [Student-t Talent-Prior RAPM](../models/student-t-talent-forward-rapm.md)
for the equations and current evaluation.
