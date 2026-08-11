---
last_updated: "2026-08-10"
---

# Audit Value HPM Reproducibility

Value-Conditioned Aging HPM is deterministic: it has no seeded initialization,
random data split, minibatch ordering, or sampling step. A rerun should
therefore reproduce the prior artifact up to ordinary floating-point tolerance.

```bash
uv run nba-build-value-conditioned-aging-reproducibility-audit \
  --reference-run-dir artifacts/models/forward_centered_value_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm/2025-26/<prior-run-id> \
  --season 2025-26
```

By default, the candidate is the latest Value HPM artifact. The audit compares
historical player coefficients, frozen cohort metrics, game predictions, team
NetRtg predictions, and team-win predictions at an absolute tolerance of
`1e-10`. It writes an immutable report under
`artifacts/analysis/value_conditioned_aging_reproducibility/2025-26/` and
refreshes the model page.
