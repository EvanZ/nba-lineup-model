---
last_updated: "2026-08-12"
---

# Train Forward Context-Reattributed HPM

This recursive candidate carries a controlled fraction of each completed
season's player-projectable context into the next-season player prior, while
leaving the untransferred context as the portable composition-plus-matchup
offset:

```bash
uv run nba-train-forward-context-reattributed-hpm \
  --through-season 2025-26 \
  --context-reattribution-weight 0.5
```

For a handoff weight \(\rho\), the season-to-season contract is

\[
\mu_{t+1}=\mu^{\mathrm{aging/cold}}_{t+1}+\rho\gamma_t,
\qquad
C^{\mathrm{residual}}_t=C_t-\rho X\gamma_t.
\]

The same \(\rho\) is added to the player prior and removed from the frozen
context offset. This prevents double counting. The initial exemplar uses 0.5;
the next validation pass will select \(\rho\) using expanding historical
seasons only.
