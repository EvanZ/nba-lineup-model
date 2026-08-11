---
last_updated: "2026-08-10"
---

# Train Era-Conditioned Aging HPM

This candidate retains the centered, value-conditioned aging HIPSTER PM prior
and allows its smooth age function to evolve across NBA eras. For target season
\(t\), the known season index is

\[
e_t = \frac{\operatorname{startYear}(t)-2010}{10}.
\]

The aging regression includes an age-spline basis \(B(a)\) and the penalized
interaction \(e_t B(a)\):

\[
\mu_{i,t} = B(a_{i,t})\gamma + e_tB(a_{i,t})\delta
+ \beta_r r_{i,t-1}
+ \beta_{ar}(a_{i,t}-27)r_{i,t-1} + \cdots.
\]

All interaction coefficients are ridge-regularized. The era index is known
before a season starts, so this adds no target-season leakage. The model does
not introduce a three-way age-by-era-by-value interaction in this first pass.

```bash
uv run nba-train-forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm \
  --through-season 2025-26
```

## Outputs

```text
artifacts/models/forward_centered_era_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm/2025-26/
```

In addition to the standard frozen evaluation tables, the run publishes the
annual player rating, seasonal fit metadata, serialized aging pipelines, and
population aging curve grid described in [Forward model artifacts](../data/forward-model-artifacts.md).
