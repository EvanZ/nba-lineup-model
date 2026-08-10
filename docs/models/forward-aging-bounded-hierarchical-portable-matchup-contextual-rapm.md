---
last_updated: "2026-08-09"
---

# Forward Aging Bounded Hierarchical Portable-Matchup Contextual RAPM

This candidate tests whether a strictly forward age-informed transition improves
the returning-player prior in the bounded hierarchical portable contextual
model. It preserves the exposure-gated draft/replacement branch for first-NBA-
season players and the existing portable context contract:

\[
C(A,B)=h(x(A))-h(x(B))+q(x(A),x(B)).
\]

For a returning player (i) entering season (t), the prior is the output of
an aging ridge model trained only on completed candidate-state transitions:

\[
\mu_{i,t}^{\mathrm{age}} =
f_{<t}\left(
r_{i,t-1},\; \log(1+p_{i,t-1}),\; a_{i,t},\; e_{i,t},\; d_i,\; b_i
\right).
\]

Here (r_{i,t-1}) is the prior completed candidate RAPM, (p_{i,t-1}) is its
on-court possession exposure, (a_{i,t}) is known age, (e_{i,t}) is NBA
experience, (d_i) is draft information, and (b_i) is the physical profile.
The aging model uses the existing spline-age ridge specification and performs
its own expanding historical regularization selection inside every recursive
season.

## Frozen 2025-26 Result

The completed through-2025-26 run uses a `0.1` aging-ridge regularization for
the 2025-26 prior. It trains on 10,794 prior completed player transitions
through 2024-25, then forecasts 462 returning players; the 100 first-year
players continue through the exposure-gated cold-start branch.

| Cohort | Possession RMSE | Eligible-game margin RMSE | Full-game margin RMSE |
| --- | ---: | ---: | ---: |
| Regular season | 1.198831 | **14.3654** | **14.6265** |
| Playoffs | 1.192544 | **16.5388** | - |

At the team level, the candidate records a NetRtg RMSE of **3.7737** and a
Pythagorean-win RMSE of **7.7289**. These are the best currently published
frozen values for each of those metrics. The complete comparison is maintained
in the [Frozen Preseason Leaderboard](preseason-leaderboard.md).

Artifact:
`forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm-2025-26-20260810T024224Z-a4e478fe`.

Use [the training guide](../guides/train-forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm.md)
to reproduce it.
