---
last_updated: "2026-08-08"
---

# Forward Portable-Matchup Contextual RAPM

Forward Portable-Matchup Contextual RAPM is a rolling contextual RAPM model
that preserves a total lineup-context correction while making two parts of it
identifiable: a portable unit score and an opponent-specific matchup residual.
It uses the same forward exposure-gated player priors, cold starts, player
profiles, composition features, spline basis, and Ridge alpha as Forward
Contextual RAPM.

For two five-player units \(A\) and \(B\), the completed context state is:

\[
C(A,B) =
\underbrace{h(x(A)) - h(x(B))}_{\text{portable composition advantage}}
+
\underbrace{q(x(A),x(B))}_{\text{specific matchup interaction}}.
\]

Here \(x(A)\) is the same side-level feature vector used by the prior
contextual models: stabilized prior-season shooting, passing, usage, turnover,
rebounding, steal, and block rates plus the published composition summaries.

## Total Context State

The total \(C\) is fitted as a relative spline Ridge function using the same
contextual feature family as the original model. Each training observation is
paired with its reversed orientation, including the negated residual target.
At prediction time the model returns:

\[
C(A,B) = \frac{g(x(A)-x(B))-g(x(B)-x(A))}{2}.
\]

Consequently, orientation symmetry is exact:

\[
C(A,B)=-C(B,A).
\]

This is the critical difference from the pure decomposed ablation. The model
does not require every useful contextual pattern to be expressible as a
difference of independent side scores. It retains a nonseparable \(q\) term
for actual opponent-specific effects.

## Identifying h And q

After every completed season \(t\), the model stores a possession-weighted
reference distribution \(\mathcal R_t\) of all observed home and away unit
feature vectors. For any candidate unit \(A\):

\[
h_t(x(A)) =
\mathbb E_{R\sim\mathcal R_t}[C_t(A,R)].
\]

The matchup component is then the exact residual:

\[
q_t(A,B) =
C_t(A,B)-h_t(x(A))+h_t(x(B)).
\]

Because the same reference distribution is used on both sides and \(C\) is
antisymmetric, \(q\) has no average portable effect:

\[
\mathbb E_{R\sim\mathcal R_t}[q_t(A,R)] = 0.
\]

Thus \(h\) is a unit's expected context against the completed-season field,
not a player rating or an absolute context level. \(q\) reports what changes
for this exact opponent after both units' portable context values have been
removed.

## Rolling Forecast Boundary

For a stint in season \(t\), only the completed prior-season total context is
subtracted before fitting the same prior-centered additive RAPM state:

\[
y^{\mathrm{adj}}_{s,t}=y_{s,t}-C_{t-1}(H_s,A_s).
\]

The frozen 2025-26 forecast uses the completed 2024-25 \(C\) model and the
2024-25 empirical reference field. The 2025-26 model is fit only after that
forecast has been scored, and is retained for the next season's state.

## Frozen 2025-26 Result

This model improves the primary regular eligible-game margin metric and both
full-game margin metrics while retaining the side/matchup decomposition:

| Metric | Forward contextual RAPM | Portable-matchup contextual RAPM |
| --- | ---: | ---: |
| Regular eligible-game margin RMSE | 14.6525 | **14.6349** |
| Regular full-game margin RMSE | 15.0190 | **14.9602** |
| Regular full-game margin MAE | 11.8474 | **11.7780** |
| Regular team NetRtg RMSE | **4.1572** | 4.1864 |
| Regular Pythagorean-win RMSE | **9.3153** | 9.3999 |
| Playoff eligible-game margin RMSE | 17.5493 | **17.5261** |

It is therefore a competitive contextual candidate rather than an
across-the-board replacement. The full metric record is maintained in the [Frozen
Preseason Leaderboard](preseason-leaderboard.md).

## Artifact

The completed through-2025-26 run is
forward-portable-matchup-contextual-rapm-2025-26-20260808T165815Z-0d9d87f0
under artifacts/models/forward_portable_matchup_contextual_rapm/2025-26/.
Its forecast_reference_units.parquet file exposes the frozen 2024-25 unit
distribution that identifies \(h\) and centers \(q\). The model artifact also
stores every seasonal context state, player-prior state, and frozen evaluation
table.

Before proposing another contextual function family, inspect the saved
seasonal total components in the [Context Function Audit](context-function-audit.md).
