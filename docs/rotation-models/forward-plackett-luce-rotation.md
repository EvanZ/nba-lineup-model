---
last_updated: "2026-09-11"
---

# Forward Player Plackett-Luce Rotation v0.2

> **Status: in-season only.** Forward Conditional Minutes (FCM) is the fixed
> preseason prior. Plackett-Luce (PL) is a forward update after completed
> games. The selected model has no cross-season residual carry, so it has no
> additional information to apply at game zero of a new season.

Forward Player PL models how medically available players compete for a team's
240 regulation minutes. It uses the promoted preseason minutes stack for the
starting forecast and learns a **global player residual** from completed games.
Unlike the superseded v0.1 prototype, the residual belongs to a player, not a
team: it follows a player to a destination roster after an in-season trade.

## Fixed Prior And Player State

Forward Conditional Minutes v0.2 supplies the preseason expected minutes per
available game:

\[
\mu_{i,t}=E[\mathrm{MPG}_{i,t}\mid\mathrm{available}].
\]

Before game \(g\) in season \(t\), player \(i\)'s allocation utility is:

\[
u_{i,t,g}=\log\!\left(\max(\mu_{i,t},10^{-3})\right)+r_{i,t,g}.
\]

The FCM prior is computed only from information before season \(t\). The PL
residual \(r_{i,t,g}\) is zero for a player with no completed current-season
games. It is updated only after the game being predicted, so the target game's
minutes cannot affect its own forecast.

### Trades And Season Boundaries

The residual has one state per player across the league. If a player appears
for a different team later in the season, the destination game receives the
state learned with the prior team. This is the intended RAPM-like transfer
signal: a completed role observation belongs to the player, while the current
roster determines the choice set.

At a season boundary, only the prior-season mean may carry:

\[
r_{i,t,0}=\rho r_{i,t-1,\mathrm{final}}.
\]

The posterior precision always resets to the selected Gaussian prior
precision \(\lambda_{\mathrm{role}}\); past game exposure is never counted
twice. The tuning grid selected \(\rho=0\), so v0.2 retains transfer
continuity **within** a season but does not use a previous-season PL residual.

## Rank Likelihood And Online Update

For an observed available roster \(A_g\), players who receive positive
regulation minutes form the partial ordering \(\pi_g\), from most to fewest
minutes. Every unselected available player, including a coach DNP, remains in
the denominator:

\[
P(\pi_g\mid u)=
\prod_k
\frac{\exp(u_{\pi_{g,k},t,g})}
{\sum_{j\in A_g\setminus\{\pi_{g,1},\ldots,\pi_{g,k-1}\}}\exp(u_{j,t,g})}.
\]

Each team-game has total likelihood weight one, divided equally over its
positive-minute ranks. Positive-minute ties use stable player-ID order; zero
minute players are not ranked against each other.

An exact league-wide posterior would retain covariance for every pair of
players who ever appeared in a choice set. That state is impractical at NBA
scale. v0.2 therefore uses a diagonal online-Laplace approximation. At each
observed choice, it accumulates a player's gradient and diagonal curvature,
then updates that player's residual using its own posterior precision. The
approximation preserves the forward contract and player transfer state while
omitting cross-player posterior covariance.

## Allocation

Utilities rank the available roster. The top 15 players are retained and
normalized to the 240 regulation minutes:

\[
\widehat y_{i,g}=\begin{cases}
\dfrac{\exp(u_{i,t,g})}{\sum_{j\in R_g}\exp(u_{j,t,g})}, & i\in R_g,\\[6pt]
0, & i\notin R_g,
\end{cases}
\qquad
\widehat m_{i,g}=240\widehat y_{i,g},
\]

where \(R_g\) is the current top-15 set. This matches the product's active
rotation limit and exact regulation-minute constraint.

## Frozen Conditional Evaluation

The model was tuned only on 2020-21 through 2022-23, using earlier seasons to
construct the forward player state. It froze the selected configuration for
2023-24 through 2025-26:

\[
\lambda_{\mathrm{role}}\in
\{0.01,0.03,0.1,0.3,1,3,10,30,100\},
\qquad
\rho\in\{0,0.1,0.25,0.5,0.75,0.9,1\}.
\]

The winner is the interior precision \(\lambda_{\mathrm{role}}=10\) with
\(\rho=0\). The separately selected no-carry control is therefore identical,
which is a result: cross-season role carry did not add validated value. The
FCM control uses the same observed available roster and top-15/240 allocation
but has no PL update.

| Mean frozen metric | FCM v0.2 conditional control | **Forward Player PL v0.2** |
| --- | ---: | ---: |
| Allocation total variation | 0.20705 | **0.19440** |
| Brier score | 0.02078 | **0.01893** |
| Cross-entropy | 2.50501 | **2.45883** |
| Player-share MAE | 0.03128 | **0.02939** |
| Player-share RMSE | 0.03959 | **0.03781** |
| Pairwise rank accuracy | 76.82% | **79.27%** |
| Positive-rotation recall at 15 | 99.60% | **99.73%** |

PL improves every reported metric in each frozen season. The transfer audit
contains 263 in-season destination entries across the frozen seasons; every
one began with a nonzero player state learned before the move.

Artifact:
`artifacts/rotation/forward_plackett_luce_rotation/forward-pl-20260912T053400Z-b624068`.

## Availability-Integrated Frozen Evaluation

The conditional result above assumes the actual medical-availability mask is
known. The end-to-end replay instead supplies each player a Forward
Availability v0.2 probability. For each game, 1,024 deterministic scrambled
Sobol draws create feasible available rosters:

\[
I_{i,g}^{(s)}\sim\operatorname{Bernoulli}(p_{i,t}),
\qquad
A_g^{(s)}=\{i:I_{i,g}^{(s)}=1\},
\qquad |A_g^{(s)}|\geq5.
\]

Each scenario receives a 240-minute FCM or PL allocation; the forecast is the
mean across scenarios. Observed availability and minutes are revealed only
after the game, when they update the global player state for future games.
Both models use identical scenario draws.

| Mean frozen metric | Availability v0.2 + FCM control | **Availability v0.2 + Player PL v0.2** |
| --- | ---: | ---: |
| Allocation total variation | 0.35735 | **0.34097** |
| Brier score | 0.04383 | **0.04203** |
| Cross-entropy | 2.75430 | **2.73426** |
| Player-share MAE | 0.04065 | **0.03878** |
| Player-share RMSE | 0.04989 | **0.04886** |

The player-state PL model improves all five metrics in all three frozen
seasons, even after availability uncertainty is introduced.

Artifact:
`artifacts/rotation/forward_plackett_luce_availability/availability-pl-20260912T053604Z-5611b02`.

## Reproduce

```bash
uv run nba-run-forward-plackett-luce-rotation \
  | tee artifacts/logs/forward-player-pl-v02.log

uv run nba-evaluate-forward-availability-pl-rotation \
  --role-prior-precision 10 \
  --season-role-retention 0 \
  --no-carry-role-prior-precision 10 \
  | tee artifacts/logs/forward-player-pl-availability-v02.log
```
