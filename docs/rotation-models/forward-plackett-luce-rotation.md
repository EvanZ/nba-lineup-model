---
last_updated: "2026-09-11"
---

# Forward Plackett-Luce Rotation v0.1

Forward Plackett-Luce Rotation is the first game-level rotation model built on
the promoted preseason minutes stack. Given a team's **observed medically
available roster** before a historical regulation game, it predicts the
distribution of that team's 240 regulation minutes.

It is deliberately conditional on the availability mask. Forward Availability
v0.2 remains responsible for a player's preseason availability probability;
this model learns how the available players compete for a place in the
rotation.

## Prior And Team Role State

Forward Conditional Minutes v0.2 supplies each player a preseason prior:

\[
\mu_{i,t}=E[\mathrm{MPG}_{i,t}\mid\mathrm{available}].
\]

For a player (i) on team (T) before game (g), the rotation utility is:

\[
u_{i,T,g}=\log\!\left(\max(\mu_{i,t}, 10^{-3})\right)+r_{i,T,g}.
\]

The team-role adjustment (r_{i,T,g}) starts at zero and is updated only after
completed earlier games for that same team and season. It has a Gaussian prior
with precision (lambda_{\mathrm{role}}). The preseason FCM prior therefore
travels with the player, while the coach- and roster-specific role does not.

### Trades And Team Changes

When a player first appears for a destination team, their destination
adjustment is exactly zero. The player still competes using their FCM prior
against the destination roster, but no role learned with the old team is
carried forward. Subsequent destination-team games update the new state.

The frozen replay found 263 in-season destination-team entries across 2023-24
through 2025-26. All 263 began with a zero destination role adjustment, as
required by the contract. This is what lets a player with a large prior from a
weak roster fall behind stronger players on a new roster without declaring the
player intrinsically worse.

## Rank Likelihood

For an available roster (A_g), players with positive observed regulation
minutes form a partial ranking (π_g), ordered from most to fewest minutes.
At each rank, all unselected available players remain in the choice set:

\[
P(\pi_g\mid u)=
\prod_k
\frac{\exp(u_{\pi_{g,k},T,g})}
{\sum_{j\in A_g\setminus\{\pi_{g,1},\ldots,\pi_{g,k-1}\}}
\exp(u_{j,T,g})}.
\]

A coach DNP remains in every denominator but is never selected, which is
evidence that the player ranked below the players who received minutes. The
v0.1 likelihood is normalized by the number of positive-minute ranks, so each
team-game contributes equal total weight. Exact ties among positive minute
totals are resolved by stable player ID order; zero-minute players are not
ranked against one another.

The posterior state is updated with a sequential Laplace/Newton step after
each completed game. Before the next game, that updated state supplies the
new role utilities. No target-game minutes, later games, or destination-team
outcomes are used in a prediction.

## Allocation

The utilities first rank the available roster. The top 15 candidates are
retained, matching the product's maximum active rotation. Their utilities are
converted to exact regulation-minute shares:

\[
\widehat y_{i,g}=
\begin{cases}
\dfrac{\exp(u_{i,T,g})}
{\sum_{j\in R_g}\exp(u_{j,T,g})}, & i\in R_g,\\[6pt]
0, & i\notin R_g,
\end{cases}
\qquad
\widehat m_{i,g}=240\widehat y_{i,g},
\]

where (R_g) is the top-15 set. This preserves both the fixed 240-minute
constraint and the existing active-15 product rule.

## Frozen Evaluation

The only fitted hyperparameter is the role-state prior precision. It was
selected on completed 2020-21 through 2022-23 seasons, then frozen for
2023-24 through 2025-26. The selected value, (lambda_{\mathrm{role}}=10),
is an interior point of the tested grid:

\[
\{0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100\}.
\]

The matched control is not a lagged-minutes model. It is exactly the promoted
Forward Conditional Minutes v0.2 prior restricted to the same observed
available roster and converted into the same top-15, 240-minute allocation.
Because availability is already given in this conditional evaluation,
(P(\mathrm{available})) is not multiplied into either model.

| Mean frozen metric | FCM v0.2 conditional control | **Forward PL Rotation v0.1** |
| --- | ---: | ---: |
| Allocation total variation | 0.20705 | **0.19461** |
| Brier score | 0.02078 | **0.01894** |
| Cross-entropy | 2.50501 | **2.46177** |
| Player-share MAE | 0.03128 | **0.02942** |
| Player-share RMSE | 0.03959 | **0.03782** |
| Pairwise rank accuracy | 76.82% | **79.22%** |
| Positive-rotation recall at 15 | 99.60% | **99.71%** |

Each individual frozen season improves allocation total variation, Brier
score, cross-entropy, share MAE, share RMSE, and pairwise rank accuracy. This
validates the team-specific forward role state as a game-level addition to the
preseason FCM prior. It does not yet forecast a game-level availability mask
or incorporate rest, opponent, coach, or lineup-size features.

Artifacts:
`artifacts/rotation/forward_plackett_luce_rotation/forward-pl-20260912T030302Z-96ee2be`.

Run:

```bash
uv run nba-run-forward-plackett-luce-rotation \
  | tee artifacts/logs/forward-plackett-luce-rotation.log
```

## Availability-Integrated Frozen Evaluation

The conditional result above holds the available roster fixed. The next test
uses the promoted Forward Availability v0.2 probability for every player on
the existing game-day roster, rather than the observed availability label.
For each game, 1,024 deterministic scrambled-Sobol scenarios draw a feasible
roster:

\[
I_{i,g}^{(s)} \sim \operatorname{Bernoulli}(p_{i,t}),
\qquad
A_g^{(s)}=\{i:I_{i,g}^{(s)}=1\},
\qquad
|A_g^{(s)}|\geq 5.
\]

The five-player condition is a feasibility constraint: an NBA team must be
able to allocate 240 regulation minutes. It is not an additional availability
feature. For each scenario, the FCM control or PL model allocates 240 minutes
conditional on (A_g^{(s)}); the final forecast is the scenario average:

\[
E[m_{i,g}]\approx\frac{1}{1{,}024}
\sum_{s=1}^{1{,}024}\hat m_{i,g}\!\left(A_g^{(s)}\right).
\]

The candidate and control use the identical deterministic scenarios, so the
comparison isolates the forward PL team-role state. The actual availability
and minutes are revealed only after a game, when they update the PL state for
future games. Thus the availability mart is read as the historical outcome
label; it is not rebuilt or used as a pregame availability input.

| Mean frozen metric | Availability v0.2 + FCM v0.2 control | **Availability v0.2 + PL Rotation v0.1** |
| --- | ---: | ---: |
| Allocation total variation | 0.35735 | **0.34126** |
| Brier score | 0.04383 | **0.04202** |
| Cross-entropy | 2.75430 | **2.73455** |
| Player-share MAE | 0.04065 | **0.03882** |
| Player-share RMSE | 0.04989 | **0.04885** |

The candidate wins all five metrics in every frozen season. The allocation-TV
improvement is 4.5% after availability uncertainty is introduced. This is the
validated end-to-end minutes stack for an in-season game forecast:

\[
P(\mathrm{available})
\longrightarrow
\text{available-roster scenarios}
\longrightarrow
\text{Forward PL Rotation}
\longrightarrow
E[\mathrm{player\ minutes}].
\]

Artifact:
`artifacts/rotation/forward_plackett_luce_availability/availability-pl-20260912T032303Z-1436e58`.

Run:

```bash
uv run nba-evaluate-forward-availability-pl-rotation \
  | tee artifacts/logs/forward-availability-pl-rotation.log
```
