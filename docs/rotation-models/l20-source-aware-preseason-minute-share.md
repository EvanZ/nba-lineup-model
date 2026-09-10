---
last_updated: "2026-09-08"
---

# L20-NAIL-MSP v0.3

**Source-Aware Preseason Minute-Share Projection** is the production successor to
[v0.2](l20-preseason-nail-forecast-minute-share.md). Its implementation and
validation history are tracked in
[GitHub issue #1](https://github.com/EvanZ/nba-lineup-model/issues/1).

The current baseline relies primarily on a player's total minutes in the
immediately preceding season. That is a useful availability-and-role signal,
but it conflates three distinct questions: how much a player played when
available, how often the player was available, and whether the player was
trusted to start. It also incorrectly turns a full injury absence into a zero
role signal.

## Player States

Every preseason roster candidate will use exactly one evidence pathway:

| State | Preseason evidence |
| --- | --- |
| Continuous player | Prior total minutes; shrunken MPG, GP%, and GS%; preseason NAIL |
| Gap returner | Decayed last-observed versions of the same role inputs; gap length; preseason NAIL |
| Rookie or cold start | Learned conservative role prior using draft capital, undrafted status, age, position, and preseason NAIL |

A missed season is missing role evidence, not evidence that a player should
receive zero minutes. For a gap of (k) seasons, the last-observed role input
(x) is attenuated before it enters the allocation model:

\[
x^{\mathrm{gap}}_{i,t} = d^k x^{\mathrm{last\ observed}}_{i,t-k}.
\]

The separate gap-state and gap-length features allow the ridge-softmax model
to learn how much additional role adjustment that attenuation requires. The
decay and cold-start pathway use only historical preseason information, never
realized target-season minutes.

## Allocation Contract

The model will convert the source-aware player role scores into a constrained
team allocation. The frozen evaluator assigns probability across every
transaction-derived opening-roster candidate, matching its first-20-game target.
The website then applies an explicit product constraint: it retains the 15
highest-scoring players for each team, renormalizes their shares to 240
regulation minutes per game, and starts all remaining rostered players at zero.
Preseason NAIL remains a modest within-roster signal rather than a claim that
coaches mechanically allocate minutes by player value.

Manual minutes overrides remain the final product layer for known current
information, including an injury restriction, delayed availability, or an
explicit coaching-role judgment.

## Replacement-Token Experiment

The opening roster is necessarily incomplete over a 20-game window: trades,
two-way conversions, signings, and emergency call-ups can receive minutes after
opening day. The baseline treats those players as zero-probability outcomes.
That is correct for the known-player allocation but makes the model incapable
of representing the missing roster mass.

The current experiment augments each team distribution with one anonymous
replacement token:

\[
\sum_{i \in \mathrm{opening\ roster}} s_{i,t} + s_{\mathrm{replacement},t} = 1.
\]

For a frozen target, the token target is the actual minute share of all players
absent from the transaction-derived opening roster. The token has a single
unpenalized global intercept; it does not know the future identity of an
arrival. This first slice tests whether making that uncertainty explicit helps
the full team allocation before adding any roster-churn predictors.

Player-level diagnostics remain calculated over real players, so the token
cannot conceal individual allocation mistakes. Team total variation, Brier
score, and cross-entropy are calculated on the augmented player-plus-token
distribution. The experiment is isolated from the v0.3 production artifact and
will be reported here after the frozen replay completes.

### Result

The replacement-token replay completed all ten frozen holdouts, `2016-17`
through `2025-26`. It modestly improved the augmented team allocation, but it
does not yet identify which individual teams will need replacement minutes well
enough to change the product model.

| Metric | Source-aware v0.3 | v0.3 + replacement token | Difference |
| --- | ---: | ---: | ---: |
| Mean allocation total variation | 0.20313 | **0.20101** | **-0.00212** |
| Mean Brier score | 0.01736 | **0.01721** | **-0.00015** |
| Player-share MAE | **0.02472** | 0.02478 | +0.00006 |
| Player-share RMSE | 0.03242 | **0.03215** | **-0.00027** |

Total variation improved in eight of ten holdouts, but Brier improved in only
five. The token's mean prediction was 5.52% against an actual mean
outside-opening-roster share of 5.08%; its median prediction was 5.40% while
the actual median was only 2.71%. The actual and predicted token shares had a
0.47 team-level correlation, so the token has some roster-specific variation
but remains close to a pooled league reserve.

Accordingly, the experiment is **not promoted**. It validates the missing-mass
contract and produces a small team-allocation gain, but a useful product model
should also forecast which teams are exposed to arrivals and departures. That
requires a separately validated roster-churn feature set rather than treating
the global token intercept as a finished availability model.

Artifact:
`artifacts/rotation/l20_source_aware_preseason_minute_share_replacement_token/l20-source-aware-replacement-token-20260909T052927Z-f726c5c`.

## Frozen Evaluation

The target remains each player's cumulative minute share over a team's first
20 regular-season games. Every evaluation fold must use the opening roster and
information available before that season begins. v0.3 will be compared with
v0.2 overall and separately for continuous players, gap returners, and cold
starts. It is eligible to replace the current baseline only if it is at least
competitive overall without creating a material failure in any player-state
group.

## Promotion Result

The first full rolling replay evaluated ten target seasons, `2016-17` through
`2025-26`. Each holdout used only earlier target seasons to select the role-rate
shrinkage, gap decay, and ridge penalty. It then compared v0.3 with the
similarly rolling [v0.2](l20-preseason-nail-forecast-minute-share.md) control.

| Metric | L20-NAIL-MSP v0.2 | Source-aware v0.3 | Difference |
| --- | ---: | ---: | ---: |
| Mean team allocation total variation | 0.21980 | **0.20366** | **-0.01614** |
| Mean Brier score | 0.01977 | **0.01743** | **-0.00234** |
| Player-share MAE | 0.02677 | **0.02479** | **-0.00198** |
| Player-share RMSE | 0.03464 | **0.03252** | **-0.00212** |

v0.3 improved allocation total variation in all ten frozen holdouts. The
initial artifact is
`artifacts/rotation/l20_source_aware_preseason_minute_share/l20-source-aware-20260909T005439Z-0cf9b7b`.

### Player-State Diagnostics

The candidate is trained conditionally on the transaction-derived opening
roster. Players who later appeared but were absent from that roster remain an
explicit evaluation-only `outside_opening_roster` category; no opening-game
reconciliation is used during training.

| Player state | Player rows | Mean absolute share error | Mean actual share | Mean predicted share |
| --- | ---: | ---: | ---: | ---: |
| Continuous | 3,570 | 0.02547 | 0.07355 | 0.07747 |
| Gap returner | 51 | 0.02698 | 0.03352 | 0.03907 |
| Rookie/cold start | 635 | 0.02280 | 0.03521 | 0.03751 |
| Outside opening roster | 661 | 0.02381 | 0.02381 | 0.00000 |

The gap-returner prediction is intentionally conservative but is still mildly
high on average. That is visible in the artifact rather than concealed by a
zero-minute fallback.

### Expanded Shrinkage Grid

The initial selector landed at the lower bounds for both the ridge penalty and
role-rate shrinkage. A second full replay therefore expanded the candidate grid
to (q \in \{1, 2, 3, 4, 5, 15, 30\}\) games and
\(\alpha \in \{0.01, 0.02, 0.05, 0.10, 1, 10\}\), while retaining the same
gap-decay choices.

| Metric | Initial grid | Expanded grid |
| --- | ---: | ---: |
| Mean team allocation total variation | 0.20366 | **0.20313** |
| Mean Brier score | 0.01743 | **0.01736** |
| Player-share MAE | 0.02479 | **0.02472** |

The expanded grid selected (q=1) in every frozen holdout. It selected
\(\alpha=0.10\) in nine of ten holdouts, establishing that the ridge penalty
is not at its lower boundary. Gap retention remained conservative early,
typically (0.50), and reached (0.75) for the final two holdouts.

`q=1` is intentionally accepted as the minimum practical role-evidence
threshold. It leaves a 10-game MPG observation at 91% of its raw value and a
60-game observation at 98%, while avoiding the claim that a one-game role
sample should be trusted with no attenuation. The selected result is
`artifacts/rotation/l20_source_aware_preseason_minute_share/l20-source-aware-20260909T013520Z-cd15741`.

## Production Artifact

For the 2026-27 forecast, the final model selected one configuration by
running time-ordered internal validation over all eleven completed source
seasons, then fitting the allocation model on all of them. It does not read
2026-27 game outcomes.

| Parameter | Selected value |
| --- | ---: |
| Role-rate shrinkage (q) | 1 game |
| Gap-role retention (d) | 0.75 per missed season |
| Ridge penalty (alpha) | 0.10 |

The frozen production artifact is
`artifacts/rotation/l20_source_aware_preseason_minute_share/l20-source-aware-production-20260909T042413Z-550aa6a`.
It is the allocation baseline used by the local Win Projections page; explicit
user minute overrides remain a product-layer adjustment above this model.
