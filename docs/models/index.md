---
last_updated: "2026-08-09"
---

# Models

The modeling program starts with transparent predictive baselines before adding
player priors, separate offensive and defensive effects, or nonlinear lineup
interactions.

## Model Evolution

This view organizes the modeling program as a primary-parent tree. A branch
represents a deliberate change to its parent, not a claim that every model is
strictly nested. The selector redraws the vertical ordering by the selected
out-of-sample metric; rank 1 is best among models eligible for that metric.
The exact score and parent delta remain visible on every node.

The default view uses frozen 2025-26 regular-season game-margin RMSE because it
matches the project’s preseason forecasting contract. Models without a result
under the selected contract remain visible in the unranked lane to preserve the
full methodological lineage. Use the in-season choices to compare the
one-season neural and tree branches on their separate chronological holdout.

<div class="model-tree" data-model-tree data-source="../assets/data/model-tree.json?v=20260809-bounded-portable-context">
  <div class="model-tree__loading" role="status">Loading model evolution…</div>
</div>

### Reading The Tree

- **Horizontal position** is the number of primary-parent changes from the
  one-season ridge RAPM trunk.
- **Vertical position** is rank for the selected metric, not the raw metric
  scale. This prevents very small RMSE differences from producing an unreadable
  chart while preserving their exact values in node labels and tooltips.
- **Solid links** identify the primary parent used for the teaching-oriented
  lineage. A model can share components with other branches; its model page is
  the authoritative implementation record.
- **Unranked nodes** were not evaluated under the selected cohort. They are not
  treated as zeroes or assigned an artificial score.

The canonical metric definitions, evaluation boundaries, and full tables are
maintained in the [Frozen Preseason Leaderboard](preseason-leaderboard.md) and
the [in-season Leaderboard](leaderboard.md).

### Registry

The visualization is driven by
[`docs/assets/data/model-tree.json`](../assets/data/model-tree.json). Add a
new node only after its evaluation artifact and model documentation exist.
Each node must identify one primary parent, its documented change, a model-page
link, and only metrics produced under their named evaluation contract. The
registry integrity test prevents duplicate IDs, dangling parents, unknown
metrics, and malformed metric values.

## Baseline ladder

| Model | Information available |
| --- | --- |
| Mean | Training-window average home net rating |
| Team | Signed home and away team identities |
| RAPM | Signed identities of all ten players |
| Bayesian RAPM | RAPM posterior uncertainty under an explicit Gaussian model |
| RAPM aging model | Forward player-season priors from prior RAPM and age |
| Age-informed prior RAPM | Lineup RAPM centered on the frozen aging forecast |
| Forward RAPM calibration | Frozen lagged RAPM mapped to team wins using realized lineup seconds |
| Frozen 1-year no-prior RAPM | Completed 2024-25 zero-centered RAPM scored on 2025-26 oracle lineups |
| Frozen pooled 3-year no-prior RAPM | One zero-centered coefficient fit across 2022-23 through 2024-25 |
| Frozen lagged RAPM | Preseason 2024-25 player values scored on all 2025-26 oracle lineups |
| Frozen aging prior | Preseason age, experience, prior RAPM, draft, and physical-profile values scored on the same oracle lineups |
| Frozen O/D RAPM | Forward offense and defense player values scored on the same oracle lineups |
| Box-score RAPM prior | Returning-player forecast from lagged RAPM and possession-native box profile |
| Draft-informed cold starts | First-NBA-season RAPM diagnostic using draft profile fields only |
| Replacement-level exposure study | Rejected coefficient-average diagnostic for the low-exposure player pool |
| Pooled replacement-token RAPM | Retrospective shared-token estimate for the low-exposure player pool |
| Cold-start exposure gate | First-NBA-season preseason probability of entering the low-exposure pool |
| Exposure-gated cold-start prior | Continuous draft-rate/replacement blend for first-year players |
| Exposure-gated O/D cold starts | Separate O/D draft-rate/replacement blend for first-year players |
| Forward exposure-gated RAPM | Recursive regular-only preseason RAPM state with cold starts |
| Student-t forward RAPM | Robust stint-error likelihood for the same recursive forward state |
| Student-t talent-prior RAPM | Heavy-tailed player adjustments with Gaussian stint errors |
| Forward contextual RAPM | Recursive RAPM with the prior season's contextual offset |
| Forward decomposed contextual RAPM | Identifiable home-side minus away-side contextual offset |
| Forward portable-matchup contextual RAPM | Antisymmetric total context decomposed into portable units and matchups |
| Forward hierarchical P-spline contextual RAPM | Portable-matchup context with curvature and projected prior-season function shrinkage |
| Forward bounded hierarchical P-spline contextual RAPM | Original relative-context model with bounded feature support, P-spline curvature, and a projected prior-season function hierarchy |
| Forward bounded hierarchical portable-matchup contextual RAPM | Portable composition and matchup context with model-level bounded feature support, P-spline curvature, and a projected prior-season function hierarchy |
| Student-t talent-prior contextual RAPM | Contextual RAPM with heavy-tailed player-prior departures |
| Additive neural RAPM | Possession-level signed scalar player embeddings |
| Deep Sets | Nonlinear permutation-invariant lineup aggregation |
| CatBoost | Categorical player states and boosted-tree interactions |
| RAPM + Transformer | Frozen ridge prediction plus contextual player-player attention residual |

The first four use the same regular-season stint target, chronological game
splits, possession weights, and test window. The first three compare predictive
information sets. Bayesian RAPM deliberately retains the ridge point estimate
and adds coefficient, rank, and predictive uncertainty. Neural models move to
single-lineup possession rows while retaining chronological game boundaries.
RAPM + Transformer reconnects the two samples by converting a leakage-safe,
stint-weighted ridge prediction into the possession target and learning only
its nonlinear residual.

See [Baseline methodology](baselines.md) for the model contract, the
[2025-26 RAPM case study](2025-26-rapm-case-study.md) for a worked diagnostic
review, the [Bayesian RAPM methodology](bayesian-rapm.md) and
[2025-26 Bayesian case study](2025-26-bayesian-rapm-case-study.md) for the
probabilistic baseline, and [Promoted rankings](rankings.md) for reviewed
public releases. [Neural Networks](neural-networks.md) defines the staged
additive, Deep Sets, and RAPM + Transformer program. [Tree Models](tree-models.md)
defines the orthogonal CatBoost baseline. [Leaderboard](leaderboard.md) defines
the shared regular-holdout and playoff metrics and maintains the cross-model
scoreboard. [Forward RAPM Calibration](forward-calibration.md) maps frozen
player priors to regular-season team wins. The [Frozen Preseason
Leaderboard](preseason-leaderboard.md) holds out the complete target regular
season and playoffs without a player refit. [RAPM Aging Model](aging-model.md) defines the first temporal player
prior. [Draft-Informed Cold Starts](draft-prior.md) is an interpretable
first-NBA-season diagnostic; it is not yet eligible for the Frozen Preseason
Leaderboard. The [Replacement-Level Exposure Study](replacement-level.md)
records why separately ridged low-exposure coefficients fail as a replacement
estimate. [Pooled Replacement-Token RAPM](replacement-token.md) estimates that
group directly before a cold-start gate is introduced. The [Cold-Start Exposure
Gate](cold-start-exposure.md) provides a calibrated first-year-only probability
for a replacement-token blend. The [Exposure-Gated Cold-Start Prior](exposure-gated-cold-start.md)
uses that blend in the frozen preseason evaluation. The [Modeling Roadmap](roadmap.md)
records the approved joint dynamic RAPM extension.

[Exposure-Gated O/D Cold Starts](exposure-gated-offense-defense.md) applies
the same cold-start logic without collapsing offense and defense into a single
replacement number.

[Forward Exposure-Gated RAPM](forward-exposure-gated-rapm.md) applies that
one-number cold-start prior recursively across completed seasons.

[Student-t Forward RAPM](student-t-forward-rapm.md) holds that state and its
lambda schedule fixed while replacing the Gaussian stint-error likelihood with
a robust Student-t alternative.

[Student-t Talent-Prior RAPM](student-t-talent-forward-rapm.md) reverses that
ablation: it keeps Gaussian stint errors and uses a heavy-tailed Student-t
prior for player departures from the forward state.

[Forward Contextual RAPM](forward-contextual-rapm.md) carries the completed
lineup-composition state into the next season's RAPM target before updating its
player coefficients.

[Forward Decomposed Contextual RAPM](forward-decomposed-contextual-rapm.md)
constrains that state to an identifiable home-side minus away-side score. It is
currently an interpretability ablation: the less constrained relative-context
function has stronger frozen predictive results.

[Forward Portable-Matchup Contextual RAPM](forward-portable-matchup-contextual-rapm.md)
retains an antisymmetric total contextual state while identifying a
reference-anchored portable unit score and a centered opponent-specific matchup
residual.

[Forward Hierarchical P-spline Contextual RAPM](forward-hierarchical-pspline-contextual-rapm.md)
extends that portable-matchup state with second-difference spline shrinkage and
a forward-projected prior-season context-function prior.

[Forward Bounded Hierarchical P-spline Contextual RAPM](forward-bounded-hierarchical-pspline-contextual-rapm.md)
returns to the original relative-context function, clips each feature to its
forward-safe 5th--95th percentile support, and combines P-spline curvature
regularization with a projected prior-season function hierarchy.

[Forward Bounded Hierarchical Portable-Matchup Contextual RAPM](forward-bounded-hierarchical-portable-matchup-contextual-rapm.md)
applies the same forward-safe clipping contract to the portable-unit plus
matchup decomposition used by the Lineup Lab.

[Frozen No-Prior Window RAPM](frozen-window-rapm.md) records the one-year and
pooled three-year controls that distinguish historical pooling from an explicit
forward player prior.
