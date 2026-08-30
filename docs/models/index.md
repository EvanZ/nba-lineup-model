---
last_updated: "2026-08-30"
---

# Ratings Models

The modeling program starts with transparent predictive baselines before adding
player priors, separate offensive and defensive effects, or nonlinear lineup
interactions.

## Model Evolution

This view organizes the modeling program as a primary-parent tree. A branch
represents a deliberate change to its parent, not a claim that every model is
strictly nested. The selector redraws the vertical ordering by the selected
out-of-sample metric; rank 1 is best among models eligible for that metric.
The exact score and parent delta remain visible on every node.

The default view uses the recovered-coverage rolling three-season regular
game-margin RMSE, the project’s current rigorous preseason contract. Models
without a result under the selected contract remain visible in the unranked
lane to preserve the full methodological lineage. The older frozen-2025-26 and
in-season selectors remain available for their respective historical snapshots.

<div class="model-tree" data-model-tree data-source="../assets/data/model-tree.json?v=20260830-continuity-replacement">
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
| Forward RAPM memory baselines | Strict rolling one- and three-season RAPM priors across three frozen target seasons |
| Complete player-prior RAPM baseline | Aging, value conditioning, and exposure-gated cold starts without context or box-score priors |
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
| Controlled no-context value-conditioned RAPM | HPM's centered value-conditioned aging and cold-start prior with all contextual corrections set to zero |
| Forward context-reattributed HPM | Experimental recursive transfer of a fixed fraction of player-projectable context into the next-season player prior, retaining residual context |
| Forward box-score residual HPM | HPM prior plus a strictly lagged box-score residual for returning players |
| Additive profile-prior RAPM | HPM x3's additive player-profile terms moved into the lagged player prior, with context disabled |
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

[Controlled No-Context Value-Conditioned RAPM](forward-centered-value-conditioned-aging-no-context-rapm.md)
is the direct HPM ablation: it preserves the current player-prior state but
removes only the recursive lineup-context state.

[HIPSTER PM v2: Depth-Aware Shooting](hpm-v2.md) is the current contextual
feature experiment. It replaces raw lineup three-point totals with depth,
capped capacity, and concentration representations while preserving the
incumbent HPM training contract.

[HIPSTER PM v2.1: Empirical Rebound Capacity](hpm-v21.md) retains that shooting
representation and replaces rebound-volume summaries with capped player ORB%
and DRB% unit capacity.

[HIPSTER PM v2.2: Usage Allocation](hpm-v22.md) keeps v2.1's empirical rebound
capacity and replaces raw usage and turnover volume with four continuous,
forward-calibrated terminal-action allocation features.

[NAIL-RAPM Attribution Contract](linear-hpm-x3-compilation-audit.md) separates
the direct linear context into player-attributable additive effects and the
remaining non-additive lineup effects, with an exact reconstruction check over the
three frozen seasons.

[Linear HPM x3 Quadratic Side Context](linear-hpm-x3-quadratic-side-context.md)
tests the smallest nonlinear extension of that contract: one squared side-total
term per additive feature, with no spline basis.

[NAIL-RAPM v1.0](nail-rapm-v1.md) is the canonical
linear model: `imputed_count` and `replacement_weight` were retired because
they are profile-quality diagnostics rather than basketball context. They remain
only in the immutable historical artifact used for the decision.

[NAIL-RAPM v1.1](nail-rapm-v11-profile-padding.md) keeps that architecture and
replaces universal 300-possession profile shrinkage with statistic-specific
stabilization constants.

[NAIL-RAPM v1.2](nail-rapm-v12-gap-returners.md) carries an established
player's state through a missed season with the existing annual aging model and
uses the player's last observed padded profile on return.

[NAIL-RAPM v1.2.1](nail-rapm-v121-pruned-nonadditive.md) retains only the two
historically resolved non-additive terms: usage concentration and top-two
assists. It clears its direct bootstrap gate versus v1.2 and establishes the
non-additive contract inherited by the current production release.

[Critical Spacing](nail-critical-spacing.md) adds one non-additive indicator to
v1.2.1: whether a unit has at least two
players below that season's prior-derived lower-tercile threshold for shrunk
three-point makes per 100 possessions. It clears the no-material-harm gate but
is not yet promoted: it has no meaningful pooled lift. Its coefficient has a
79.9% negative total one-sided mass index despite several small positive-sign
seasons, supporting the hypothesized long-run direction.

[Lower-Quintile Critical Spacing](nail-critical-spacing-quintile.md) tests a
stricter version of the same threshold and was rejected after the frozen
comparison and full non-additive coefficient audit.

[NAIL-RAPM v1.2.1.1](nail-rapm-v1211-standard-usage.md) replaces the internal
usage-events rate in v1.2.1 with conventional box-score USG%. It passes the
same no-material-harm bootstrap gate, improving interpretability while leaving
the frozen predictive result effectively unchanged; it is not promoted because
the standard convention does not materially alter the lineup interpretation.

[NAIL-RAPM v1.2.1.2](nail-rapm-v1212-back-to-back.md) adds a known-before-tipoff
back-to-back control. It is retained as the prior production release.

[NAIL-RAPM v1.2.1.3](nail-rapm-v1213-residualized-lambda.md) keeps the complete
v1.2.1.2 contract but selects the player penalty on each source season's
context- and schedule-residualized target. It is the current production model.

[Prior Teammate Continuity](nail-teammate-continuity.md) adds a strictly lagged
relationship feature: the mean log prior-season shared possessions over a
unit's ten player pairs. Its coefficient is strongly and consistently positive,
but the full recursive frozen replay finds no material incremental lift, so the
candidate is not promoted.

[Teammate-Continuity Replacement](nail-teammate-continuity-replacement.md)
removes `top_two_assists` and retains continuity beside usage concentration.
The more stable relationship feature can replace shared playmaking without a
material full-game loss, but possession MAE and team NetRtg error worsen and
the feature is not portable to hypothetical lineups, so it is not promoted.

[NAIL-RAPM v1.2.2](nail-rapm-v122-defensive-rebound-profile.md) tests
defensive rebound percentage as a ninth additive profile coordinate. Its
bootstrap is non-inferior, but the feature is positive in only 72.4% of
source seasons, below the 80% stability gate, so it is retained as a rejected
experiment rather than a promoted release.

[NAIL-RAPM v1.2.3](nail-rapm-v123-free-throw-profile.md) instead tests
free-throw attempts per 100 as the ninth additive coordinate. FTA/100 is
directionally stable, but it does not improve the primary three-season frozen
margin metrics, so it is also retained as a non-promoted sibling experiment.

[NAIL-RAPM v1.2.4](nail-rapm-v124-free-throw-replacement.md) replaces
additive Usage/100 with FTA/100 to remove their conditional collinearity.
FTA/100 remains highly stable, but the frozen margin metrics still do not
improve, so v1.2.1.2 remains the promoted model.

[Quartile Critical Spacing Plus Standard USG%](nail-critical-spacing-quartile-standard-usage.md)
tests a narrower 25th-percentile threshold while replacing the internal
usage-events coordinate with conventional box-score USG%. It is effectively
tied at possession level but worse on the primary pooled game metrics, so it
is not promoted.

[NAIL-RAPM v1.3](nail-rapm-v13-additive-profiles.md) adds four additive
box-score profile measures, including self-created rim and three-point makes.
It remains visible in the leaderboard and model tree, but is not promoted: its
paired full-game bootstrap gate failed.

[NAIL-RAPM v1.3.1](nail-rapm-v131-pruned-additive-profiles.md) removes only
three-point attempts per 100 and usage per 100 from v1.3 after the partial-effect
stability audit. It is the preferred parsimonious contract within that branch:
the direct bootstrap comparison found no practically material frozen-prediction
loss, though v1.2.1.2 is the global regular-season leader.

[NAIL-RAPM Context Regularization](nail-context-regularization.md) replaces the
inherited raw context `alpha=10000` with a season-size-invariant penalty chosen
over 23 pre-frozen seasons. Its playoff replay improves slightly, but the
regular-season frozen bootstrap rejects it, so v1.2.1.2 remains promoted.

[Forward Compiled-Additive-Prior HPM x3](forward-compiled-additive-prior-hpm-x3.md)
tests the resulting attribution contract recursively: learned additive player
profile effects enter the next player prior, while the six true non-additive lineup
effects remain in carried context. Its full transfer is a useful negative
regular-season result, so the NAIL v1 family remains the reference.

[Additive Prior Plus Linear Non-Additive Context](additive-profile-linear-shape-context.md)
is the controlled test of the attribution boundary: additive player profiles
are learned in the lagged prior, while six nonlinear non-additive lineup coordinates
enter a plain linear Ridge context layer.

[Forward Context-Reattributed HPM](forward-context-reattributed-hpm.md)
tests whether some player-projectable context should persist in the next-season
player prior while the remaining context stays lineup-specific. Its first
\(\rho=0.5\) candidate is retained as a negative predictive ablation.

[Forward Box-Score Residual HPM](forward-box-score-residual-hpm.md) uses the
existing possession-native box-score panel to predict only what the completed
HPM prior misses for returning players. Cold starts remain on the exposure-gated
branch, and the frozen 2025-26 outcome decides whether the residual earns a
Leaderboard branch.

[Additive Profile-Prior RAPM](additive-profile-prior-rapm.md) is the controlled
test that transfers HPM x3's additive lineup signals into the player prior and
removes context altogether.

[Forward Box-Score Interaction HPM](forward-box-score-interaction-hpm.md)
keeps that residual boundary and adds six declared lagged box-score products
for high-volume creation, shooting, scoring pressure, rebounding, and
defensive-event profiles. It is evaluated as a direct additive-residual
extension, not as a target-season box-score model.

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
