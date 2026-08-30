---
last_updated: "2026-08-30"
---

# Non-Additive Context Feature Registry

This page is the working backlog for proposed NAIL lineup-context features.
It exists to preserve the basketball hypothesis, the exact mathematical
contract, and the result of every test, including negative results.

The promoted [NAIL-RAPM v1.2.1.3](nail-rapm-v1213-residualized-lambda.md)
currently retains two player-composition terms: `usage_concentration` and
`top_two_assists`. Home-court advantage and back-to-back status are separate
schedule controls, not player-composition features.

## What Counts As Non-Additive

Let \(U\) be a five-player unit, \(O\) its opponent, and \(p_i\) the frozen,
shrunken profile for player \(i\). A proposed side feature is
\(\phi_k(U)\), and its matchup coordinate is

\[
x_k(U,O)=\phi_k(U)-\phi_k(O).
\]

A feature belongs in this registry only when it cannot be written as

\[
\phi_k(U)=\sum_{i\in U}g_k(p_i).
\]

Pure sums belong in the additive player-profile layer. Normalized shares,
order statistics, thresholds, pairwise terms, cross-player products, and
prior teammate relationships can be genuinely non-additive. A nonlinear
formula is not sufficient by itself: a transformed statistic computed
independently for each player and then summed remains additive.

All player inputs below must be frozen before the target season and use the
same leakage-safe shrinkage contract as the incumbent model unless a candidate
explicitly pre-registers a different source.

## Status Key

- [x] The proposed test is resolved. The accompanying decision says whether it
  was retained, rejected, or completed without promotion.
- [ ] The candidate remains pending. Checking the box requires a persisted
  screen or model artifact and a documented decision.

## Resolved Tests

- [x] **Usage concentration: retained.** Share of all five usage events
  supplied by the two highest-usage players. This is the strongest retained
  non-additive term and a positive control for the screening harness.
- [x] **Top-two assists: retained.** Sum of the two highest shrunken assist
  profiles. It is less influential than usage concentration but has stable
  incremental signal. See
  [NAIL-RAPM v1.2.1](nail-rapm-v121-pruned-nonadditive.md).
- [x] **Median three-point makes: rejected at frozen screen.** The residual
  deciles were flat and the pooled weighted correlation was **+0.0003**.
  See [Screen a Frozen Feature](../guides/screen-frozen-feature.md#initial-result-median-lineup-3pm-100).
- [x] **Rim-protection ceiling: rejected at frozen screen.** Maximum shrunken
  blocks per 100 did not produce a stable residual relationship. See
  [Screen a Frozen Feature](../guides/screen-frozen-feature.md#candidate-result-rim-protection-ceiling).
- [x] **Lower-tercile Critical Spacing: tested, not promoted.** Its coefficient
  was directionally credible, but it did not improve pooled full-game RMSE.
  See [NAIL Critical Spacing](nail-critical-spacing.md).
- [x] **Lower-quintile Critical Spacing: rejected.** The stricter threshold was
  less stable and predictively worse. See
  [Lower-Quintile Critical Spacing](nail-critical-spacing-quintile.md).
- [x] **Lower-quartile Critical Spacing with standard USG%: not promoted.**
  The joint spacing-and-usage variant did not add sufficient predictive value.
  See [Quartile Critical Spacing](nail-critical-spacing-quartile-standard-usage.md).

## Priority Candidates

The definitions below are initial contracts. Any change to a formula after
looking at outcomes creates a new candidate and receives a separate registry
entry.

### Creation-Spacing Alignment

- [x] **Frozen residual screen: rejected**
- [x] **Recursive candidate fit: did not advance**
- [x] **Coefficient and bootstrap decision: not applicable**

Let \(u_i\) be usage events per 100 and \(s_i\) be shrunken three-point makes
per 100. Define

\[
\phi_{\text{creation-spacing}}(U)
=\sum_{i\in U}\frac{u_i}{\sum_{j\in U}u_j}s_i.
\]

This asks whether a unit's shooting is concentrated in the players expected
to use possessions. The normalization makes it a lineup allocation feature,
not a sum of five independent shooting values.

The three-season frozen screen found no stable conditional residual signal:
the weighted correlations were `-0.0028`, `-0.0000`, and `+0.0070` for
2023-24 through 2025-26, respectively (`+0.0017` pooled). The lowest-to-highest
decile residual spread also reversed direction, from `-2.07` in 2023-24 to
`+3.43` and `+2.26` in the next two seasons. It therefore does **not** advance
to a recursive fit. See [Screen a Frozen Feature](../guides/screen-frozen-feature.md#candidate-result-creation-spacing-alignment).

Screen artifact:
`artifacts/models/analysis/frozen_feature_screen/creation_spacing_alignment/frozen-feature-screen-creation_spacing_alignment-20260830T195938Z-24613d9e`.

### Secondary-Creator Floor

- [x] **Frozen residual screen: rejected**
- [x] **Recursive candidate fit: did not advance**
- [x] **Coefficient and bootstrap decision: not applicable**

Let \(a_{(2)}(U)\) be the second-highest shrunken assists-per-100 profile among
the five players:

\[
\phi_{\text{secondary creator}}(U)=a_{(2)}(U).
\]

This tests whether the current `top_two_assists` term is identifying a more
specific mechanism: the presence of a credible second creator. The first test
uses assists alone; adding a turnover conversion constant would be a separate
candidate.

The three-season frozen screen did not support this narrower interpretation of
the retained `top_two_assists` term. Weighted residual correlations were
`+0.0132`, `+0.0005`, and `-0.0037` from 2023-24 through 2025-26 (`+0.0032`
pooled); the endpoint decile spread similarly reversed from `+4.45` to `-1.58`.
The feature therefore does **not** advance to a recursive fit. See
[Screen a Frozen Feature](../guides/screen-frozen-feature.md#candidate-result-secondary-creator-floor).

Screen artifact:
`artifacts/models/analysis/frozen_feature_screen/secondary_creator_floor/frozen-feature-screen-secondary_creator_floor-20260830T202300Z-2d67c5e4`.

### Rim Pressure By Spacing Floor

- [x] **Frozen residual screen: rejected**
- [x] **Recursive candidate fit: did not advance**
- [x] **Coefficient and bootstrap decision: not applicable**

Let \(r_i\) be shrunken unassisted rim makes per 100 and let \(s_{(1)}(U)\) and
\(s_{(2)}(U)\) be the two lowest shrunken three-point-make profiles. Define

\[
\phi_{\text{rim-spacing}}(U)
=\left(\sum_{i\in U}r_i\right)
  \left(\frac{s_{(1)}(U)+s_{(2)}(U)}{2}\right).
\]

The hypothesis is that rim pressure is more productive when the lineup's two
weakest spacers still provide credible gravity. This is a continuous successor
to the rejected hard Critical Spacing thresholds.

The three-season frozen screen did not support the interaction. Weighted
residual correlations were `+0.0038`, `-0.0065`, and `+0.0016` from 2023-24
through 2025-26 (`-0.0001` pooled), while the lowest-to-highest decile spread
changed direction in each season. It therefore does **not** advance to a
recursive fit. See [Screen a Frozen Feature](../guides/screen-frozen-feature.md#candidate-result-rim-pressure-by-spacing-floor).

Screen artifact:
`artifacts/models/analysis/frozen_feature_screen/rim_pressure_by_spacing_floor/frozen-feature-screen-rim_pressure_by_spacing_floor-20260830T202922Z-0678e0aa`.

### Defensive Anchor By Perimeter Pressure

- [x] **Frozen residual screen: rejected**
- [x] **Recursive candidate fit: did not advance**
- [x] **Coefficient and bootstrap decision: not applicable**

Let \(m\) be the player with the highest shrunken blocks per 100, \(b_m\), and
let \(d_j\) be steals per 100 for the other four players. Define

\[
\phi_{\text{anchor-pressure}}(U)
=b_m\sum_{j\in U,\,j\ne m}d_j.
\]

Maximum blocks alone failed its frozen screen. This different hypothesis asks
whether a rim protector becomes more useful when the other players generate
perimeter disruption.

The three-season frozen screen did not support the interaction. Weighted
residual correlations were `-0.0067`, `+0.0004`, and `+0.0099` from 2023-24
through 2025-26 (`+0.0021` pooled), and the endpoint decile direction reversed
from `-3.26` in 2023-24 to `+4.92` in 2025-26. It therefore does **not**
advance to a recursive fit. See [Screen a Frozen Feature](../guides/screen-frozen-feature.md#candidate-result-defensive-anchor-by-perimeter-pressure).

Screen artifact:
`artifacts/models/analysis/frozen_feature_screen/defensive_anchor_by_perimeter_pressure/frozen-feature-screen-defensive_anchor_by_perimeter_pressure-20260830T203515Z-3b3a6ee0`.

### Offensive Role Redundancy

- [x] **Profile coordinates and scaling locked before outcomes are inspected**
- [x] **Frozen residual screen: rejected**
- [x] **Recursive candidate fit: did not advance**
- [x] **Coefficient and bootstrap decision: not applicable**

For target season (t), use each player's frozen, shrunken prior-season rates
for usage events, assists, three-point attempts, unassisted rim makes,
offensive rebounds, and free-throw attempts per 100 possessions. For each
coordinate (k), let (c_{k,t-1}) be its possession-weighted 90th percentile
across all source-season players, weighted by source RAPM possessions. Define

\[
z_{i,k,t}=p_{i,k,t-1}/c_{k,t-1}.
\]

After player-level unit-length normalization, define average pairwise cosine
similarity:

\[
\phi_{\text{redundancy}}(U)
=\frac{1}{10}\sum_{i<j}\frac{z_i^\mathsf{T}z_j}
{\lVert z_i\rVert_2\lVert z_j\rVert_2}.
\]

High values represent five players with unusually similar offensive roles.
The home-minus-away feature edge enters the screen. The pre-registered
basketball hypothesis is negative: high role similarity may leave a unit with
less complementary offensive coverage after player ratings and retained
non-additive terms have been accounted for.

The frozen screen was directionally negative in 2023-24 and 2024-25, but
reversed in 2025-26. Weighted residual correlations were `-0.0096`, `-0.0144`,
and `+0.0027` (`-0.0069` pooled), so the candidate fails the pre-registered
three-season stability gate and does **not** advance to a recursive fit. See
[Screen a Frozen Feature](../guides/screen-frozen-feature.md#candidate-result-offensive-role-redundancy).

Screen artifact:
`artifacts/models/analysis/frozen_feature_screen/offensive_role_redundancy/frozen-feature-screen-offensive_role_redundancy-20260830T204638Z-222eb12b`.

## Secondary Candidates

### Foul-Pressure Diversity

- [ ] **Frozen residual screen**
- [ ] **Recursive candidate fit if supported**

For \(f_i=\mathrm{FTA100}_i\), let \(q_i=f_i/\sum_j f_j\). Define

\[
\phi_{\text{foul diversity}}(U)
=\left(\sum_i f_i\right)\left(1-\sum_i q_i^2\right),
\]

with zero assigned when total FTA is zero. It distinguishes several credible
foul-pressure sources from the same total generated by one player.

### Size-Skill Coverage

- [ ] **Lock a single basketball-motivated contract**
- [ ] **Frozen residual screen**
- [ ] **Recursive candidate fit if supported**

The candidate should combine a lower-tail size statistic with a middle or
lower-tail perimeter-skill statistic. No formula is registered yet because an
arbitrary product of height and shooting would be difficult to defend. The
contract must specify the exact order statistics and season scaling before any
outcome is inspected.

### Defensive Weak Link

- [ ] **Identify a forward-safe defensive profile that is not derived from the
  target being predicted**
- [ ] **Lock the order statistic**
- [ ] **Frozen residual screen**

The intended hypothesis is that a defense can be attacked through its weakest
member. Blocks plus steals is not accepted as a general defensive-quality
score, and using a fitted NAIL defensive split could double-count the rating
state. This candidate remains a design problem rather than a registered
formula.

## Prediction-Only Relationship Features

These candidates may help real-team forecasting but are not intrinsic,
portable descriptions of five-player composition. They should not silently
penalize hypothetical or mixed-era units for never having played together.

### Prior Teammate Continuity

- [x] **Build prior shared-possession pair table**
- [x] **Frozen residual screen: advances**
- [x] **Recursive candidate fit: tested, not promoted**
- [x] **Coefficient and bootstrap decision: stable but no incremental lift**

For the ten player pairs and strictly prior shared possessions \(c_{ij,t-1}\),
define

\[
\phi_{\text{continuity}}(U,t)
=\frac{1}{10}\sum_{i<j}\log\left(1+c_{ij,t-1}\right).
\]

This tests repeatable coordination and familiarity. It belongs in a prediction
layer or optional Lab control, not the portable lineup-composition score.

The v1.2.1.3 frozen screen found a positive residual relationship in all three
target seasons. Weighted correlations were `+0.0093`, `+0.0081`, and `+0.0106`
for 2023-24 through 2025-26, with a pooled value of `+0.0093`. The residual
spread from the lowest to highest continuity decile was positive in every
season.

The full 30-season recursive fit retained a positive standardized coefficient
in 27 of 29 states and a 99.75% positive one-sided mass share. The three-season
frozen replay passed the paired non-inferiority gate but did not materially
improve full-game RMSE; possession MAE and team net-rating RMSE worsened. The
feature is therefore **tested, not promoted**: stable and probably real, but
largely redundant with the incumbent forecast state. See the
[complete model result](nail-teammate-continuity.md) and the original
[screening result](../guides/screen-frozen-feature.md#candidate-result-prior-season-teammate-continuity).

### Prior Teammate Continuity Replacing Top-Two Assists

- [x] **Controlled recursive replacement fit**
- [x] **Frozen three-season and playoff replay**
- [x] **Coefficient and paired-bootstrap decision: not promoted**

The follow-up candidate retains `usage_concentration` but replaces
`top_two_assists` with prior teammate continuity. Continuity remains positive
in 28 of 29 fitted states with a 99.999% positive directional-mass share and a
median standardized weight of `+1.12`. Pooled full-game RMSE improves by only
`0.0047`, with a paired 95% interval of `[-0.0364, +0.0272]`. Possession MAE
worsens by `0.000082`, team NetRtg RMSE worsens from `3.2351` to `3.2853`, and
the feature remains unavailable for genuinely hypothetical teammate groups.
The replacement is therefore stable and non-inferior, but not promoted. See
the [complete replacement result](nail-teammate-continuity-replacement.md).

### Shrunken Prior Pair Residual

- [ ] **Specify pair-effect shrinkage and minimum support**
- [ ] **Frozen residual screen**
- [ ] **Recursive candidate fit if supported**

This is the direct chemistry candidate: estimate prior player-pair residuals,
shrink them strongly toward zero as a function of pair exposure, and sum the
ten frozen pair effects for a unit. It is more expressive than continuity but
has the highest sparsity and multiple-testing risk in the registry.

## Testing Contract

Every candidate follows the same sequence:

1. **Pre-register the formula.** Record the player inputs, shrinkage source,
   missing-value behavior, expected direction, portability, and support.
2. **Check structural additivity.** Reject or move the feature to the additive
   profile layer if it can be exactly decomposed into five player terms.
3. **Run the frozen residual screen.** Use the promoted model and 2023-24,
   2024-25, and 2025-26 without refitting the candidate. Inspect weighted
   residual deciles and each season separately.
4. **Run one controlled recursive fit.** Add only the screened feature to the
   incumbent contract. The context correction must roll forward through every
   source season because it affects subsequent player states.
5. **Audit coefficient history.** Publish the full non-additive coefficient
   panel, directional one-sided mass, support, and displacement of the two
   incumbent terms.
6. **Evaluate the frozen holdouts.** Report the standard three-season regular
   and pooled playoff metrics and update the model tree for every completed
   full model, promoted or not.
7. **Apply the paired game-block bootstrap gate.** Report pooled and
   season-specific full-game RMSE differences. Passing the non-inferiority gate
   is necessary but does not establish improvement.
8. **Record the decision here.** Link the immutable screen, recursive fit,
   coefficient audit, bootstrap artifact, and model page before checking the
   candidate as resolved.

The residual screen is a compute-saving diagnostic, not a substitute for the
recursive frozen replay. A feature can appear useful in a fixed residual and
lose that signal after all player and context coefficients are refit jointly.

## Result Template

Copy this block when a pending candidate is tested:

```markdown
### Candidate Name

- [x] Frozen residual screen: `artifact path`
- [x] Recursive fit: `artifact path` or `did not advance`
- [x] Coefficient audit: `artifact path` or `not applicable`
- [x] Paired bootstrap: `artifact path` or `not applicable`
- **Decision:** retained / rejected / tested, not promoted
- **Reason:** one-sentence predictive and stability conclusion
```
