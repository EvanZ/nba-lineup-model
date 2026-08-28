---
last_updated: "2026-08-27"
---

# NAIL Model Contract

This is the assumption register for the production release,
**NAIL-RAPM v1.2.1.3**, which supersedes v1.2.1.2 by selecting each source-season
player penalty directly on its residualized source target. It applies the same standard as the
[Split NAIL Model Contract](split-nail-model-contract.md): a model can be
strictly forward and still include choices that are inherited, fixed, or not
identified by the data.

Each immutable audit run is stored under `artifacts/audits/nail_model_contract/`
as Parquet and JSON, together with the matrix hash, status counts, and
promotion-blocker flags.

## Current Verdict

Production NAIL has substantially more empirical support than Split NAIL: its
gap-returner policy, profile-padding contract, retained non-additive bundle,
linear antisymmetric context, B2B feature, and player-lambda provenance each
have documented candidate comparisons. It is still **not a fully validated
final specification**.

The audit identifies several decisions that should remain explicit in every
public model card:

1. Context alpha `10,000` and B2B alpha `10,000` are inherited numerical
   choices, not jointly selected for the current release.
2. The current sequential player -> schedule -> context protocol is not a
   joint or iterated optimum; it is a deliberate allocation rule that needs a
   controlled comparison.
3. The three frozen seasons have informed iterative research decisions and are
   no longer a pristine final holdout.
4. Player ratings have no published uncertainty intervals.

This does not invalidate the existing frozen results. It says exactly what
those results establish and what they do not.

## Status Definitions

| Status | Meaning |
| --- | --- |
| `historically_selected_for_scalar_release` | Directly compared earlier in the scalar NAIL lineage and retained in this release. |
| `inherited_not_jointly_revalidated_current_release` | Has prior evidence, but was not jointly reselected with the complete v1.2.1.2 contract. |
| `fixed_choice_not_validated_current_release` | A current numerical or algorithmic choice without a direct release-level validation. |
| `structural_identifiability_or_attribution_limit` | A decomposition convention, not a uniquely estimated causal quantity. |
| `established_data_contract` | A required data/identifiability convention rather than a tunable release feature. |
| `data_or_evaluation_boundary` | A stated limitation of the evaluation scope or available evidence. |

## Contract Matrix

### Objective And Player Prior

| Choice | Current setting | Status | Required validation |
| --- | --- | --- | --- |
| Stint target | Possession-weighted reconstructed home net rating | Established data contract | Maintain reconstruction and possession-conservation tests. |
| Season type | Regular season trains; playoffs are a frozen cohort | Inherited | Confirm regular-only vs regular-plus-playoffs on a fresh window. |
| Aging | Value-conditioned forward total prior | Inherited | Revalidate only as an explicit release component. |
| Cold start | Exposure-gated draft/replacement total prior | Inherited | Lock a rookie/low-exposure cohort test. |
| Replacement token | Forward low-exposure replacement pool | Inherited | Test cutoff/policy only in a dedicated cold-start experiment. |
| Gap returner | Last-observed annual aging bridge | Historically selected | Retest only with a forward gap-returner ablation. |
| Rating origin | Prior-season exposure centering | Established data contract | Preserve exact centering/reconstruction tests. |
| Player lambda | v1.2.1.3 uses source-season \(s=t-1\) three-fold CV on the context- and B2B-residualized \(s\) target, over an explicit 13-value grid from `1e-5` to `10`; v1.2.1.2 imported a forward RAPM schedule | Historically selected | v1.2.1.3 beat the imported schedule on frozen regular full-game RMSE, winner accuracy, team net-rating RMSE, Pythagorean-win RMSE, and playoff eligible-margin RMSE. |
| Player prior precision | Uniform Ridge precision | Historically selected | Retain unless state precision wins a locked comparison. |

### Profile And Context Contract

| Choice | Current setting | Status | Required validation |
| --- | --- | --- | --- |
| Profile timing | Strictly lagged, last-observed returner profile | Inherited | Test only in gap/cold-start work. |
| Profile shrinkage | Stat-specific Medvedovsky-style padding | Historically selected | Treat as fixed unless a new padding candidate clears frozen evaluation. |
| Additive profile bundle | Eight coordinates: `three_pa_per_100`, `three_pm_per_100`, `assists_per_100`, `turnovers_per_100`, `usage_pct`, `steals_per_100`, `blocks_per_100`, `offensive_rebound_claim_total` | Inherited | Confirm any changed bundle against a locked scalar baseline. |
| Non-additive bundle | `top_two_assists`, `usage_concentration` | Historically selected | Require residual screens, stability, and a frozen ablation for changes. |
| Context family | Linear Ridge on antisymmetric unit differences | Historically selected | New nonlinear families must be separate candidates. |
| Context alpha | Common `10,000` alpha after feature standardization | **Fixed, unvalidated** | Test a pre-registered common-alpha grid after lambda provenance is resolved. |
| Context block penalties | Additive and non-additive terms share alpha | **Fixed, unvalidated** | Test one additive-vs-non-additive penalty split only after the baseline is locked. |
| Residualization protocol | Prior context -> player RAPM -> B2B -> current context, one pass | **Fixed, unvalidated** | Compare with a controlled alternating/joint-fit formulation. |
| Additive player credit | Post-fit attribution of lineup-level additive features | Structural attribution limit | Keep reconstruction checks; do not label it a causal player estimate. |
| Portable h/q explanation | Possession-weighted completed-season reference field | Structural attribution limit | Persist/reference the source field with every interpretation. |

### Schedule And Evaluation

| Choice | Current setting | Status | Required validation |
| --- | --- | --- | --- |
| B2B definition | Previous competitive game was exactly one calendar day earlier | Historically selected | Treat other rest variables as separate candidates. |
| B2B alpha | `10,000`, inherited from context alpha | **Fixed, unvalidated** | Run a small B2B-alpha sensitivity after player-lambda validation. |
| B2B fitting order | Fit after player RAPM; carry forward and subtract next season | **Fixed, unvalidated** | Include it in the joint/alternating allocation control. |
| Home court | Season-specific scalar residual intercept | Established data contract | Keep separate from player credit. |
| Target exposure | Actual target-season lineups and possessions | Evaluation boundary | Do not call these full-season forecasts. |
| Reused frozen window | 2023-24 through 2025-26 informed iteration | Evaluation boundary | Confirm on a locked new season/window. |
| Uncertainty | No player-level intervals | Evaluation boundary | Add clustered-game bootstrap or posterior intervals after contract lock. |

## Next Audit Sequence

The sensible order is not to tune everything simultaneously:

1. Test the common context and B2B penalty choices under the selected player
   lambda protocol.
2. Then test whether the one-pass residualization contract should be
   replaced by an alternating or joint fit.
3. Reserve a new confirmation period before promoting a changed production
   contract.

This sequence preserves interpretability: each experiment answers one
identified question rather than burying multiple new choices in a new model.
