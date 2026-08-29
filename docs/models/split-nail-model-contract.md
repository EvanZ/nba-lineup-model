---
last_updated: "2026-08-27"
---

# Split NAIL Model Contract

This is the formal assumption register for standalone **Split NAIL-RAPM**. It
exists because a forward model can be leakage-free yet still rest on arbitrary
or unidentified design choices. A choice is not considered validated merely
because it is plausible, inherited from scalar NAIL, or produces reasonable
rankings.

The immutable machine-readable audit is persisted under
`artifacts/audits/split_nail_model_contract/`. Each run contains a Parquet and
JSON contract matrix, a hash, status counts, and an explicit promotion-blocker
flag for every row.

## Status Definitions

| Status | Meaning |
| --- | --- |
| `selected_within_split_contract` | Selected chronologically inside each Split season, conditional on the rest of the current contract. |
| `inherited_scalar_contract_not_revalidated_for_split` | Previously accepted scalar NAIL choice that has not yet been jointly tested with O/D splitting. |
| `fixed_split_choice_not_validated` | New numerical or state-transition choice that is currently a heuristic. |
| `structural_identifiability_or_attribution_limit` | The data/model parameterization cannot separately identify the reported quantity. |
| `data_or_evaluation_boundary` | A stated limit of the available data or evaluation design rather than a fitted parameter. |

## Current Verdict

**Split NAIL is an audited research candidate, not a promotable O/D rating
model.** Its no-refit replay and component reconstruction are valid, but the
O/D allocation contains unresolved promotion blockers. The most consequential
ones are the 4x specialization penalty, the 1.5x feature precision, the
neutral/gap-returner side-state policy, and the unidentifiable O/D split of
home-court advantage.

The scalar total-prior ingredients are not being discarded. They retain their
prior NAIL evidence, but that is intentionally not treated as evidence that
their interaction with the O/D state is correct.

## Contract Matrix

### Objective And Evaluation

| Choice | Current setting | Status | Required next validation |
| --- | --- | --- | --- |
| Scoring target | Two per-100 scoring rows per stint, weighted by offensive possessions | Structural limit | Compare against an otherwise identical net-margin parameterization. |
| Season type | Regular season trains; playoffs are frozen evaluation | Inherited scalar contract | Pre-register regular-only vs. regular-plus-playoffs training on a fresh window. |
| Target exposure | Realized target lineups and possessions | Evaluation boundary | Keep separate from rotation/minutes forecasting. |
| Reused leaderboard window | 2023-24 through 2025-26 informed iteration | Evaluation boundary | Confirm on a truly untouched season or locked historical window. |

### Scalar Total Prior

| Choice | Current setting | Status | Required next validation |
| --- | --- | --- | --- |
| Aging | Value-conditioned forward aging, total rating only | Inherited scalar contract | Jointly validate aging with the O/D state. |
| Cold starts | Exposure-gated draft/replacement total prior | Inherited scalar contract | Test O/D cold-start policies on rookie/low-exposure cohorts. |
| Replacement level | Forward low-exposure replacement token, total only | Inherited scalar contract | Test neutral versus learned O/D replacement state. |
| Gap returns | Last-observed annual aging bridge for total rating | Inherited scalar contract | Validate total and specialization gap transitions together. |
| Centering | Prior-season exposure-centered totals | Inherited scalar contract | Verify equivalent O+D recentering and retain as an identifiability invariant. |

### O/D State And Features

| Choice | Current setting | Status | Required next validation |
| --- | --- | --- | --- |
| Side persistence | Carry immediate-prior (s=O-D) unchanged | **Fixed, unvalidated** | Select a small persistence/forgetting grid on all frozen seasons. |
| Gap specialization | Missing immediate side state resets to (s=0) | **Fixed, unvalidated** | Compare neutral reset with decayed last-observed specialization. |
| Cold-start specialization | New players use (O=D=P/2) | **Fixed, unvalidated** | Compare against pre-specified O/D cold-start alternatives. |
| O/D aging | No side-specific aging process | **Fixed, unvalidated** | Test only after specialization persistence is resolved. |
| Profile source | Strictly lagged, last-observed profile; cold starts use replacement blend | Inherited scalar contract | Restrict any test to a forward gap/cold-start study. |
| Profile padding | Medvedovsky-style rate padding | Inherited scalar contract | Treat as registered default; compare alternatives only in a separate frozen study. |
| Additive feature sides | All eight profile features get O and D coefficients | **Fixed, unvalidated** | Compare a small semantic constraint set to all-sides-free. |
| Non-additive feature sides | `top_two_assists` and `usage_concentration` get O and D coefficients | **Fixed, unvalidated** | Run individual O/D ablations after penalty selection. |

The eight additive features are `three_pa_per_100`, `three_pm_per_100`,
`assists_per_100`, `turnovers_per_100`, conventional `usage_pct`,
`steals_per_100`, `blocks_per_100`, and `offensive_rebound_claim_total`.

### Penalties And Schedule

| Choice | Current setting | Status | Required next validation |
| --- | --- | --- | --- |
| Player lambda | Chronological within-season selection over the standard grid | Selected conditionally | Reselect inside every season whenever O/D penalties change. |
| Player O/D penalty | (\operatorname{precision}(s)=4\operatorname{precision}(m)) | **Sensitivity-tested, not selected** | An exploratory \(r\in\{1,2,4,8\}\) frozen replay had mixed winners; pre-register development selection before changing the control. |
| Feature precision | Feature (m) precision (=1.5\times) player (m); feature (s=6\times) player (m) | **Fixed, unvalidated** | Jointly test a small feature-precision grid with the specialization ratio. |
| Feature block sharing | Additive and non-additive terms share one multiplier | **Fixed, unvalidated** | After global selection, test one additive-vs-non-additive block split. |
| B2B | Calendar-day flag, known before tipoff; O/D terms use the feature penalty | Inherited plus **unvalidated O/D split** | Test a scalar B2B term before retaining a side split. |
| Home court | Separate O and D coefficients | **Structurally unidentified** | Collapse to a scalar net home-court term before promotion. |

## Required Sequence Before Promotion

1. Collapse home court to a scalar net term; it cannot support an O/D claim.
2. Pre-register and evaluate the player specialization-precision ratio with
   lambda reselected inside every completed season.
3. Evaluate the feature-block precision only after the player O/D penalty is
   fixed.
4. Test side-state persistence, gap handling, and cold-start allocation on
   their relevant cohorts.
5. Confirm the chosen contract on an untouched season or a locked backtest
   window, then add uncertainty intervals for published O/D values.

Until that sequence is complete, the scalar NAIL production model remains the
only promotable player-value model. Split NAIL's current frozen replay and
attribution audit are useful diagnostics, not evidence that its offense and
defense allocations are uniquely correct.
