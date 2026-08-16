---
last_updated: "2026-08-14"
---

# HPM v1 Feature Knockouts

This study tests whether each original HPM v1 contextual feature family earns
its complexity. Every candidate retains the same player prior, bounded
hierarchical P-spline procedure, temporal hierarchy, training seasons, and
frozen evaluation contract as Original HPM v1. It removes only one family from
the contextual design matrix.

| Candidate | Removed signals |
| --- | --- |
| Without shooting | 3PA, 3PM, bottom-two shooting, credible-shooter count, shooting-by-usage, shooter-by-passing |
| Without creation/passing | assists and top-two assists |
| Without rebounding | offensive/defensive rebounds, square-root rebound summaries, rebounding-by-usage |
| Without defensive events | steals and blocks |
| Without uncertainty/replacement | imputed-profile count and replacement-profile weight |

Each completed candidate will face Original HPM v1 in the
[Frozen Model Tournament](frozen-model-tournament.md). A knockout is promoted
only if it has a paired 95% bootstrap interval below zero for full-game margin
RMSE; otherwise the original feature family remains in the production model.

## Fixed-Model Reliance Screen

Before paying for five full recursive refits, we ran a faster, fixed-model
screen. For each frozen target season from 2023-24 through 2025-26, it holds
the completed Original HPM v1 model and player priors fixed, replaces one
feature family's value on **both** units with its possession-weighted reference
value, and replays the realized regular-season stints and possessions. This
measures how much the saved model relies on that family; it does **not** measure
the incremental predictive value of a family after the remaining correlated
signals have been retrained to compensate.

Positive deltas below mean the neutralized model is worse, so the retained
family supports the original forecast. The reported interval is a paired 95%
bootstrap interval over the three pooled frozen regular seasons.

| Neutralized family | Full-game margin RMSE delta | Paired 95% interval | Initial read |
| --- | ---: | ---: | --- |
| Shooting | -0.0052 | [-0.0483, +0.0380] | No material reliance signal |
| Creation/passing | +0.0530 | [-0.0099, +0.1171] | Suggestive, but inconclusive |
| Rebounding | +0.0700 | [+0.0400, +0.1003] | Clear reliance signal |
| Defensive events | +0.0467 | [-0.0003, +0.0948] | Borderline reliance signal |
| Uncertainty/replacement | +0.0149 | [-0.0084, +0.0382] | No material reliance signal |

The same replay records winner accuracy, possession RMSE, and possession MAE.
Its immutable artifact is
`artifacts/models/analysis/hpm_v1_feature_inference_audit/hpm-v1-feature-inference-audit-20260815T021616Z-597607aa/`.
Rebounding is the first family worth prioritizing for a full recursive
ablation; the screen alone is not grounds for deleting any other family.

## Rebounding Components

We then used the same no-refit, three-season screen to neutralize each input
within the original rebounding family individually. This identifies which
saved HPM v1 terms the forecast actually depends on, while retaining the same
important limitation: correlated terms cannot adapt because the model is not
retrained.

| Neutralized input | Full-game margin RMSE delta | Paired 95% interval | Initial read |
| --- | ---: | ---: | --- |
| Offensive rebounds per 100 | +0.0340 | [+0.0137, +0.0546] | Clear reliance signal |
| Defensive rebounds per 100 | +0.0011 | [-0.0053, +0.0074] | No standalone signal |
| Square-root offensive rebounds | -0.0021 | [-0.0130, +0.0089] | No standalone signal |
| Square-root defensive rebounds | +0.0009 | [-0.0048, +0.0065] | No standalone signal |
| Rebounding-by-usage interaction | +0.0337 | [+0.0182, +0.0495] | Clear reliance signal |

The two clear terms nearly account for the full family result together. This
does not prove that defensive rebounding or the square-root summaries are
useless after a refit; it says that the saved HPM v1 forecast primarily draws
its current rebounding signal from offensive rebounding and how rebounding is
paired with lineup usage.

### What “rebounding-by-usage” Means

The saved v1 feature is more precisely named **offensive-rebound × usage
concentration**. For a unit (U), it is

\[
r(U) = \sqrt{\sum_{i \in U} \operatorname{ORB100}_i},\qquad
u(U) = \frac{\operatorname{USG100}_{(1)} + \operatorname{USG100}_{(2)}}
{\sum_{i \in U} \operatorname{USG100}_i},\qquad
x(U) = r(U)u(U).
\]

Here “(1)” and “(2)” are the two highest-usage players in the unit.
For example, a lineup with total profile offensive rebounds of 9 per 100 has
(r=3). If its top two players account for 57% of the unit's profile usage,
then (x=3\times0.57=1.71). With the same rebounding but a more evenly shared
43% top-two usage share, (x=1.29).

It is a **lineup-level proxy**, not a player attribution and not a direct net
rating coefficient: the fitted spline learns the response to this value along
with all other context features. The term was intended to test whether
offensive rebounding has a different lineup effect when offensive creation is
concentrated. Its reliance result makes it worth retaining for now, but it is
also exactly the sort of hand-crafted feature that merits a later retrained
ablation or replacement by a more principled possession-allocation feature.
The immutable artifact is
`artifacts/models/analysis/hpm_v1_feature_inference_audit/hpm-v1-rebounding-components-inference-audit-20260815T022949Z-f87b9b04/`.

## Next Group: Defensive Events

The next component-level screen separates the original defensive-event family
into its two raw inputs: steals per 100 possessions and blocks per 100
possessions. This is a diagnostic first, not a claim that either event is an
individual player's context value. The follow-up candidate will be designed
only after this audit determines whether either saved HPM v1 response is
materially relied on.

## Run

```bash
uv run nba-audit-hpm-v1-context
```

This no-refit screen typically completes in minutes because it caches each
distinct five-player unit and unit pair. To run a full retrained knockout:

```bash
uv run nba-train-hpm-v1-knockout shooting --through-season 2025-26
```

Replace `shooting` with `creation`, `rebounding`, `defensive_events`, or
`uncertainty`. Each run writes an immutable recursive artifact under
`artifacts/models/forward_hpm_v1_without_<family>/`.

To repeat the individual rebounding screen:

```bash
uv run nba-audit-hpm-v1-rebounding
```

To audit steals and blocks separately:

```bash
uv run nba-audit-hpm-v1-defensive-events
```

| Neutralized input | Full-game margin RMSE delta | Paired 95% interval | Initial read |
| --- | ---: | --- | --- |
| Steals per 100 | +0.0199 | [-0.0206, +0.0603] | Inconclusive |
| Blocks per 100 | +0.0237 | [+0.0028, +0.0446] | Clear fixed-model reliance |

The immutable artifact is
`artifacts/models/analysis/hpm_v1_feature_inference_audit/hpm-v1-defensive-event-components-inference-audit-20260815T155811Z-b186e6b7/`.

## Next Group: Creation and Passing

The final unexamined original HPM v1 group separates raw assists per 100,
top-two assists, and the credible-shooter-by-top-two-assists interaction. The
last signal belongs here even though it appears in the historical shooting
bundle: its meaning is explicitly conditional on concentrated passing and
therefore needs its own component screen.

```bash
uv run nba-audit-hpm-v1-creation
```
