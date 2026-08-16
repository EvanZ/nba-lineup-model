---
title: NAIL-RAPM Attribution Contract
---

# NAIL-RAPM Attribution Contract

Last updated: 2026-08-16

## Result

NAIL-RAPM v1.0 assigns additive features to players and retains only
non-additive features at the lineup level. Moving the fitted additive component
from one term to the other produces the same forecast. The reconstruction error
is at floating-point rounding scale, not an approximation.

This establishes a clean distinction:

- **Additive context** can be attributed to the five players in a unit.
- **Non-additive lineup context** remains a property of the combination and is not
  reducible to five independent player values.

This is the attribution contract for NAIL-RAPM v1.0. The underlying artifact
and CLI retain their earlier `hpm_x3` identifier for reproducibility only.

## Non-Additive Lineup Edge

Player rankings report **Non-Additive Lineup Edge** separately from NAIL-RAPM.
For player \(i\) in season \(t\), it is the possession-weighted average
of the residual non-additive lineup edge from regular-season stints in which the
player appeared:

\[
E_{i,t}=\frac{\sum_{s\ni i}w_s d_{i,s} C_{\mathrm{nonadd},s}}
{\sum_{s\ni i}w_s}.
\]

Here \(w_s\) is the stint's possessions and \(d_{i,s}\) is \(+1\) for the
home unit and \(-1\) for the away unit. The quantity uses only the six
six non-additive NAIL-RAPM coordinates below. It describes the non-additive lineup context
of units a player actually shared; it is not an additional player rating or
causal allocation of that unit effect.

## Identity

Let \(H\) and \(A\) be the home and away units. NAIL-RAPM v1.0 fits:

\[
C(H,A) = \beta^\top\big(x(H)-x(A)\big).
\]

For an additive feature, \(x_f(U)=\sum_{i\in U} z_{if}\). Its fitted
coefficient therefore compiles into a player adjustment:

\[
\delta_i = \sum_{f\in\mathcal{A}} \beta_f z_{if}.
\]

The additive context edge is then exactly:

\[
C_{\mathrm{add}}(H,A)=\sum_{i\in H}\delta_i-\sum_{j\in A}\delta_j.
\]

The full model equals that player edge plus the non-additive lineup remainder:

\[
C(H,A)=C_{\mathrm{add}}(H,A)+C_{\mathrm{nonadd}}(H,A).
\]

### Why This Does Not Double Count

The player RAPM fit first produces a raw player state:

\[
R_{i,\mathrm{raw}} = R_{i,\mathrm{prior}} + \Delta R_{i,\mathrm{season}}.
\]

The context model is fit to the residual after this raw player edge is
removed. Its additive component is then compiled into the published player
rating, \(R_i^{\mathrm{NAIL}}=R_{i,\mathrm{raw}}+\delta_i\), while
\(C_{\mathrm{nonadd}}\) remains assigned to the full unit. Therefore:

\[
\sum_{i\in H}R_i^{\mathrm{NAIL}}-\sum_{j\in A}R_j^{\mathrm{NAIL}}
+C_{\mathrm{nonadd}}(H,A)
=
\sum_{i\in H}R_{i,\mathrm{raw}}-\sum_{j\in A}R_{j,\mathrm{raw}}
+C_{\mathrm{add}}(H,A)+C_{\mathrm{nonadd}}(H,A).
\]

This is a reparameterization, not an additional fitted effect. The canonical
artifact reconstructs it to floating-point precision.

## Feature Split

| Compiled into player adjustments | Remains non-additive lineup context |
| --- | --- |
| Three-point attempts | Bottom-two three-point makes |
| Three-point makes | Credible-shooter count |
| Assists | Top-two assists |
| Turnovers | Usage concentration |
| Usage | Shooting-by-usage interaction |
| Steals | Shooter-by-passing interaction |
| Blocks |  |
| Offensive-rebound claim total |  |

## Forecast-Equivalence Check

Every frozen 2023-24 through 2025-26 regular-season and playoff possession
was evaluated with the matching prior-season linear context state.

| Target season | Cohort | Possessions | Additive compilation error | Context reconstruction error | Full forecast-component error | RMS additive context | RMS non-additive lineup edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-24 | Regular season | 182,729 | 1.47e-14 | 1.49e-14 | 1.82e-14 | 3.94 | 1.89 |
| 2023-24 | Playoffs | 12,570 | 1.73e-14 | 1.82e-14 | 1.93e-14 | 3.90 | 1.91 |
| 2024-25 | Regular season | 183,431 | 1.74e-14 | 1.78e-14 | 1.95e-14 | 4.60 | 2.00 |
| 2024-25 | Playoffs | 13,144 | 1.55e-14 | 1.69e-14 | 1.82e-14 | 3.75 | 2.16 |
| 2025-26 | Regular season | 218,810 | 1.07e-14 | 1.07e-14 | 1.19e-14 | 3.60 | 2.12 |
| 2025-26 | Playoffs | 14,253 | 9.77e-15 | 8.88e-15 | 1.17e-14 | 3.42 | 1.80 |

The full forecast-component check compares:

\[
\underbrace{\text{base player prior}}_{\theta}
+\underbrace{C_{\mathrm{add}}(H,A)+C_{\mathrm{nonadd}}(H,A)}_{\text{original context}}
\]

against:

\[
\underbrace{\text{base player prior}+\delta}_{\text{compiled player prior}}
+\underbrace{C_{\mathrm{nonadd}}(H,A)}_{\text{non-additive context only}}.
\]

They are exactly the same prediction. Common league-average and home-court
terms are unchanged, so the complete possession-margin forecast is also
identical.

The audit artifact is stored under
`artifacts/models/analysis/linear_hpm_x3_compilation_audit/` and includes the
target-season compiled player-adjustment table.

## What This Does Not Say

This identity applies when the additive player adjustment is compiled using the
same pre-existing linear context coefficients. It does not imply that a
separate model which refits raw profile fields as a player prior will recover
the same predictions or player values. That separate model learns a
player-season residual from outcomes with its own regularization and a
different competition for attribution.

## Reproduce

```bash
uv run nba-audit-linear-hpm-x3-compilation
```
