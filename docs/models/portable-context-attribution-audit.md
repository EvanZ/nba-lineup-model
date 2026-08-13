---
last_updated: "2026-08-11"
---

# Portable Context Attribution Audit

Value-Conditioned Aging HPM is intentionally predictive: it can place a
lineup-level pattern in the portable composition term (h(U)) or the specific
matchup term (q(U,V)), rather than forcing all of it into five player
coefficients. This audit asks a separate accounting question: conditional on a
specific unit, how does the frozen context function change as each player's
preseason profile enters the unit?

The results are **model attribution**, not causal credit. They explain how HPM
uses player profiles in its context layer; they do not prove how much a player
caused the observed result.

For a whole-season player reallocation of frozen context, see
[Context-Reattributed RAPM](context-reattributed-rapm.md). This Shapley audit
instead explains a selected unit's composition and matchup ledger.

## Method

The frozen 2024-25 context state evaluates a 2025-26 matchup as

\[
C(A,B) = h(A) - h(B) + q(A,B).
\]

For composition accounting, each player in a unit is replaced by one of five
distinct synthetic league-average profiles. Those profiles are possession
weighted from the completed 2024-25 player pool. For every subset (S) of the
five real players, (v(S)) is the portable score after players in (S) enter
the otherwise-reference unit. Player (i)'s exact Shapley value is

\[
\phi_i = \sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(|N|-|S|-1)!}{|N|!}
\left[v(S\cup\{i\})-v(S)\right].
\]

Thus the five values add exactly to (h(A)-h(R)), where (R) is that
synthetic reference unit. They do **not** necessarily add to (h(A)), because
HPM defines (h(A)) against its empirical reference-unit distribution rather
than against (R). The reference cancels when sides are compared:

\[
\sum_{i\in A}\phi_i^{\mathrm{comp}}
- \sum_{j\in B}\phi_j^{\mathrm{comp}}
= h(A)-h(B).
\]

The matchup residual uses the same calculation across all ten players, with
both sides initially set to (R). Its ten Shapley values add exactly to
(q(A,B)). Therefore every player row has a signed contribution to the focal
unit's edge and all ten rows reconcile to (C(A,B)).

`HPM player rating + context contribution` is included as a useful unit ledger,
not a newly calibrated or leaderboard-eligible player metric.

## Selected Units

These are frozen 2025-26 evaluations using realized units. The possession and
game counts only describe how often the exact matchup occurred; they are not
used in the frozen context score itself.

| Unit matchup | Realized possessions | Games | Player edge | Composition edge | Matchup edge | Total context | Predicted net rating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Warriors unit at Spurs unit | 37.3 | 2 | -11.18 | +1.40 | -2.39 | -0.99 | -12.17 |
| Rockets unit vs Clippers unit | 46.5 | 2 | +2.08 | -1.21 | +0.61 | -0.60 | +1.48 |
| Knicks unit vs Hawks unit | 36.5 | 1 | +0.45 | +0.47 | -0.20 | +0.27 | +0.72 |

### Warriors Unit At Spurs Unit

Focal unit: Stephen Curry, Jimmy Butler III, Draymond Green, Moses Moody, and
Will Richard. Opponent: Harrison Barnes, De'Aaron Fox, Devin Vassell, Victor
Wembanyama, and Stephon Castle.

| Player | Side | HPM player rating | Player edge | Composition contribution | Matchup contribution | Total context contribution | Combined unit ledger |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stephen Curry | Warriors | +1.01 | +1.01 | +3.40 | -0.53 | **+2.87** | +3.88 |
| Draymond Green | Warriors | -2.87 | -2.87 | +1.83 | -0.28 | **+1.55** | -1.32 |
| Jimmy Butler III | Warriors | +4.37 | +4.37 | +1.33 | -0.45 | **+0.89** | +5.26 |
| Moses Moody | Warriors | +1.45 | +1.45 | +0.26 | +0.02 | **+0.27** | +1.72 |
| Will Richard | Warriors | -2.09 | -2.09 | -1.56 | -0.49 | **-2.05** | -4.13 |
| Victor Wembanyama | Spurs | +7.18 | -7.18 | -2.27 | -0.47 | **-2.74** | -9.91 |
| De'Aaron Fox | Spurs | +0.94 | -0.94 | -1.08 | -0.09 | **-1.17** | -2.11 |
| Devin Vassell | Spurs | +3.15 | -3.15 | -0.68 | -0.04 | **-0.72** | -3.87 |
| Stephon Castle | Spurs | +1.34 | -1.34 | +0.21 | -0.09 | **+0.12** | -1.22 |
| Harrison Barnes | Spurs | +0.46 | -0.46 | -0.04 | +0.03 | **-0.01** | -0.47 |

This is the central interpretive example. Curry's HPM player term is modest in
the 2025-26 completed fit, but the frozen context model assigns a large positive
composition contribution to his profile in this particular unit. It does not
mean Curry alone created all of that environment: the Shapley calculation shares
overlapping nonlinear profile effects across Curry, Green, Butler, and the
other players in proportion to their average marginal effect.

### Rockets Unit Vs Clippers Unit

For Kevin Durant, Alperen Sengun, Jabari Smith Jr., Tari Eason, and Amen
Thompson against Brook Lopez, Kawhi Leonard, Kris Dunn, Derrick Jones Jr., and
John Collins, the frozen portable edge is (-1.21) and the matchup residual is
(+0.61). Eason receives the largest positive Rockets context contribution
((+1.68)), followed by Sengun ((+1.18)); Durant's profile contribution is
(+0.23). This is a useful reminder that an excellent player rating and a
contextual profile contribution answer different questions.

### Knicks Unit Vs Hawks Unit

For Karl-Anthony Towns, OG Anunoby, Josh Hart, Mikal Bridges, and Jalen Brunson
against CJ McCollum, Nickeil Alexander-Walker, Onyeka Okongwu, Jalen Johnson,
and Dyson Daniels, the frozen total context edge is only (+0.27). Hart
((+2.01)), Towns ((+1.75)), and Brunson ((+1.45)) are the largest positive
Knicks context contributors, while Bridges is negative ((-1.19)). The Hawks'
largest offsetting context terms are Dyson Daniels ((-1.56)) and Okongwu
((-1.31)).

## Interpretation Boundaries

- This is a profile-based counterfactual inside HPM's frozen context function.
  It does not observe a player being literally replaced by an average player.
- The synthetic reference unit is only an accounting zero point. It does not
  replace HPM's empirical, possession-weighted reference field used to define
  portable composition ratings.
- A positive opponent contribution helps the focal unit's edge; a negative one
  helps the opponent. This sign convention lets all ten rows sum directly.
- The values are lineup- and opponent-dependent. A player can have a different
  context contribution in another five-man unit even with the same HPM player
  rating.
- Low exact-matchup exposure means these selected examples are pedagogical
  decompositions, not estimates of how those specific units performed on court.

## Artifact

Immutable artifact:

```text
artifacts/models/portable_context_attribution_audit/2025-26/
```

`case_summary.parquet` contains the ledger totals and realized matchup exposure.
`player_context_attribution.parquet` contains all ten player rows for every
case. Regenerate it with the [audit guide](../guides/build-portable-context-attribution-audit.md).
