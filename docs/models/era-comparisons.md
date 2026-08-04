# Era-Adjusted Player Comparisons

## Objective

Raw RAPM values are not directly comparable across seasons. In particular,
prior-centered RAPM inherits an arbitrary league-wide location and scale from
its prior. This report converts completed player-seasons to a common relative
scale and gives that scale a team-win interpretation.

It compares **player-seasons**, not careers. A player's best qualified season
is a peak estimate; career value requires a separate longevity and availability
model.

## Standardization

For a player-season RAPM estimate \(r_{i,t}\), define \(m_{i,t}\) as the
player's reconstructed lineup-stint seconds. The season reference moments are
exposure weighted:

\[
\mu_t = \frac{\sum_i m_{i,t}r_{i,t}}{\sum_i m_{i,t}},
\qquad
\sigma_t = \sqrt{
\frac{\sum_i m_{i,t}(r_{i,t}-\mu_t)^2}{\sum_i m_{i,t}}}.
\]

The era-adjusted player rating is

\[
z_{i,t} = \frac{r_{i,t}-\mu_t}{\sigma_t}.
\]

A value of +1 means one exposure-weighted player standard deviation above that
season's league average. This is the primary cross-era comparison quantity;
raw RAPM remains more appropriate for lineup prediction inside its own season.

## Win Conversion

The forward RAPM calibration estimates \(\beta=0.3254\) win-percentage points
per standardized team-strength unit. For a fixed \(M\)-minute player role,
the incremental regular-season wins estimate is

\[
\mathrm{WinsAboveAverage}_{i,t}(M) =
\frac{\beta z_{i,t} M}{5 \times 48}.
\]

The published rate uses \(M=2{,}000\) minutes. A +1.0 standardized player is
therefore worth about 2.71 wins above an average player in a 2,000-minute role,
conditional on the rest of the roster and allocation remaining fixed.

`wins_above_average_actual_minutes` uses a player's observed minutes instead.
It is useful descriptively and sums to the team's un-clipped calibrated win
difference from the average-team baseline, but it incorporates availability,
coaching, transactions, and role selection.

## Qualification And Limits

The initial peak-season table requires 2,000 reconstructed regular-season
minutes. The threshold reduces small-sample ranking artifacts; it is not a
claim that players below it lack value.

This report does not make player effects context free. It does not account for
opponent quality, playoff translation, injuries, rule changes beyond their
observed effect on the historical calibration, or an ex ante minute projection.
It should be read as a common-unit summary of the existing RAPM evidence, not
as a final all-time-player ranking.

## Initial Peak-Season Results

Run `era-comparison-2025-26-20260804T184941Z-8af20a00` covers 14,560
player-seasons from 1996-97 through 2025-26. Of those, 2,294 clear the
2,000-minute qualification threshold. The table ranks the fixed-role rate,
not actual-minute totals. It uses the forward-prior RAPM specification.

| Rank | Season | Player | Team | Minutes | Era RAPM (z) | Wins above average / 2,000 min |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | 2006-07 | Tim Duncan | SAS | 2,649 | 4.906 | 13.30 |
| 2 | 2004-05 | Tim Duncan | SAS | 2,098 | 4.817 | 13.06 |
| 3 | 2007-08 | Tim Duncan | SAS | 2,617 | 4.671 | 12.67 |
| 4 | 2005-06 | Tim Duncan | SAS | 2,601 | 4.530 | 12.29 |
| 5 | 2025-26 | Nikola Jokic | DEN | 2,265 | 4.325 | 11.73 |
| 6 | 2024-25 | Nikola Jokic | DEN | 2,157 | 4.291 | 11.64 |
| 7 | 2016-17 | LeBron James | CLE | 2,566 | 4.084 | 11.07 |
| 8 | 2003-04 | Kevin Garnett | MIN | 3,031 | 4.062 | 11.02 |
| 9 | 2003-04 | Tim Duncan | SAS | 2,365 | 4.008 | 10.87 |
| 10 | 2023-24 | Nikola Jokic | DEN | 2,277 | 3.972 | 10.77 |
| 11 | 2015-16 | LeBron James | CLE | 2,496 | 3.893 | 10.56 |
| 12 | 2008-09 | Tim Duncan | SAS | 2,275 | 3.824 | 10.37 |
| 13 | 2020-21 | Chris Paul | PHX | 2,057 | 3.781 | 10.25 |
| 14 | 2009-10 | LeBron James | CLE | 2,848 | 3.737 | 10.13 |
| 15 | 2019-20 | Chris Paul | OKC | 2,047 | 3.722 | 10.09 |
| 16 | 2006-07 | Dirk Nowitzki | DAL | 2,742 | 3.650 | 9.90 |
| 17 | 2014-15 | LeBron James | CLE | 2,399 | 3.639 | 9.87 |
| 18 | 2010-11 | Tim Duncan | SAS | 2,063 | 3.622 | 9.82 |
| 19 | 2023-24 | Stephen Curry | GSW | 2,065 | 3.598 | 9.76 |
| 20 | 2019-20 | LeBron James | LAL | 2,140 | 3.596 | 9.75 |
| 21 | 2002-03 | Tim Duncan | SAS | 2,701 | 3.593 | 9.74 |
| 22 | 2007-08 | Kevin Garnett | BOS | 2,272 | 3.590 | 9.74 |
| 23 | 2009-10 | Tim Duncan | SAS | 2,260 | 3.589 | 9.73 |
| 24 | 2016-17 | Stephen Curry | GSW | 2,491 | 3.530 | 9.57 |
| 25 | 2002-03 | Kevin Garnett | MIN | 3,043 | 3.510 | 9.52 |

This is a method smoke test, not a promoted all-time ranking. In particular,
the recurring prior RAPM state can reward sustained past estimates.

## Canonical One-Season Cross-Check

The same report standardizes the existing zero-centered, one-season canonical
RAPM panel using identical exposure weights and qualification. The 2,294
qualified player-seasons common to both specifications have a 0.761 correlation
in standardized RAPM. Only seven player-seasons overlap between their top-25
tables, so the extreme tail remains specification-sensitive.

| Rank | Season | Player | Team | Canonical RAPM (z) | Common-unit wins / 2,000 min |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | 2004-05 | Tim Duncan | SAS | 4.195 | 11.38 |
| 2 | 2024-25 | Shai Gilgeous-Alexander | OKC | 3.692 | 10.01 |
| 3 | 2015-16 | Draymond Green | GSW | 3.679 | 9.98 |
| 4 | 2016-17 | Stephen Curry | GSW | 3.657 | 9.92 |
| 5 | 2002-03 | Kevin Garnett | MIN | 3.457 | 9.38 |
| 6 | 2002-03 | Dirk Nowitzki | DAL | 3.443 | 9.34 |
| 7 | 2010-11 | Dirk Nowitzki | DAL | 3.414 | 9.26 |
| 8 | 2008-09 | LeBron James | CLE | 3.310 | 8.98 |
| 9 | 2004-05 | Manu Ginobili | SAS | 3.273 | 8.88 |
| 10 | 2009-10 | LeBron James | CLE | 3.268 | 8.86 |

This directly answers the Duncan-Curry example. Duncan's canonical peak is
2004-05 at +4.195 standard deviations; Curry's is 2016-17 at +3.657. The
forward-prior model estimates those same seasons at +4.817 and +3.530,
respectively. Both models put Duncan ahead, but the forward prior materially
widens the gap. The conclusion that Duncan's peak was more exceptional in this
data is robust to this particular specification check; the *size* of that lead
is not.

The common-unit wins conversion was fit on the forward-prior team calibration.
It makes the canonical table readable on the same scale, but it is not a
separately validated canonical-RAPM win forecast. Further promotion still
requires minutes-threshold and calibration-slope sensitivity, plus uncertainty
intervals.
