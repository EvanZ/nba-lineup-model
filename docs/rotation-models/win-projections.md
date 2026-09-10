---
last_updated: "2026-09-10"
---

# Win Projections

**W0 minute-weighted NAIL** turns the preseason player ratings and playing-time
forecasts into schedule-aware expected win totals. It is a downstream planning
model: it does not refit NAIL-RAPM, availability, or conditional minutes.

## Inputs

Each player starts with three frozen preseason inputs:

\[
R_i,\qquad
p_i=P(\mathrm{available}_i),\qquad
m_i=E[\mathrm{MPG}_i\mid\mathrm{available}_i].
\]

\(R_i\) is the player's projected NAIL rating. \(p_i\) is the probability
that the player is medically available for a typical regular-season game,
expressed in the interface as \(G_{\mathrm{available}}=82p_i\). \(m_i\) is
the projected NBA minutes in a game conditional on that medical availability;
it is not the player's all-schedule average MPG. Their raw expected season
minutes are therefore:

\[
r_i=82p_i m_i.
\]

Forward Availability produces \(p_i\), and Forward Conditional Minutes
produces \(m_i\). The roster-squashing step below reconciles the independent
raw totals \(r_i\) into a 240-regulation-minute team rotation. See
[Forward Availability](forward-availability.md) and
[Forward Conditional Minutes](forward-conditional-minutes.md) for the fitted
player-level models.

For team \(k\), its neutral-court strength is the minutes-weighted player sum:

\[
S_k = \sum_{i\in k}
\frac{\widehat{\mathrm{MPG}}_i}{48} R_i.
\]

Because each team receives 240 projected minutes, the player weights sum to
five. The 30 team strengths are then league-centered, so a positive value is a
projected net-rating advantage over the preseason league average.

## Roster Squashing

The availability and conditional-minutes models create independent player
forecasts. Those raw forecasts cannot be used directly for team strength: they
will not generally sum to a team's fixed 19,680 regulation minutes. W0 first
reconciles every opening roster to one feasible preseason rotation.

### 1. Form A Raw Rotation Weight

For every player on the roster, W0 uses:

\[
r_{i,t}=82\,P(\mathrm{available}_{i,t})
\widehat{\mathrm{MPG}}^{\mathrm{avail}}_{i,t}.
\]

This is a rotation *weight*, not the displayed team-minute allocation. A low
availability forecast reduces the weight even when the player projects for a
large role in the games they can play. The raw totals are intentionally
unconstrained because they do not yet model teammates absorbing minutes when a
player is unavailable.

### 2. Select The Active 15

Let \(R_t\) be the 15 players with the largest non-negative raw totals on a
team's opening roster. Players outside \(R_t\) receive zero baseline rotation
minutes. This hard active-15 constraint prevents small residual forecasts from
allocating minutes across every two-way and deep-bench player on a preseason
roster.

The selection is rerun after an availability or conditional-MPG override. An
outside-rotation player enters only when their adjusted raw total reaches the
top 15, at which point the player displaced at the cutoff receives zero
baseline rotation minutes.

### 3. Normalize To The Team-Minute Budget

Within the selected group, W0 preserves the raw relative weights and scales
them to the 82-game regulation-minute budget:

\[
w_{i,t}=
\frac{r_{i,t}}{\sum_{j\in R_t}r_{j,t}},
\qquad
\widehat M_{i,t}=19{,}680\,w_{i,t},
\qquad
\widehat{\mathrm{MPG}}_{i,t}=\frac{\widehat M_{i,t}}{82}.
\]

This is equivalent to \(\operatorname{softmax}(\log r)\) over positive
eligible weights, but it has no fitted temperature or additional model
parameter. It is a deterministic conservation step: selected players always
sum to 19,680 total regulation minutes, or 240 MPG. A player with zero
availability or conditional MPG has zero raw weight; the other selected
players absorb the released minutes proportionally.

For example, a player with 1,840 raw minutes in a selected group totaling
11,500 raw minutes has a \(1{,}840/11{,}500=16\%\) share. W0 converts that to
\(0.16\times19{,}680=3{,}148.8\) season minutes, or 38.4 Squashed MPG. This
does not say the player is projected to play 38.4 minutes in every available
game. It says they receive 16% of the team's fixed preseason rotation after
all availability-adjusted player weights are reconciled.

The Win Projections controls expose \(G_{\mathrm{available}}\) and
\(E[\mathrm{MPG}\mid\mathrm{available}]\), the two inputs to \(r_{i,t}\).
The \(+/-\) control is applied only after squashing, when W0 constructs the
minutes-weighted team strength; changing a rating does not change a player's
minutes share.

## Game Win Probability

For scheduled game \(g\), the home team's game edge is:

\[
d_g = S_{\mathrm{home}(g)} - S_{\mathrm{away}(g)}
+ h + b\left(
\mathrm{B2B}_{\mathrm{home}(g)} - \mathrm{B2B}_{\mathrm{away}(g)}
\right).
\]

\(h\) and \(b\) are the home-court and back-to-back controls from the
published NAIL release. The current values are \(h=2.84\) net-rating points
and \(b=-1.51\) points for a home back-to-back relative to an away
back-to-back.

The edge becomes a home-win probability through a one-parameter logistic
calibration with no additional intercept:

\[
P(\mathrm{home\ win}\mid g) = \sigma(\beta d_g)
= \frac{1}{1+e^{-\beta d_g}}.
\]

The calibration fits \(\beta\) on completed 2024-25 game outcomes, tests it
once on frozen 2025-26 outcomes, then refits the production value using those
two completed preseason snapshots. The frozen 2025-26 calibration produced a
Brier score of 0.2269, log loss of 0.6447, and 61.0% winner accuracy. The
current production scale is \(\beta=0.1053\).

## Season Win Total

The expected wins from every published scheduled game are summed directly. The
current schedule contains 1,200 known regular-season games, or 80 per team.
The two NBA Cup-dependent regular-season games are still unresolved, so W0
adds one home and one away game against a neutral-strength opponent:

\[
U_k = 2\,\frac{
\sigma\!\left(\beta(S_k+h)\right)
+ \sigma\!\left(\beta(S_k-h)\right)}{2}.
\]

The final point forecast is:

\[
\widehat W_k =
\sum_{g\in\mathrm{scheduled}(k)} P(\mathrm{win}\mid g) + U_k.
\]

The neutral-opponent placeholder expectations are league-centered after this
calculation so their combined value is exactly 30 wins, one per team. Therefore
the table always sums to 1,230 projected wins: 1,200 from published games plus
30 from the unresolved NBA Cup games.

Once the NBA publishes the two remaining matchups, the placeholders can be
replaced with their actual opponents, dates, home/away assignments, and
back-to-back status without changing the model contract.

## Win-Loss Envelope

The Win-Loss Envelope uses the same per-game probabilities, but simulates the
82-game schedule 10,000 times in the browser. Each trial draws an independent
Bernoulli win/loss outcome for every game and accumulates wins by game number.
The displayed bands are the 1st--99th and 5th--95th percentile paths across
those trials; the median is the 50th percentile path. It is an uncertainty
view of the same W0 forecast, not a separate model.

At release time, W0 also runs a seeded 10,000-trial final-total simulation for
all 30 baseline schedules and stores each team's P1, P5, P50, P95, and P99 in
the release bundle. The Win Projections table shows P5--P95 and highlights a
BetMGM total below P5 in green or above P95 in red. Clicking `Calculate
projection` reruns those 30 interval calculations locally for the user's
override scenario; it never changes the published baseline cache.

Changing a player's availability, conditional MPG, or \(+/-\) override causes
the page to re-squash that team's minutes, recompute team strengths and every
affected game probability, then rerun the local simulation.
