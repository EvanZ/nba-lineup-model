---
last_updated: "2026-09-08"
---

# L20-MSP v0.1

**Gap-Returner Preseason Minute-Share Persistence** extends the static L20
baseline for players who are on the transaction-derived opening roster but had
zero NBA minutes in the immediately prior regular season.

## Contract

For an incumbent, the score remains total prior-season minutes. For an opening
candidate with zero prior-season minutes but an earlier NBA season with minutes,
the model substitutes the player's most recent active-season minutes, scaled by
\(\gamma\):

\[
S_i =
\begin{cases}
M_{i,s-1}, & M_{i,s-1}>0 \\
\gamma M_{i,a(i)}, & M_{i,s-1}=0,\ M_{i,a(i)}>0 \\
0, & \text{otherwise.}
\end{cases}
\]

Here \(a(i)\) is the latest completed season before \(s-1\) where the player
logged regular-season minutes. The scores are roster-normalized and blended
with the same uniform cold-start mass \(\epsilon\) as L20-MSP v0.0.

This is deliberately not an injury classifier. It is a strictly pregame proxy
for a player returning from any full-season NBA absence, including injury,
international play, or time out of the league.

## Evaluation

Tune \(\epsilon\) and \(\gamma\) jointly on 2024-25 by mean team allocation
total variation, then freeze both values for 2025-26. The target remains each
team's cumulative minute allocation over its first 20 games. A lookback of up
to three prior seasons supports longer absences without using target-season
data.

```bash
uv run nba-evaluate-l20-gap-returner-minute-share \
  --tune-season 2024-25 \
  --holdout-season 2025-26 \
  --team-games 20
```

The result will determine whether this recovery prior supersedes the uniform
cold-start treatment for returning players.

## Initial Result

The 2024-25 source grid selected \(\epsilon=0.32\) and \(\gamma=0\). Thus,
within this one-season tuning slice, carrying forward earlier minutes for a
zero-minute returner did not improve mean team allocation total variation. The
frozen 2025-26 result is consequently identical to L20-MSP v0.0: mean total
variation \(0.252\) and Brier score \(0.024\).

This is a preliminary non-promotion. Only six qualifying gap returners were in
the source opening rosters, and players absent for a full season are a mixed
population: injury returners, players who left the NBA, and players returning
with substantially different roles. The next defensible test, if pursued, is a
rolling multi-season selection focused on the gap-returner subset rather than
changing the baseline from this sparse result.
