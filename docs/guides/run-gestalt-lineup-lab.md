---
last_updated: "2026-08-22"
---

# Run NBA GESTALT Locally

NBA GESTALT is a local interactive Matchup Lab built around the completed
2025-26 NAIL-RAPM v1.2 state. This release carries a returner's last observed
profile through an injury or other non-playing gap instead of replacing it
with a generic cold-start profile.
The browser app and API are separate processes during development.

## Install dependencies

From the repository root, install the Python environment:

```bash
uv sync --group dev
```

Install the frontend dependencies once:

```bash
cd apps/lineup-explorer
npm install
```

## Start the API

In a terminal at the repository root:

```bash
uv run nba-gestalt-api --port 8001
```

The API listens at `http://127.0.0.1:8001`. It loads the latest published
`forward_centered_value_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm/2025-26`
artifact on its first request. It retains that completed portable-model state
in memory for the lifetime of the process. It does not read raw possession
data or retrain a model for each browser session.

## Materialize observed lineup rankings

The **Lineups** page reads precomputed, retrospective regular-season five-man
tables. Build the current season with:

```bash
uv run nba-build-gestalt-lineup-rankings --season 2025-26
```

Build the historical archive, from 1997-98 through 2025-26, with:

```bash
uv run nba-build-gestalt-lineup-rankings --all-seasons
```

Each table uses that season's completed-fit player ratings and contextual
model, while player and context edges are weighted by the opponents a unit
actually faced. The initial 1996-97 fit is excluded from the Lineups page
because it initializes the forward context state and has no completed
contextual model. Player Rankings nevertheless include 1996-97 as the
initial, zero-prior RAPM fit; its additive-profile adjustment is zero.

## Start the Lineup Lab

In a second terminal:

```bash
cd apps/lineup-explorer
npm run dev
```

Open [http://127.0.0.1:5174](http://127.0.0.1:5174). Vite proxies `/api`
requests to the local API on port 8001, while the Zensical documentation site
continues to use port 8000.

Both lineup columns start empty. Each has a dice control that fills that
side's empty slots with a random sample from the upper possession quartile,
preserving already selected players and excluding the other side's selected
players. The dice control disables once the side is full. Its trash control
clears that side only.

## Historical and mixed-era matchups

Each lineup column has an independent **Season** selector. A selected unit is
always drawn from exactly one completed-fit season, but the two units may come
from different seasons. The **Evaluation era** control chooses which
season's contextual functions score their profiles:

- **Your unit** applies your unit's season-specific composition and matchup
  functions.
- **Opponent unit** applies the opponent's season-specific functions.
- **Neutral** averages those two directional estimates. It is symmetric, but
  is not yet a separately trained era-neutral model.

Player ratings remain tied to each unit's own completed season. The Lab uses
the selected season's realized, statistically padded player profiles for its
context calculation, while the environment changes only the contextual response
surface. This makes a historical matchup internally retrospective rather than
combining completed player ratings with prior-season profile inputs.

Materialize the compact cache of realized profiles for every completed season:

```bash
uv run nba-build-gestalt-realized-profiles
```

This cache is display-only. The frozen prior-season profiles used by leaderboard
and preseason-prediction workflows remain a separate artifact contract.

Warm all completed historical response surfaces once with:

```bash
uv run nba-build-gestalt-response-cache --all-seasons
```

## Materialize player-team splits

Player profile histories show the regular-season team split for players who
appeared for more than one club. The primary label remains the team with the
most reconstructed on-court possessions, but the table lists every team with
its on-court possessions and games. Build the cached profile metadata with:

```bash
uv run nba-build-gestalt-player-team-splits
```

This is a display artifact only: a player's NAIL-RAPM remains one completed
season-wide estimate across all of that player's team stints.

## Current model contract

The initial Lineup Lab is intentionally defined as a retrospective 2025-26
explorer. A neutral-court matchup estimate is:

\[
\widehat{NR}_{A,B} =
\sum_{i \in A}\hat{r}_i -
\sum_{j \in B}\hat{r}_j +
h_{2025\text{-}26}(A)-h_{2025\text{-}26}(B)+
q_{2025\text{-}26}(A,B).
\]

- `\hat{r}_i` is the completed 2025-26 bounded portable-model player
  coefficient for player `i`.
- `h(A)-h(B)` is the portable composition advantage against the completed
  2025-26 possession-weighted reference unit field.
- `q(A,B)` is the opponent-specific matchup adjustment after portable
  composition has been removed.
- No home-court term is applied in the Lineup Lab.

This is not a 2026-27 forecast and is not an observed lineup net rating. It is
a portable, retrospective model estimate for a user-selected matchup. The
result panel displays additive player value, portable composition, and the
specific matchup adjustment separately. Its context signals show all material
feature contributions to each term: lineup composition factors sum to
`h(A)-h(B)`, while matchup factors sum to `q(A,B)`. They are exact
orientation-symmetrized spline-Ridge contributions over the original context
features, with each `h` contribution averaged over the frozen reference-unit
field. Composition sparklines show the portable per-feature response against
the reference field, with orange for the selected unit and a dark outlined
marker for the opponent. Matchup sparklines hold the opponent's feature value
fixed and vary the selected unit. The shaded band is the fitted
possession-weighted 5th-to-95th side-feature support.

## API contract

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Loaded artifact identity and player count. |
| `GET /api/players?q=<query>&season=<season>` | Accent-insensitive player search over one completed-fit season pool. |
| `GET /api/players/{player_id}` | One display profile. |
| `GET /api/rankings` | Completed-fit player rankings. |
| `GET /api/lineups` | Materialized observed five-man rankings; accepts `season`, `minimum_possessions`, and repeated `player_id` parameters. |
| `GET /api/default-opponent?season=<season>` | Random five-player sample from that season's upper possession quartile. |
| `POST /api/matchups` | Score two distinct five-player units. |

`POST /api/matchups` accepts:

```json
{
  "unit_player_ids": [203999, 201939, 202691, 203110, 202710],
  "opponent_player_ids": [2544, 1629029, 1630162, 203507, 203954],
  "unit_season": "2017-18",
  "opponent_season": "2025-26",
  "environment": "neutral"
}
```

The response includes the additive margin, portable composition margin,
specific matchup adjustment, total contextual adjustment, final net-rating
estimate, player profiles, and separate feature-level contributions for the
composition edge and matchup bonus. Each contribution list reconstructs its
corresponding displayed context term.

## Verify changes

Run the backend contract tests and build the frontend before committing:

```bash
uv run ruff check src/nba_lineup_model/web_api tests/test_web_api.py
uv run pytest tests/test_web_api.py
cd apps/lineup-explorer && npm run build
```
