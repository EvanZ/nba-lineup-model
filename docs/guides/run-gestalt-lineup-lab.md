---
last_updated: "2026-08-09"
---

# Run NBA GESTALT Locally

NBA GESTALT is a local interactive Lineup Lab built around the completed
2025-26 Forward Bounded Hierarchical Portable-Matchup Contextual RAPM state.
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
`forward_bounded_hierarchical_portable_matchup_contextual_rapm/2025-26`
artifact on its first request. It retains that completed portable-model state
in memory for the lifetime of the process. It does not read raw possession
data or retrain a model for each browser session.

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
| `GET /api/players?q=<query>` | Accent-insensitive player search over the current model universe. |
| `GET /api/players/{player_id}` | One display profile. |
| `GET /api/default-opponent` | Random five-player sample from the upper possession quartile. |
| `POST /api/matchups` | Score two distinct five-player units. |

`POST /api/matchups` accepts:

```json
{
  "unit_player_ids": [203999, 201939, 202691, 203110, 202710],
  "opponent_player_ids": [2544, 1629029, 1630162, 203507, 203954]
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
