---
last_updated: "2026-08-07"
---

# Build and Serve the Documentation

The project documentation is a Zensical static site configured by
`zensical.toml`.

Run all commands from the repository root.

## Install dependencies

Install the development and documentation dependency groups:

```bash
uv sync --group dev --group docs
```

Activating `.venv` is optional because `uv run` selects the project environment.

## Start the preview server

```bash
uv run --group docs zensical serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Zensical watches the
documentation and Python source trees, rebuilds changed pages, and serves the
updated site. Stop the server with `Ctrl+C`.

Use a different address when port 8000 is occupied:

```bash
uv run --group docs zensical serve \
  --dev-addr 127.0.0.1:8001
```

Pass `--open` to open the preview in the default browser automatically.

## Build the static site

Before a build, update each page's displayed date from its most recent Git
commit. The deployment workflow performs the same step using full repository
history:

```bash
uv run python scripts/update_doc_dates.py
```

Build with strict validation:

```bash
uv run --group docs zensical build --strict
```

Strict mode fails on documentation warnings, including invalid navigation and
API-reference problems. Generated files are written to `site/`, which is
ignored by Git.

Force a clean build when checking cache-sensitive changes:

```bash
uv run --group docs zensical build --clean --strict
```

## Documentation layout

| Path | Role |
| --- | --- |
| `zensical.toml` | Site metadata, navigation, theme, extensions, and plugins |
| `docs/` | Markdown source pages |
| `docs/stylesheets/extra.css` | Project-specific visual styling |
| `docs/javascripts/last-updated.js` | Places each page's update date below its main heading |
| `overrides/main.html` | Shared page metadata element containing the Git-derived update date |
| `scripts/update_doc_dates.py` | Writes per-page `last_updated` metadata from Git history |

## Interactive Model Evolution

The [Model Evolution](../models/#model-evolution) section is a static D3
visualization. Its versioned data registry is
`docs/assets/data/model-tree.json`; update it after a model has both an
evaluation artifact and a documentation page. Each node must use metrics from
the same declared data-recovery and evaluation contract as its parent branch.
Run the focused registry check after editing it:

```bash
uv run --group dev pytest tests/test_model_tree_registry.py
```

The docs configuration loads D3 before `docs/javascripts/model-tree.js`.
Both local serving and the GitHub Pages build use the same static assets; no
separate visualization service is required.
| `src/nba_lineup_model/` | Python API source consumed by MkDocstrings |
| `site/` | Generated static site |

## Sortable rankings

Ranking tables and model leaderboards are enhanced by
`docs/javascripts/sortable-tables.js` at page load. Any table with a rank-style
column heading, located under a ranking/leaderboard heading, or with a `Model`
column plus a standard evaluation metric receives sortable column headers.
Metric cells may append a rank such as `1.198010 (1)` and still sort by the
metric value. Keep the source table as ordinary Markdown; no per-page HTML or
JavaScript is required. In draft tables, an undrafted `Pick` value (`-`)
remains after all numeric draft picks in either sort direction.

Before committing documentation changes, run the strict build and inspect the
affected pages in the preview server.
