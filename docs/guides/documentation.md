---
last_updated: "2026-08-05"
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
| `src/nba_lineup_model/` | Python API source consumed by MkDocstrings |
| `site/` | Generated static site |

## Sortable rankings

Ranking tables are enhanced by `docs/javascripts/sortable-tables.js` at page
load. Any table with a rank-style column heading, or located under a heading
containing "ranking", receives sortable column headers automatically. Keep
the source table as ordinary Markdown; no per-page HTML or JavaScript is
required. In draft tables, an undrafted `Pick` value (`-`) remains after all
numeric draft picks in either sort direction.

Before committing documentation changes, run the strict build and inspect the
affected pages in the preview server.
