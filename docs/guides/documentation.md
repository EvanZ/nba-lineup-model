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
