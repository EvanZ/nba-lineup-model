# Use the Prefect Web UI

Prefect's local web UI displays flow runs, individual tasks, timing, retries,
logs, and failures.

## Start the server

From the repository root, start the persistent local Prefect server:

```bash
uv run prefect server start
```

Leave that terminal running and open:

[http://127.0.0.1:4200](http://127.0.0.1:4200)

Stop the server with `Ctrl+C`.

Prefect stores local history in `~/.prefect/prefect.db` by default. Runs created
by Prefect's automatic temporary server use the same database unless its home or
database settings were overridden, so earlier local runs remain visible.

## Connect a flow run

With the server running, launch a flow from another terminal:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
  uv run nba-fetch-season 2025-26 --max-workers 4
```

The flow and its game tasks appear in the UI while they execute.

The same connection applies to processing and compaction:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
  uv run nba-process-season 2025-26 --max-workers 4

PREFECT_API_URL=http://127.0.0.1:4200/api \
  uv run nba-compact-season 2025-26 --max-workers 4
```

Season compaction appears as one flow with one task for each table and
season-type partition.

## Save the connection

Store the local API URL in the active Prefect profile:

```bash
uv run prefect config set \
  PREFECT_API_URL=http://127.0.0.1:4200/api
```

After saving this setting, the local server must be running whenever a flow is
launched. Inspect the active configuration with:

```bash
uv run prefect config view --show-sources
```

Restore automatic temporary-server behavior with:

```bash
uv run prefect config unset PREFECT_API_URL
```

See Prefect's official guides for
[running a local server](https://docs.prefect.io/v3/how-to-guides/self-hosted/server-cli)
and [managing settings](https://docs.prefect.io/v3/concepts/settings-and-profiles).
