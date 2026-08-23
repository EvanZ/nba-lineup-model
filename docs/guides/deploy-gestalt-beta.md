# Deploy NBA GESTALT Beta

NBA GESTALT beta is served from `gestalt.toplines.app` on the existing EC2 instance.
Nginx serves the static Vite frontend and proxies `/api/` to a loopback-only FastAPI
service on port `8010`.

## Storage

`s3://nba-gestalt-data` is the canonical artifact store.

- `nba-gestalt/releases/<release-id>/` contains the compact API runtime bundle.
- `nba-gestalt/archive/` is reserved for raw and historical parquet data.

The API never reads S3 on a request path. A release is synced to local EC2 disk before
the service is restarted.

## Publish a runtime release

From the repository root:

```bash
./scripts/publish_gestalt_release.sh
```

The default release is the promoted NAIL-RAPM v1.2.1 artifact. Every numerical
cache is namespaced by the immutable model run ID. Materialize all of them before a
release so rankings, biographies, historical Lab queries, observed lineups, profile
decompositions, and preseason forecasts all use the same fitted model:

```bash
uv run nba-build-gestalt-exposure-cohort
uv run nba-build-gestalt-lineup-rankings --all-seasons
uv run nba-build-gestalt-player-team-splits
uv run nba-build-gestalt-response-cache --all-seasons
uv run nba-build-gestalt-historical-profiles
uv run nba-build-gestalt-realized-profiles
uv run nba-build-gestalt-preseason-rankings
uv run nba-validate-gestalt-release
```

The validator checks model identity, context regularization, the exact profile-padding
coefficients, all historical season coverage, lineup-score arithmetic, and consistency
between the trained target profiles and their web cache. It then writes a SHA-256 and
row-count inventory to
`artifacts/web/releases/<model-artifact>/<run-id>/bundle_manifest.json`. The publish
script runs this validation again and refuses to upload an incomplete or mixed bundle.
Override the run with `GESTALT_RUN_ID` only after that run's manifest succeeds.

## Deploy on EC2

The server keeps the source checkout at `~/nba-lineup-model` and releases under
`/opt/nba-gestalt/releases/`. `current` is a symlink to the active release.

```bash
ssh gestalt-web-server
cd ~/nba-lineup-model
git pull --ff-only
sudo cp deploy/gestalt-web.service /etc/systemd/system/gestalt-web.service
sudo cp deploy/gestalt.nginx.conf /etc/nginx/sites-available/gestalt
sudo cp deploy/gestalt-rate-limits.conf /etc/nginx/conf.d/gestalt-rate-limits.conf
sudo ln -sfn /etc/nginx/sites-available/gestalt /etc/nginx/sites-enabled/gestalt
sudo systemctl daemon-reload
sudo nginx -t
sudo systemctl reload nginx
```

For a published release:

```bash
GESTALT_RELEASE_ID=<release-id> ./scripts/deploy_gestalt_beta.sh
```

After the Route 53 record for `gestalt.toplines.app` resolves to the EC2 instance,
obtain the TLS certificate:

```bash
sudo certbot --nginx -d gestalt.toplines.app
```

## API throttling

Nginx applies per-IP limits before proxying requests to FastAPI:

- General `/api/` routes: 120 requests per minute, with a burst of 30.
- `POST /api/matchups`: 30 requests per minute, with a burst of 10.
- Concurrent API connections: 20 per IP generally and 8 on matchup evaluation.

Excess requests receive `429 Too Many Requests`. These limits are deployed by
`scripts/deploy_gestalt_beta.sh` and live in `deploy/gestalt-rate-limits.conf`.
The release script intentionally does not overwrite the Nginx virtual-host file,
because Certbot manages its TLS directives after the initial setup.

## Verify

```bash
curl -fsS http://127.0.0.1:8010/api/health
systemctl status gestalt-web
journalctl -u gestalt-web -f
```
