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

The default release is the currently published NAIL-RAPM v1.0 artifact. Override the
model run with `GESTALT_RUN_ID` only after materializing its web rankings and response
curve caches. Build the compact historical exposure cache before publishing so mixed-era
Lab queries do not require the full stint archive on EC2:

```bash
uv run nba-build-gestalt-exposure-cohort
```

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
