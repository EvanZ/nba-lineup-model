# Deploy the isolated NBA GESTALT development site

`gestalt-dev.toplines.app` is a review environment on the existing EC2 host. It
does not duplicate the lake and it cannot write to production schedule or model
state.

## Isolation contract

- source releases: `/opt/nba-gestalt-dev/releases/`
- active source: `/opt/nba-gestalt-dev/current`
- generated dev state: `/var/lib/nba-gestalt-dev/`
- static site: `/var/www/nba-gestalt-dev/`
- API: loopback port `8011`
- production artifacts/data: linked read-only from `/opt/nba-gestalt/current`
- production current-season and Schedule state: mounted read-only by systemd
- collectors and Dagster sensors: not installed or enabled for dev

Current-season serving and lineup materialization run on EC2 against the existing
updater states, CDN lake, and final-game archive. Only their compact derived
artifacts are written under the dev state root.

## Deploy a review branch

Push the branch, fetch it in the existing EC2 source checkout, then run:

```bash
ssh gestalt-web-server
cd ~/nba-lineup-model
git fetch origin
GESTALT_DEV_REF=origin/<branch> ./scripts/deploy_gestalt_dev.sh
```

The first deployment emits a generated HTTP Basic Auth username and password.
Store them in the password manager; later deployments retain the same file.

The script creates a detached Git worktree, reuses production's immutable model
bundle and Python environment, materializes current lineups into the dev state
root, builds the branch frontend, and restarts only `gestalt-web-dev`.

## DNS and TLS

Create an A record for `gestalt-dev.toplines.app` pointing at the existing EC2
public address. After it resolves:

```bash
sudo certbot --nginx -d gestalt-dev.toplines.app
```

## Verify

```bash
curl -u gestalt:<password> -fsS https://gestalt-dev.toplines.app/api/health
systemctl status gestalt-web-dev
journalctl -u gestalt-web-dev -f
```

Production promotion remains a separate release and deployment action.
