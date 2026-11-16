# EC2 Dagster runtime configuration

These files reproduce the running NBA GESTALT Dagster configuration on the existing EC2 deployment. They preserve one permanent instance at `/var/lib/nba-gestalt/collection-dagster` and the single-worker queue. They do not provision a blank machine or restore databases, model artifacts, or private credentials.

| Service | Manager | Purpose |
| --- | --- | --- |
| `gestalt-collection-webserver.service` | System | Dagster UI, `127.0.0.1:3000` |
| `gestalt-collection-daemon.service` | System | Schedules, sensors and queued runs |
| `gestalt-pipelines-code-server.service` | Ubuntu user | Reloadable `nba_gestalt_pipelines` code location, `127.0.0.1:4003` |

The code service loads all new pipelines. All three services use `Restart=always`. The Ubuntu user manager has lingering enabled so the code service starts at boot without an SSH login. Reloading code remains available in the Dagster UI.

`retired-services.txt` lists the seven retired system services. Keep them stopped and disabled. The older `gestalt-dagster-code-server`, `gestalt-dagster-daemon` and `gestalt-dagster-webserver` services also remain deprecated and disabled. Do not run the deprecated orchestration installer.

## Prerequisites and secrets

- The checkout and its installed pipeline environment exist at `/home/ubuntu/nba-lineup-model` and `pipelines/gestalt/.venv`.
- The UI/daemon environment remains at `/opt/nba-gestalt/current/.venv`; its installed runtime files are still needed even though the old code locations are retired.
- Preserve the existing permanent Dagster home and its run history. Do not replace it with a temporary instance.
- Private settings remain in `/etc/nba-gestalt/dagster.env` and `/home/ubuntu/nba-lineup-model/pipelines/gestalt/code-location.env`. Their contents are deliberately excluded. Use the tracked `../code-location.env.example` as the starting template only on a new host, then supply private values securely.
- The service files contain deployment paths and non-secret environment settings. They reference private configuration rather than embedding credentials.

## Apply on EC2

Run as `ubuntu` from `/home/ubuntu/nba-lineup-model`. Back up existing unit files, `dagster.yaml` and `workspace.yaml` before replacing them. Keep application/website deployment separate from this runtime setup.

```bash
sudo install -m 0644 pipelines/gestalt/runtime/gestalt-collection-webserver.service /etc/systemd/system/gestalt-collection-webserver.service
sudo install -m 0644 pipelines/gestalt/runtime/gestalt-collection-daemon.service /etc/systemd/system/gestalt-collection-daemon.service
install -d -m 0755 /home/ubuntu/.config/systemd/user
install -m 0644 pipelines/gestalt/runtime/gestalt-pipelines-code-server.service /home/ubuntu/.config/systemd/user/gestalt-pipelines-code-server.service
sudo install -m 0644 pipelines/gestalt/runtime/dagster.yaml /var/lib/nba-gestalt/collection-dagster/dagster.yaml
sudo install -m 0644 pipelines/gestalt/runtime/workspace.yaml /var/lib/nba-gestalt/collection-dagster/workspace.yaml
sudo systemctl daemon-reload
systemctl --user daemon-reload
sudo loginctl enable-linger ubuntu
# When migrating from the previous name, first wait until no pipeline run is active.
# Skip this command on a fresh host where the old service is not installed.
# systemctl --user disable --now gestalt-collection-injury-review.service
systemctl --user enable --now gestalt-pipelines-code-server.service
```

Before retiring the old collector server, verify that the UI and scheduler no longer require it. The saved scheduler startup check points to port **4003**, not 4001. Inspect any existing unit drop-ins; older overrides must not reintroduce the retired dependency.

```bash
systemctl show gestalt-collection-webserver.service gestalt-collection-daemon.service -p Requires -p ExecStartPre
xargs -r sudo systemctl disable --now < pipelines/gestalt/runtime/retired-services.txt
sudo systemctl enable --now gestalt-collection-webserver.service gestalt-collection-daemon.service
curl --fail --silent --show-error http://127.0.0.1:3000/graphql -H 'Content-Type: application/json' --data '{"query":"mutation { reloadWorkspace { __typename } }"}'
```

The daemon refreshes the workspace automatically every 60 seconds. The only configured code location should be `nba_gestalt_pipelines`. On an already running host, enabling an active service does not restart it; reapplying this configuration does not restart active services. Migrating the old code-service name requires one brief code-server handoff; the UI and scheduler stay running.

## Verify

```bash
systemctl is-active gestalt-collection-webserver.service gestalt-collection-daemon.service
systemctl is-enabled gestalt-collection-webserver.service gestalt-collection-daemon.service
systemctl --user is-active gestalt-pipelines-code-server.service
systemctl --user is-enabled gestalt-pipelines-code-server.service
loginctl show-user ubuntu -p Linger
xargs -r systemctl is-enabled < pipelines/gestalt/runtime/retired-services.txt
xargs -r systemctl is-active < pipelines/gestalt/runtime/retired-services.txt
```

Expect the three retained services to be active/enabled, `Linger=yes`, and all seven retired services to be inactive/disabled. The last two commands return a nonzero status when the retired services are correctly disabled/inactive.

Check the Dagster UI, enabled schedules/sensors, and actual recent injury-PDF, Odds API and scoreboard captures. After a reboot, an interrupted run can remain marked `STARTED` and hold the single-worker queue. Compare its start time with the host boot time and cancel the confirmed interrupted run through Dagster. Service auto-restart does not itself reconcile abandoned run history. Historical/model backfills require their own approval; preserve genuine saved pregame projections.
