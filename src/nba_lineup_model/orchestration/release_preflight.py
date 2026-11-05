"""Exercise the sealed application against proposed data without moving live state."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


def fetch(api, path):
    with urllib.request.urlopen(api + path, timeout=90) as response:
        return json.load(response)


@contextmanager
def candidate_api(release: Path, report_root: Path, *, observation_root: Path, quality_root: Path):
    bindings = json.loads((release / "serving-bindings.json").read_text())
    report_root.mkdir(parents=True, exist_ok=True)
    pointer = report_root / "preflight-release"
    if not pointer.is_symlink():
        pointer.symlink_to(release, target_is_directory=True)
    if pointer.resolve() != release.resolve():
        raise ValueError("Preflight link points at another release")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(release / "code/src"),
           "GESTALT_ACTIVE_RELEASE_ROOT": str(pointer),
           "GESTALT_LIVE_SCORE_STATE_DIR": str(observation_root),
           "GESTALT_DATA_QUALITY_ROOT": str(quality_root)}
    for variable, relative in {
        "CURRENT_SEASON_SERVING_ROOT": "current-season-serving",
        "CURRENT_SEASON_LINEUPS_ROOT": "current-season-lineups",
        "SCHEDULE_STATE_DIR": "schedule", "GAME_CATALOG_PATH": "catalog/games.parquet",
        "SITE_RELEASE_ROOT": "site-releases", "PRESEASON_RANKINGS_PATH": "preseason/rankings.parquet",
        "PRESEASON_PROFILES_PATH": "preseason/profiles.parquet",
    }.items():
        env["GESTALT_" + variable] = str(release / relative)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    api = f"http://127.0.0.1:{port}"
    with (report_root / "api.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn",
            "nba_lineup_model.web_api.app:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=bindings["historical_runtime"], env=env, stdout=log, stderr=subprocess.STDOUT,
            user="ubuntu" if os.geteuid() == 0 else None,
            group="ubuntu" if os.geteuid() == 0 else None)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Candidate API exited; see {report_root / 'api.log'}")
                try:
                    fetch(api, "/api/health")
                    break
                except OSError:
                    time.sleep(.5)
            else:
                raise TimeoutError("Candidate API did not start")
            yield api
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
