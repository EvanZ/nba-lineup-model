"""One ordered, fail-closed nightly release pipeline per environment.

The pipeline prepares a fully validated immutable candidate. Activation is a
separate, explicit operation; a dev run cannot implicitly write production.
Frozen forecasts are copied byte-for-byte, never recomputed by nightly replay.
"""
from __future__ import annotations

import fcntl
import hashlib
import html
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.season.slate_time import PACIFIC
from nba_lineup_model.season.source_policy import require_clean_artifact_sources


@dataclass(frozen=True)
class DailyPipelineConfig:
    environment_root: Path
    pipeline_root: Path
    data_root: Path
    model_root: Path
    schedule_root: Path
    coverage_path: Path
    catalog_path: Path
    preseason_rankings_path: Path
    preseason_minutes_path: Path
    preseason_profiles_path: Path
    release_root: Path
    code_root: Path
    frontend_root: Path
    live_pointer: Path
    matchup_url: str
    season_start: date = date(2026, 10, 20)
    season: str = "2026-27"
    activate_after_validation: bool = False

    @classmethod
    def read(cls, path: Path | str) -> "DailyPipelineConfig":
        raw = json.loads(Path(path).read_text())
        for key in cls.__dataclass_fields__:
            if key.endswith(("_root", "_path", "_pointer")) and key in raw:
                raw[key] = Path(raw[key])
        if "season_start" in raw:
            raw["season_start"] = date.fromisoformat(raw["season_start"])
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        boundary = self.environment_root.resolve(strict=True)
        if boundary == Path("/") or boundary.name not in {"nba-gestalt", "nba-gestalt-dev"}:
            raise ValueError("Pipeline must name an explicit Gestalt environment root")
        for path in (self.pipeline_root, self.data_root, self.model_root, self.live_pointer):
            if not path.absolute().is_relative_to(boundary):
                raise ValueError(f"Pipeline output escapes its environment: {path}")
            if path.exists() and path != self.live_pointer and not path.resolve().is_relative_to(boundary):
                raise ValueError(f"Pipeline output symlink escapes its environment: {path}")
        if not self.schedule_root.resolve().is_relative_to(boundary):
            raise ValueError("Dev and production cannot share a mutable schedule")


def expected_slate_ids(config: DailyPipelineConfig, game_date: date) -> set[str]:
    frame = pd.read_parquet(config.catalog_path)
    selected = frame.loc[frame.season.astype(str).eq(config.season) & frame.season_type.eq("regular")]
    dates = pd.to_datetime(selected.game_time_utc, utc=True).dt.tz_convert(PACIFIC).dt.date
    return set(selected.loc[dates.eq(game_date), "game_id"].astype(str))


def require_complete_slate(config: DailyPipelineConfig, game_date: date, *, now: datetime | None = None) -> set[str]:
    """The official catalog, not a partial scoreboard, defines completeness."""
    now = now or datetime.now(UTC)
    if game_date > now.astimezone(PACIFIC).date():
        raise ValueError("Cannot publish a future slate")
    expected = expected_slate_ids(config, game_date)
    if not expected and game_date >= now.astimezone(PACIFIC).date():
        raise ValueError("An empty date can advance only after Pacific midnight")
    coverage = json.loads(config.coverage_path.read_text())
    if coverage.get("season") != config.season:
        raise ValueError("CDN coverage season mismatch")
    errors = []
    for game_id in sorted(expected):
        summary = config.schedule_root / "final" / f"{game_id}.summary.json"
        box = config.schedule_root / "final" / f"{game_id}.nba.json"
        frozen = config.schedule_root / "frozen" / f"{game_id}.json"
        if not all(path.is_file() for path in (summary, box, frozen)):
            errors.append(f"{game_id}: missing final/official box score/frozen forecast")
            continue
        record = json.loads(summary.read_text())
        if str(record.get("game_id")) != game_id or not record.get("final"):
            errors.append(f"{game_id}: invalid final summary")
        if not coverage.get("games", {}).get(game_id):
            errors.append(f"{game_id}: missing official play-by-play coverage")
    if errors:
        raise ValueError("Slate incomplete: " + "; ".join(errors))
    return expected


def _frozen_hashes(root: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((root / "frozen").glob("*.json"))}


def schedule_lock_path(root: Path) -> Path:
    """One stable inode per environment, outside every swappable release."""
    configured = os.environ.get("GESTALT_SCHEDULE_LOCK_PATH")
    if configured:
        return Path(configured)
    for parent in (root.absolute(), *root.absolute().parents):
        if parent.name in {"nba-gestalt", "nba-gestalt-dev"}:
            return parent / ".schedule-writer.lock"
    return root.absolute().parent / ".schedule-writer.lock"


@contextmanager
def schedule_lock(root: Path):
    lock = schedule_lock_path(root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _report(root: Path, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=str)
    temporary = root / ".report.json.tmp"
    temporary.write_text(text + "\n")
    temporary.replace(root / "report.json")
    page = '<!doctype html><meta charset="utf-8"><title>Daily pipeline report</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Daily pipeline — '+html.escape(str(payload["game_date"]))+'</h1><p>'+html.escape(str(payload["status"]))+'</p><pre>'+html.escape(text)+'</pre>'
    temporary = root / ".report.html.tmp"
    temporary.write_text(page)
    temporary.replace(root / "report.html")


class CanonicalPipelineSteps:
    """The only production backend; tests replace I/O at this boundary."""
    def __init__(self, config: DailyPipelineConfig, candidate: Path, revision: str):
        self.config, self.candidate, self.revision = config, candidate, revision
        self.env = {**os.environ, "PYTHONPATH": str(config.code_root / "src"),
            "GESTALT_CURRENT_SEASON_SERVING_ROOT": str(candidate / "serving"),
            "GESTALT_CURRENT_SEASON_LINEUPS_ROOT": str(candidate / "lineups"),
            "GESTALT_SCHEDULE_STATE_DIR": str(candidate / "schedule"),
            "GESTALT_GAME_CATALOG_PATH": str(config.catalog_path),
            "GESTALT_SITE_RELEASE_ROOT": str(candidate / "site-releases"),
            "GESTALT_PRESEASON_RANKINGS_PATH": str(config.preseason_rankings_path),
            "GESTALT_PRESEASON_PROFILES_PATH": str(config.preseason_profiles_path)}

    def command(self, module: str, *arguments: object) -> None:
        subprocess.run([sys.executable, "-m", module, *map(str, arguments)], cwd=self.config.release_root, env=self.env, check=True, timeout=1800)

    def data(self, target: date) -> None:
        from nba_lineup_model.season.daily_data_replay import replay_daily_data_chain
        config = self.config
        pointer = config.data_root / "latest.json"
        start = date.fromisoformat(json.loads(pointer.read_text())["game_date_pacific"]) + timedelta(days=1) if pointer.exists() else config.season_start
        if start <= target:
            replay_daily_data_chain(start_date=start, end_date=target, coverage_path=config.coverage_path, output_root=config.data_root, game_catalog_path=config.catalog_path, season=config.season)
        elif start != target + timedelta(days=1):
            raise ValueError("Daily data tip is ahead of the requested release")

    def model(self, target: date) -> None:
        from nba_lineup_model.season.daily_model_replay import DailyModelReplayConfig, replay_daily_model_chain
        config = self.config
        replay_daily_model_chain(start_date=config.season_start, end_date=target, preseason_rankings_path=config.preseason_rankings_path, preseason_minutes_path=config.preseason_minutes_path, data_root=config.data_root, output_root=config.model_root, config=DailyModelReplayConfig(season=config.season, matchup_url=config.matchup_url), code_revision=self.revision)

    def serving(self) -> None:
        c = self.config
        self.command("nba_lineup_model.modeling.current_season_serving_materialization", "--daily-model-root", c.model_root, "--daily-data-root", c.data_root, "--release-root", c.release_root, "--output-root", self.candidate / "serving")

    def lineups(self) -> None:
        self.command("nba_lineup_model.modeling.current_season_lineup_materialization", "--daily-data-root", self.config.data_root, "--output-root", self.candidate / "lineups")

    @contextmanager
    def api(self):
        with socket.socket() as bound:
            bound.bind(("127.0.0.1", 0))
            port = bound.getsockname()[1]
        root = f"http://127.0.0.1:{port}"
        import urllib.request
        with (self.candidate / "api.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "nba_lineup_model.web_api.app:app", "--host", "127.0.0.1", "--port", str(port)], cwd=self.config.release_root, env=self.env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Candidate API exited; inspect api.log")
                    try:
                        with urllib.request.urlopen(root + "/api/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(0.5)
                else:
                    raise RuntimeError("Candidate API startup timed out")
                yield root
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def validate_and_stage(self, target: date) -> Path:
        from nba_lineup_model.season.dev_site_release import collect_api_validation, publish_dev_site_release
        from nba_lineup_model.season.production_site_release import stage_production_site_release
        from nba_lineup_model.season.schedule_artifact_contract import validate_schedule_artifacts
        from nba_lineup_model.season.official_schedule_runtime import refresh_official_schedule_window
        c, out = self.config, self.candidate
        before = _frozen_hashes(out / "schedule")
        with self.api() as api:
            # Only forward boards use the updated ratings/minutes. No historical
            # replay, changed frozen pregame, or postgame rating substitution.
            refresh_official_schedule_window(start_date=(target + timedelta(days=1)).isoformat(), days_ahead=7, state_dir=out / "schedule", game_catalog_path=c.catalog_path, projection_url=api + "/api/win-projections", s3_uri=None)
            if any(_frozen_hashes(out / "schedule").get(key) != value for key, value in before.items()):
                raise ValueError("Nightly refresh changed a frozen forecast")
            validate_schedule_artifacts(out / "schedule")
            validation = collect_api_validation(api, evidence_date=target.isoformat())
            publish_dev_site_release(daily_data_root=c.data_root, daily_model_root=c.model_root, serving_root=out / "serving", lineups_root=out / "lineups", schedule_state_dir=out / "schedule", catalog_path=c.catalog_path, preseason_rankings_path=c.preseason_rankings_path, preseason_profiles_path=c.preseason_profiles_path, code_revision=self.revision, api_validation=validation, output_root=out / "site-releases")
        return stage_production_site_release(dev_site_release_root=out / "site-releases", serving_root=out / "serving", lineups_root=out / "lineups", schedule_snapshot_root=out / "schedule", schedule_overlay_path=out / "schedule", catalog_path=c.catalog_path, preseason_rankings_path=c.preseason_rankings_path, preseason_profiles_path=c.preseason_profiles_path, code_root=c.code_root, frontend_root=c.frontend_root, output_root=c.pipeline_root / "releases", live_pointer=c.live_pointer)


def prepare_daily_release(config: DailyPipelineConfig, target: date, *, steps_class=CanonicalPipelineSteps) -> dict:
    """No live pointer changes on success or failure; failed stages are visible."""
    config.validate()
    config.pipeline_root.mkdir(parents=True, exist_ok=True)
    report_root = config.pipeline_root / "reports" / target.isoformat()
    report = {"game_date": target.isoformat(), "status": "running", "stage": "preflight", "completed_stages": []}
    with (config.pipeline_root / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another daily pipeline is already running") from error
        try:
            require_complete_slate(config, target)
            revision = subprocess.check_output(["git", "-C", str(config.code_root), "rev-parse", "HEAD"], text=True).strip()
            # The source tree must match the revision recorded in the artifact.
            if subprocess.check_output(["git", "-C", str(config.code_root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
                raise ValueError("Commit the tested pipeline code before materialization")
            candidate = config.pipeline_root / "candidates" / f"{target}-{uuid4().hex}"
            candidate.mkdir(parents=True)
            with schedule_lock(config.schedule_root):
                require_clean_artifact_sources(config.schedule_root)
                from nba_lineup_model.season.dev_site_release import _tree_sha256
                report["source_schedule_sha256"] = _tree_sha256(config.schedule_root)
                shutil.copytree(config.schedule_root.resolve(), candidate / "schedule")
            report.update(code_revision=revision, candidate=str(candidate))
            steps = steps_class(config, candidate, revision)
            for name in ("data", "model", "serving", "lineups", "validate_and_stage"):
                report["stage"] = name
                _report(report_root, report)
                result = getattr(steps, name)(target) if name in {"data", "model", "validate_and_stage"} else getattr(steps, name)()
                report["completed_stages"].append(name)
            report.update(status="validated_candidate", runtime_release=str(result), stage="awaiting_activation")
            _report(report_root, report)
            if config.activate_after_validation:
                from nba_lineup_model.orchestration.daily_activation import activate_daily_release
                activate_daily_release(config, report)
            return report
        except Exception as error:
            if report.get("status") not in {"activation_failed", "rollback_failed"}:
                report.update(status="failed", error=f"{type(error).__name__}: {error}")
            _report(report_root, report)
            raise
