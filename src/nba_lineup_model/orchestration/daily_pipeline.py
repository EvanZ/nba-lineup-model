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

from nba_lineup_model.season.release_publisher import read_active_generation
from nba_lineup_model.season.slate_time import PACIFIC
from nba_lineup_model.season.source_policy import require_clean_artifact_sources

CAUSAL_HCA_MODEL_TRACK = "causal_hca_replacement_candidate_v2"
MODEL_IMPLEMENTATION_PATHS = (
    "src/nba_lineup_model/modeling",
    "src/nba_lineup_model/season/daily_artifact_common.py",
    "src/nba_lineup_model/season/daily_data_artifact.py",
    "src/nba_lineup_model/season/daily_model_artifact.py",
    "src/nba_lineup_model/season/daily_model_replay.py",
    "src/nba_lineup_model/web_api/inference.py",
)

GIT_PREFLIGHT_TIMEOUT_SECONDS = 30


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
    model_track: str = "official_nba_daily_model_v1"
    model_code_revision: str | None = None
    model_configuration_path: Path | None = None
    model_configuration_sha256: str | None = None
    model_provenance_matchup_url: str | None = None
    season: str = "2026-27"
    activate_after_validation: bool = False
    historical_runtime_root: Path | None = None
    observation_root: Path | None = None
    preserve_schedule_selections: bool = False
    forecast_date: str | None = None

    @classmethod
    def read(cls, path: Path | str) -> "DailyPipelineConfig":
        raw = json.loads(Path(path).read_text())
        for key in cls.__dataclass_fields__:
            if key.endswith(("_root", "_path", "_pointer")) and raw.get(key) is not None:
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
        if self.model_track == CAUSAL_HCA_MODEL_TRACK:
            if not all(
                (
                    self.model_code_revision,
                    self.model_configuration_path,
                    self.model_configuration_sha256,
                    self.model_provenance_matchup_url,
                )
            ):
                raise ValueError("Accepted HCA model provenance is incomplete")
            assert self.model_configuration_path is not None
            if not self.model_configuration_path.resolve(strict=True).is_relative_to(boundary):
                raise ValueError("HCA configuration escapes its environment")
        elif self.model_track != "official_nba_daily_model_v1":
            raise ValueError(f"Unsupported daily model track: {self.model_track}")


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
            missing = [label for label, path in (("final summary", summary), ("official box score", box), ("frozen forecast", frozen)) if not path.is_file()]
            errors.append(f"{game_id}: missing {', '.join(missing)}")
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




def causal_schedule_sha256(root: Path) -> str:
    """Hash only inputs that can change a completed-game model state.

    Live boards, odds, injuries, clocks, and pending captures are deliberately
    excluded so protected collectors can continue while a nightly candidate builds.
    """

    digest = hashlib.sha256()
    for directory in ("frozen", "final"):
        base = root / directory
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
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


def require_clean_code_revision(code_root: Path) -> str:
    """Return the tested revision or fail quickly on a blocked Git preflight."""

    try:
        revision = subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"],
            text=True,
            timeout=GIT_PREFLIGHT_TIMEOUT_SECONDS,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(code_root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            timeout=GIT_PREFLIGHT_TIMEOUT_SECONDS,
        ).strip()
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Daily pipeline Git preflight exceeded "
            f"{GIT_PREFLIGHT_TIMEOUT_SECONDS} seconds"
        ) from error
    if dirty:
        raise ValueError("Commit the tested pipeline code before materialization")
    return revision



def require_model_implementation_compatible(
    code_root: Path, accepted_revision: str
) -> None:
    """Refuse to extend a causal chain with changed model implementation code."""

    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(code_root),
                "merge-base",
                "--is-ancestor",
                accepted_revision,
                "HEAD",
            ],
            check=True,
            timeout=GIT_PREFLIGHT_TIMEOUT_SECONDS,
        )
        result = subprocess.run(
            [
                "git",
                "-C",
                str(code_root),
                "diff",
                "--quiet",
                f"{accepted_revision}..HEAD",
                "--",
                *MODEL_IMPLEMENTATION_PATHS,
            ],
            timeout=GIT_PREFLIGHT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Model implementation compatibility check timed out") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "Accepted model revision is not an ancestor of the tested code"
        ) from error
    if result.returncode != 0:
        raise ValueError(
            "Model implementation changed after the accepted revision; "
            "an explicitly approved causal replay is required"
        )


def causal_hca_replay_config(config: DailyPipelineConfig):
    """Reconstruct and verify the immutable accepted HCA replay settings."""

    from nba_lineup_model.season.daily_artifact_common import file_sha256
    from nba_lineup_model.season.daily_model_replay import CausalHcaReplayConfig

    if config.model_track != CAUSAL_HCA_MODEL_TRACK:
        raise ValueError("Daily model track is not the accepted causal HCA model")
    assert config.model_configuration_path is not None
    assert config.model_provenance_matchup_url is not None
    payload = json.loads(config.model_configuration_path.read_text())
    if payload.get("candidate_id") != CAUSAL_HCA_MODEL_TRACK:
        raise ValueError("HCA configuration candidate identity mismatch")
    if payload.get("target_season") != config.season:
        raise ValueError("HCA configuration season mismatch")
    replay_config = CausalHcaReplayConfig(
        season=config.season,
        home_court_prior_mean=float(payload["prior_mean"]),
        home_court_prior_equivalent_possessions=float(
            payload["prior_equivalent_possessions"]
        ),
        home_court_observation_variance_per_100=float(
            payload["observation_variance_per_100"]
        ),
        home_court_prior_source=(
            "artifact://causal-hca-prior/"
            + file_sha256(config.model_configuration_path)
        ),
        matchup_url=config.model_provenance_matchup_url,
    )
    if replay_config.sha256 != config.model_configuration_sha256:
        raise ValueError(
            "HCA replay configuration does not match the accepted provenance"
        )
    return replay_config

class CanonicalPipelineSteps:
    """The only production backend; tests replace I/O at this boundary."""
    def __init__(self, config: DailyPipelineConfig, candidate: Path, revision: str):
        self.config, self.candidate, self.revision = config, candidate, revision
        self.runtime_root = config.release_root.resolve(strict=True)
        self.data_replay_dates: tuple[date, ...] = ()
        self.approved_historical_model_replay_from: date | None = None
        self.env = {**os.environ, "PYTHONPATH": str(config.code_root / "src"),
            "GESTALT_CURRENT_SEASON_SERVING_ROOT": str(candidate / "serving"),
            "GESTALT_CURRENT_SEASON_LINEUPS_ROOT": str(candidate / "lineups"),
            "GESTALT_SCHEDULE_STATE_DIR": str(candidate / "schedule"),
            "GESTALT_GAME_CATALOG_PATH": str(config.catalog_path),
            "GESTALT_SITE_RELEASE_ROOT": str(candidate / "site-releases"),
            "GESTALT_PRESEASON_RANKINGS_PATH": str(config.preseason_rankings_path),
            "GESTALT_PRESEASON_PROFILES_PATH": str(config.preseason_profiles_path)}

    def command(self, module: str, *arguments: object) -> None:
        subprocess.run([sys.executable, "-m", module, *map(str, arguments)], cwd=self.runtime_root, env=self.env, check=True, timeout=1800)

    def data(self, target: date) -> None:
        from nba_lineup_model.season.daily_data_replay import replay_daily_data_chain
        config = self.config
        pointer = config.data_root / "latest.json"
        repair_dates = tuple(sorted(set(self.data_replay_dates)))
        if any(day < config.season_start or day > target for day in repair_dates):
            raise ValueError("Explicit data replay dates must fall within the release range")
        for repaired_date in repair_dates:
            replay_daily_data_chain(
                start_date=repaired_date,
                end_date=repaired_date,
                coverage_path=config.coverage_path,
                output_root=config.data_root,
                game_catalog_path=config.catalog_path,
                season=config.season,
            )
        start = (
            date.fromisoformat(json.loads(pointer.read_text())["through_date"])
            + timedelta(days=1)
            if pointer.exists()
            else config.season_start
        )
        if start <= target:
            replay_daily_data_chain(start_date=start, end_date=target, coverage_path=config.coverage_path, output_root=config.data_root, game_catalog_path=config.catalog_path, season=config.season)
        elif start != target + timedelta(days=1):
            raise ValueError("Daily data tip is ahead of the requested release")

    def model(self, target: date) -> None:
        from nba_lineup_model.season.daily_model_replay import DailyModelReplayConfig, _http_context_provider, replay_daily_model_chain
        from nba_lineup_model.season.daily_artifact_common import file_sha256
        config = self.config
        if config.model_track == CAUSAL_HCA_MODEL_TRACK:
            assert config.model_code_revision is not None
            require_model_implementation_compatible(
                config.code_root, config.model_code_revision
            )
            base = causal_hca_replay_config(config)
            context_court = "neutral"
            provenance_revision = config.model_code_revision
        else:
            base = DailyModelReplayConfig(
                season=config.season, matchup_url=config.matchup_url
            )
            context_court = "unit_home"
            provenance_revision = self.revision
        # Context is evaluated against fixed profiles and the installed immutable
        # model bundle, never whichever player universe the public API serves.
        context_identity = {
            "model_version": base.context_model_version,
            "model_release": str(self.runtime_root),
            "profiles_sha256": file_sha256(config.preseason_profiles_path),
        }
        replay_config = (
            base
            if config.model_track == CAUSAL_HCA_MODEL_TRACK
            else DailyModelReplayConfig(
                season=config.season,
                matchup_url="isolated://frozen-context",
                context_model_version=json.dumps(context_identity, sort_keys=True),
            )
        )
        with self.api(log_name="context-api.log") as api:
            provider = (
                _http_context_provider(api + "/api/matchups", court=context_court)
                if config.model_track == CAUSAL_HCA_MODEL_TRACK
                else _http_context_provider(api + "/api/matchups")
            )
            def strict_context(stints, home_b2b, away_b2b):
                offsets, warnings = provider(stints, home_b2b, away_b2b)
                if warnings:
                    raise ValueError("Pinned context inputs incomplete: " + "; ".join(warnings))
                return offsets, warnings
            replay_daily_model_chain(
                start_date=config.season_start,
                end_date=target,
                preseason_rankings_path=config.preseason_rankings_path,
                preseason_minutes_path=config.preseason_minutes_path,
                data_root=config.data_root,
                output_root=config.model_root,
                config=replay_config,
                code_revision=provenance_revision,
                context_provider=strict_context,
                approved_historical_replay_from=self.approved_historical_model_replay_from,
            )

    def serving(self) -> None:
        c = self.config
        self.command("nba_lineup_model.modeling.current_season_serving_materialization", "--daily-model-root", c.model_root, "--daily-data-root", c.data_root, "--release-root", c.release_root, "--output-root", self.candidate / "serving")

    def lineups(self) -> None:
        self.command("nba_lineup_model.modeling.current_season_lineup_materialization", "--daily-data-root", self.config.data_root, "--output-root", self.candidate / "lineups")

    @contextmanager
    def api(self, *, log_name="api.log"):
        with socket.socket() as bound:
            bound.bind(("127.0.0.1", 0))
            port = bound.getsockname()[1]
        root = f"http://127.0.0.1:{port}"
        import urllib.request
        with (self.candidate / log_name).open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "nba_lineup_model.web_api.app:app", "--host", "127.0.0.1", "--port", str(port)], cwd=self.runtime_root, env=self.env, stdout=log, stderr=subprocess.STDOUT)
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
        from nba_lineup_model.season.official_schedule_runtime import reproject_captured_schedule_window, _fetch_json
        from nba_lineup_model.season.serving_selection import INDEX_FILENAME, read_object
        from nba_lineup_model.season.sealed_code_package import materialize_sealed_code_package, verify_sealed_code_package
        from nba_lineup_model.season.upcoming_forecasts import refresh_open_selection
        c, out = self.config, self.candidate
        schedule = out / "schedule"
        before = _frozen_hashes(schedule)
        first_forecast = date.fromisoformat(c.forecast_date) if c.forecast_date else target + timedelta(days=1)
        days_ahead = 0 if c.forecast_date else 7
        with self.api() as api:
            if (schedule / INDEX_FILENAME).is_file():
                code_root = out / "forecast-code"
                code_root.mkdir(exist_ok=True)
                package = (verify_sealed_code_package(code_root) if (code_root / "code").exists()
                    else materialize_sealed_code_package(c.code_root, code_root, expected_revision=self.revision))
                pointer = read_object(c.model_root / "latest.json")
                source_index = read_object(schedule / INDEX_FILENAME)
                registry = Path(source_index.get("forecast_registry_root") or c.environment_root / "forecast-registry")
                refreshed = out / "refreshed-schedule"
                evidence = refresh_open_selection(source=schedule, target=refreshed,
                    model_bundle=c.model_root / pointer["bundle_path"],
                    forecast_response=_fetch_json(api + "/api/win-projections"),
                    start_date=first_forecast, days_ahead=days_ahead, registry_root=registry,
                    catalog=c.catalog_path, code=package["identity"],
                    environment="dev" if c.environment_root.name == "nba-gestalt-dev" else "prod")
                (out / "forecast-refresh.json").write_text(json.dumps(evidence, indent=2) + "\n")
                schedule = refreshed
            else:
                reproject_captured_schedule_window(start_date=first_forecast.isoformat(),
                    days_ahead=days_ahead, state_dir=schedule, projection_url=api + "/api/win-projections")
        if any(_frozen_hashes(schedule).get(key) != value for key, value in before.items()):
            raise ValueError("Nightly refresh changed a frozen forecast")
        self.env["GESTALT_SCHEDULE_STATE_DIR"] = str(schedule)
        with self.api() as api:
            validate_schedule_artifacts(schedule)
            validation = collect_api_validation(api, evidence_date=target.isoformat(), daily_model_root=c.model_root)
            publish_dev_site_release(daily_data_root=c.data_root, daily_model_root=c.model_root, serving_root=out / "serving", lineups_root=out / "lineups", schedule_state_dir=schedule, catalog_path=c.catalog_path, preseason_rankings_path=c.preseason_rankings_path, preseason_profiles_path=c.preseason_profiles_path, code_revision=self.revision, api_validation=validation, output_root=out / "site-releases")
        return stage_production_site_release(dev_site_release_root=out / "site-releases", serving_root=out / "serving", lineups_root=out / "lineups", schedule_snapshot_root=schedule, schedule_overlay_path=c.schedule_root, catalog_path=c.catalog_path, preseason_rankings_path=c.preseason_rankings_path, preseason_profiles_path=c.preseason_profiles_path, code_root=c.code_root, frontend_root=c.frontend_root, output_root=c.pipeline_root / "releases", live_pointer=c.live_pointer, historical_runtime_root=c.historical_runtime_root, observation_root=c.observation_root, environment="dev" if c.environment_root.name == "nba-gestalt-dev" else "prod")


def prepare_daily_release(
    config: DailyPipelineConfig,
    target: date,
    *,
    steps_class=CanonicalPipelineSteps,
    data_replay_dates: tuple[date, ...] = (),
    approved_historical_model_replay_from: date | None = None,
) -> dict:
    """No live pointer changes on success or failure; failed stages are visible."""
    config.validate()
    replay_dates = tuple(sorted(set(data_replay_dates)))
    if any(day < config.season_start or day > target for day in replay_dates):
        raise ValueError("Explicit data replay dates must fall within the release range")
    if approved_historical_model_replay_from is not None and not (
        config.season_start <= approved_historical_model_replay_from <= target
    ):
        raise ValueError("Approved model replay date must fall within the release range")
    config.pipeline_root.mkdir(parents=True, exist_ok=True)
    report_root = config.pipeline_root / "reports" / target.isoformat()
    report = {
        "activation_operation_id": (
            f"daily-{config.environment_root.name}-{target.isoformat()}-"
            f"{uuid4().hex}"
        ),
        "expected_previous_generation": read_active_generation(
            config.live_pointer
        ),
        "game_date": target.isoformat(),
        "data_replay_dates": [day.isoformat() for day in replay_dates],
        "approved_historical_model_replay_from": (
            approved_historical_model_replay_from.isoformat()
            if approved_historical_model_replay_from else None
        ),
        "status": "running",
        "stage": "preflight",
        "completed_stages": [],
    }
    with (config.pipeline_root / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another daily pipeline is already running") from error
        try:
            _report(report_root, report)
            require_complete_slate(config, target)
            revision = require_clean_code_revision(config.code_root)
            candidate = config.pipeline_root / "candidates" / f"{target}-{uuid4().hex}"
            candidate.mkdir(parents=True)
            with schedule_lock(config.schedule_root):
                require_clean_artifact_sources(config.schedule_root)
                from nba_lineup_model.season.dev_site_release import _tree_sha256
                report["source_schedule_sha256"] = _tree_sha256(config.schedule_root)
                report["source_causal_schedule_sha256"] = causal_schedule_sha256(
                    config.schedule_root
                )
                shutil.copytree(config.schedule_root.resolve(), candidate / "schedule")
            report.update(code_revision=revision, candidate=str(candidate))
            steps = steps_class(config, candidate, revision)
            steps.data_replay_dates = replay_dates
            steps.approved_historical_model_replay_from = approved_historical_model_replay_from
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
        except BaseException as error:
            if report.get("status") not in {"activation_failed", "rollback_failed"}:
                report.update(status="failed", error=f"{type(error).__name__}: {error}")
            _report(report_root, report)
            raise
