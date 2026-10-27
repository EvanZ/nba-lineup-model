"""Immutable serving bundles for the active NBA season.

The public site must never read partially advanced rating and minutes states.
This module validates one consistent pair, writes a content-addressed bundle,
and advances a tiny local pointer only after the immutable bundle is complete.
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.in_season_nail import validate_player_state
from nba_lineup_model.season.schema import validate_season

SERVING_SCHEMA_VERSION = 1
POINTER_SCHEMA_VERSION = 1
SERVING_STATE_FILENAME = "serving_state.parquet"
RATING_TRAJECTORY_FILENAME = "rating_trajectory.parquet"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "report.html"

_MINUTES_REQUIRED_COLUMNS = {
    "season",
    "updater_version",
    "player_id",
    "player_name",
    "team",
    "preseason_participation_prior",
    "current_participation_estimate",
    "participation_observations",
    "preseason_conditional_minutes_prior",
    "preseason_normalized_minutes",
    "current_conditional_minutes_estimate",
    "conditional_observations",
    "effective_conditional_weight",
    "conditional_window_observations",
    "included_game_ids_json",
    "contribution_lineage_json",
    "cutoff_time",
    "prior_participation_weight",
    "prior_conditional_weight",
}


@dataclass(frozen=True)
class ServingPromotionConfig:
    """Locked updater parameters and frozen-preseason lineage."""

    preseason_run_id: str
    rating_prior_possessions: float = 6000.0
    minutes_participation_prior_games: float = 4.0
    minutes_conditional_prior_games: float = 0.2
    minutes_conditional_lookback_games: int = 3
    minutes_role_cutoff_minutes: int = 44

    def validate(self) -> None:
        if not self.preseason_run_id.strip():
            raise ValueError("Preseason run ID must be non-empty")
        positive = {
            "rating prior possessions": self.rating_prior_possessions,
            "minutes participation prior games": self.minutes_participation_prior_games,
            "minutes conditional prior games": self.minutes_conditional_prior_games,
        }
        for label, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label.title()} must be finite and positive")
        if self.minutes_conditional_lookback_games <= 0:
            raise ValueError("Minutes conditional lookback games must be positive")
        if not 0 < self.minutes_role_cutoff_minutes <= 48:
            raise ValueError("Minutes role cutoff must lie in regulation")


@dataclass(frozen=True)
class PublishedServingBundle:
    path: Path
    created: bool
    manifest: dict[str, Any]


def build_serving_state(
    rating_state: pd.DataFrame,
    minutes_state: pd.DataFrame,
    *,
    season: str,
    evidence_through_date: date,
    as_of_cutoff_time: datetime,
    config: ServingPromotionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and combine one synchronized rating/minutes state pair."""

    season = validate_season(season)
    config.validate()
    cutoff = _aware_utc(as_of_cutoff_time, "as-of cutoff time")
    validate_player_state(rating_state)
    _validate_minutes_state(minutes_state)
    _require_uniform(rating_state, "season", season, "rating season")
    _require_uniform(minutes_state, "season", season, "minutes season")
    _require_numeric_uniform(
        rating_state,
        "prior_equivalent_possessions",
        config.rating_prior_possessions,
    )
    _require_numeric_uniform(
        minutes_state,
        "prior_participation_weight",
        config.minutes_participation_prior_games,
    )
    _require_numeric_uniform(
        minutes_state,
        "prior_conditional_weight",
        config.minutes_conditional_prior_games,
    )

    rating_games = _uniform_json_list(
        rating_state, "included_game_ids_json", "rating included games"
    )
    minutes_games = _uniform_json_list(
        minutes_state, "included_game_ids_json", "minutes included games"
    )
    if rating_games != minutes_games:
        raise ValueError("Rating and minutes states do not contain the same finalized games")
    rating_lineage = _uniform_json_list(
        rating_state, "contribution_lineage_json", "rating contribution lineage"
    )
    minutes_lineage = _uniform_json_list(
        minutes_state, "contribution_lineage_json", "minutes contribution lineage"
    )

    rating_ids = set(rating_state["player_id"].astype(int))
    minutes_ids = set(minutes_state["player_id"].astype(int))
    missing_ratings = sorted(minutes_ids - rating_ids)
    if missing_ratings:
        raise ValueError(
            "Rostered minutes-state players are missing rating priors: "
            + ", ".join(map(str, missing_ratings))
        )

    rating_columns = [
        "player_id",
        "team_at_cutoff",
        "core_model_version",
        "state_updater_version",
        "frozen_preseason_nail",
        "current_season_residual",
        "updated_nail",
        "raw_on_court_possessions",
        "effective_information",
        "prior_equivalent_possessions",
        "prior_weight",
        "evidence_weight",
        "posterior_standard_error",
    ]
    minutes_columns = [
        "player_id",
        "player_name",
        "team",
        "updater_version",
        "preseason_participation_prior",
        "current_participation_estimate",
        "participation_observations",
        "preseason_conditional_minutes_prior",
        "preseason_normalized_minutes",
        "current_conditional_minutes_estimate",
        "conditional_observations",
        "effective_conditional_weight",
        "conditional_window_observations",
    ]
    serving = minutes_state.loc[:, minutes_columns].merge(
        rating_state.loc[:, rating_columns],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    serving = serving.rename(
        columns={
            "team_at_cutoff": "rating_team_at_cutoff",
            "state_updater_version": "rating_updater_version",
            "updater_version": "minutes_updater_version",
        }
    )
    serving.insert(0, "schema_version", SERVING_SCHEMA_VERSION)
    serving.insert(1, "season", season)
    serving.insert(2, "evidence_through_date", evidence_through_date.isoformat())
    serving.insert(3, "as_of_cutoff_time", cutoff.isoformat().replace("+00:00", "Z"))
    serving["team"] = serving["team"].astype(str).str.strip().str.upper()
    serving["rating_team_at_cutoff"] = (
        serving["rating_team_at_cutoff"].fillna("").astype(str).str.strip().str.upper()
    )
    if serving["team"].eq("").any():
        raise ValueError("Every serving-state player must have a current team")

    numeric_columns = [
        column
        for column in serving.columns
        if column
        not in {
            "season",
            "evidence_through_date",
            "as_of_cutoff_time",
            "player_name",
            "team",
            "rating_team_at_cutoff",
            "core_model_version",
            "rating_updater_version",
            "minutes_updater_version",
        }
    ]
    for column in numeric_columns:
        values = pd.to_numeric(serving[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Serving-state column is not finite: {column}")
    if not np.allclose(
        serving["updated_nail"],
        serving["frozen_preseason_nail"] + serving["current_season_residual"],
        atol=1e-12,
    ):
        raise ValueError("Serving NAIL does not reconstruct from preseason plus residual")
    no_rating_evidence = serving["raw_on_court_possessions"].eq(0.0)
    if not np.allclose(
        serving.loc[no_rating_evidence, "updated_nail"],
        serving.loc[no_rating_evidence, "frozen_preseason_nail"],
        atol=1e-12,
    ):
        raise ValueError("Zero-evidence ratings must reproduce the preseason prior")
    no_minutes_evidence = serving["conditional_observations"].eq(0)
    if not np.allclose(
        serving.loc[no_minutes_evidence, "current_conditional_minutes_estimate"],
        serving.loc[no_minutes_evidence, "preseason_conditional_minutes_prior"],
        atol=1e-12,
    ):
        raise ValueError("Zero-evidence minutes must reproduce the preseason prior")

    serving = serving.sort_values("player_id", kind="stable").reset_index(drop=True)
    audit = {
        "source_game_ids": rating_games,
        "rating_contribution_lineage": rating_lineage,
        "minutes_contribution_lineage": minutes_lineage,
        "rating_player_count": int(len(rating_state)),
        "serving_player_count": int(len(serving)),
        "team_count": int(serving["team"].nunique()),
        "rating_team_mismatch_count": int(
            serving["rating_team_at_cutoff"].ne(serving["team"]).sum()
        ),
    }
    return serving, audit


def publish_serving_bundle(
    rating_state_path: Path | str,
    minutes_state_path: Path | str,
    *,
    root: Path | str,
    season: str,
    evidence_through_date: date,
    as_of_cutoff_time: datetime,
    config: ServingPromotionConfig,
    rating_source_uri: str,
    minutes_source_uri: str,
    rating_trajectory_path: Path | str | None = None,
    rating_trajectory_source_uri: str | None = None,
    generated_at: datetime | None = None,
) -> PublishedServingBundle:
    """Write one immutable serving bundle and return its validated identity."""

    rating_path = Path(rating_state_path)
    minutes_path = Path(minutes_state_path)
    rating_state = pd.read_parquet(rating_path)
    minutes_state = pd.read_parquet(minutes_path)
    trajectory_path = (
        Path(rating_trajectory_path) if rating_trajectory_path is not None else None
    )
    if (trajectory_path is None) != (rating_trajectory_source_uri is None):
        raise ValueError("Rating trajectory path and source URI must be provided together")
    rating_trajectory = (
        pd.read_parquet(trajectory_path) if trajectory_path is not None else None
    )
    if rating_trajectory is not None:
        _validate_rating_trajectory(
            rating_trajectory,
            season=season,
            rating_state=rating_state,
        )
    serving, audit = build_serving_state(
        rating_state,
        minutes_state,
        season=season,
        evidence_through_date=evidence_through_date,
        as_of_cutoff_time=as_of_cutoff_time,
        config=config,
    )
    root_path = Path(root)
    staging = root_path / f".serving-bundle.tmp-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        state_path = staging / SERVING_STATE_FILENAME
        serving.to_parquet(state_path, index=False)
        state_sha = _file_sha256(state_path)
        trajectory_sha = None
        if rating_trajectory is not None:
            serving_trajectory_path = staging / RATING_TRAJECTORY_FILENAME
            rating_trajectory.to_parquet(serving_trajectory_path, index=False)
            trajectory_sha = _file_sha256(serving_trajectory_path)
        report_path = staging / REPORT_FILENAME
        report_path.write_text(_render_report(serving, audit, config), encoding="utf-8")
        report_sha = _file_sha256(report_path)
        identity = {
            "schema_version": SERVING_SCHEMA_VERSION,
            "season": validate_season(season),
            "evidence_through_date": evidence_through_date.isoformat(),
            "as_of_cutoff_time": _aware_utc(
                as_of_cutoff_time, "as-of cutoff time"
            ).isoformat().replace("+00:00", "Z"),
            "preseason_run_id": config.preseason_run_id,
            "configuration": {
                "rating_prior_possessions": config.rating_prior_possessions,
                "minutes_participation_prior_games": (
                    config.minutes_participation_prior_games
                ),
                "minutes_conditional_prior_games": config.minutes_conditional_prior_games,
                "minutes_conditional_lookback_games": (
                    config.minutes_conditional_lookback_games
                ),
                "minutes_role_cutoff_minutes": config.minutes_role_cutoff_minutes,
            },
            "sources": {
                "rating_state": {
                    "uri": rating_source_uri,
                    "sha256": _file_sha256(rating_path),
                    "updater_version": _uniform_string(
                        rating_state, "state_updater_version"
                    ),
                    "child_cutoff_time": _uniform_timestamp(
                        rating_state, "cutoff_time"
                    ),
                    "contribution_lineage": audit["rating_contribution_lineage"],
                },
                "minutes_state": {
                    "uri": minutes_source_uri,
                    "sha256": _file_sha256(minutes_path),
                    "updater_version": _uniform_string(minutes_state, "updater_version"),
                    "child_cutoff_time": _uniform_timestamp(minutes_state, "cutoff_time"),
                    "contribution_lineage": audit["minutes_contribution_lineage"],
                },
            },
            "coverage": {
                "source_game_ids": audit["source_game_ids"],
                "game_count": len(audit["source_game_ids"]),
                "rating_player_count": audit["rating_player_count"],
                "serving_player_count": audit["serving_player_count"],
                "team_count": audit["team_count"],
                "rating_team_mismatch_count": audit["rating_team_mismatch_count"],
            },
            "files": {
                SERVING_STATE_FILENAME: state_sha,
                REPORT_FILENAME: report_sha,
            },
        }
        if trajectory_path is not None and rating_trajectory_source_uri is not None:
            identity["sources"]["rating_trajectory"] = {
                "uri": rating_trajectory_source_uri,
                "sha256": _file_sha256(trajectory_path),
                "row_count": len(rating_trajectory),
                "player_count": int(rating_trajectory["player_id"].nunique()),
            }
            identity["files"][RATING_TRAJECTORY_FILENAME] = trajectory_sha
        bundle_sha = hashlib.sha256(_canonical_json(identity)).hexdigest()
        manifest = {
            "bundle_sha256": bundle_sha,
            "generated_at": _aware_utc(
                generated_at or datetime.now(UTC), "generated at"
            ).isoformat().replace("+00:00", "Z"),
            "identity": identity,
        }
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        target = (
            root_path
            / f"season={season}"
            / f"through_date={evidence_through_date.isoformat()}"
            / f"bundle_sha256={bundle_sha}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(staging)
            existing = verify_serving_bundle(target)
            return PublishedServingBundle(target, False, existing)
        staging.replace(target)
        verified = verify_serving_bundle(target)
        return PublishedServingBundle(target, True, verified)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_serving_bundle(path: Path | str) -> dict[str, Any]:
    """Validate an immutable serving bundle and return its manifest."""

    root = Path(path)
    manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Serving manifest identity is missing")
    expected_bundle_sha = hashlib.sha256(_canonical_json(identity)).hexdigest()
    if manifest.get("bundle_sha256") != expected_bundle_sha:
        raise ValueError("Serving bundle checksum does not match its identity")
    if root.name != f"bundle_sha256={expected_bundle_sha}":
        raise ValueError("Serving bundle directory does not match its checksum")
    files = identity.get("files")
    if not isinstance(files, dict):
        raise ValueError("Serving manifest file inventory is missing")
    required_files = [SERVING_STATE_FILENAME, REPORT_FILENAME]
    if RATING_TRAJECTORY_FILENAME in files:
        required_files.append(RATING_TRAJECTORY_FILENAME)
    for filename in required_files:
        expected = files.get(filename)
        if not isinstance(expected, str) or _file_sha256(root / filename) != expected:
            raise ValueError(f"Serving bundle file checksum changed: {filename}")
    state = pd.read_parquet(root / SERVING_STATE_FILENAME)
    _validate_serving_frame(state, identity)
    if RATING_TRAJECTORY_FILENAME in files:
        _validate_rating_trajectory(
            pd.read_parquet(root / RATING_TRAJECTORY_FILENAME),
            season=str(identity["season"]),
            rating_state=state,
        )
    return manifest


def promote_serving_bundle(
    bundle_path: Path | str,
    *,
    root: Path | str,
    promoted_at: datetime | None = None,
) -> tuple[dict[str, Any], bool]:
    """Atomically advance the local latest pointer after full validation."""

    root_path = Path(root).resolve()
    bundle = Path(bundle_path).resolve()
    try:
        relative = bundle.relative_to(root_path)
    except ValueError as error:
        raise ValueError("Serving bundle must live beneath its promotion root") from error
    manifest = verify_serving_bundle(bundle)
    timestamp = _aware_utc(
        promoted_at or datetime.now(UTC), "promoted at"
    ).isoformat().replace("+00:00", "Z")
    pointer = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "season": manifest["identity"]["season"],
        "evidence_through_date": manifest["identity"]["evidence_through_date"],
        "bundle_sha256": manifest["bundle_sha256"],
        "bundle_path": relative.as_posix(),
        "promoted_at": timestamp,
    }
    latest_path = root_path / "latest.json"
    if latest_path.is_file():
        current = json.loads(latest_path.read_text(encoding="utf-8"))
        if current.get("bundle_sha256") == pointer["bundle_sha256"]:
            resolve_pointer(root_path, current)
            return current, False
        resolve_pointer(root_path, current)
        _atomic_write_json(root_path / "previous.json", current)
    _atomic_write_json(latest_path, pointer)
    history = root_path / "pointer_history" / f"{pointer['bundle_sha256']}.json"
    if not history.exists():
        _atomic_write_json(history, pointer)
    return pointer, True


def rollback_serving_bundle(*, root: Path | str) -> dict[str, Any]:
    """Swap latest with the previously promoted, still-immutable bundle."""

    root_path = Path(root).resolve()
    latest_path = root_path / "latest.json"
    previous_path = root_path / "previous.json"
    if not latest_path.is_file() or not previous_path.is_file():
        raise ValueError("Both latest and previous serving pointers are required")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    resolve_pointer(root_path, latest)
    resolve_pointer(root_path, previous)
    _atomic_write_json(previous_path, latest)
    _atomic_write_json(latest_path, previous)
    return previous


def resolve_latest_bundle(root: Path | str) -> Path:
    root_path = Path(root).resolve()
    latest = json.loads((root_path / "latest.json").read_text(encoding="utf-8"))
    return resolve_pointer(root_path, latest)


def resolve_pointer(root: Path, pointer: dict[str, Any]) -> Path:
    if pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise ValueError("Unsupported serving pointer schema")
    raw_path = pointer.get("bundle_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Serving pointer path is missing")
    target = (root / raw_path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Serving pointer escapes its root") from error
    manifest = verify_serving_bundle(target)
    if manifest["bundle_sha256"] != pointer.get("bundle_sha256"):
        raise ValueError("Serving pointer checksum does not match its bundle")
    return target


def _validate_minutes_state(state: pd.DataFrame) -> None:
    missing = sorted(_MINUTES_REQUIRED_COLUMNS - set(state.columns))
    if missing:
        raise ValueError("Minutes state is missing required columns: " + ", ".join(missing))
    if state.empty or state["player_id"].duplicated().any():
        raise ValueError("Minutes state must contain one row per player")
    if not state["current_participation_estimate"].between(0.0, 1.0).all():
        raise ValueError("Current participation estimates must lie in [0, 1]")
    for column in (
        "preseason_participation_prior",
        "preseason_conditional_minutes_prior",
        "preseason_normalized_minutes",
        "current_conditional_minutes_estimate",
        "effective_conditional_weight",
    ):
        values = pd.to_numeric(state[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Minutes-state column is invalid: {column}")


def _validate_serving_frame(state: pd.DataFrame, identity: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "season",
        "player_id",
        "player_name",
        "team",
        "frozen_preseason_nail",
        "current_season_residual",
        "updated_nail",
        "current_conditional_minutes_estimate",
    }
    missing = sorted(required - set(state.columns))
    if missing:
        raise ValueError("Serving state is missing required columns: " + ", ".join(missing))
    if state.empty or state["player_id"].duplicated().any():
        raise ValueError("Serving state must contain one row per player")
    if not state["schema_version"].eq(SERVING_SCHEMA_VERSION).all():
        raise ValueError("Serving-state schema version is inconsistent")
    if not state["season"].astype(str).eq(str(identity.get("season"))).all():
        raise ValueError("Serving-state season does not match its manifest")
    if len(state) != int(identity["coverage"]["serving_player_count"]):
        raise ValueError("Serving-state row count does not match its manifest")
    if not np.allclose(
        state["updated_nail"],
        state["frozen_preseason_nail"] + state["current_season_residual"],
        atol=1e-12,
    ):
        raise ValueError("Serving-state NAIL arithmetic failed validation")


def _validate_rating_trajectory(
    trajectory: pd.DataFrame,
    *,
    season: str,
    rating_state: pd.DataFrame,
) -> None:
    required = {
        "season",
        "updater_version",
        "player_id",
        "game_number",
        "game_id",
        "game_date",
        "preseason_rating",
        "in_season_delta",
        "updated_rating",
        "game_on_court_possessions",
        "cumulative_on_court_possessions",
        "effective_information",
        "evidence_weight",
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise ValueError("Rating trajectory is missing columns: " + ", ".join(missing))
    if trajectory.empty:
        raise ValueError("Rating trajectory cannot be empty")
    if not trajectory["season"].astype(str).eq(validate_season(season)).all():
        raise ValueError("Rating trajectory season does not match its serving state")
    if trajectory.duplicated(["player_id", "game_number"]).any():
        raise ValueError("Rating trajectory contains duplicate player game numbers")
    ordered = trajectory.sort_values(["player_id", "game_number"], kind="stable")
    game_numbers = ordered.groupby("player_id", sort=False)["game_number"]
    if not game_numbers.apply(
        lambda values: values.astype(int).tolist() == list(range(1, len(values) + 1))
    ).all():
        raise ValueError("Rating trajectory game numbers must be contiguous per player")
    if not np.allclose(
        trajectory["updated_rating"],
        trajectory["preseason_rating"] + trajectory["in_season_delta"],
        atol=1e-12,
    ):
        raise ValueError("Rating trajectory arithmetic failed validation")
    if (trajectory["game_on_court_possessions"] <= 0).any():
        raise ValueError("Rating trajectory games require positive on-court possessions")
    if not trajectory["evidence_weight"].between(0.0, 1.0).all():
        raise ValueError("Rating trajectory evidence weights must lie in [0, 1]")
    baselines = rating_state.set_index("player_id")["frozen_preseason_nail"]
    expected = trajectory["player_id"].astype(int).map(baselines)
    known = expected.notna()
    if not np.allclose(
        trajectory.loc[known, "preseason_rating"],
        expected.loc[known],
        atol=1e-12,
    ):
        raise ValueError("Rating trajectory preseason ratings changed")


def _render_report(
    state: pd.DataFrame,
    audit: dict[str, Any],
    config: ServingPromotionConfig,
) -> str:
    ranked = state.sort_values("updated_nail", ascending=False, kind="stable").head(25)
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.player_name))}</td>"
        f"<td>{html.escape(str(row.team))}</td>"
        f"<td>{float(row.frozen_preseason_nail):+.2f}</td>"
        f"<td>{float(row.current_season_residual):+.2f}</td>"
        f"<td>{float(row.updated_nail):+.2f}</td>"
        f"<td>{float(row.current_conditional_minutes_estimate):.1f}</td>"
        "</tr>"
        for row in ranked.itertuples(index=False)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2026-27 current-season serving audit</title>
<style>
body {{ font: 15px/1.45 system-ui; margin: 32px; color: #17201c; background: #f8f6ef; }}
main {{ max-width: 1050px; margin: auto; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
.card {{ padding: 16px; background: #fff; border: 1px solid #d8d4c8; }}
strong {{ display: block; font-size: 24px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; margin-top: 22px; }}
th, td {{ padding: 9px; border-bottom: 1px solid #ddd; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
code {{ font-size: 13px; }}
</style>
</head>
<body><main>
<h1>Current-season serving-state audit</h1>
<p>Immutable promotion candidate combining the possession-based scalar NAIL updater with
the calibrated early-game minutes updater. Additive profiles and offense/defense
decompositions remain frozen.</p>
<div class="cards">
<div class="card"><strong>{audit['serving_player_count']}</strong>rostered players</div>
<div class="card"><strong>{audit['team_count']}</strong>teams</div>
<div class="card"><strong>{len(audit['source_game_ids'])}</strong>final games</div>
<div class="card"><strong>{audit['rating_team_mismatch_count']}</strong>team audit differences</div>
</div>
<p><code>Preseason run: {html.escape(config.preseason_run_id)}</code></p>
<table>
<thead><tr><th>Player</th><th>Team</th><th>Preseason</th><th>Season delta</th>
<th>Updated NAIL</th><th>Minutes if available</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</main></body></html>"""


def _uniform_json_list(frame: pd.DataFrame, column: str, label: str) -> list[str]:
    values = frame[column].astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"{label.title()} must be uniform")
    parsed = json.loads(values[0])
    if not isinstance(parsed, list) or any(not isinstance(value, str) for value in parsed):
        raise ValueError(f"{label.title()} must be a JSON string list")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{label.title()} contains duplicates")
    return sorted(parsed)


def _require_uniform(frame: pd.DataFrame, column: str, value: str, label: str) -> None:
    values = frame[column].astype(str).unique()
    if len(values) != 1 or values[0] != value:
        raise ValueError(f"{label.title()} does not match {value}")


def _require_numeric_uniform(frame: pd.DataFrame, column: str, value: float) -> None:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values, value, atol=1e-12):
        raise ValueError(f"{column} does not match the locked promotion configuration")


def _uniform_string(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].astype(str).unique()
    if len(values) != 1 or not values[0].strip():
        raise ValueError(f"{column} must contain one non-empty value")
    return values[0]


def _uniform_timestamp(frame: pd.DataFrame, column: str) -> str:
    values = pd.to_datetime(frame[column], utc=True).unique()
    if len(values) != 1:
        raise ValueError(f"{column} must contain one timestamp")
    return pd.Timestamp(values[0]).to_pydatetime().astimezone(UTC).isoformat()


def _aware_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label.title()} must include a timezone")
    return value.astimezone(UTC)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
