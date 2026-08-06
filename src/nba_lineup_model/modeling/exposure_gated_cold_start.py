"""Blend a draft RAPM rate with the pooled replacement token using an exposure gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.cold_start_exposure import (
    validate_cold_start_exposure_study,
)
from nba_lineup_model.modeling.draft_prior import validate_draft_prior_study
from nba_lineup_model.modeling.replacement_token import validate_replacement_token_study
from nba_lineup_model.modeling.stints import modeling_code_fingerprint
from nba_lineup_model.season.schema import validate_season

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")


@dataclass(frozen=True)
class ExposureGatedColdStartPrior:
    """Immutable preseason cold-start prior and revised first-year rankings."""

    run_dir: Path
    run_id: str
    player_count: int
    replacement_rapm: float


def build_exposure_gated_cold_start_prior(
    *,
    season: str = "2025-26",
    draft_run_id: str | None = None,
    exposure_run_id: str | None = None,
    replacement_run_id: str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ExposureGatedColdStartPrior:
    """Create a target-season cold-start prior without target outcomes."""

    target_season = validate_season(season)
    source_season = _previous_season(target_season)
    artifact_root = Path(artifacts_dir)
    draft_root = _resolve_run(artifact_root / "draft_prior" / target_season, draft_run_id)
    exposure_root = _resolve_run(
        artifact_root / "cold_start_exposure" / target_season, exposure_run_id
    )
    replacement_root = _resolve_run(
        artifact_root / "replacement_token" / source_season, replacement_run_id
    )
    draft_metadata = validate_draft_prior_study(draft_root)
    exposure_metadata = validate_cold_start_exposure_study(exposure_root)
    replacement_metadata = validate_replacement_token_study(replacement_root)
    _validate_sources(
        target_season=target_season,
        source_season=source_season,
        draft_metadata=draft_metadata,
        exposure_metadata=exposure_metadata,
        replacement_metadata=replacement_metadata,
    )
    rankings = blend_cold_start_prior_components(
        pd.read_parquet(draft_root / "rookie_rankings.parquet"),
        pd.read_parquet(exposure_root / "target_exposure_predictions.parquet"),
        replacement_rapm=_replacement_rapm(replacement_root),
    )
    return _write_prior(
        target_season=target_season,
        source_season=source_season,
        rankings=rankings,
        replacement_rapm=_replacement_rapm(replacement_root),
        draft_root=draft_root,
        exposure_root=exposure_root,
        replacement_root=replacement_root,
        artifacts_dir=artifact_root,
    )


def blend_cold_start_prior_components(
    draft_rankings: pd.DataFrame,
    exposure_predictions: pd.DataFrame,
    *,
    replacement_rapm: float,
) -> pd.DataFrame:
    """Blend draft-rate and replacement-rate forecasts using P(low exposure)."""

    draft_required = {
        "player_id",
        "player_name",
        "listed_position",
        "draft_status",
        "draft_number",
        "draft_prior",
    }
    exposure_required = {
        "player_id",
        "draft_age",
        "predicted_replacement_probability",
        "predicted_rotation_probability",
    }
    missing_draft = draft_required - set(draft_rankings)
    missing_exposure = exposure_required - set(exposure_predictions)
    if missing_draft or missing_exposure:
        raise ValueError(
            "Cold-start blend inputs missing columns: "
            f"draft={sorted(missing_draft)}, exposure={sorted(missing_exposure)}"
        )
    if not np.isfinite(replacement_rapm):
        raise ValueError("Replacement RAPM must be finite")
    draft = draft_rankings.loc[:, sorted(draft_required)].copy()
    exposure = exposure_predictions.loc[:, sorted(exposure_required)].copy()
    for name, frame in (("draft", draft), ("exposure", exposure)):
        frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
        if frame["player_id"].duplicated().any():
            raise ValueError(f"{name} cold-start input has duplicate player IDs")
    output = draft.merge(exposure, on="player_id", how="inner", validate="one_to_one")
    if len(output) != len(draft) or len(output) != len(exposure):
        raise ValueError("Draft and exposure target cohorts do not match")
    probability = pd.to_numeric(
        output["predicted_replacement_probability"], errors="raise"
    ).to_numpy(dtype=float)
    rate = pd.to_numeric(output["draft_prior"], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(rate).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise ValueError("Cold-start blend components must be finite valid probabilities and rates")
    output["replacement_rapm"] = replacement_rapm
    output["blended_cold_start_prior"] = probability * replacement_rapm + (1.0 - probability) * rate
    output["rank"] = (
        output["blended_cold_start_prior"].rank(method="first", ascending=False).astype(int)
    )
    columns = [
        "rank",
        "player_id",
        "player_name",
        "listed_position",
        "draft_status",
        "draft_number",
        "draft_age",
        "draft_prior",
        "predicted_replacement_probability",
        "predicted_rotation_probability",
        "replacement_rapm",
        "blended_cold_start_prior",
    ]
    return output.loc[:, columns].sort_values("rank", kind="stable").reset_index(drop=True)


def validate_exposure_gated_cold_start_prior(run_dir: Path | str) -> dict[str, object]:
    """Validate immutable blended cold-start artifacts and their no-leakage contract."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("target_outcomes_used_for_fit") is not False:
        raise ValueError("Exposure-gated cold-start prior uses target outcomes")
    if manifest.get("scope") != "first_nba_season_players_only":
        raise ValueError("Exposure-gated prior must be restricted to first-year players")
    required = {"revised_rookie_rankings.parquet", "metadata.json"}
    records = {record["filename"]: record for record in manifest["artifacts"]}
    if not required <= set(records):
        raise ValueError("Exposure-gated cold-start prior is missing required files")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"Exposure-gated cold-start prior changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Exposure-gated cold-start prior hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"Exposure-gated cold-start prior row count changed: {filename}")
    rankings = pd.read_parquet(root / "revised_rookie_rankings.parquet")
    if rankings["player_id"].duplicated().any() or not rankings["rank"].is_unique:
        raise ValueError("Exposure-gated rankings have duplicate IDs or ranks")
    if not np.isfinite(rankings["blended_cold_start_prior"].to_numpy(dtype=float)).all():
        raise ValueError("Exposure-gated rankings contain non-finite priors")
    return manifest


def _validate_sources(
    *,
    target_season: str,
    source_season: str,
    draft_metadata: dict[str, object],
    exposure_metadata: dict[str, object],
    replacement_metadata: dict[str, object],
) -> None:
    if draft_metadata.get("target_season") != target_season:
        raise ValueError("Draft-prior target season does not match cold-start blend")
    if draft_metadata.get("training_last_season") != source_season:
        raise ValueError("Draft-prior study does not end with the source season")
    if exposure_metadata.get("target_season") != target_season:
        raise ValueError("Exposure-gate target season does not match cold-start blend")
    if exposure_metadata.get("training_last_season") != source_season:
        raise ValueError("Exposure-gate study does not end with the source season")
    if replacement_metadata.get("through_season") != source_season:
        raise ValueError("Replacement-token study must end with the source season")
    if replacement_metadata.get("replacement_share_cutoff") != 0.05:
        raise ValueError("Cold-start blend requires the canonical 5% replacement cutoff")


def _replacement_rapm(replacement_root: Path) -> float:
    payload = json.loads((replacement_root / "replacement_token_summary.json").read_text())
    value = float(payload["season_balanced_replacement_token_rapm"])
    if not np.isfinite(value):
        raise ValueError("Replacement-token summary has non-finite RAPM")
    return value


def _write_prior(
    *,
    target_season: str,
    source_season: str,
    rankings: pd.DataFrame,
    replacement_rapm: float,
    draft_root: Path,
    exposure_root: Path,
    replacement_root: Path,
    artifacts_dir: Path,
) -> ExposureGatedColdStartPrior:
    now = datetime.now(UTC)
    run_id = f"exposure-gated-cold-start-{target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "exposure_gated_cold_start" / target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        rankings_path = temporary / "revised_rookie_rankings.parquet"
        rankings.to_parquet(rankings_path, index=False)
        draft_run_id = json.loads((draft_root / "manifest.json").read_text())["run_id"]
        exposure_run_id = json.loads((exposure_root / "manifest.json").read_text())["run_id"]
        replacement_run_id = json.loads((replacement_root / "manifest.json").read_text())["run_id"]
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "exposure_gated_draft_and_replacement_cold_start_prior",
            "season": target_season,
            "target_season": target_season,
            "source_season": source_season,
            "scope": "first_nba_season_players_only",
            "player_count": len(rankings),
            "replacement_share_cutoff": 0.05,
            "replacement_rapm": replacement_rapm,
            "formula": "p_low_exposure * replacement_rapm + (1-p_low_exposure) * draft_prior",
            "target_outcomes_used_for_fit": False,
            "draft_prior_run_id": draft_run_id,
            "draft_prior_manifest_sha256": _sha256_file(draft_root / "manifest.json"),
            "exposure_gate_run_id": exposure_run_id,
            "exposure_gate_manifest_sha256": _sha256_file(exposure_root / "manifest.json"),
            "replacement_token_run_id": replacement_run_id,
            "replacement_token_manifest_sha256": _sha256_file(replacement_root / "manifest.json"),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {
                "filename": path.name,
                "row_count": len(rankings) if path.name == rankings_path.name else None,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        validate_exposure_gated_cold_start_prior(output)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return ExposureGatedColdStartPrior(
            run_dir=output,
            run_id=run_id,
            player_count=len(rankings),
            replacement_rapm=replacement_rapm,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _resolve_run(parent: Path, run_id: str | None) -> Path:
    if run_id is None:
        run_id = str(json.loads((parent / "latest.json").read_text())["run_id"])
    root = parent / run_id
    if not root.is_dir():
        raise ValueError(f"Model run not found: {root}")
    return root


def _previous_season(season: str) -> str:
    year = int(season[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an exposure-gated cold-start prior")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--draft-run-id")
    parser.add_argument("--exposure-run-id")
    parser.add_argument("--replacement-run-id")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    args = parser.parse_args()
    prior = build_exposure_gated_cold_start_prior(
        season=args.season,
        draft_run_id=args.draft_run_id,
        exposure_run_id=args.exposure_run_id,
        replacement_run_id=args.replacement_run_id,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(prior.run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(f"Exposure-gated cold-start prior: run={prior.run_dir}{tracking_text}")


if __name__ == "__main__":
    main()
