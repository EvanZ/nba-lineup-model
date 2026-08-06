"""Publish preseason drafted-rookie priors from official NBA Draft History."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.cold_start_exposure import (
    FEATURE_COLUMNS as EXPOSURE_FEATURE_COLUMNS,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    _feature_reference as exposure_feature_reference,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    _prepare_features as prepare_exposure_features,
)
from nba_lineup_model.modeling.cold_start_exposure import validate_cold_start_exposure_study
from nba_lineup_model.modeling.draft_prior import FEATURE_COLUMNS as DRAFT_FEATURE_COLUMNS
from nba_lineup_model.modeling.draft_prior import _feature_reference as draft_feature_reference
from nba_lineup_model.modeling.draft_prior import _prepare_features as prepare_draft_features
from nba_lineup_model.modeling.draft_prior import validate_draft_prior_study
from nba_lineup_model.modeling.replacement_token import validate_replacement_token_study
from nba_lineup_model.season.schema import validate_season

DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_DOCS_PAGE = Path("docs/models/2026-27-draft-history-rankings.md")


@dataclass(frozen=True)
class DraftHistoryColdStartRanking:
    """Immutable draft-history cold-start ranking outputs."""

    run_dir: Path
    run_id: str
    ranking_count: int


def build_draft_history_cold_start_ranking(
    *,
    season: str = "2026-27",
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    draft_run_id: str | None = None,
    exposure_run_id: str | None = None,
    replacement_run_id: str | None = None,
) -> DraftHistoryColdStartRanking:
    """Score one official draft class with frozen pre-season components.

    The source components are intentionally not refit here.  This command only
    scores the NBA's direct ``drafthistory`` response, so it remains usable
    before a new class appears in the player-season panel.
    """

    target_season = validate_season(season)
    target_year = int(target_season[:4])
    curated_root = Path(curated_dir)
    artifacts_root = Path(artifacts_dir)
    draft_root = _resolve_run(artifacts_root / "draft_prior" / "2025-26", draft_run_id)
    exposure_root = _resolve_run(
        artifacts_root / "cold_start_exposure" / "2025-26", exposure_run_id
    )
    replacement_root = _resolve_run(
        artifacts_root / "replacement_token" / "2024-25", replacement_run_id
    )
    draft_metadata = validate_draft_prior_study(draft_root)
    exposure_metadata = validate_cold_start_exposure_study(exposure_root)
    replacement_metadata = validate_replacement_token_study(replacement_root)
    _validate_temporal_boundary(
        target_year, draft_metadata, exposure_metadata, replacement_metadata
    )

    source_path = curated_root / "draft_history" / target_season / "part-00000.parquet"
    source_manifest_path = source_path.with_name("_manifest.json")
    if not source_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Draft History for {target_season} is missing; run nba-fetch-draft-history first"
        )
    draft_history = pd.read_parquet(source_path)
    roster_path = curated_root / "team_rosters" / target_season / "part-00000.parquet"
    roster_manifest_path = roster_path.with_name("_manifest.json")
    roster_profiles = pd.read_parquet(roster_path) if roster_path.is_file() else None
    profiles = draft_history_profiles(
        draft_history, season=target_season, roster_profiles=roster_profiles
    )
    rankings = score_draft_history_profiles(
        profiles,
        draft_model=joblib.load(draft_root / "model.joblib"),
        exposure_model=joblib.load(exposure_root / "model.joblib"),
        draft_training=pd.read_parquet(draft_root / "training_first_nba_season_players.parquet"),
        exposure_training=pd.read_parquet(
            exposure_root / "training_first_nba_season_exposure.parquet"
        ),
        replacement_rapm=_replacement_rapm(replacement_root),
    )
    return _write_ranking(
        target_season=target_season,
        source_path=source_path,
        source_manifest_path=source_manifest_path,
        roster_path=roster_path if roster_path.is_file() else None,
        roster_manifest_path=roster_manifest_path if roster_manifest_path.is_file() else None,
        rankings=rankings,
        draft_root=draft_root,
        exposure_root=exposure_root,
        replacement_root=replacement_root,
        artifacts_dir=artifacts_root,
    )


def draft_history_profiles(
    draft_history: pd.DataFrame,
    *,
    season: str,
    roster_profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create draft-only first-season profiles while preserving NBA player IDs."""

    required = {
        "season",
        "draft_year",
        "player_id",
        "player_name",
        "draft_round",
        "draft_round_pick",
        "draft_number",
        "draft_team_abbreviation",
        "affiliation",
    }
    missing = required - set(draft_history)
    if missing:
        raise ValueError(f"Draft History table missing columns: {sorted(missing)}")
    if (
        draft_history["player_id"].duplicated().any()
        or draft_history["draft_number"].duplicated().any()
    ):
        raise ValueError("Draft History profiles require unique players and draft slots")
    output = draft_history.copy()
    output["season"] = season
    output["season_start_year"] = int(season[:4])
    output["is_undrafted"] = False
    # DraftHistory has no player bio fields. A current CommonTeamRoster extract
    # fills these when available; the frozen models impute any remaining gaps.
    output["age"] = np.nan
    output["height_inches"] = np.nan
    output["weight_pounds"] = np.nan
    output["listed_position"] = "Unknown"
    if roster_profiles is not None:
        output = _join_roster_profiles(output, roster_profiles)
    return output


def score_draft_history_profiles(
    profiles: pd.DataFrame,
    *,
    draft_model: object,
    exposure_model: object,
    draft_training: pd.DataFrame,
    exposure_training: pd.DataFrame,
    replacement_rapm: float,
) -> pd.DataFrame:
    """Apply the frozen rate and gate models to draft-only profiles."""

    draft_features = prepare_draft_features(
        profiles, reference=draft_feature_reference(draft_training)
    )
    exposure_features = prepare_exposure_features(
        profiles, reference=exposure_feature_reference(exposure_training)
    )
    output = profiles.copy()
    output["draft_rate_rapm"] = np.asarray(
        draft_model.predict(draft_features.loc[:, DRAFT_FEATURE_COLUMNS]), dtype=float
    )
    output["low_exposure_probability"] = np.asarray(
        exposure_model.predict_proba(exposure_features.loc[:, EXPOSURE_FEATURE_COLUMNS])[:, 1],
        dtype=float,
    )
    output["replacement_rapm"] = float(replacement_rapm)
    output["cold_start_rapm_prior"] = (
        output["low_exposure_probability"] * output["replacement_rapm"]
        + (1.0 - output["low_exposure_probability"]) * output["draft_rate_rapm"]
    )
    output = output.sort_values(
        ["cold_start_rapm_prior", "draft_number", "player_name"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    output.insert(0, "rank", np.arange(1, len(output) + 1, dtype=int))
    return output.loc[
        :,
        [
            "rank",
            "player_id",
            "player_name",
            "roster_team_id",
            "roster_team_abbreviation",
            "listed_position",
            "age",
            "height_inches",
            "weight_pounds",
            "draft_team_abbreviation",
            "affiliation",
            "draft_round",
            "draft_round_pick",
            "draft_number",
            "draft_rate_rapm",
            "low_exposure_probability",
            "replacement_rapm",
            "cold_start_rapm_prior",
        ],
    ]


def render_draft_history_rankings_page(
    run_dir: Path | str,
    *,
    page_path: Path | str = DEFAULT_DOCS_PAGE,
) -> Path:
    """Render the sortable 2026-27 drafted-rookie table from an immutable run."""

    root = Path(run_dir)
    metadata = validate_draft_history_cold_start_ranking(root)
    rankings = pd.read_parquet(root / "drafted_rookie_rankings.parquet")
    forward_source = metadata.get("forward_run")
    source_line = (
        f"- Forward-state source: `{forward_source}`." if forward_source is not None else None
    )
    lines = [
        "# 2026-27 Drafted Rookie Rankings",
        "",
        "These are preseason cold-start RAPM priors for the 60 players in the "
        "official 2026 NBA Draft History response. They are not retrospective "
        "player evaluations and they do not include undrafted free agents, two-way "
        "signings, or other late entrants.",
        "",
        "The table is sortable by every column. The prior continuously blends a "
        "draft-profile RAPM rate with the chance that a first-year player finishes "
        "below 5% of team possession opportunities, using the pooled replacement "
        "token for that low-exposure state.",
        "",
        "## Inputs",
        "",
        f"- Draft class: official NBA `drafthistory` response for {metadata['season']}.",
        f"- Draft-rate and exposure-gate training end at {metadata['training_last_season']}.",
        f"- Replacement token: {metadata['replacement_rapm']:+.2f} RAPM.",
        *([source_line] if source_line is not None else []),
        "- Active roster profiles come from direct NBA `commonteamroster` responses when "
        "the player is on a listed team roster. Any remaining missing bio field uses its "
        "frozen historical drafted-player reference profile.",
        "",
        "See [Exposure-Gated Cold-Start Prior](exposure-gated-cold-start.md) for the "
        "underlying blend and [Fetch Draft History](../guides/fetch-draft-history.md) "
        "for the direct-source ingestion contract.",
        "",
        "## Rankings",
        "",
        (
            "| Rank | Cold-start RAPM prior | Player | Roster team | Pos. | Age | Ht. | Wt. | "
            "Pick | Draft rate | "
            "P(low exposure) | Affiliation |"
        ),
        "| ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rankings.itertuples(index=False):
        affiliation = "" if pd.isna(row.affiliation) else str(row.affiliation).replace("|", "\\|")
        team = row.roster_team_abbreviation
        if pd.isna(team):
            team = row.draft_team_abbreviation
        team = "" if pd.isna(team) else str(team)
        position = "" if pd.isna(row.listed_position) else str(row.listed_position)
        age = "" if pd.isna(row.age) else f"{row.age:.0f}"
        height = "" if pd.isna(row.height_inches) else _format_height(int(row.height_inches))
        weight = "" if pd.isna(row.weight_pounds) else f"{row.weight_pounds:.0f}"
        lines.append(
            "| "
            f"{row.rank} | {row.cold_start_rapm_prior:+.2f} | {row.player_name} | {team} | "
            f"{position} | {age} | {height} | {weight} | {row.draft_number} | "
            f"{row.draft_rate_rapm:+.2f} | {row.low_exposure_probability:.1%} | "
            f"{affiliation} |"
        )
    destination = Path(page_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return destination


def validate_draft_history_cold_start_ranking(run_dir: Path | str) -> dict[str, object]:
    """Validate artifact hashes and the frozen pre-season source boundary."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    required = {"drafted_rookie_rankings.parquet", "metadata.json"}
    records = {record["filename"]: record for record in manifest["artifacts"]}
    if not required <= set(records):
        raise ValueError("Draft-history cold-start ranking is missing required files")
    if manifest.get("target_outcomes_used_for_fit") is not False:
        raise ValueError("Draft-history ranking used target outcomes")
    if int(str(manifest["training_last_season"])[:4]) >= int(str(manifest["season"])[:4]):
        raise ValueError("Draft-history ranking has no pre-season temporal boundary")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"Draft-history ranking changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Draft-history ranking hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"Draft-history ranking row count changed: {filename}")
    return manifest


def _resolve_run(root: Path, run_id: str | None) -> Path:
    if run_id is not None:
        path = root / run_id
    else:
        latest = json.loads((root / "latest.json").read_text())
        path = root / str(latest["run_id"])
    if not path.is_dir():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    return path


def _join_roster_profiles(
    draft_history: pd.DataFrame, roster_profiles: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "player_id",
        "team_id",
        "team_abbreviation",
        "listed_position",
        "age",
        "height_inches",
        "weight_pounds",
    }
    missing = required - set(roster_profiles)
    if missing:
        raise ValueError(f"Team roster profiles missing columns: {sorted(missing)}")
    roster = (
        roster_profiles.loc[:, sorted(required)]
        .copy()
        .rename(
            columns={
                "team_id": "roster_team_id",
                "team_abbreviation": "roster_team_abbreviation",
            }
        )
    )
    roster["player_id"] = roster["player_id"].astype("string")
    duplicates = roster["player_id"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = sorted(roster.loc[duplicates, "player_id"].unique())
        raise ValueError(f"Team roster profiles duplicate player IDs: {duplicate_ids[:5]}")
    output = draft_history.merge(
        roster,
        on="player_id",
        how="left",
        suffixes=("", "_roster"),
        validate="one_to_one",
    )
    for column in ("listed_position", "age", "height_inches", "weight_pounds"):
        roster_column = f"{column}_roster"
        output[column] = output[roster_column].combine_first(output[column])
        output = output.drop(columns=roster_column)
    return output


def _validate_temporal_boundary(
    target_year: int,
    draft_metadata: dict[str, object],
    exposure_metadata: dict[str, object],
    replacement_metadata: dict[str, object],
) -> None:
    for name, metadata in {
        "draft rate": draft_metadata,
        "exposure gate": exposure_metadata,
        "replacement token": replacement_metadata,
    }.items():
        last_year = int(_training_last_season(metadata)[:4])
        if last_year >= target_year:
            raise ValueError(f"{name} training reaches target season")


def _replacement_rapm(root: Path) -> float:
    summary = json.loads((root / "replacement_token_summary.json").read_text())
    value = float(summary["season_balanced_replacement_token_rapm"])
    if not np.isfinite(value):
        raise ValueError("Replacement-token estimate is not finite")
    return value


def _training_last_season(metadata: dict[str, object]) -> str:
    for key in ("training_last_season", "through_season", "season"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    raise ValueError("Source artifact has no season cutoff")


def _format_height(height_inches: int) -> str:
    return f"{height_inches // 12}-{height_inches % 12}"


def _write_ranking(
    *,
    target_season: str,
    source_path: Path,
    source_manifest_path: Path,
    roster_path: Path | None,
    roster_manifest_path: Path | None,
    rankings: pd.DataFrame,
    draft_root: Path,
    exposure_root: Path,
    replacement_root: Path,
    artifacts_dir: Path,
) -> DraftHistoryColdStartRanking:
    now = datetime.now(UTC)
    run_id = f"draft-history-cold-start-{target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "draft_history_cold_start" / target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        rankings.to_parquet(temporary / "drafted_rookie_rankings.parquet", index=False)
        draft_metadata = json.loads((draft_root / "metadata.json").read_text())
        exposure_metadata = json.loads((exposure_root / "metadata.json").read_text())
        replacement_metadata = json.loads((replacement_root / "metadata.json").read_text())
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "draft_history_exposure_gated_cold_start_prior",
            "season": target_season,
            "ranking_count": len(rankings),
            "training_last_season": max(
                _training_last_season(draft_metadata),
                _training_last_season(exposure_metadata),
                _training_last_season(replacement_metadata),
                key=lambda value: int(value[:4]),
            ),
            "target_outcomes_used_for_fit": False,
            "source": {
                "draft_history_parquet": str(source_path),
                "draft_history_manifest_sha256": _sha256_file(source_manifest_path),
                "team_roster_parquet": str(roster_path) if roster_path is not None else None,
                "team_roster_manifest_sha256": (
                    _sha256_file(roster_manifest_path) if roster_manifest_path is not None else None
                ),
                "draft_rate_run": str(draft_root),
                "exposure_gate_run": str(exposure_root),
                "replacement_token_run": str(replacement_root),
            },
            "replacement_rapm": float(rankings["replacement_rapm"].iloc[0]),
            "roster_profile_match_count": int(rankings["height_inches"].notna().sum()),
            "missing_bio_strategy": (
                "use frozen historical drafted-player reference values for missing bio fields"
            ),
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {
                "filename": path.name,
                "row_count": len(rankings) if path.suffix == ".parquet" else None,
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
        validate_draft_history_cold_start_ranking(output)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return DraftHistoryColdStartRanking(output, run_id, len(rankings))
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build draft-history cold-start rookie rankings")
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--draft-run-id")
    parser.add_argument("--exposure-run-id")
    parser.add_argument("--replacement-run-id")
    parser.add_argument(
        "--render-docs-page",
        action="store_true",
        help="Render docs/models/2026-27-draft-history-rankings.md from the new artifact",
    )
    args = parser.parse_args()
    ranking = build_draft_history_cold_start_ranking(
        season=args.season,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        draft_run_id=args.draft_run_id,
        exposure_run_id=args.exposure_run_id,
        replacement_run_id=args.replacement_run_id,
    )
    if args.render_docs_page:
        render_draft_history_rankings_page(ranking.run_dir)
    print(f"run={ranking.run_dir}; rankings={ranking.ranking_count}")


if __name__ == "__main__":
    main()
