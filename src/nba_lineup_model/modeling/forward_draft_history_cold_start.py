"""Score an external draft class with completed forward exposure-gated RAPM state."""

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
    _prepare_features as prepare_exposure_features,
)
from nba_lineup_model.modeling.cold_start_exposure import fit_exposure_model
from nba_lineup_model.modeling.cold_start_exposure import (
    select_regularization as select_exposure_regularization,
)
from nba_lineup_model.modeling.draft_history_cold_start import (
    draft_history_profiles,
    render_draft_history_rankings_page,
    score_draft_history_profiles,
)
from nba_lineup_model.modeling.draft_prior import (
    _prepare_features as prepare_draft_features,
)
from nba_lineup_model.modeling.draft_prior import fit_draft_prior_model
from nba_lineup_model.modeling.draft_prior import (
    select_regularization as select_draft_regularization,
)
from nba_lineup_model.modeling.replacement_level import player_exposure_shares
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.season.schema import validate_season

DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_REPLACEMENT_SHARE_CUTOFF = 0.05


@dataclass(frozen=True)
class ForwardDraftHistoryColdStartRanking:
    """Immutable external-class ranking scored from a forward RAPM state."""

    run_dir: Path
    run_id: str
    ranking_count: int


def build_forward_draft_history_cold_start_ranking(
    *,
    season: str = "2026-27",
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardDraftHistoryColdStartRanking:
    """Score one direct NBA draft class using state completed before that season."""

    target_season = validate_season(season)
    source_season = _previous_season(target_season)
    artifacts_root = Path(artifacts_dir)
    forward_root = _latest_run(artifacts_root / "forward_exposure_gated_rapm" / source_season)
    panel = pd.read_parquet(player_season_panel_path)
    coefficients = pd.read_parquet(forward_root / "historical_player_coefficients.parquet")
    rate_source = forward_rookie_rate_training(panel, coefficients, through_season=source_season)
    rate_training = prepare_draft_features(rate_source)
    draft_regularization, _ = select_draft_regularization(
        rate_training, regularization_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    )
    draft_model = fit_draft_prior_model(rate_training, regularization=draft_regularization)

    exposure_source = forward_rookie_exposure_training(
        panel, through_season=source_season, analytical_dir=Path(analytical_dir)
    )
    exposure_training = prepare_exposure_features(exposure_source)
    exposure_training["is_replacement_candidate"] = exposure_training["exposure_share"].lt(
        DEFAULT_REPLACEMENT_SHARE_CUTOFF
    )
    exposure_c, _ = select_exposure_regularization(
        exposure_training, c_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    )
    exposure_model = fit_exposure_model(exposure_training, c=exposure_c)

    curated_root = Path(curated_dir)
    draft_path = curated_root / "draft_history" / target_season / "part-00000.parquet"
    if not draft_path.is_file():
        raise FileNotFoundError(
            f"Draft History for {target_season} is missing; run nba-fetch-draft-history first"
        )
    roster_path = curated_root / "team_rosters" / target_season / "part-00000.parquet"
    profiles = draft_history_profiles(
        pd.read_parquet(draft_path),
        season=target_season,
        roster_profiles=pd.read_parquet(roster_path) if roster_path.is_file() else None,
    )
    replacement_rapm = forward_replacement_rapm(forward_root, through_season=source_season)
    rankings = score_draft_history_profiles(
        profiles,
        draft_model=draft_model,
        exposure_model=exposure_model,
        draft_training=rate_training,
        exposure_training=exposure_training,
        replacement_rapm=replacement_rapm,
    )
    return _write_ranking(
        target_season=target_season,
        source_season=source_season,
        forward_root=forward_root,
        draft_path=draft_path,
        roster_path=roster_path if roster_path.is_file() else None,
        rankings=rankings,
        draft_regularization=draft_regularization,
        exposure_c=exposure_c,
        replacement_rapm=replacement_rapm,
        artifacts_dir=artifacts_root,
    )


def forward_rookie_rate_training(
    panel: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    through_season: str,
) -> pd.DataFrame:
    """Use only completed forward-RAPM rookie outcomes through a season cutoff."""

    cutoff = validate_season(through_season)
    required_panel = {
        "season",
        "player_id",
        "is_rookie",
        "rapm_possessions",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
        "age",
        "height_inches",
        "weight_pounds",
    }
    required_coefficients = {"season", "player_id", "rapm"}
    if missing := required_panel - set(panel):
        raise ValueError(f"Player panel missing forward draft-rate columns: {sorted(missing)}")
    if missing := required_coefficients - set(coefficients):
        raise ValueError(f"Forward coefficients missing columns: {sorted(missing)}")
    rookies = panel.loc[panel["is_rookie"].astype(bool) & panel["season"].le(cutoff)].copy()
    values = coefficients.loc[coefficients["season"].le(cutoff), ["season", "player_id", "rapm"]]
    output = rookies.drop(columns="rapm", errors="ignore").merge(
        values, on=["season", "player_id"], how="inner", validate="one_to_one"
    )
    if output.empty or not output["rapm_possessions"].gt(0).all():
        raise ValueError("Forward draft-rate training requires positive-exposure rookie outcomes")
    output["season_start_year"] = output["season"].str[:4].astype(int)
    return output.sort_values(["season", "player_id"], kind="stable").reset_index(drop=True)


def forward_rookie_exposure_training(
    panel: pd.DataFrame,
    *,
    through_season: str,
    analytical_dir: Path,
) -> pd.DataFrame:
    """Reconstruct realized rookie exposure only from completed regular seasons."""

    cutoff = validate_season(through_season)
    roster = panel.loc[panel["season"].le(cutoff) & panel["is_rookie"].astype(bool)].copy()
    rows = []
    for season in sorted(roster["season"].unique(), key=lambda value: int(value[:4])):
        exposure = player_exposure_shares(read_rapm_stints(str(season), analytical_dir))
        rookie_season = roster.loc[roster["season"].eq(season)]
        rows.append(
            rookie_season.merge(exposure, on="player_id", how="inner", validate="one_to_one")
        )
    output = pd.concat(rows, ignore_index=True)
    output["season_start_year"] = output["season"].str[:4].astype(int)
    if output.empty or not output["on_court_possessions"].gt(0).all():
        raise ValueError("Forward exposure training requires positive-exposure rookie outcomes")
    return output.sort_values(["season", "player_id"], kind="stable").reset_index(drop=True)


def forward_replacement_rapm(forward_root: Path | str, *, through_season: str) -> float:
    """Return the equal-season replacement-token mean available before the target."""

    tokens = pd.read_parquet(Path(forward_root) / "season_replacement_tokens.parquet")
    eligible = tokens.loc[tokens["season"].le(validate_season(through_season))]
    value = float(eligible["replacement_token_rapm"].mean())
    if eligible.empty or not np.isfinite(value):
        raise ValueError("Forward replacement-token history is empty or non-finite")
    return value


def _latest_run(root: Path) -> Path:
    latest = json.loads((root / "latest.json").read_text())
    output = root / str(latest["run_id"])
    if not output.is_dir():
        raise FileNotFoundError(f"Forward RAPM artifact does not exist: {output}")
    return output


def _previous_season(season: str) -> str:
    year = int(validate_season(season)[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _write_ranking(
    *,
    target_season: str,
    source_season: str,
    forward_root: Path,
    draft_path: Path,
    roster_path: Path | None,
    rankings: pd.DataFrame,
    draft_regularization: float,
    exposure_c: float,
    replacement_rapm: float,
    artifacts_dir: Path,
) -> ForwardDraftHistoryColdStartRanking:
    now = datetime.now(UTC)
    run_id = (
        f"forward-draft-history-cold-start-{target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = artifacts_dir / "forward_draft_history_cold_start" / target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        rankings.to_parquet(temporary / "drafted_rookie_rankings.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "forward_exposure_gated_external_draft_history_cold_start",
            "season": target_season,
            "training_last_season": source_season,
            "target_outcomes_used_for_fit": False,
            "ranking_count": len(rankings),
            "forward_run": str(forward_root),
            "draft_history_parquet": str(draft_path),
            "team_roster_parquet": str(roster_path) if roster_path is not None else None,
            "draft_regularization": draft_regularization,
            "exposure_c": exposure_c,
            "replacement_rapm": replacement_rapm,
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
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
        latest_tmp = root / "latest.json.tmp"
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(root / "latest.json")
        return ForwardDraftHistoryColdStartRanking(output, run_id, len(rankings))
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward-RAPM draft-history rookie rankings")
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--curated-dir", default=str(DEFAULT_CURATED_DIR))
    parser.add_argument("--player-season-panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--analytical-dir", default=str(DEFAULT_ANALYTICAL_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--render-docs-page", action="store_true")
    args = parser.parse_args()
    result = build_forward_draft_history_cold_start_ranking(
        season=args.season,
        curated_dir=args.curated_dir,
        player_season_panel_path=args.player_season_panel_path,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
    )
    if args.render_docs_page:
        render_draft_history_rankings_page(result.run_dir)
    print(f"run={result.run_dir}; rankings={result.ranking_count}")


if __name__ == "__main__":
    main()
