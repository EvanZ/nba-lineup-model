"""Materialize current-season observed lineups from reconciled CDN play-by-play."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.modeling.in_season_nail_monitor import (
    _fetch_s3_json,
    _load_json,
    _player_name_lookup,
    _sha256_bytes,
)
from nba_lineup_model.season.daily_data_artifact import resolve_daily_data_pointer
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.web_api.current_season_lineups import (
    DEFAULT_CURRENT_SEASON_LINEUPS_ROOT,
    build_player_team_exposure,
    publish_current_season_lineup_bundle,
)
from nba_lineup_model.web_api.inference import (
    PRESEASON_PREVIEW_SEASON,
    LineupEvaluator,
    build_observed_lineup_rankings,
)

DEFAULT_SCHEDULE_STATE_DIR = Path("/var/lib/nba-gestalt/schedule")
DEFAULT_COVERAGE_PATH = Path("/var/lib/nba-gestalt/cdn/coverage.json")
DEFAULT_CDN_S3_PREFIX = "s3://nba-gestalt-data/lake/raw/cdn/playbyplay"
DEFAULT_DAILY_DATA_ROOT = Path("/var/lib/nba-gestalt-dev/daily-data-nba-v2")




def materialize_daily_data_lineups(
    *,
    evaluator: LineupEvaluator,
    daily_data_root: Path | str = DEFAULT_DAILY_DATA_ROOT,
    output_root: Path | str = DEFAULT_CURRENT_SEASON_LINEUPS_ROOT,
) -> Path:
    """Build the serving lineup catalog from the verified NBA data chain.

    This path reads only the verified official NBA daily data chain.  Every stint
    and source checksum is loaded from the same fail-closed daily-data chain
    that fed the promoted daily-model checkpoint.
    """

    metadata = evaluator.current_season_metadata
    if not metadata:
        raise ValueError("A promoted current-season serving state is required")
    source_game_ids = sorted(
        str(game_id) for game_id in metadata.get("source_game_ids", [])
    )
    if not source_game_ids:
        raise ValueError("The serving state does not declare source-game coverage")
    through_date = date.fromisoformat(str(metadata["evidence_through_date"]))
    chain = _daily_data_chain(Path(daily_data_root).resolve())
    selected = [row for row in chain if row[0] <= through_date]
    if not selected or selected[-1][0] != through_date:
        raise ValueError("Daily-data chain does not reach the serving cutoff")

    stints: list[pd.DataFrame] = []
    checksums: dict[str, str] = {}
    for _game_date, bundle, manifest in selected:
        identity = manifest["identity"]
        stints.append(
            pd.read_parquet(
                _daily_component_path(
                    bundle, identity["components"], "observed_stints"
                )
            )
        )
        for game_id, sources in identity["game_sources"].items():
            checksums[str(game_id)] = str(sources["nba_play_by_play"]["sha256"])
    combined = pd.concat(stints, ignore_index=True)
    observed_ids = sorted(combined["game_id"].astype(str).unique().tolist())
    if observed_ids != source_game_ids or sorted(checksums) != source_game_ids:
        raise ValueError("Daily-data lineups do not match serving-state game coverage")

    rankings = evaluator._active_current_season_rankings()
    serving_player_ids = set(rankings["player_id"].astype(int))
    observed_player_ids = {
        int(player_id)
        for column in ("home_player_ids", "away_player_ids")
        for lineup in combined[column]
        for player_id in lineup
    }
    missing_players = sorted(observed_player_ids - serving_player_ids)
    if missing_players:
        raise ValueError(
            "Observed lineups contain players outside the serving universe: "
            + ", ".join(map(str, missing_players))
        )
    state = evaluator._preseason_lab_state()
    lineups = build_observed_lineup_rankings(
        season=PRESEASON_PREVIEW_SEASON,
        profiles=state.profiles,
        players=state.players,
        coefficients=state.coefficients,
        context_model=evaluator.context_model,
        stints=combined,
    )
    exposure = build_player_team_exposure(combined)
    return publish_current_season_lineup_bundle(
        lineups=lineups,
        stints=combined,
        player_team_exposure=exposure,
        serving_metadata=metadata,
        source_game_checksums=checksums,
        root=output_root,
    )


def _daily_data_chain(
    root: Path,
) -> list[tuple[date, Path, dict[str, Any]]]:
    pointer = _load_json(root / "latest.json")
    collection = resolve_daily_data_pointer(root, pointer)
    rows: list[tuple[date, Path, dict[str, Any]]] = []
    for selected_date, selected in collection["identity"]["dates"].items():
        bundle = (root / str(selected["bundle_path"])).resolve()
        try:
            bundle.relative_to(root)
        except ValueError as error:
            raise ValueError("Daily-data collection escapes its root") from error
        manifest = verify_daily_data_artifact(bundle)
        identity = manifest["identity"]
        game_date = date.fromisoformat(str(identity["game_date_pacific"]))
        if game_date.isoformat() != selected_date:
            raise ValueError("Selected daily data date does not match its artifact")
        rows.append((game_date, bundle, manifest))
    return rows


def _daily_component_path(
    bundle: Path, components: dict[str, Any], name: str
) -> Path:
    entry = components.get(name)
    raw_path = str(entry.get("path", "")).strip() if isinstance(entry, dict) else ""
    if not raw_path:
        raise ValueError(f"Daily-data component is missing: {name}")
    path = (bundle / raw_path).resolve()
    try:
        path.relative_to(bundle)
    except ValueError as error:
        raise ValueError(f"Daily-data component escapes its bundle: {name}") from error
    return path




def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-state-dir", type=Path, default=DEFAULT_SCHEDULE_STATE_DIR)
    parser.add_argument("--coverage-path", type=Path, default=DEFAULT_COVERAGE_PATH)
    parser.add_argument("--cdn-s3-prefix", default=DEFAULT_CDN_S3_PREFIX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CURRENT_SEASON_LINEUPS_ROOT)
    parser.add_argument("--daily-data-root", type=Path, default=DEFAULT_DAILY_DATA_ROOT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    evaluator = LineupEvaluator.from_latest_artifact(include_current_season_lineups=False)
    path = materialize_daily_data_lineups(
        evaluator=evaluator,
        daily_data_root=args.daily_data_root,
        output_root=args.output_root,
    )
    print(path)


if __name__ == "__main__":
    main()
