"""Full frozen roster-squash test for the incumbent-team-strength FCM candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_CURATED_DIR,
    PROMOTED_AVAILABILITY_CONFIG,
    build_availability_season_summary,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    DEFAULT_PLAYER_PANEL_PATH,
    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    PROMOTED_CONDITIONAL_MINUTES_CONFIG,
    ColdStartConditionalMinutesConfig,
    attach_cold_start_biographies,
    normalize_opening_roster_minutes,
)
from nba_lineup_model.rotation.forward_conditional_team_strength import (
    DEFAULT_FROZEN_SEASONS,
    DEFAULT_TUNING_SEASONS,
    TEAM_STRENGTH_COLUMN,
    build_incumbent_team_strength_summary,
)
from nba_lineup_model.rotation.forward_nail_roster_squash import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_INITIAL_ROTATION_SIZE,
    _season_allocation_targets,
    assert_full_regular_season_minute_coverage,
    build_forward_conditional_raw_weights,
    build_static_opening_roster_bios,
)
from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    read_preseason_roster_candidates,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_ARTIFACTS_DIR,
    read_regular_game_minutes,
)
from nba_lineup_model.rotation.l20_preseason_minute_share_persistence import (
    _assert_distribution,
    _validate_game_minutes,
)
from nba_lineup_model.rotation.l20_preseason_nail_forecast_minute_share import (
    DEFAULT_MODEL_RUN_DIR,
)

MODEL_NAME = "forward_conditional_team_strength_roster_squash"
MODEL_VERSION = "v0.1"
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_912
_PAIRED_METRICS = (
    "allocation_total_variation",
    "brier_score",
    "player_share_mae",
    "player_share_mse",
    "actual_top_rotation_overlap",
)


@dataclass(frozen=True)
class TeamStrengthRosterInputs:
    """A frozen opening roster and its full regular-season allocation label."""

    season: str
    opening_roster: pd.DataFrame
    static_roster_bios: pd.DataFrame
    target_game_minutes: pd.DataFrame


@dataclass(frozen=True)
class TeamStrengthRosterSquashRun:
    """Immutable output location for one full candidate replay."""

    run_dir: Path
    run_id: str


def build_team_strength_roster_inputs(
    season: str,
    *,
    panel: pd.DataFrame,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> TeamStrengthRosterInputs:
    """Construct target inputs from preseason roster fields and held-out labels."""

    opening = read_preseason_roster_candidates(season, curated_dir=curated_dir)
    return TeamStrengthRosterInputs(
        season=season,
        opening_roster=opening,
        static_roster_bios=build_static_opening_roster_bios(
            season, opening, panel=panel, curated_dir=curated_dir
        ),
        target_game_minutes=read_regular_game_minutes(season, curated_dir=curated_dir),
    )


def attach_incumbent_team_strength_to_roster(
    inputs: TeamStrengthRosterInputs,
    *,
    team_strength_diagnostics: pd.DataFrame,
) -> TeamStrengthRosterInputs:
    """Attach the already-preseason-built team feature to every opening player."""

    team_values = team_strength_diagnostics.loc[
        team_strength_diagnostics["season"].astype(str).eq(inputs.season),
        ["team_id", TEAM_STRENGTH_COLUMN, "league_strength_fallback"],
    ].copy()
    if team_values.empty:
        raise ValueError(f"Team-strength diagnostics lack {inputs.season}")
    if team_values.duplicated("team_id").any():
        raise ValueError(f"Team-strength diagnostics duplicate teams for {inputs.season}")
    fallback_values = team_values["league_strength_fallback"].dropna().unique()
    if len(fallback_values) != 1:
        raise ValueError(f"Team-strength diagnostics lack one league fallback for {inputs.season}")
    fallback = float(fallback_values[0])
    bios = inputs.static_roster_bios.merge(
        team_values.drop(columns="league_strength_fallback"),
        on="team_id",
        how="left",
        validate="many_to_one",
    )
    bios[TEAM_STRENGTH_COLUMN] = pd.to_numeric(
        bios[TEAM_STRENGTH_COLUMN], errors="coerce"
    ).fillna(fallback)
    return TeamStrengthRosterInputs(
        season=inputs.season,
        opening_roster=inputs.opening_roster,
        static_roster_bios=bios,
        target_game_minutes=inputs.target_game_minutes,
    )


def apply_production_roster_squash(
    raw_predictions: pd.DataFrame,
    *,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> pd.DataFrame:
    """Apply the unchanged production top-15 gate and team normalization input."""

    required = {"team_id", "player_id", "player_name", "raw_expected_total_minutes"}
    missing = sorted(required - set(raw_predictions))
    if missing:
        raise ValueError("Raw predictions lack required columns: " + ", ".join(missing))
    if initial_rotation_size <= 0:
        raise ValueError("Initial rotation size must be positive")
    output = raw_predictions.copy()
    selected_ids: set[int] = set()
    for _, team in output.groupby("team_id", sort=False):
        selected_ids.update(
            team.sort_values(
                ["raw_expected_total_minutes", "player_name", "player_id"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(initial_rotation_size)["player_id"]
            .astype(int)
            .tolist()
        )
    output["is_rotation_candidate"] = output["player_id"].astype(int).isin(selected_ids)
    output.loc[~output["is_rotation_candidate"], "raw_expected_total_minutes"] = 0.0
    return output


def evaluate_team_strength_roster_squash(
    inputs: TeamStrengthRosterInputs,
    *,
    availability_summary: pd.DataFrame,
    cold_start_config: ColdStartConditionalMinutesConfig,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score an all-season opening-roster allocation using the selected FCM branch."""

    _validate_game_minutes(inputs.target_game_minutes, label="target")
    raw = build_forward_conditional_raw_weights(
        inputs,
        availability_summary=availability_summary,
        cold_start_config=cold_start_config,
    )
    squashed = apply_production_roster_squash(
        raw, initial_rotation_size=initial_rotation_size
    )
    opening = inputs.opening_roster.loc[:, ["team_id", "team", "player_id", "player_name"]]
    normalized = normalize_opening_roster_minutes(squashed, opening_roster=opening)
    predicted = opening.merge(
        squashed,
        on=["team_id", "team", "player_id", "player_name"],
        how="left",
        validate="one_to_one",
    ).merge(
        normalized.loc[:, ["player_id", "projected_minute_share", "projected_total_minutes"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    target_by_team = {
        target.team_id: target
        for target in _season_allocation_targets(inputs.target_game_minutes, team_games=None)
    }
    records: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for team_id, team_predictions in predicted.groupby("team_id", sort=True):
        target = target_by_team.get(int(team_id))
        if target is None:
            raise ValueError(f"Target allocation missing team {team_id}")
        predicted_map = team_predictions.set_index("player_id")["projected_minute_share"].to_dict()
        actual_map = target.player_shares
        player_ids = sorted(set(predicted_map) | set(actual_map))
        actual_values = np.asarray(
            [float(actual_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        predicted_values = np.asarray(
            [float(predicted_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        _assert_distribution(actual_values, label=f"actual for {target.team}")
        _assert_distribution(predicted_values, label=f"prediction for {target.team}")
        absolute_error = np.abs(actual_values - predicted_values)
        squared_error = np.square(actual_values - predicted_values)
        actual_top_ids = {
            player_id
            for player_id, _share in sorted(
                actual_map.items(), key=lambda item: (-item[1], item[0])
            )[:initial_rotation_size]
        }
        selected_ids = set(
            team_predictions.loc[team_predictions["is_rotation_candidate"], "player_id"]
            .astype(int)
            .tolist()
        )
        zero_probability_active = (actual_values > 0.0) & (predicted_values <= 0.0)
        cross_entropy = float("inf")
        if not zero_probability_active.any():
            positive_actual = actual_values > 0.0
            cross_entropy = float(
                -(actual_values[positive_actual] * np.log(predicted_values[positive_actual])).sum()
            )
        metric_rows.append(
            {
                "season": inputs.season,
                "team_id": int(team_id),
                "team": target.team,
                "target_game_count": target.target_game_count,
                "allocation_total_variation": float(0.5 * absolute_error.sum()),
                "brier_score": float(squared_error.sum()),
                "player_share_mae": float(absolute_error.mean()),
                "player_share_mse": float(squared_error.mean()),
                "cross_entropy": cross_entropy,
                "has_zero_probability_active_player": bool(zero_probability_active.any()),
                "actual_top_rotation_overlap": float(
                    len(actual_top_ids & selected_ids) / max(len(actual_top_ids), 1)
                ),
                "selected_rotation_size": int(len(selected_ids)),
            }
        )
        player_rows = team_predictions.set_index("player_id")
        for player_id, actual_share, predicted_share in zip(
            player_ids, actual_values, predicted_values, strict=True
        ):
            source = player_rows.loc[player_id] if player_id in player_rows.index else None
            records.append(
                {
                    "season": inputs.season,
                    "team_id": int(team_id),
                    "team": target.team,
                    "player_id": int(player_id),
                    "player_name": (
                        str(source["player_name"]) if source is not None else str(player_id)
                    ),
                    "actual_minute_share": float(actual_share),
                    "predicted_minute_share": float(predicted_share),
                    "absolute_error": float(abs(actual_share - predicted_share)),
                    "raw_expected_total_minutes": (
                        float(source["raw_expected_total_minutes"])
                        if source is not None
                        else 0.0
                    ),
                    TEAM_STRENGTH_COLUMN: (
                        float(source[TEAM_STRENGTH_COLUMN])
                        if source is not None and TEAM_STRENGTH_COLUMN in source.index
                        else np.nan
                    ),
                    "was_selected_rotation": (
                        bool(source["is_rotation_candidate"]) if source is not None else False
                    ),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(metric_rows)


def run_forward_conditional_team_strength_roster_squash(
    *,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    nail_run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> TeamStrengthRosterSquashRun:
    """Evaluate the FCM candidate through the unchanged production roster squash."""

    if not tuning_seasons or not frozen_seasons:
        raise ValueError("Both tuning and frozen seasons are required")
    if set(tuning_seasons) & set(frozen_seasons):
        raise ValueError("Tuning and frozen seasons must not overlap")
    all_targets = tuple(dict.fromkeys((*tuning_seasons, *frozen_seasons)))
    final_year = max(int(season[:4]) for season in all_targets)
    history_seasons = tuple(
        f"{year}-{str(year + 1)[-2:]}" for year in range(2015, final_year + 1)
    )
    print(
        f"[Team strength squash] loading history through {history_seasons[-1]}", flush=True
    )
    base_summary = attach_cold_start_biographies(
        build_availability_season_summary(history_seasons, curated_dir=curated_dir)
    )
    panel = pd.read_parquet(player_panel_path)
    print("[Team strength squash] building preseason incumbent team strength", flush=True)
    augmented_summary, diagnostics = build_incumbent_team_strength_summary(
        base_summary,
        panel=panel,
        seasons=history_seasons,
        curated_dir=curated_dir,
        run_dir=nail_run_dir,
    )
    candidate_config = ColdStartConditionalMinutesConfig(
        alpha=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG.alpha,
        draft_pick_half_life=None,
        include_incumbent_team_strength=True,
    )
    inputs_by_season: dict[str, TeamStrengthRosterInputs] = {}
    for season in all_targets:
        print(f"[Team strength squash] building frozen inputs for {season}", flush=True)
        inputs = build_team_strength_roster_inputs(season, panel=panel, curated_dir=curated_dir)
        assert_full_regular_season_minute_coverage(
            season, inputs.target_game_minutes, catalog_path=catalog_path
        )
        inputs_by_season[season] = attach_incumbent_team_strength_to_roster(
            inputs, team_strength_diagnostics=diagnostics
        )
    candidate_results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    control_results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for season in frozen_seasons:
        print(f"[Team strength squash] freezing {season} (candidate)", flush=True)
        candidate_results.append(
            evaluate_team_strength_roster_squash(
                inputs_by_season[season],
                availability_summary=augmented_summary,
                cold_start_config=candidate_config,
                initial_rotation_size=initial_rotation_size,
            )
        )
        print(f"[Team strength squash] freezing {season} (production control)", flush=True)
        control_results.append(
            evaluate_team_strength_roster_squash(
                inputs_by_season[season],
                availability_summary=base_summary,
                cold_start_config=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
                initial_rotation_size=initial_rotation_size,
            )
        )
    candidate_predictions = pd.concat(
        [result[0] for result in candidate_results], ignore_index=True
    )
    candidate_metrics = pd.concat([result[1] for result in candidate_results], ignore_index=True)
    control_predictions = pd.concat([result[0] for result in control_results], ignore_index=True)
    control_metrics = pd.concat([result[1] for result in control_results], ignore_index=True)
    comparison = candidate_metrics.loc[:, ["season", "team_id", *_PAIRED_METRICS]].merge(
        control_metrics.loc[:, ["season", "team_id", *_PAIRED_METRICS]],
        on=["season", "team_id"],
        suffixes=("_candidate", "_control"),
        validate="one_to_one",
    )
    for column in _PAIRED_METRICS:
        comparison[f"{column}_delta"] = (
            comparison[f"{column}_candidate"] - comparison[f"{column}_control"]
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"forward-conditional-team-strength-roster-squash-{timestamp}-{uuid4().hex[:7]}"
    output_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    base_summary.to_parquet(output_dir / "base_player_season_summary.parquet", index=False)
    augmented_summary.to_parquet(
        output_dir / "team_strength_player_season_summary.parquet", index=False
    )
    diagnostics.to_parquet(output_dir / "incumbent_team_strength.parquet", index=False)
    candidate_predictions.to_parquet(output_dir / "frozen_predictions.parquet", index=False)
    candidate_metrics.to_parquet(output_dir / "frozen_team_metrics.parquet", index=False)
    control_predictions.to_parquet(output_dir / "frozen_control_predictions.parquet", index=False)
    control_metrics.to_parquet(output_dir / "frozen_control_team_metrics.parquet", index=False)
    comparison.to_parquet(output_dir / "frozen_comparison.parquet", index=False)
    paired_team_bootstrap(comparison).to_parquet(
        output_dir / "paired_team_bootstrap.parquet", index=False
    )
    _summarize_metrics(candidate_metrics, frozen_seasons=frozen_seasons).to_parquet(
        output_dir / "frozen_summary.parquet", index=False
    )
    _summarize_metrics(control_metrics, frozen_seasons=frozen_seasons).to_parquet(
        output_dir / "frozen_control_summary.parquet", index=False
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "evaluation_target": "all regular-season games for each team",
                "initial_rotation_size": initial_rotation_size,
                "paired_bootstrap": {
                    "draws": DEFAULT_BOOTSTRAP_DRAWS,
                    "seed": DEFAULT_BOOTSTRAP_SEED,
                    "resampling_unit": "teams stratified within frozen season",
                },
                "candidate_cold_start_config": asdict(candidate_config),
                "control_cold_start_config": asdict(
                    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG
                ),
                "availability_config": asdict(PROMOTED_AVAILABILITY_CONFIG),
                "conditional_minutes_config": asdict(PROMOTED_CONDITIONAL_MINUTES_CONFIG),
                "contract": (
                    "The candidate differs from production only in the rookie conditional-MPG "
                    "prior: a rookie receives the raw-minute-weighted preseason NAIL strength "
                    "of returning non-rookie teammates. Availability, raw expected-minutes "
                    "formula, top-15 selection, and 240-minute normalization are identical. "
                    "There is no NAIL, draft, or rookie adjustment in the squash itself."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return TeamStrengthRosterSquashRun(run_dir=output_dir, run_id=run_id)


def paired_team_bootstrap(
    comparison: pd.DataFrame,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Return paired team-bootstrap intervals, preserving frozen-season balance."""

    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    required = {"season", *(f"{metric}_delta" for metric in _PAIRED_METRICS)}
    missing = sorted(required - set(comparison))
    if missing:
        raise ValueError("Comparison lacks paired bootstrap columns: " + ", ".join(missing))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    scopes = [
        ("pooled_frozen", comparison),
        *(
            (str(season), frame)
            for season, frame in comparison.groupby("season", sort=True)
        ),
    ]
    for scope, source in scopes:
        by_season = [
            frame.loc[:, [f"{metric}_delta" for metric in _PAIRED_METRICS]].to_numpy(float)
            for _, frame in source.groupby("season", sort=True)
        ]
        totals = np.zeros((draws, len(_PAIRED_METRICS)), dtype=float)
        total_teams = 0
        for values in by_season:
            if len(values) == 0:
                continue
            sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
            totals += sampled.sum(axis=1)
            total_teams += len(values)
        if total_teams == 0:
            raise ValueError(f"Paired bootstrap has no teams for {scope}")
        means = totals / total_teams
        for index, metric in enumerate(_PAIRED_METRICS):
            draw_values = means[:, index]
            lower_is_better = metric != "actual_top_rotation_overlap"
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "point_estimate_candidate_minus_control": float(
                        source[f"{metric}_delta"].mean()
                    ),
                    "ci_lower": float(np.quantile(draw_values, 0.025)),
                    "ci_upper": float(np.quantile(draw_values, 0.975)),
                    "candidate_better_probability": float(
                        (draw_values < 0.0).mean()
                        if lower_is_better
                        else (draw_values > 0.0).mean()
                    ),
                    "draws": int(draws),
                    "resampling_unit": "teams stratified within frozen season",
                }
            )
    return pd.DataFrame(rows)


def _summarize_metrics(
    metrics: pd.DataFrame,
    *,
    frozen_seasons: tuple[str, ...],
) -> pd.DataFrame:
    """Return per-season and pooled team-allocation summaries."""

    rows: list[dict[str, object]] = []
    for season in (*frozen_seasons, "pooled_frozen"):
        source = metrics if season == "pooled_frozen" else metrics.loc[metrics["season"].eq(season)]
        rows.append(
            {
                "season": season,
                "evaluated_teams": int(len(source)),
                "mean_allocation_total_variation": float(
                    source["allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(source["brier_score"].mean()),
                "player_share_mae": float(source["player_share_mae"].mean()),
                "player_share_rmse": float(np.sqrt(source["player_share_mse"].mean())),
                "mean_actual_top_rotation_overlap": float(
                    source["actual_top_rotation_overlap"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the full frozen incumbent-team-strength roster-squash replay."""

    parser = argparse.ArgumentParser(
        description="Evaluate incumbent team strength through the production roster squash"
    )
    parser.add_argument("--tuning-seasons", nargs="+", default=list(DEFAULT_TUNING_SEASONS))
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--nail-run-dir", type=Path, default=DEFAULT_MODEL_RUN_DIR)
    args = parser.parse_args()
    run = run_forward_conditional_team_strength_roster_squash(
        tuning_seasons=tuple(args.tuning_seasons),
        frozen_seasons=tuple(args.frozen_seasons),
        player_panel_path=args.player_panel_path,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        catalog_path=args.catalog_path,
        nail_run_dir=args.nail_run_dir,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
