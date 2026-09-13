"""Materialize compact observed and forward rotation histories for player bios."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    PROMOTED_AVAILABILITY_CONFIG,
    build_availability_season_summary,
    predict_availability_season,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    PROMOTED_CONDITIONAL_MINUTES_CONFIG,
    attach_cold_start_biographies,
    predict_conditional_minutes_season,
)
from nba_lineup_model.web_api.inference import (
    MODEL_ARTIFACT,
    LineupEvaluator,
    player_rotation_history_path,
)
from nba_lineup_model.web_api.win_projections import read_win_projection_cache

DEFAULT_HISTORY_START_SEASON = "2015-16"
DEFAULT_COMPLETED_SEASON = "2025-26"

_OUTPUT_COLUMNS = (
    "season",
    "season_start_year",
    "player_id",
    "player_name",
    "age",
    "actual_available_games",
    "known_roster_games",
    "actual_availability_share",
    "injury_or_illness_games",
    "rest_games",
    "actual_minutes_per_available_game",
    "actual_total_minutes",
    "predicted_available_share",
    "predicted_minutes_per_available_game",
    "projected_total_minutes",
    "is_preseason_forecast",
)


def build_player_rotation_history_cache(
    *,
    evaluator: LineupEvaluator | None = None,
    output_path: Path | str | None = None,
    history_start_season: str = DEFAULT_HISTORY_START_SEASON,
    completed_season: str = DEFAULT_COMPLETED_SEASON,
) -> Path:
    """Write the release-scoped player-season cache consumed by player bios.

    Historical rows contain both the realized labels and forecasts made before
    that season. The final forecast row is copied from the published FCM v0.3
    Win Projections payload so it exactly matches the product's current
    planning surface.
    """

    state = evaluator or LineupEvaluator.from_latest_artifact()
    seasons = _season_range(history_start_season, completed_season)
    summary = attach_cold_start_biographies(build_availability_season_summary(seasons))
    payload = read_win_projection_cache(model_artifact=MODEL_ARTIFACT, run_id=state.run_id)
    minutes = payload.get("minutes")
    if not isinstance(minutes, dict):
        raise ValueError("Published win projection cache lacks a minutes payload")
    history = build_player_rotation_history(
        summary,
        forecast_players=pd.DataFrame(minutes.get("players", [])),
        forecast_season=str(minutes.get("season", "")),
    )
    path = (
        Path(output_path)
        if output_path is not None
        else player_rotation_history_path(MODEL_ARTIFACT, state.run_id)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(path, index=False)
    return path


def build_player_rotation_history(
    summary: pd.DataFrame,
    *,
    forecast_players: pd.DataFrame | None = None,
    forecast_season: str | None = None,
) -> pd.DataFrame:
    """Combine observed availability/FCM targets with frozen preseason forecasts."""

    _validate_summary(summary)
    observed = summary.loc[
        :,
        [
            "season",
            "season_start_year",
            "player_id",
            "player_name",
            "age",
            "available_games",
            "known_player_games",
            "available_share",
            "injury_or_illness_games",
            "rest_games",
            "minutes_per_available_game",
            "total_nba_minutes",
        ],
    ].copy()
    observed = observed.rename(
        columns={
            "available_games": "actual_available_games",
            "known_player_games": "known_roster_games",
            "available_share": "actual_availability_share",
            "minutes_per_available_game": "actual_minutes_per_available_game",
            "total_nba_minutes": "actual_total_minutes",
        }
    )
    predictions = _historical_preseason_predictions(summary)
    output = observed.merge(
        predictions,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    output["projected_total_minutes"] = np.nan
    output["is_preseason_forecast"] = False

    if forecast_players is not None and not forecast_players.empty:
        if not forecast_season:
            raise ValueError("A forecast season is required when forecast players are supplied")
        output = pd.concat(
            [output, _forecast_rows(forecast_players, season=forecast_season)],
            ignore_index=True,
        )

    for column in _OUTPUT_COLUMNS:
        if column not in output:
            output[column] = np.nan
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    output["season_start_year"] = pd.to_numeric(
        output["season_start_year"], errors="raise"
    ).astype(int)
    if output.duplicated(["season", "player_id"]).any():
        raise ValueError("Player rotation history has duplicate player-season rows")
    return output.loc[:, list(_OUTPUT_COLUMNS)].sort_values(
        ["player_id", "season_start_year"], kind="stable"
    ).reset_index(drop=True)


def _historical_preseason_predictions(summary: pd.DataFrame) -> pd.DataFrame:
    seasons = (
        summary.loc[:, ["season", "season_start_year"]]
        .drop_duplicates()
        .sort_values("season_start_year", kind="stable")
    )
    rows: list[pd.DataFrame] = []
    for season in seasons["season"].iloc[1:].astype(str):
        availability, _age, _metadata = predict_availability_season(
            summary,
            target_season=season,
            config=PROMOTED_AVAILABILITY_CONFIG,
        )
        try:
            conditional, _conditional_age = predict_conditional_minutes_season(
                summary,
                target_season=season,
                config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
                cold_start_config=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
            )
        except ValueError as error:
            # The first historical season can predate enough observed rookies to
            # fit the draft residual. Its pooled age baseline remains valid.
            if "requires observed rookie seasons" not in str(error):
                raise
            conditional, _conditional_age = predict_conditional_minutes_season(
                summary,
                target_season=season,
                config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
            )
        rows.append(
            availability.loc[:, ["season", "player_id", "predicted_available_share"]].merge(
                conditional.loc[
                    :,
                    ["season", "player_id", "predicted_minutes_per_available_game"],
                ],
                on=["season", "player_id"],
                how="inner",
                validate="one_to_one",
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "season",
                "player_id",
                "predicted_available_share",
                "predicted_minutes_per_available_game",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _forecast_rows(players: pd.DataFrame, *, season: str) -> pd.DataFrame:
    required = {
        "player_id",
        "player_name",
        "age",
        "availability_probability",
        "conditional_minutes_per_game",
        "baseline_minutes_per_game",
    }
    missing = sorted(required - set(players))
    if missing:
        raise ValueError(
            "Win projection players lack rotation-history fields: " + ", ".join(missing)
        )
    output = players.loc[:, list(required)].copy()
    output["season"] = season
    output["season_start_year"] = int(season[:4])
    output = output.rename(
        columns={
            "availability_probability": "predicted_available_share",
            "conditional_minutes_per_game": "predicted_minutes_per_available_game",
        }
    )
    output["projected_total_minutes"] = (
        pd.to_numeric(output["baseline_minutes_per_game"], errors="raise") * 82.0
    )
    output["is_preseason_forecast"] = True
    for column in (
        "actual_available_games",
        "known_roster_games",
        "actual_availability_share",
        "injury_or_illness_games",
        "rest_games",
        "actual_minutes_per_available_game",
        "actual_total_minutes",
    ):
        output[column] = np.nan
    return output.drop(columns="baseline_minutes_per_game")


def _validate_summary(summary: pd.DataFrame) -> None:
    required = {
        "season",
        "season_start_year",
        "player_id",
        "player_name",
        "age",
        "available_games",
        "known_player_games",
        "available_share",
        "injury_or_illness_games",
        "rest_games",
        "minutes_per_available_game",
        "total_nba_minutes",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(
            "Availability summary lacks rotation-history fields: " + ", ".join(missing)
        )
    if summary.duplicated(["season", "player_id"]).any():
        raise ValueError("Availability summary has duplicate player-season rows")


def _season_range(start: str, end: str) -> tuple[str, ...]:
    first = int(start[:4])
    last = int(end[:4])
    if last < first:
        raise ValueError("Completed rotation-history season precedes the start season")
    return tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(first, last + 1))


def main() -> None:
    """Build the compact player rotation-history cache for the active release."""

    parser = argparse.ArgumentParser(description="Build player availability and FCM histories")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build_player_rotation_history_cache(output_path=args.output))


if __name__ == "__main__":
    main()
