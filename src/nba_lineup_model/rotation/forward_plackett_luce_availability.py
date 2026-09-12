"""End-to-end availability plus forward Plackett-Luce minute evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import qmc

from nba_lineup_model.rotation.ac_minute_share_projection import (
    REGULATION_TEAM_MINUTES,
    load_regulation_available_minutes,
    load_regulation_roster_minutes,
)
from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_PLAYER_PANEL_PATH,
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
from nba_lineup_model.rotation.forward_plackett_luce_rotation import (
    DEFAULT_FROZEN_SEASONS,
    DEFAULT_HISTORY_START_SEASON,
    PROMOTED_FORWARD_PL_CONFIG,
    ForwardPlackettLuceConfig,
    _PlayerRoleState,
    _season_range,
    _season_year,
    fit_forward_plackett_luce_player_roles,
    prepare_forward_plackett_luce_panel,
)

DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/rotation/forward_plackett_luce_availability")
DEFAULT_SCENARIO_COUNT = 1024
DEFAULT_SCENARIO_SEED = 20260912
MIN_REGULATION_AVAILABLE_PLAYERS = 5
_EPSILON = 1e-12


@dataclass(frozen=True)
class ForwardPlackettLuceAvailabilityRun:
    """Immutable location for a fully probabilistic rotation frozen replay."""

    run_dir: Path
    run_id: str


def prepare_availability_integrated_panel(
    roster_minutes: pd.DataFrame,
    conditional_prior: pd.DataFrame,
    availability_prior: pd.DataFrame,
) -> pd.DataFrame:
    """Join frozen FCM and availability priors to game-day roster membership."""

    required = {"player_id", "predicted_available_share"}
    missing = sorted(required - set(availability_prior))
    if missing:
        raise ValueError(f"Availability prior lacks required columns: {missing}")
    if availability_prior["player_id"].duplicated().any():
        raise ValueError("Availability prior must have one row per player")
    output = prepare_forward_plackett_luce_panel(roster_minutes, conditional_prior)
    availability = availability_prior.loc[:, ["player_id", "predicted_available_share"]].copy()
    availability["player_id"] = pd.to_numeric(
        availability["player_id"], errors="raise"
    ).astype(int)
    availability["predicted_available_share"] = pd.to_numeric(
        availability["predicted_available_share"], errors="raise"
    )
    output = output.merge(availability, on="player_id", how="left", validate="many_to_one")
    if output["predicted_available_share"].isna().any():
        missing_players = output.loc[
            output["predicted_available_share"].isna(), "player_id"
        ].unique()
        raise ValueError(
            "Availability prior is missing rostered candidates: "
            f"{missing_players[:10].tolist()}"
        )
    if not output["predicted_available_share"].between(0.0, 1.0).all():
        raise ValueError("Availability probabilities must lie in [0, 1]")
    _validate_integrated_panel(output)
    return output


def predict_availability_integrated_plackett_luce(
    panel: pd.DataFrame,
    *,
    config: ForwardPlackettLuceConfig = PROMOTED_FORWARD_PL_CONFIG,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    scenario_seed: int = DEFAULT_SCENARIO_SEED,
) -> pd.DataFrame:
    """Forecast unconditional minutes with Forward Availability and PL rotation.

    The current game's availability label is held out of the allocation. It is
    used only after prediction to update the global player role state for later
    games. The one-panel entry point is equivalent to a single fresh season.
    """

    panels = {
        str(season): frame.copy()
        for season, frame in panel.groupby("season", sort=False)
    }
    return pd.concat(
        predict_availability_integrated_plackett_luce_sequence(
            panels,
            config=config,
            scenario_count=scenario_count,
            scenario_seed=scenario_seed,
        ).values(),
        ignore_index=True,
    )


def predict_availability_integrated_plackett_luce_sequence(
    panels: Mapping[str, pd.DataFrame],
    *,
    config: ForwardPlackettLuceConfig,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    scenario_seed: int = DEFAULT_SCENARIO_SEED,
    initial_player_roles: Mapping[int, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Forecast availability-weighted minutes with global player PL updates.

    The conditional-minute prior is fixed before each target season. A player
    residual is updated only after a completed game, travels with that player
    after a trade, and carries to the next season only through the configured
    retained prior-season mean. Posterior precision resets at each boundary.
    """

    _validate_scenario_count(scenario_count)
    _validate_integrated_config(config)
    output: dict[str, pd.DataFrame] = {}
    prior_season_roles = {
        int(player_id): float(role)
        for player_id, role in (initial_player_roles or {}).items()
    }
    for season in sorted(panels, key=_season_year):
        prediction, prior_season_roles = _predict_availability_integrated_season(
            panels[season],
            config=config,
            scenario_count=scenario_count,
            scenario_seed=scenario_seed,
            prior_season_roles=prior_season_roles,
        )
        output[season] = prediction
    return output


def _predict_availability_integrated_season(
    panel: pd.DataFrame,
    *,
    config: ForwardPlackettLuceConfig,
    scenario_count: int,
    scenario_seed: int,
    prior_season_roles: Mapping[int, float],
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Forecast one season and return the final observed-availability state."""

    _validate_integrated_panel(panel)
    seeded_roles = {
        int(player_id): config.season_role_retention * float(role)
        for player_id, role in prior_season_roles.items()
    }
    state = _PlayerRoleState(
        prior_precision=config.role_prior_precision,
        initial_role_by_player=seeded_roles,
    )
    rows: list[pd.DataFrame] = []
    ordered = panel.sort_values(
        ["game_date", "game_id", "team_id", "player_id"], kind="stable"
    )
    for _key, game in ordered.groupby(["game_date", "game_id", "team_id"], sort=False):
        game = game.sort_values("player_id", kind="stable").copy()
        player_ids = game["player_id"].to_numpy(dtype=int)
        state_indices = state.ensure_players(player_ids)
        prior_utilities = game["prior_utility"].to_numpy(dtype=float)
        role_before = state.role[state_indices].copy()
        utilities = prior_utilities + role_before
        shares, selection_probability, mean_available = _expected_scenario_allocation(
            utilities=utilities,
            availability_probability=game["predicted_available_share"].to_numpy(dtype=float),
            player_ids=player_ids,
            scenario_count=scenario_count,
            seed=_game_seed(
                scenario_seed,
                str(game.iloc[0]["season"]),
                str(game.iloc[0]["game_id"]),
                int(game.iloc[0]["team_id"]),
            ),
        )
        game["predicted_minute_share"] = shares
        game["predicted_minutes"] = REGULATION_TEAM_MINUTES * shares
        game["selected_top_15_probability"] = selection_probability
        game["mean_scenario_available_players"] = mean_available
        game["player_role_adjustment"] = role_before
        game["prior_season_player_role_adjustment"] = [
            seeded_roles.get(int(player_id), 0.0) for player_id in player_ids
        ]
        game["has_prior_season_player_role"] = [
            int(player_id) in seeded_roles for player_id in player_ids
        ]
        game["plackett_luce_utility"] = utilities
        game["player_role_games_observed"] = state.player_games_observed[state_indices]
        rows.append(game)

        observed_available = game["available"].astype(bool).to_numpy()
        state.update(
            candidate_indices=state_indices[observed_available],
            prior_utilities=prior_utilities[observed_available],
            minutes=game.loc[observed_available, "minutes"].to_numpy(dtype=float),
            player_ids=player_ids[observed_available],
        )
    final_roles = {
        player_id: float(state.role[index])
        for player_id, index in state.player_index.items()
        if state.player_games_observed[index] > 0
    }
    return _finalize_integrated_predictions(pd.concat(rows, ignore_index=True)), final_roles


def predict_availability_integrated_fcm_control(
    panel: pd.DataFrame,
    *,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    scenario_seed: int = DEFAULT_SCENARIO_SEED,
) -> pd.DataFrame:
    """Apply the production availability and FCM components without PL updates."""

    _validate_integrated_panel(panel)
    _validate_scenario_count(scenario_count)
    rows: list[pd.DataFrame] = []
    ordered = panel.sort_values(
        ["season", "team_id", "game_date", "game_id", "player_id"], kind="stable"
    )
    for (_season, team_id, game_id), game in ordered.groupby(
        ["season", "team_id", "game_id"], sort=False
    ):
        game = game.sort_values("player_id", kind="stable").copy()
        player_ids = game["player_id"].to_numpy(dtype=int)
        utilities = game["prior_utility"].to_numpy(dtype=float)
        shares, selection_probability, mean_available = _expected_scenario_allocation(
            utilities=utilities,
            availability_probability=game["predicted_available_share"].to_numpy(dtype=float),
            player_ids=player_ids,
            scenario_count=scenario_count,
            seed=_game_seed(scenario_seed, str(game.iloc[0]["season"]), str(game_id), int(team_id)),
        )
        game["predicted_minute_share"] = shares
        game["predicted_minutes"] = REGULATION_TEAM_MINUTES * shares
        game["selected_top_15_probability"] = selection_probability
        game["mean_scenario_available_players"] = mean_available
        game["player_role_adjustment"] = 0.0
        game["plackett_luce_utility"] = utilities
        game["player_role_games_observed"] = 0
        rows.append(game)
    return _finalize_integrated_predictions(pd.concat(rows, ignore_index=True))


def summarize_integrated_rotation_metrics(
    predictions: pd.DataFrame, *, model: str, season: str
) -> pd.DataFrame:
    """Score unconditional player-game minute forecasts on complete roster support."""

    required = {
        "season",
        "game_id",
        "team_id",
        "actual_minute_share",
        "predicted_minute_share",
        "mean_scenario_available_players",
    }
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"Integrated predictions lack required columns: {missing}")
    work = predictions.copy()
    work["absolute_error"] = np.abs(work["actual_minute_share"] - work["predicted_minute_share"])
    work["squared_error"] = np.square(work["actual_minute_share"] - work["predicted_minute_share"])
    work["cross_entropy_term"] = -work["actual_minute_share"] * np.log(
        work["predicted_minute_share"].clip(_EPSILON, None)
    )
    team_games = work.groupby(["season", "game_id", "team_id"], as_index=False).agg(
        allocation_total_variation=("absolute_error", lambda values: 0.5 * float(values.sum())),
        brier_score=("squared_error", "sum"),
        cross_entropy=("cross_entropy_term", "sum"),
        mean_scenario_available_players=("mean_scenario_available_players", "first"),
        roster_candidates=("player_id", "size"),
    )
    return pd.DataFrame(
        [
            {
                "model": model,
                "season": season,
                "evaluated_team_games": int(len(team_games)),
                "mean_allocation_total_variation": float(
                    team_games["allocation_total_variation"].mean()
                ),
                "median_allocation_total_variation": float(
                    team_games["allocation_total_variation"].median()
                ),
                "mean_brier_score": float(team_games["brier_score"].mean()),
                "mean_cross_entropy": float(team_games["cross_entropy"].mean()),
                "player_share_mae": float(work["absolute_error"].mean()),
                "player_share_rmse": float(np.sqrt(work["squared_error"].mean())),
                "mean_scenario_available_players": float(
                    team_games["mean_scenario_available_players"].mean()
                ),
                "mean_roster_candidates": float(team_games["roster_candidates"].mean()),
            }
        ]
    )


def run_availability_integrated_plackett_luce(
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    history_start_season: str = DEFAULT_HISTORY_START_SEASON,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    plackett_luce_config: ForwardPlackettLuceConfig = PROMOTED_FORWARD_PL_CONFIG,
    no_carry_plackett_luce_config: ForwardPlackettLuceConfig | None = None,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    scenario_seed: int = DEFAULT_SCENARIO_SEED,
) -> ForwardPlackettLuceAvailabilityRun:
    """Run a frozen end-to-end replay with global player PL role updates.

    Observed-availability panels before the first frozen season establish the
    player residual state. Frozen predictions use forecast availability for
    allocation, then update that same state only after each completed game.
    """

    _validate_scenario_count(scenario_count)
    season_range = _season_range(history_start_season, max(frozen_seasons, key=_season_year))
    print(
        f"Availability + PL: building shared history from {len(season_range)} seasons",
        flush=True,
    )
    summary = build_availability_season_summary(
        season_range, curated_dir=curated_dir, player_panel_path=player_panel_path
    )
    summary = attach_cold_start_biographies(summary, player_panel_path=player_panel_path)
    earliest_frozen = min(frozen_seasons, key=_season_year)
    panel_seasons = season_range[1:]
    observed_source_panels: dict[str, pd.DataFrame] = {}
    frozen_panels: dict[str, pd.DataFrame] = {}
    for index, season in enumerate(panel_seasons, start=1):
        print(
            f"Availability + PL: preparing {index}/{len(panel_seasons)} {season}",
            flush=True,
        )
        conditional_prior, _ = predict_conditional_minutes_season(
            summary,
            target_season=season,
            config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
            cold_start_config=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
        )
        if _season_year(season) < _season_year(earliest_frozen):
            available_minutes = load_regulation_available_minutes(season, curated_dir=curated_dir)
            observed_source_panels[season] = prepare_forward_plackett_luce_panel(
                available_minutes, conditional_prior
            )
        else:
            availability_prior, _age_model, _metadata = predict_availability_season(
                summary,
                target_season=season,
                config=PROMOTED_AVAILABILITY_CONFIG,
            )
            roster_minutes = load_regulation_roster_minutes(season, curated_dir=curated_dir)
            frozen_panels[season] = prepare_availability_integrated_panel(
                roster_minutes, conditional_prior, availability_prior
            )

    no_carry_config = no_carry_plackett_luce_config or ForwardPlackettLuceConfig(
        role_prior_precision=plackett_luce_config.role_prior_precision,
        season_role_retention=0.0,
    )
    source_player_roles = fit_forward_plackett_luce_player_roles(
        observed_source_panels,
        config=plackett_luce_config,
    )

    run_id = (
        f"availability-pl-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    )
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print("Availability + PL: forecasting frozen sequence", flush=True)
    candidate_predictions = predict_availability_integrated_plackett_luce_sequence(
        frozen_panels,
        config=plackett_luce_config,
        initial_player_roles=source_player_roles,
        scenario_count=scenario_count,
        scenario_seed=scenario_seed,
    )
    no_carry_predictions = predict_availability_integrated_plackett_luce_sequence(
        frozen_panels,
        config=no_carry_config,
        scenario_count=scenario_count,
        scenario_seed=scenario_seed,
    )
    metrics_rows: list[pd.DataFrame] = []
    for index, season in enumerate(frozen_seasons, start=1):
        print(f"Availability + PL: scoring {index}/{len(frozen_seasons)} {season}", flush=True)
        fitted = candidate_predictions[season]
        no_carry = no_carry_predictions[season]
        control = predict_availability_integrated_fcm_control(
            frozen_panels[season],
            scenario_count=scenario_count,
            scenario_seed=scenario_seed,
        )
        fitted.to_parquet(run_dir / f"{season}_predictions.parquet", index=False)
        no_carry.to_parquet(
            run_dir / f"{season}_no_carry_control_predictions.parquet", index=False
        )
        control.to_parquet(run_dir / f"{season}_fcm_v02_control_predictions.parquet", index=False)
        metrics_rows.extend(
            [
                summarize_integrated_rotation_metrics(
                    fitted,
                    model="Forward Availability v0.2 + Forward Player PL v0.2",
                    season=season,
                ),
                summarize_integrated_rotation_metrics(
                    no_carry,
                    model="Forward Availability v0.2 + Forward Player PL no-carry control",
                    season=season,
                ),
                summarize_integrated_rotation_metrics(
                    control,
                    model="Forward Availability v0.2 + FCM v0.2 control",
                    season=season,
                ),
            ]
        )
    metrics = pd.concat(metrics_rows, ignore_index=True)
    metrics.to_parquet(run_dir / "frozen_metrics.parquet", index=False)
    summary_metrics = (
        metrics.groupby("model", as_index=False)
        .agg(
            frozen_seasons=("season", "size"),
            evaluated_team_games=("evaluated_team_games", "sum"),
            mean_allocation_total_variation=("mean_allocation_total_variation", "mean"),
            mean_brier_score=("mean_brier_score", "mean"),
            mean_cross_entropy=("mean_cross_entropy", "mean"),
            player_share_mae=("player_share_mae", "mean"),
            player_share_rmse=("player_share_rmse", "mean"),
        )
        .sort_values(["mean_allocation_total_variation", "mean_brier_score"], kind="stable")
        .reset_index(drop=True)
    )
    summary_metrics.to_parquet(run_dir / "frozen_summary.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": "forward_availability_plackett_luce_rotation",
                "version": "v0.2",
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "frozen_seasons": list(frozen_seasons),
                "availability_config": asdict(PROMOTED_AVAILABILITY_CONFIG),
                "conditional_minutes_config": asdict(PROMOTED_CONDITIONAL_MINUTES_CONFIG),
                "cold_start_config": asdict(PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG),
                "plackett_luce_config": asdict(plackett_luce_config),
                "no_carry_plackett_luce_config": asdict(no_carry_config),
                "scenario_method": (
                    "scrambled Sobol availability masks conditioned on at least five players"
                ),
                "scenario_count": scenario_count,
                "scenario_seed": scenario_seed,
                "availability_mask": (
                    "forecast from Forward Availability v0.2; not observed in prediction"
                ),
                "pl_state": (
                    "league-wide player residual with a diagonal online-Laplace precision "
                    "approximation; it travels across in-season team changes"
                ),
                "season_carryover": (
                    "the final observed prior-season player role mean is retained according "
                    "to the configured factor; precision resets at each season boundary"
                ),
                "pl_state_update": "observed availability and minutes used only after the game",
                "control": (
                    "same availability scenarios with FCM v0.2 utilities and no PL role state"
                ),
                "target": (
                    "unconditional player regulation minutes / 240 on complete game-day roster"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return ForwardPlackettLuceAvailabilityRun(run_dir=run_dir, run_id=run_id)


def _expected_scenario_allocation(
    *,
    utilities: np.ndarray,
    availability_probability: np.ndarray,
    player_ids: np.ndarray,
    scenario_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return mean 240-minute share across feasible availability scenarios."""

    masks = _sample_feasible_availability_masks(
        availability_probability=availability_probability,
        scenario_count=scenario_count,
        seed=seed,
    )
    selected_count = min(len(player_ids), 15)
    masked_utilities = np.where(masks, utilities[None, :], -np.inf)
    order = np.argsort(-masked_utilities, axis=1, kind="stable")[:, :selected_count]
    selected_utilities = np.take_along_axis(masked_utilities, order, axis=1)
    valid = np.isfinite(selected_utilities)
    maxima = np.max(np.where(valid, selected_utilities, -np.inf), axis=1, keepdims=True)
    weights = np.where(valid, np.exp(selected_utilities - maxima), 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    scenario_shares = np.zeros_like(masked_utilities, dtype=float)
    np.put_along_axis(scenario_shares, order, weights, axis=1)
    return (
        scenario_shares.mean(axis=0),
        (scenario_shares > 0.0).mean(axis=0),
        float(masks.sum(axis=1).mean()),
    )


def _sample_feasible_availability_masks(
    *,
    availability_probability: np.ndarray,
    scenario_count: int,
    seed: int,
) -> np.ndarray:
    """Generate deterministic QMC masks conditioned on a legal five-player team."""

    probabilities = np.asarray(availability_probability, dtype=float)
    if len(probabilities) < MIN_REGULATION_AVAILABLE_PLAYERS:
        raise ValueError("A regulation roster must contain at least five candidates")
    if not np.isfinite(probabilities).all() or not np.logical_and(
        probabilities >= 0.0, probabilities <= 1.0
    ).all():
        raise ValueError("Availability probabilities must be finite values in [0, 1]")
    draw_count = scenario_count
    for attempt in range(6):
        engine = qmc.Sobol(d=len(probabilities), scramble=True, seed=seed + attempt)
        uniforms = engine.random_base2(int(np.log2(draw_count)))
        masks = uniforms < probabilities[None, :]
        feasible = masks[masks.sum(axis=1) >= MIN_REGULATION_AVAILABLE_PLAYERS]
        if len(feasible) >= scenario_count:
            return feasible[:scenario_count]
        draw_count *= 2
    raise ValueError(
        "Availability probabilities rarely produce a feasible five-player roster; "
        "cannot condition the scenario simulation"
    )


def _finalize_integrated_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    output["absolute_error"] = np.abs(
        output["actual_minute_share"] - output["predicted_minute_share"]
    )
    return output


def _game_seed(base_seed: int, season: str, game_id: str, team_id: int) -> int:
    payload = f"{base_seed}|{season}|{game_id}|{team_id}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _validate_scenario_count(scenario_count: int) -> None:
    if scenario_count < 2 or scenario_count & (scenario_count - 1):
        raise ValueError("Scenario count must be a power of two and at least two")


def _validate_integrated_config(config: ForwardPlackettLuceConfig) -> None:
    if config.role_prior_precision <= 0.0:
        raise ValueError("Role prior precision must be positive")
    if not 0.0 <= config.season_role_retention <= 1.0:
        raise ValueError("Season role retention must lie in [0, 1]")


def _validate_integrated_panel(panel: pd.DataFrame) -> None:
    required = {
        "season",
        "game_id",
        "game_date",
        "team_id",
        "player_id",
        "minutes",
        "available",
        "actual_minute_share",
        "prior_utility",
        "predicted_available_share",
    }
    missing = sorted(required - set(panel))
    if missing:
        raise ValueError(f"Integrated panel lacks required columns: {missing}")
    total = panel.groupby(["season", "game_id", "team_id"])["actual_minute_share"].sum()
    if not np.isclose(total, 1.0, atol=1e-8).all():
        raise ValueError("Every rostered team-game target must sum to one")


def main() -> None:
    """Run the fixed-contract availability-integrated frozen evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate availability plus PL rotation")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--history-start-season", default=DEFAULT_HISTORY_START_SEASON)
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    parser.add_argument(
        "--role-prior-precision",
        type=float,
        default=PROMOTED_FORWARD_PL_CONFIG.role_prior_precision,
    )
    parser.add_argument(
        "--season-role-retention",
        type=float,
        default=PROMOTED_FORWARD_PL_CONFIG.season_role_retention,
    )
    parser.add_argument("--no-carry-role-prior-precision", type=float)
    parser.add_argument("--scenario-count", type=int, default=DEFAULT_SCENARIO_COUNT)
    parser.add_argument("--scenario-seed", type=int, default=DEFAULT_SCENARIO_SEED)
    args = parser.parse_args()
    run = run_availability_integrated_plackett_luce(
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel_path,
        artifacts_dir=args.artifacts_dir,
        history_start_season=args.history_start_season,
        frozen_seasons=tuple(args.frozen_seasons),
        plackett_luce_config=ForwardPlackettLuceConfig(
            role_prior_precision=args.role_prior_precision,
            season_role_retention=args.season_role_retention,
        ),
        no_carry_plackett_luce_config=(
            ForwardPlackettLuceConfig(args.no_carry_role_prior_precision, 0.0)
            if args.no_carry_role_prior_precision is not None
            else None
        ),
        scenario_count=args.scenario_count,
        scenario_seed=args.scenario_seed,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
