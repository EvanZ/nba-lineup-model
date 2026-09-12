"""Forward, player-state Plackett-Luce regulation-minute model.

The promoted Forward Conditional Minutes model supplies a preseason utility for
each player. This module updates one player-level residual role state only
after observed games; that state travels with a player across teams.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.ac_minute_share_projection import (
    DEFAULT_HISTORY_START_SEASON,
    REGULATION_TEAM_MINUTES,
    load_regulation_available_minutes,
)
from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_PLAYER_PANEL_PATH,
    build_availability_season_summary,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    PROMOTED_CONDITIONAL_MINUTES_CONFIG,
    attach_cold_start_biographies,
    predict_conditional_minutes_season,
)

DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/rotation/forward_plackett_luce_rotation")
DEFAULT_TUNING_SEASONS = ("2020-21", "2021-22", "2022-23")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_ROLE_PRIOR_PRECISION_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_SEASON_ROLE_RETENTION_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
MAX_ROTATION_PLAYERS = 15
_MINUTE_WEIGHT_FLOOR = 1e-3
_TIE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ForwardPlackettLuceConfig:
    """Season role prior precision and prior-season mean retention."""

    role_prior_precision: float
    season_role_retention: float = 0.0


PROMOTED_FORWARD_PL_CONFIG = ForwardPlackettLuceConfig(10.0, 0.0)


@dataclass(frozen=True)
class ForwardPlackettLuceRun:
    """Immutable artifact location for a forward frozen replay."""

    run_dir: Path
    run_id: str


class _PlayerRoleState:
    """Diagonal online-Laplace approximation for league-wide player residuals.

    A dense posterior would couple every pair of players who ever shared a
    choice set and becomes impractical for a league-wide state. The diagonal
    approximation retains each player's own curvature while dropping those
    posterior correlations. It lets the residual travel across teams without
    changing the forward information set.
    """

    def __init__(
        self,
        *,
        prior_precision: float,
        initial_role_by_player: Mapping[int, float] | None = None,
    ) -> None:
        if prior_precision <= 0.0:
            raise ValueError("Role prior precision must be positive")
        self.prior_precision = float(prior_precision)
        self.player_index: dict[int, int] = {}
        self.role = np.empty(0, dtype=float)
        self.precision = np.empty(0, dtype=float)
        self.player_games_observed = np.empty(0, dtype=int)
        self.initial_role_by_player = {
            int(player_id): float(role)
            for player_id, role in (initial_role_by_player or {}).items()
        }

    def ensure_players(self, player_ids: np.ndarray) -> np.ndarray:
        """Add players, seeding only their independently available prior mean."""

        indices: list[int] = []
        for player_id in player_ids.astype(int):
            index = self.player_index.get(int(player_id))
            if index is None:
                index = len(self.player_index)
                self.player_index[int(player_id)] = index
                self.role = np.append(
                    self.role, self.initial_role_by_player.get(int(player_id), 0.0)
                )
                self.precision = np.append(self.precision, self.prior_precision)
                self.player_games_observed = np.append(self.player_games_observed, 0)
            indices.append(index)
        return np.asarray(indices, dtype=int)

    def update(
        self,
        *,
        candidate_indices: np.ndarray,
        prior_utilities: np.ndarray,
        minutes: np.ndarray,
        player_ids: np.ndarray,
    ) -> None:
        """Apply one equally weighted regulation-game PL Laplace update.

        Only players with positive minutes appear in the observed partial order.
        All medically available candidates remain in every choice denominator, so
        a coach DNP is evidence that the player ranked below the selected group.
        """

        positive = np.flatnonzero(minutes > _TIE_TOLERANCE)
        if not len(positive):
            raise ValueError("A regulation team-game requires at least one positive-minute player")
        order = positive[
            np.lexsort((player_ids[positive], -minutes[positive]))
        ]
        weight = 1.0 / len(order)
        gradient = np.zeros_like(self.role)
        curvature = np.zeros_like(self.precision)
        remaining = np.arange(len(candidate_indices), dtype=int)
        for selected_local in order:
            chosen_position = np.flatnonzero(remaining == selected_local)
            if len(chosen_position) != 1:
                raise ValueError("Observed PL order contains a duplicate player")
            active_local = remaining
            active_indices = candidate_indices[active_local]
            logits = prior_utilities[active_local] + self.role[active_indices]
            probabilities = _softmax(logits)
            selected_index = candidate_indices[selected_local]
            gradient[active_indices] -= weight * probabilities
            gradient[selected_index] += weight
            curvature[active_indices] += weight * probabilities * (1.0 - probabilities)
            remaining = np.delete(remaining, chosen_position[0])

        self.precision += curvature
        self.role += gradient / self.precision
        self.player_games_observed[candidate_indices] += 1


def prepare_forward_plackett_luce_panel(
    available_minutes: pd.DataFrame,
    conditional_prior: pd.DataFrame,
) -> pd.DataFrame:
    """Join the target-season FCM prior to exact-240 observed roster candidates."""

    required_prior = {"player_id", "predicted_minutes_per_available_game"}
    missing_prior = sorted(required_prior - set(conditional_prior))
    if missing_prior:
        raise ValueError(f"Conditional prior lacks required columns: {missing_prior}")
    if conditional_prior["player_id"].duplicated().any():
        raise ValueError("Conditional prior must have one row per player")
    base = available_minutes.copy()
    base["player_id"] = pd.to_numeric(base["player_id"], errors="raise").astype(int)
    prior = conditional_prior.loc[:, list(required_prior)].copy()
    prior["player_id"] = pd.to_numeric(prior["player_id"], errors="raise").astype(int)
    prior["predicted_minutes_per_available_game"] = pd.to_numeric(
        prior["predicted_minutes_per_available_game"], errors="raise"
    )
    output = base.merge(prior, on="player_id", how="left", validate="many_to_one")
    if output["predicted_minutes_per_available_game"].isna().any():
        missing = output.loc[
            output["predicted_minutes_per_available_game"].isna(), "player_id"
        ].unique()
        raise ValueError(
            f"Conditional prior is missing available candidates: {missing[:10].tolist()}"
        )
    if output["predicted_minutes_per_available_game"].lt(0.0).any():
        raise ValueError("Conditional minute prior cannot be negative")
    output["prior_utility"] = np.log(
        output["predicted_minutes_per_available_game"].clip(_MINUTE_WEIGHT_FLOOR)
    )
    return output.sort_values(
        ["season", "team_id", "game_date", "game_id", "player_id"], kind="stable"
    ).reset_index(drop=True)


def predict_forward_plackett_luce(
    panel: pd.DataFrame,
    *,
    config: ForwardPlackettLuceConfig,
) -> pd.DataFrame:
    """Predict an already-assembled panel with a fresh player role state."""

    panels = {
        str(season): frame.copy()
        for season, frame in panel.groupby("season", sort=False)
    }
    return pd.concat(
        predict_forward_plackett_luce_sequence(panels, config=config).values(),
        ignore_index=True,
    )


def predict_forward_plackett_luce_sequence(
    panels: Mapping[str, pd.DataFrame],
    *,
    config: ForwardPlackettLuceConfig,
    initial_player_roles: Mapping[int, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Predict chronologically, carrying player role means across team changes.

    Every panel uses a target-season FCM prior constructed before that season.
    At a season boundary the role precision is reset to its configured Gaussian
    prior. The final prior-season player role mean, multiplied by the selected
    retention factor, seeds that player wherever the player appears next. This
    prevents posterior exposure from one season from being counted again while
    retaining the player-level residual identified through team changes.
    """

    _validate_config(config)
    output: dict[str, pd.DataFrame] = {}
    prior_season_roles = {
        int(player_id): float(role)
        for player_id, role in (initial_player_roles or {}).items()
    }
    for season in sorted(panels, key=_season_year):
        prediction, prior_season_roles = _predict_forward_plackett_luce_season(
            panels[season],
            config=config,
            prior_season_roles=prior_season_roles,
        )
        output[season] = prediction
    return output


def fit_forward_plackett_luce_player_roles(
    panels: Mapping[str, pd.DataFrame],
    *,
    config: ForwardPlackettLuceConfig,
    initial_player_roles: Mapping[int, float] | None = None,
) -> dict[int, float]:
    """Fit chronological player residuals and return the final season state.

    This is useful when a downstream evaluator has a different prediction
    surface, such as forecast availability, but must begin from exactly the
    same forward PL information state.  The returned state contains only
    players who appeared in at least one observed choice set during its final
    processed season.
    """

    _validate_config(config)
    prior_season_roles = {
        int(player_id): float(role)
        for player_id, role in (initial_player_roles or {}).items()
    }
    for season in sorted(panels, key=_season_year):
        _prediction, prior_season_roles = _predict_forward_plackett_luce_season(
            panels[season],
            config=config,
            prior_season_roles=prior_season_roles,
        )
    return prior_season_roles


def _predict_forward_plackett_luce_season(
    panel: pd.DataFrame,
    *,
    config: ForwardPlackettLuceConfig,
    prior_season_roles: Mapping[int, float],
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Return one season's predictions and its final player role means."""

    _validate_panel(panel)
    rows: list[pd.DataFrame] = []
    seeded_roles = {
        int(player_id): config.season_role_retention * float(role)
        for player_id, role in prior_season_roles.items()
    }
    state = _PlayerRoleState(
        prior_precision=config.role_prior_precision,
        initial_role_by_player=seeded_roles,
    )
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
        shares, selected = _allocate_top_rotation(utilities, player_ids)
        game["predicted_minute_share"] = shares
        game["predicted_minutes"] = REGULATION_TEAM_MINUTES * shares
        game["player_role_adjustment"] = role_before
        game["prior_season_player_role_adjustment"] = [
            seeded_roles.get(int(player_id), 0.0) for player_id in player_ids
        ]
        game["has_prior_season_player_role"] = [
            int(player_id) in seeded_roles for player_id in player_ids
        ]
        game["plackett_luce_utility"] = utilities
        game["player_role_games_observed"] = state.player_games_observed[state_indices]
        game["selected_top_15"] = selected
        rows.append(game)
        state.update(
            candidate_indices=state_indices,
            prior_utilities=prior_utilities,
            minutes=game["minutes"].to_numpy(dtype=float),
            player_ids=player_ids,
        )
    output = pd.concat(rows, ignore_index=True)
    output["absolute_error"] = np.abs(
        output["actual_minute_share"] - output["predicted_minute_share"]
    )
    final_roles = {
        player_id: float(state.role[index])
        for player_id, index in state.player_index.items()
    }
    return output, final_roles


def predict_forward_conditional_minutes_control(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply production FCM v0.2 to the current observed available roster.

    The control deliberately does not multiply by availability probability: the
    candidate roster is already conditioned on actual medical availability.
    """

    _validate_panel(panel)
    rows: list[pd.DataFrame] = []
    for _key, game in panel.groupby(["season", "game_id", "team_id"], sort=False):
        game = game.sort_values("player_id", kind="stable").copy()
        player_ids = game["player_id"].to_numpy(dtype=int)
        utilities = game["prior_utility"].to_numpy(dtype=float)
        shares, selected = _allocate_top_rotation(utilities, player_ids)
        game["predicted_minute_share"] = shares
        game["predicted_minutes"] = REGULATION_TEAM_MINUTES * shares
        game["player_role_adjustment"] = 0.0
        game["plackett_luce_utility"] = utilities
        game["player_role_games_observed"] = 0
        game["selected_top_15"] = selected
        rows.append(game)
    output = pd.concat(rows, ignore_index=True)
    output["absolute_error"] = np.abs(
        output["actual_minute_share"] - output["predicted_minute_share"]
    )
    return output


def summarize_rotation_metrics(
    predictions: pd.DataFrame, *, model: str, season: str
) -> pd.DataFrame:
    """Summarize matched-support allocation and rotation-order metrics."""

    required = {
        "game_id",
        "team_id",
        "player_id",
        "minutes",
        "actual_minute_share",
        "predicted_minute_share",
        "plackett_luce_utility",
        "selected_top_15",
    }
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"Predictions lack required columns: {missing}")
    work = predictions.copy()
    work["absolute_error"] = np.abs(work["actual_minute_share"] - work["predicted_minute_share"])
    work["squared_error"] = np.square(work["actual_minute_share"] - work["predicted_minute_share"])
    work["cross_entropy_term"] = -work["actual_minute_share"] * np.log(
        work["predicted_minute_share"].clip(1e-12, None)
    )
    game_rows: list[dict[str, float]] = []
    for _key, game in work.groupby(["season", "game_id", "team_id"], sort=False):
        actual_minutes = game["minutes"].to_numpy(dtype=float)
        utilities = game["plackett_luce_utility"].to_numpy(dtype=float)
        player_ids = game["player_id"].to_numpy(dtype=int)
        pairwise_total = 0
        pairwise_correct = 0.0
        for left in range(len(game)):
            for right in range(left + 1, len(game)):
                difference = actual_minutes[left] - actual_minutes[right]
                if abs(difference) <= _TIE_TOLERANCE:
                    continue
                predicted_difference = utilities[left] - utilities[right]
                pairwise_total += 1
                if abs(predicted_difference) <= _TIE_TOLERANCE:
                    pairwise_correct += 0.5
                elif np.sign(predicted_difference) == np.sign(difference):
                    pairwise_correct += 1.0
        selected_ids = set(player_ids[game["selected_top_15"].to_numpy(dtype=bool)])
        actual_positive = set(player_ids[actual_minutes > _TIE_TOLERANCE])
        game_rows.append(
            {
                "allocation_total_variation": 0.5 * float(game["absolute_error"].sum()),
                "brier_score": float(game["squared_error"].sum()),
                "cross_entropy": float(game["cross_entropy_term"].sum()),
                "pairwise_rank_accuracy": (
                    pairwise_correct / pairwise_total if pairwise_total else np.nan
                ),
                "positive_rotation_recall_at_15": (
                    len(selected_ids & actual_positive) / len(actual_positive)
                    if actual_positive
                    else np.nan
                ),
                "candidate_count": float(len(game)),
            }
        )
    team_games = pd.DataFrame(game_rows)
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
                "pairwise_rank_accuracy": float(team_games["pairwise_rank_accuracy"].mean()),
                "positive_rotation_recall_at_15": float(
                    team_games["positive_rotation_recall_at_15"].mean()
                ),
                "mean_available_candidates": float(team_games["candidate_count"].mean()),
            }
        ]
    )


def tune_forward_plackett_luce_season_carryover(
    panels: Mapping[str, pd.DataFrame],
    *,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    role_prior_precision_grid: tuple[float, ...] = DEFAULT_ROLE_PRIOR_PRECISION_GRID,
    season_role_retention_grid: tuple[float, ...] = DEFAULT_SEASON_ROLE_RETENTION_GRID,
) -> tuple[ForwardPlackettLuceConfig, ForwardPlackettLuceConfig, pd.DataFrame]:
    """Select player-role precision and prior-season mean retention forwardly.

    The sequence includes source seasons before each tuning season solely to
    construct the role mean available at that season's opening game. Metrics
    are computed only for ``tuning_seasons``.
    """

    missing = sorted(set(tuning_seasons) - set(panels))
    if missing:
        raise ValueError(f"Carryover tuning panels are missing: {missing}")
    rows: list[dict[str, float]] = []
    precisions = sorted(set(role_prior_precision_grid))
    retentions = sorted(set(season_role_retention_grid))
    for precision_index, precision in enumerate(precisions, start=1):
        for retention_index, retention in enumerate(retentions, start=1):
            print(
                "Forward PL carryover: tuning "
                f"precision {precision_index}/{len(precisions)} ({precision:g}), "
                f"retention {retention_index}/{len(retentions)} ({retention:g})",
                flush=True,
            )
            config = ForwardPlackettLuceConfig(
                role_prior_precision=float(precision),
                season_role_retention=float(retention),
            )
            predictions = predict_forward_plackett_luce_sequence(panels, config=config)
            metrics = pd.concat(
                [
                    summarize_rotation_metrics(
                        predictions[season],
                        model="Forward PL carryover tuning",
                        season=season,
                    )
                    for season in tuning_seasons
                ],
                ignore_index=True,
            )
            rows.append(
                {
                    "role_prior_precision": float(precision),
                    "season_role_retention": float(retention),
                    "validation_season_count": float(len(metrics)),
                    "mean_allocation_total_variation": float(
                        metrics["mean_allocation_total_variation"].mean()
                    ),
                    "mean_brier_score": float(metrics["mean_brier_score"].mean()),
                    "mean_cross_entropy": float(metrics["mean_cross_entropy"].mean()),
                    "player_share_mae": float(metrics["player_share_mae"].mean()),
                    "player_share_rmse": float(metrics["player_share_rmse"].mean()),
                    "pairwise_rank_accuracy": float(
                        metrics["pairwise_rank_accuracy"].mean()
                    ),
                    "positive_rotation_recall_at_15": float(
                        metrics["positive_rotation_recall_at_15"].mean()
                    ),
                }
            )
    grid = pd.DataFrame(rows).sort_values(
        [
            "mean_allocation_total_variation",
            "mean_brier_score",
            "mean_cross_entropy",
            "season_role_retention",
            "role_prior_precision",
        ],
        kind="stable",
    ).reset_index(drop=True)
    selected = ForwardPlackettLuceConfig(
        role_prior_precision=float(grid.loc[0, "role_prior_precision"]),
        season_role_retention=float(grid.loc[0, "season_role_retention"]),
    )
    no_carry = grid.loc[grid["season_role_retention"].eq(0.0)].reset_index(drop=True)
    if no_carry.empty:
        raise ValueError("Season role retention grid must include zero for the no-carry control")
    no_carry_config = ForwardPlackettLuceConfig(
        role_prior_precision=float(no_carry.loc[0, "role_prior_precision"]),
        season_role_retention=0.0,
    )
    return selected, no_carry_config, grid


def run_forward_plackett_luce_rotation(
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    history_start_season: str = DEFAULT_HISTORY_START_SEASON,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    role_prior_precision_grid: tuple[float, ...] = DEFAULT_ROLE_PRIOR_PRECISION_GRID,
    season_role_retention_grid: tuple[float, ...] = DEFAULT_SEASON_ROLE_RETENTION_GRID,
) -> ForwardPlackettLuceRun:
    """Validate global player PL updates and optional prior-season carry."""

    target_seasons = (*tuning_seasons, *frozen_seasons)
    season_range = _season_range(history_start_season, max(target_seasons, key=_season_year))
    print(
        f"Forward PL rotation: building FCM history from {len(season_range)} seasons",
        flush=True,
    )
    summary = build_availability_season_summary(
        season_range, curated_dir=curated_dir, player_panel_path=player_panel_path
    )
    summary = attach_cold_start_biographies(summary, player_panel_path=player_panel_path)
    panels: dict[str, pd.DataFrame] = {}
    panel_seasons = season_range[1:]
    for index, season in enumerate(panel_seasons, start=1):
        print(
            f"Forward PL rotation: preparing {index}/{len(panel_seasons)} {season}",
            flush=True,
        )
        conditional_prior, _ = predict_conditional_minutes_season(
            summary,
            target_season=season,
            config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
            cold_start_config=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
        )
        available_minutes = load_regulation_available_minutes(season, curated_dir=curated_dir)
        panels[season] = prepare_forward_plackett_luce_panel(available_minutes, conditional_prior)

    print("Forward PL rotation: tuning player-role carry", flush=True)
    config, no_carry_config, tuning_grid = tune_forward_plackett_luce_season_carryover(
        panels,
        tuning_seasons=tuning_seasons,
        role_prior_precision_grid=role_prior_precision_grid,
        season_role_retention_grid=season_role_retention_grid,
    )
    print(
        "Forward PL rotation: selected "
        f"precision={config.role_prior_precision:g}, "
        f"retention={config.season_role_retention:g}; "
        "no-carry control "
        f"precision={no_carry_config.role_prior_precision:g}",
        flush=True,
    )
    run_id = f"forward-pl-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tuning_grid.to_parquet(run_dir / "tuning_grid.parquet", index=False)

    candidate_predictions = predict_forward_plackett_luce_sequence(panels, config=config)
    no_carry_predictions = predict_forward_plackett_luce_sequence(
        panels, config=no_carry_config
    )
    metric_rows: list[pd.DataFrame] = []
    candidate_rows: list[pd.DataFrame] = []
    for index, season in enumerate(frozen_seasons, start=1):
        print(f"Forward PL rotation: frozen {index}/{len(frozen_seasons)} {season}", flush=True)
        fitted = candidate_predictions[season]
        no_carry = no_carry_predictions[season]
        fcm_prior = predict_forward_conditional_minutes_control(panels[season])
        fitted.to_parquet(run_dir / f"{season}_predictions.parquet", index=False)
        no_carry.to_parquet(run_dir / f"{season}_no_carry_control_predictions.parquet", index=False)
        fcm_prior.to_parquet(run_dir / f"{season}_fcm_v02_prior_predictions.parquet", index=False)
        metric_rows.extend(
            [
                summarize_rotation_metrics(
                    fitted, model="Forward Player PL v0.2", season=season
                ),
                summarize_rotation_metrics(
                    no_carry, model="Forward Player PL no-carry control", season=season
                ),
                summarize_rotation_metrics(
                    fcm_prior, model="Forward Conditional Minutes v0.2 prior", season=season
                ),
            ]
        )
        candidate_rows.append(fitted)
    metrics = pd.concat(metric_rows, ignore_index=True)
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
            pairwise_rank_accuracy=("pairwise_rank_accuracy", "mean"),
            positive_rotation_recall_at_15=("positive_rotation_recall_at_15", "mean"),
        )
        .sort_values(["mean_allocation_total_variation", "mean_brier_score"], kind="stable")
        .reset_index(drop=True)
    )
    summary_metrics.to_parquet(run_dir / "frozen_summary.parquet", index=False)
    transfer_audit = _build_player_transfer_carry_audit(
        pd.concat(candidate_rows, ignore_index=True)
    )
    transfer_audit.to_parquet(run_dir / "transfer_carry_audit.parquet", index=False)
    carry_audit = _build_season_carryover_audit(pd.concat(candidate_rows, ignore_index=True))
    carry_audit.to_parquet(run_dir / "season_carryover_audit.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "selected_config": asdict(config),
                "no_carry_control_config": asdict(no_carry_config),
                "history_start_season": history_start_season,
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "target": (
                    "player regulation minutes / 240 among observed medically available roster"
                ),
                "availability": "observed historical medical-availability mask; not forecasted",
                "overtime": "excluded; no rescaling",
                "prior": "Forward Conditional Minutes v0.2 conditional MPG, without P(available)",
                "role_state": (
                    "league-wide player residual state with a diagonal online-Laplace "
                    "precision approximation; player state travels across teams"
                ),
                "season_carryover": (
                    "every player receives retention times the final prior-season player "
                    "role mean, regardless of the player's next team; precision resets each season"
                ),
                "game_weighting": (
                    "each team's game likelihood is normalized by its positive-minute ranks"
                ),
                "positive_minute_ties": "resolved by stable player-ID order in v0.2",
                "allocation": f"top {MAX_ROTATION_PLAYERS} by PL utility, normalized to 240",
                "control": (
                    "Forward Conditional Minutes v0.2 prior restricted to the same observed "
                    "available roster and top-15 allocation"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return ForwardPlackettLuceRun(run_dir=run_dir, run_id=run_id)


def _allocate_top_rotation(
    utilities: np.ndarray, player_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(utilities) != len(player_ids):
        raise ValueError("Utilities and player IDs must have equal length")
    order = np.lexsort((player_ids, -utilities))
    chosen = order[: min(MAX_ROTATION_PLAYERS, len(order))]
    shares = np.zeros(len(utilities), dtype=float)
    shares[chosen] = _softmax(utilities[chosen])
    selected = np.zeros(len(utilities), dtype=bool)
    selected[chosen] = True
    return shares, selected


def _validate_config(config: ForwardPlackettLuceConfig) -> None:
    if config.role_prior_precision <= 0.0:
        raise ValueError("Role prior precision must be positive")
    if not 0.0 <= config.season_role_retention <= 1.0:
        raise ValueError("Season role retention must lie in [0, 1]")


def _softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    return weights / weights.sum()


def _build_player_transfer_carry_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    """Show that a player residual travels to an in-season destination team."""

    ordered = predictions.sort_values(
        ["season", "player_id", "game_date", "game_id", "team_id"], kind="stable"
    ).copy()
    prior_teams = ordered.groupby(["season", "player_id"], sort=False)["team_id"].transform(
        lambda values: values.ne(values.shift()).cumsum().sub(1)
    )
    ordered["team_stint_index"] = prior_teams.astype(int)
    first_destination_game = ordered.groupby(
        ["season", "player_id", "team_id"], sort=False
    )["game_date"].transform("min").eq(ordered["game_date"])
    result = ordered.loc[
        first_destination_game & ordered["team_stint_index"].gt(0),
        [
            "season",
            "game_id",
            "game_date",
            "team_id",
            "team",
            "player_id",
            "player_name",
            "predicted_minutes_per_available_game",
            "player_role_adjustment",
            "player_role_games_observed",
            "minutes",
            "predicted_minutes",
        ],
    ].copy()
    return result.reset_index(drop=True)


def _build_season_carryover_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    """Expose each player's opening-season carry state for no-leakage review."""

    required = {
        "season",
        "team_id",
        "player_id",
        "game_id",
        "game_date",
        "prior_season_player_role_adjustment",
        "has_prior_season_player_role",
        "player_role_adjustment",
    }
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"Carryover audit requires: {missing}")
    ordered = predictions.sort_values(
        ["season", "player_id", "game_date", "game_id", "team_id"], kind="stable"
    ).copy()
    first_target_game = ordered.groupby(["season", "player_id"], sort=False).cumcount().eq(0)
    return ordered.loc[
        first_target_game,
        [
            "season",
            "game_id",
            "game_date",
            "team_id",
            "team",
            "player_id",
            "player_name",
            "has_prior_season_player_role",
            "prior_season_player_role_adjustment",
            "player_role_adjustment",
            "predicted_minutes_per_available_game",
            "predicted_minutes",
        ],
    ].reset_index(drop=True)


def _validate_panel(panel: pd.DataFrame) -> None:
    required = {
        "season",
        "game_id",
        "game_date",
        "team_id",
        "player_id",
        "minutes",
        "actual_minute_share",
        "predicted_minutes_per_available_game",
        "prior_utility",
    }
    missing = sorted(required - set(panel))
    if missing:
        raise ValueError(f"Forward PL panel lacks required columns: {missing}")
    totals = panel.groupby(["season", "game_id", "team_id"])["actual_minute_share"].sum()
    if not np.isclose(totals, 1.0, atol=1e-8).all():
        raise ValueError("Every team-game target must sum to one")
    if not np.isfinite(panel["prior_utility"].to_numpy(dtype=float)).all():
        raise ValueError("Forward PL utilities must be finite")


def _season_range(first: str, last: str) -> tuple[str, ...]:
    return tuple(
        f"{year}-{str(year + 1)[-2:]}"
        for year in range(_season_year(first), _season_year(last) + 1)
    )


def _season_year(season: str) -> int:
    return int(season[:4])


MODEL_NAME = "forward_plackett_luce_rotation"
MODEL_VERSION = "v0.2"


def main() -> None:
    """Run a frozen forward Plackett-Luce rotation evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate forward PL rotation allocations")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--history-start-season", default=DEFAULT_HISTORY_START_SEASON)
    parser.add_argument("--tuning-seasons", nargs="+", default=list(DEFAULT_TUNING_SEASONS))
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    parser.add_argument(
        "--role-prior-precision-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_ROLE_PRIOR_PRECISION_GRID),
    )
    parser.add_argument(
        "--season-role-retention-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_SEASON_ROLE_RETENTION_GRID),
    )
    args = parser.parse_args()
    run = run_forward_plackett_luce_rotation(
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel_path,
        artifacts_dir=args.artifacts_dir,
        history_start_season=args.history_start_season,
        tuning_seasons=tuple(args.tuning_seasons),
        frozen_seasons=tuple(args.frozen_seasons),
        role_prior_precision_grid=tuple(args.role_prior_precision_grid),
        season_role_retention_grid=tuple(args.season_role_retention_grid),
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
