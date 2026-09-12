"""Forward, roster-conditional Plackett-Luce regulation-minute model.

The promoted Forward Conditional Minutes model supplies a preseason utility for
each player.  This module updates a separate team-season role state only after
each observed game.  A transfer therefore retains the player prior but starts
with no team-role adjustment on the destination team.
"""

from __future__ import annotations

import argparse
import json
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
MAX_ROTATION_PLAYERS = 15
_MINUTE_WEIGHT_FLOOR = 1e-3
_TIE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ForwardPlackettLuceConfig:
    """Gaussian prior precision for team-specific log minute-role adjustments."""

    role_prior_precision: float


PROMOTED_FORWARD_PL_CONFIG = ForwardPlackettLuceConfig(10.0)


@dataclass(frozen=True)
class ForwardPlackettLuceRun:
    """Immutable artifact location for a forward frozen replay."""

    run_dir: Path
    run_id: str


class _TeamRoleState:
    """Sequential Laplace approximation for one team-season's PL role state."""

    def __init__(self, *, prior_precision: float) -> None:
        if prior_precision <= 0.0:
            raise ValueError("Role prior precision must be positive")
        self.prior_precision = float(prior_precision)
        self.player_index: dict[int, int] = {}
        self.role = np.empty(0, dtype=float)
        self.precision = np.empty((0, 0), dtype=float)
        self.games_completed = 0

    def ensure_players(self, player_ids: np.ndarray) -> np.ndarray:
        """Add destination-team players with a zero team-role adjustment."""

        indices: list[int] = []
        for player_id in player_ids.astype(int):
            index = self.player_index.get(int(player_id))
            if index is None:
                index = len(self.player_index)
                self.player_index[int(player_id)] = index
                self.role = np.append(self.role, 0.0)
                expanded = np.zeros((index + 1, index + 1), dtype=float)
                if index:
                    expanded[:index, :index] = self.precision
                expanded[index, index] = self.prior_precision
                self.precision = expanded
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
            covariance = np.diag(probabilities) - np.outer(probabilities, probabilities)
            curvature[np.ix_(active_indices, active_indices)] += weight * covariance
            remaining = np.delete(remaining, chosen_position[0])

        self.precision += curvature
        try:
            step = np.linalg.solve(self.precision, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(self.precision, gradient, rcond=None)[0]
        self.role += step
        self.games_completed += 1


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
    """Allocate game minutes before each game updates its team-specific role state."""

    _validate_panel(panel)
    rows: list[pd.DataFrame] = []
    ordered = panel.sort_values(
        ["season", "team_id", "game_date", "game_id", "player_id"], kind="stable"
    )
    for (_, _team_id), team_rows in ordered.groupby(["season", "team_id"], sort=False):
        state = _TeamRoleState(prior_precision=config.role_prior_precision)
        for _game_id, game in team_rows.groupby("game_id", sort=False):
            game = game.sort_values("player_id", kind="stable").copy()
            player_ids = game["player_id"].to_numpy(dtype=int)
            state_indices = state.ensure_players(player_ids)
            prior_utilities = game["prior_utility"].to_numpy(dtype=float)
            role_before = state.role[state_indices].copy()
            utilities = prior_utilities + role_before
            shares, selected = _allocate_top_rotation(utilities, player_ids)
            game["predicted_minute_share"] = shares
            game["predicted_minutes"] = REGULATION_TEAM_MINUTES * shares
            game["team_role_adjustment"] = role_before
            game["plackett_luce_utility"] = utilities
            game["team_role_games_completed"] = state.games_completed
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
    return output


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
        game["team_role_adjustment"] = 0.0
        game["plackett_luce_utility"] = utilities
        game["team_role_games_completed"] = 0
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


def tune_forward_plackett_luce(
    panels: dict[str, pd.DataFrame],
    *,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    role_prior_precision_grid: tuple[float, ...] = DEFAULT_ROLE_PRIOR_PRECISION_GRID,
) -> tuple[ForwardPlackettLuceConfig, pd.DataFrame]:
    """Select team-role precision exclusively from completed tuning seasons."""

    rows: list[dict[str, float]] = []
    precisions = sorted(set(role_prior_precision_grid))
    for index, precision in enumerate(precisions, start=1):
        print(
            "Forward PL rotation: tuning precision "
            f"{index}/{len(precisions)} ({precision:g})",
            flush=True,
        )
        metrics = [
            summarize_rotation_metrics(
                predict_forward_plackett_luce(
                    panels[season],
                    config=ForwardPlackettLuceConfig(role_prior_precision=float(precision)),
                ),
                model=MODEL_NAME,
                season=season,
            )
            for season in tuning_seasons
        ]
        aggregate = pd.concat(metrics, ignore_index=True)
        rows.append(
            {
                "role_prior_precision": float(precision),
                "validation_season_count": float(len(aggregate)),
                "mean_allocation_total_variation": float(
                    aggregate["mean_allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(aggregate["mean_brier_score"].mean()),
                "mean_cross_entropy": float(aggregate["mean_cross_entropy"].mean()),
                "player_share_mae": float(aggregate["player_share_mae"].mean()),
                "player_share_rmse": float(aggregate["player_share_rmse"].mean()),
                "pairwise_rank_accuracy": float(aggregate["pairwise_rank_accuracy"].mean()),
                "positive_rotation_recall_at_15": float(
                    aggregate["positive_rotation_recall_at_15"].mean()
                ),
            }
        )
    grid = pd.DataFrame(rows).sort_values(
        [
            "mean_allocation_total_variation",
            "mean_brier_score",
            "mean_cross_entropy",
            "role_prior_precision",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return ForwardPlackettLuceConfig(float(grid.loc[0, "role_prior_precision"])), grid


def run_forward_plackett_luce_rotation(
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    history_start_season: str = DEFAULT_HISTORY_START_SEASON,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    role_prior_precision_grid: tuple[float, ...] = DEFAULT_ROLE_PRIOR_PRECISION_GRID,
) -> ForwardPlackettLuceRun:
    """Run the forward team-role PL replay against the production FCM control."""

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
    for index, season in enumerate(target_seasons, start=1):
        print(
            f"Forward PL rotation: preparing {index}/{len(target_seasons)} {season}",
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

    print("Forward PL rotation: tuning team-role prior precision", flush=True)
    config, tuning_grid = tune_forward_plackett_luce(
        panels,
        tuning_seasons=tuning_seasons,
        role_prior_precision_grid=role_prior_precision_grid,
    )
    print(
        f"Forward PL rotation: selected role prior precision={config.role_prior_precision:g}",
        flush=True,
    )
    run_id = f"forward-pl-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tuning_grid.to_parquet(run_dir / "tuning_grid.parquet", index=False)

    metric_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    for index, season in enumerate(frozen_seasons, start=1):
        print(f"Forward PL rotation: frozen {index}/{len(frozen_seasons)} {season}", flush=True)
        fitted = predict_forward_plackett_luce(panels[season], config=config)
        control = predict_forward_conditional_minutes_control(panels[season])
        fitted.to_parquet(run_dir / f"{season}_predictions.parquet", index=False)
        control.to_parquet(run_dir / f"{season}_fcm_v02_control_predictions.parquet", index=False)
        metric_rows.extend(
            [
                summarize_rotation_metrics(
                    fitted, model="Forward PL Rotation v0.1", season=season
                ),
                summarize_rotation_metrics(
                    control, model="Forward Conditional Minutes v0.2 control", season=season
                ),
            ]
        )
        prediction_rows.append(fitted)
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
    transfer_audit = _build_transfer_reset_audit(pd.concat(prediction_rows, ignore_index=True))
    transfer_audit.to_parquet(run_dir / "transfer_reset_audit.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "selected_config": asdict(config),
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
                    "team-season-specific sequential Plackett-Luce Laplace state; "
                    "new team starts at zero adjustment"
                ),
                "game_weighting": (
                    "each team's game likelihood is normalized by its positive-minute ranks"
                ),
                "positive_minute_ties": "resolved by stable player-ID order in v0.1",
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


def _softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    return weights / weights.sum()


def _build_transfer_reset_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    """Verify in artifacts that a destination-team role begins at zero."""

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
            "team_role_adjustment",
            "team_role_games_completed",
            "minutes",
            "predicted_minutes",
        ],
    ].copy()
    result["destination_role_reset"] = np.isclose(result["team_role_adjustment"], 0.0)
    return result.reset_index(drop=True)


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
MODEL_VERSION = "v0.1"


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
    args = parser.parse_args()
    run = run_forward_plackett_luce_rotation(
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel_path,
        artifacts_dir=args.artifacts_dir,
        history_start_season=args.history_start_season,
        tuning_seasons=tuple(args.tuning_seasons),
        frozen_seasons=tuple(args.frozen_seasons),
        role_prior_precision_grid=tuple(args.role_prior_precision_grid),
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
