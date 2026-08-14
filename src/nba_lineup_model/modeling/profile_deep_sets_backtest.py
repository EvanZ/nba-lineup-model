"""Three-season frozen evaluation harness for profile-aware Deep Sets.

The implementation deliberately keeps all target-season regular and playoff
outcomes outside model fitting. Training corpus assembly is separated here so
the trainer can consume lazy season/player token tables.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from scipy.stats import pearsonr, spearmanr

from nba_lineup_model.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    possession_game_margin_rmse,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.frozen_game_outcomes import score_full_game_outcomes
from nba_lineup_model.modeling.frozen_prior_evaluation import (
    _historical_team_seasons,
    _read_playoff_possessions,
    _read_regular_possessions,
    _team_win_evaluation,
    fit_pythagorean_win_model,
)
from nba_lineup_model.modeling.neural import NeuralTrainingConfig, _trainer
from nba_lineup_model.modeling.neural_data import player_vocabulary
from nba_lineup_model.modeling.profile_deep_sets import (
    LazyProfilePossessionTensorDataset,
    ProfilePossessionDataModule,
    profile_possession_loader,
    profile_token_tables,
    read_profile_tokens,
)
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.models.neural import ProfileDeepSetsRapmModule

DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
MODEL_NAME = "profile_deep_sets"
LABEL = "Profile-aware Deep Sets"


class _PilotEpochLogger(Callback):
    """Write compact epoch progress to the durable pilot log."""

    def __init__(self, target_season: str, total_epochs: int) -> None:
        super().__init__()
        self.target_season = target_season
        self.total_epochs = total_epochs

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del pl_module
        value = trainer.callback_metrics.get("train_mse")
        train_mse = float(value.detach().cpu()) if value is not None else float("nan")
        print(
            f"target={self.target_season} epoch={trainer.current_epoch + 1}/"
            f"{self.total_epochs} train_mse={train_mse:.6f}",
            flush=True,
        )


@dataclass(frozen=True)
class FrozenProfileDeepSetsCohort:
    """One target season with an outcome-isolated historical training corpus."""

    target_season: str
    training_seasons: tuple[str, ...]
    train_regular: pd.DataFrame
    target_regular: pd.DataFrame
    target_playoffs: pd.DataFrame
    tokens: pd.DataFrame


def build_frozen_profile_deep_sets_cohort(
    target_season: str,
    *,
    token_dir: Path | str = Path("data/analytical/profile_tokens"),
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
) -> FrozenProfileDeepSetsCohort:
    """Load a target's prior regular corpus plus isolated regular/playoff tests."""

    print(f"target={target_season} reading profile tokens", flush=True)
    all_tokens = read_profile_tokens_for_backtest(token_dir)
    available = tuple(sorted(all_tokens["target_season"].astype(str).unique()))
    training_seasons = tuple(season for season in available if season < target_season)
    if not training_seasons:
        raise ValueError(f"No profile-token seasons precede {target_season}")
    print(f"target={target_season} reading {len(training_seasons)} training seasons", flush=True)
    train_regular = pd.concat(
        [
            _read_regular_possessions(
                season,
                analytical_dir=analytical_dir,
                curated_dir=curated_dir,
            )[0]
            for season in training_seasons
        ],
        ignore_index=True,
    )
    target_regular, _ = _read_regular_possessions(
        target_season,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    target_playoffs, _ = _read_playoff_possessions(target_season, curated_dir)
    required_seasons = (*training_seasons, target_season)
    tokens = all_tokens.loc[
        all_tokens["target_season"].astype(str).isin(required_seasons)
    ].copy()
    for frame in (train_regular, target_regular, target_playoffs):
        _validate_cohort_token_coverage(frame, tokens)
    print(f"target={target_season} token coverage passed", flush=True)
    return FrozenProfileDeepSetsCohort(
        target_season=target_season,
        training_seasons=training_seasons,
        train_regular=train_regular,
        target_regular=target_regular,
        target_playoffs=target_playoffs,
        tokens=tokens,
    )


def read_profile_tokens_for_backtest(token_dir: Path | str) -> pd.DataFrame:
    """Read all validated season/player profile tokens once for a backtest."""

    root = Path(token_dir)
    if not root.is_dir():
        raise ValueError(f"Profile token directory does not exist: {root}")
    manifest = root / "_manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Profile token manifest is missing: {manifest}")
    # Validate through the public single-season reader, then load all rows.
    first_season = pd.read_parquet(root / "player_profile_tokens.parquet")[
        "target_season"
    ].astype(str).min()
    read_profile_tokens(first_season, profile_dir=root)
    return pd.read_parquet(root / "player_profile_tokens.parquet")


def _validate_cohort_token_coverage(
    possessions: pd.DataFrame,
    tokens: pd.DataFrame,
) -> None:
    available = set(
        zip(
            tokens["target_season"].astype(str),
            tokens["player_id"].astype(int),
            strict=True,
        )
    )
    required = {
        (str(season), int(player_id))
        for season, offense, defense in zip(
            possessions["season"],
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            strict=True,
        )
        for player_id in [*offense, *defense]
    }
    if missing := sorted(required - available):
        raise ValueError(f"Profile token mart is missing lineup tokens: {missing[:10]}")


def build_frozen_profile_deep_sets_cohorts(
    seasons: Sequence[str] = DEFAULT_FROZEN_SEASONS,
    **kwargs: object,
) -> tuple[FrozenProfileDeepSetsCohort, ...]:
    """Build validated, outcome-isolated cohorts for every frozen target."""

    return tuple(
        build_frozen_profile_deep_sets_cohort(season, **kwargs)
        for season in seasons
    )


def run_frozen_profile_deep_sets_pilot(
    *,
    seasons: Sequence[str] = DEFAULT_FROZEN_SEASONS,
    epochs: int = 3,
    batch_size: int = 8_192,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    token_dir: Path | str = Path("data/analytical/profile_tokens"),
    output_dir: Path | str = Path("artifacts/models/profile_deep_sets_frozen_candidate"),
) -> pd.DataFrame:
    """Run a fixed-budget frozen candidate with the full leaderboard contract."""

    if epochs < 1:
        raise ValueError("Pilot epochs must be positive")
    cohort_metrics: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    full_game_frames: list[pd.DataFrame] = []
    team_frames: list[pd.DataFrame] = []
    win_frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    for target_season in seasons:
        cohort = build_frozen_profile_deep_sets_cohort(
            target_season,
            token_dir=token_dir,
        )
        print(
            f"target={cohort.target_season} loading complete "
            f"train_seasons={len(cohort.training_seasons)} "
            f"train_possessions={len(cohort.train_regular):,}",
            flush=True,
        )
        vocabulary = player_vocabulary(cohort.train_regular)
        train_games = tuple(cohort.train_regular["game_id"].astype(str).unique())
        data = ProfilePossessionDataModule(
            cohort.train_regular,
            vocabulary,
            cohort.tokens,
            train_game_ids=train_games,
            batch_size=batch_size,
        )
        train_mean = float(cohort.train_regular["target_offense_margin"].mean())
        L.seed_everything(17, workers=True, verbose=False)
        module = ProfileDeepSetsRapmModule(
            len(vocabulary),
            profile_feature_count=47,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        with torch.no_grad():
            module.model.intercept.fill_(train_mean)
        trainer = _trainer(
            NeuralTrainingConfig(
                batch_size=batch_size,
                max_epochs=epochs,
                learning_rates=(learning_rate,),
                weight_decays=(weight_decay,),
            ),
            max_epochs=epochs,
            callbacks=[_PilotEpochLogger(cohort.target_season, epochs)],
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        data.setup("fit")
        trainer.fit(module, train_dataloaders=data.train_dataloader())
        print(f"target={cohort.target_season} training complete", flush=True)
        for cohort_name, possessions in (
            ("regular_season", cohort.target_regular),
            ("playoffs", cohort.target_playoffs),
        ):
            predictions = _predict_profile(module, possessions, vocabulary, cohort.tokens, data)
            metrics, frame = _score_possession_cohort(
                possessions,
                predictions,
                cohort=cohort_name,
                target_season=cohort.target_season,
                training_mean=train_mean,
                train_season_count=len(cohort.training_seasons),
                train_game_count=len(train_games),
            )
            cohort_metrics.append(metrics)
            prediction_frames.append(frame)
        stints = read_rapm_stints(cohort.target_season)
        games, teams, missing_profile_slots = _score_full_regular_stints(
            module,
            stints,
            vocabulary,
            cohort.tokens,
            data,
        )
        calibration = _historical_team_seasons(
            analytical_dir=Path("data/analytical"),
            through_season=cohort.training_seasons[-1],
        )
        pythagorean = fit_pythagorean_win_model(calibration)
        team_wins, _ = _team_win_evaluation(games, teams, pythagorean, model=MODEL_NAME)
        full_game_frames.append(games.assign(season=cohort.target_season))
        team_frames.append(teams.assign(season=cohort.target_season))
        win_frames.append(team_wins.assign(season=cohort.target_season))
        source_rows.append(
            {
                "target_season": cohort.target_season,
                "training_first_season": cohort.training_seasons[0],
                "training_last_season": cohort.training_seasons[-1],
                "train_season_count": len(cohort.training_seasons),
                "train_game_count": len(train_games),
                "target_regular_outcomes_used_for_fit": False,
                "target_playoff_outcomes_used_for_fit": False,
                "pythagorean_calibration_last_season": pythagorean.training_seasons[-1],
                "pythagorean_calibration_team_seasons": pythagorean.training_team_season_count,
                "full_game_missing_profile_slots": missing_profile_slots,
            }
        )
        print(f"target={cohort.target_season} scoring complete", flush=True)
    outputs = {
        "cohort_metrics": pd.DataFrame(cohort_metrics),
        "possession_predictions": pd.concat(prediction_frames, ignore_index=True),
        "regular_game_predictions": pd.concat(full_game_frames, ignore_index=True),
        "team_net_rating_predictions": pd.concat(team_frames, ignore_index=True),
        "team_win_predictions": pd.concat(win_frames, ignore_index=True),
        "source_states": pd.DataFrame(source_rows),
    }
    outputs["aggregate_metrics"] = _aggregate_metrics(outputs)
    run_dir = _write_candidate_run(
        outputs,
        target_seasons=tuple(seasons),
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        token_dir=Path(token_dir),
        output_dir=Path(output_dir),
    )
    print(f"wrote={run_dir}", flush=True)
    return outputs["aggregate_metrics"]


def _predict_profile(
    module: ProfileDeepSetsRapmModule,
    possessions: pd.DataFrame,
    vocabulary: dict[int, int],
    tokens: pd.DataFrame,
    training_data: ProfilePossessionDataModule,
    *,
    allow_missing_profiles: bool = False,
) -> np.ndarray:
    if training_data.scaler is None:
        raise RuntimeError("Profile scaler is unavailable")
    dataset = LazyProfilePossessionTensorDataset(
        possessions,
        vocabulary,
        profile_token_tables(
            tokens,
            training_data.scaler,
            tuple(possessions["season"].astype(str).unique()),
        ),
        allow_missing_profiles=allow_missing_profiles,
    )
    loader = profile_possession_loader(
        dataset,
        batch_size=8_192,
        shuffle=False,
    )
    values: list[np.ndarray] = []
    module.eval()
    with torch.inference_mode():
        for batch in loader:
            values.append(
                module(
                    batch["offense_player_indices"].to(module.device),
                    batch["defense_player_indices"].to(module.device),
                    batch["offense_profiles"].to(module.device),
                    batch["defense_profiles"].to(module.device),
                    batch["home_offense_sign"].to(module.device),
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(values)


def _score_possession_cohort(
    possessions: pd.DataFrame,
    predictions: np.ndarray,
    *,
    cohort: str,
    target_season: str,
    training_mean: float,
    train_season_count: int,
    train_game_count: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Persist one frozen possession cohort and score its mean-reference metrics."""

    actual = possessions["target_offense_margin"].to_numpy(dtype=float)
    signs = possessions["home_offense_sign"].to_numpy(dtype=float)
    mean_values = np.full(len(possessions), training_mean, dtype=float)
    mean_mse = mean_squared_error(actual, mean_values)
    mean_game_rmse = possession_game_margin_rmse(
        possessions["game_id"], actual, mean_values, signs
    )
    game_rmse = possession_game_margin_rmse(possessions["game_id"], actual, predictions, signs)
    frame = possessions.loc[
        :,
        [
            "season",
            "season_type",
            "game_id",
            "game_date",
            "possession_id",
            "possession_index",
            "period",
            "offense_team_id",
            "defense_team_id",
            "home_offense_sign",
            "target_offense_margin",
        ],
    ].copy()
    frame.insert(0, "cohort", cohort)
    frame.insert(1, "model", MODEL_NAME)
    frame["target_home_margin"] = actual * signs
    frame["prediction_offense_margin"] = predictions
    frame["prediction_home_margin"] = predictions * signs
    frame["residual_offense_margin"] = actual - predictions
    return (
        {
            "target_season": target_season,
            "cohort": cohort,
            "model": MODEL_NAME,
            "label": LABEL,
            "train_season_count": train_season_count,
            "train_game_count": train_game_count,
            "game_count": int(possessions["game_id"].nunique()),
            "possession_count": len(possessions),
            "possession_mse": mean_squared_error(actual, predictions),
            "possession_rmse": rmse(actual, predictions),
            "possession_mae": mean_absolute_error(actual, predictions),
            "eligible_possession_game_margin_rmse": game_rmse,
            "possession_skill_vs_mean": skill_score(
                mean_squared_error(actual, predictions), mean_mse
            ),
            "eligible_game_margin_skill_vs_mean": skill_score(
                game_rmse**2, mean_game_rmse**2
            ),
            "mean_reference_possession_rmse": float(np.sqrt(mean_mse)),
            "mean_reference_game_margin_rmse": mean_game_rmse,
        },
        frame,
    )


def _score_full_regular_stints(
    module: ProfileDeepSetsRapmModule,
    stints: pd.DataFrame,
    vocabulary: dict[int, int],
    tokens: pd.DataFrame,
    training_data: ProfilePossessionDataModule,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Score every reconstructed regular-season stint from both offensive views."""

    home = _stint_offense_frame(stints, offense="home")
    away = _stint_offense_frame(stints, offense="away")
    predicted_home_offense = _predict_profile(
        module, home, vocabulary, tokens, training_data, allow_missing_profiles=True
    )
    predicted_away_offense = _predict_profile(
        module, away, vocabulary, tokens, training_data, allow_missing_profiles=True
    )
    target_season = str(stints["season"].iat[0])
    available = set(
        tokens.loc[
            tokens["target_season"].astype(str).eq(target_season), "player_id"
        ].astype(int)
    )
    missing_profile_slots = sum(
        int(player_id) not in available
        for lineup in [*stints["home_player_ids"], *stints["away_player_ids"]]
        for player_id in lineup
    )
    base = stints.loc[
        :,
        [
            "game_id",
            "home_team_id",
            "away_team_id",
            "home_team_tricode",
            "away_team_tricode",
            "home_margin",
            "home_offensive_possessions",
            "away_offensive_possessions",
        ],
    ].copy()
    base["predicted_home_margin"] = (
        predicted_home_offense * base["home_offensive_possessions"].to_numpy(dtype=float)
        - predicted_away_offense * base["away_offensive_possessions"].to_numpy(dtype=float)
    )
    game_columns = [
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
    ]
    games = base.groupby(game_columns, as_index=False, sort=False).agg(
        actual_home_margin=("home_margin", "sum"),
        predicted_home_margin=("predicted_home_margin", "sum"),
    )
    if games["actual_home_margin"].eq(0.0).any():
        raise ValueError("Full-game scoring encountered an actual tied game")
    games["actual_home_win"] = games["actual_home_margin"].gt(0.0)
    games["predicted_home_win"] = games["predicted_home_margin"].gt(0.0)
    games["predicted_tie"] = np.isclose(games["predicted_home_margin"], 0.0)
    games["margin_error"] = games["predicted_home_margin"] - games["actual_home_margin"]
    possession_columns = ["home_offensive_possessions", "away_offensive_possessions"]
    home_teams = base.rename(
        columns={
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "home_margin": "actual_margin",
            "predicted_home_margin": "predicted_margin",
            "home_offensive_possessions": "possessions",
        }
    ).loc[:, ["team_id", "team_tricode", "possessions", "actual_margin", "predicted_margin"]]
    away_teams = base.rename(
        columns={
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "home_margin": "actual_margin",
            "predicted_home_margin": "predicted_margin",
            "away_offensive_possessions": "possessions",
        }
    ).loc[:, ["team_id", "team_tricode", "possessions", "actual_margin", "predicted_margin"]]
    del possession_columns
    away_teams["actual_margin"] *= -1.0
    away_teams["predicted_margin"] *= -1.0
    teams = pd.concat([home_teams, away_teams], ignore_index=True).groupby(
        ["team_id", "team_tricode"], as_index=False, sort=False
    ).agg(
        possessions=("possessions", "sum"),
        actual_total_margin=("actual_margin", "sum"),
        predicted_total_margin=("predicted_margin", "sum"),
    )
    teams["actual_net_rating"] = 100.0 * teams["actual_total_margin"] / teams["possessions"]
    teams["predicted_net_rating"] = (
        100.0 * teams["predicted_total_margin"] / teams["possessions"]
    )
    teams["net_rating_error"] = teams["predicted_net_rating"] - teams["actual_net_rating"]
    return (
        games.sort_values("game_id", kind="stable").reset_index(drop=True),
        teams.sort_values("team_id", kind="stable").reset_index(drop=True),
        missing_profile_slots,
    )


def _stint_offense_frame(stints: pd.DataFrame, *, offense: str) -> pd.DataFrame:
    """Translate stints into offense-oriented rows accepted by the neural model."""

    if offense not in {"home", "away"}:
        raise ValueError("Stint offense must be home or away")
    is_home = offense == "home"
    return pd.DataFrame(
        {
            "season": stints["season"].astype(str).to_numpy(),
            "game_id": stints["game_id"].astype(str).to_numpy(),
            "offense_player_ids": stints[
                "home_player_ids" if is_home else "away_player_ids"
            ].to_list(),
            "defense_player_ids": stints[
                "away_player_ids" if is_home else "home_player_ids"
            ].to_list(),
            "home_offense_sign": np.full(len(stints), 1.0 if is_home else -1.0),
            "target_offense_margin": np.zeros(len(stints), dtype=float),
        }
    )


def _aggregate_metrics(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pool the strict three-season regular and playoff leaderboard metrics."""

    predictions = outputs["possession_predictions"]
    metrics = outputs["cohort_metrics"]
    rows: list[dict[str, object]] = []
    for cohort, frame in predictions.groupby("cohort", sort=False):
        actual = frame["target_offense_margin"].to_numpy(dtype=float)
        predicted = frame["prediction_offense_margin"].to_numpy(dtype=float)
        games = (
            frame.assign(
                game_key=frame["season"].astype(str) + ":" + frame["game_id"].astype(str)
            )
            .groupby("game_key", as_index=False, sort=False)
            .agg(
                actual_home_margin=("target_home_margin", "sum"),
                predicted_home_margin=("prediction_home_margin", "sum"),
            )
        )
        errors = games["actual_home_margin"] - games["predicted_home_margin"]
        cohort_metrics = metrics.loc[metrics["cohort"].eq(cohort)]
        possession_weights = cohort_metrics["possession_count"].to_numpy(dtype=float)
        game_weights = cohort_metrics["game_count"].to_numpy(dtype=float)
        mean_mse = np.average(
            np.square(cohort_metrics["mean_reference_possession_rmse"]),
            weights=possession_weights,
        )
        mean_game_mse = np.average(
            np.square(cohort_metrics["mean_reference_game_margin_rmse"]),
            weights=game_weights,
        )
        rows.append(
            {
                "model": MODEL_NAME,
                "label": LABEL,
                "scope": f"pooled_{cohort}",
                "season_count": int(frame["season"].nunique()),
                "game_count": len(games),
                "possession_count": len(frame),
                "possession_rmse": rmse(actual, predicted),
                "possession_mae": mean_absolute_error(actual, predicted),
                "eligible_game_margin_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "possession_skill_vs_mean": skill_score(
                    mean_squared_error(actual, predicted), mean_mse
                ),
                "eligible_game_margin_skill_vs_mean": skill_score(
                    float(np.mean(np.square(errors))), mean_game_mse
                ),
            }
        )
    regular_games = outputs["regular_game_predictions"].copy()
    regular_games["game_id"] = (
        regular_games["season"].astype(str) + ":" + regular_games["game_id"].astype(str)
    )
    full = score_full_game_outcomes(regular_games)
    teams = outputs["team_net_rating_predictions"]
    team_wins = outputs["team_win_predictions"]
    actual_net = teams["actual_net_rating"].to_numpy(dtype=float)
    predicted_net = teams["predicted_net_rating"].to_numpy(dtype=float)
    rows.append(
        {
            "model": MODEL_NAME,
            "label": LABEL,
            "scope": "pooled_regular_full_games_and_teams",
            "season_count": int(regular_games["season"].nunique()),
            "game_count": len(regular_games),
            "full_game_margin_rmse": full["full_game_margin_rmse"],
            "full_game_margin_mae": full["full_game_margin_mae"],
            "game_winner_accuracy": full["game_winner_accuracy"],
            "team_net_rating_rmse": rmse(actual_net, predicted_net),
            "team_net_rating_mae": mean_absolute_error(actual_net, predicted_net),
            "team_net_rating_pearson": float(pearsonr(actual_net, predicted_net).statistic),
            "team_net_rating_spearman": float(spearmanr(actual_net, predicted_net).statistic),
            "pythagorean_win_rmse": rmse(
                team_wins["wins"], team_wins["pythagorean_wins"]
            ),
            "pythagorean_win_mae": mean_absolute_error(
                team_wins["wins"], team_wins["pythagorean_wins"]
            ),
            "pythagorean_win_spearman": float(
                spearmanr(team_wins["wins"], team_wins["pythagorean_wins"]).statistic
            ),
        }
    )
    return pd.DataFrame(rows)


def _write_candidate_run(
    outputs: dict[str, pd.DataFrame],
    *,
    target_seasons: tuple[str, ...],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    token_dir: Path,
    output_dir: Path,
) -> Path:
    """Write an immutable frozen Deep Sets candidate artifact."""

    now = datetime.now(UTC)
    run_id = (
        f"profile-deep-sets-{target_seasons[0]}-to-{target_seasons[-1]}-"
        f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = output_dir / f"{target_seasons[0]}_to_{target_seasons[-1]}"
    run_dir = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for name, frame in outputs.items():
            frame.to_parquet(temporary / f"{name}.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "label": LABEL,
            "created_at": now.isoformat(),
            "target_seasons": list(target_seasons),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "profile_token_dir": str(token_dir),
            "information_boundary": (
                "each target model is trained only on completed regular seasons before the "
                "target; target regular and playoff outcomes are used only for scoring"
            ),
            "evaluation_contract": (
                "possession, eligible-game, reconstructed full-game, team-net-rating, and "
                "Pythagorean-win metrics pooled over three frozen target seasons"
            ),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        records = [
            {
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(run_dir)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return run_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
