"""Frozen residual token models over the additive-only NAIL baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset

from nba_lineup_model.modeling.contextual_prior import (
    _contextual_stint_predictions,
    _lineup_effects,
    _score_possessions,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_PANEL_PATH
from nba_lineup_model.modeling.forward_nail_additive_only_context import (
    MODEL_NAME as ADDITIVE_ONLY_MODEL_NAME,
)
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    DEFAULT_ARTIFACTS_DIR,
)
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    _aggregate_metrics,
    _latest_recursive_run,
    _load_state,
    _target_contextual_profiles,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import (
    _game_prediction_frame,
    _historical_team_seasons,
    _read_playoff_possessions,
    _read_regular_possessions,
    _recover_home_intercept,
    _team_net_rating_metrics,
    _team_win_evaluation,
    fit_pythagorean_win_model,
    score_possession_cohort,
)
from nba_lineup_model.modeling.neural import NeuralTrainingConfig, _trainer
from nba_lineup_model.modeling.neural_data import read_neural_possessions
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.models.nail_token_residual import (
    NailTokenResidualModule,
    TokenResidualArchitecture,
)

NAIL_TOKEN_FEATURE_COLUMNS = (
    "three_pa_per_100",
    "three_pm_per_100",
    "assists_per_100",
    "turnovers_per_100",
    "usage_per_100",
    "offensive_rebound_pct",
    "steals_per_100",
    "blocks_per_100",
)
MODEL_NAMES = {
    "token_mlp": "nail_token_mlp_residual",
    "set_attention": "nail_set_attention_residual",
}
MODEL_LABELS = {
    "token_mlp": "NAIL token-MLP residual",
    "set_attention": "NAIL within-unit Set Attention residual",
}
DEFAULT_ARCHITECTURES: tuple[TokenResidualArchitecture, ...] = (
    "token_mlp",
    "set_attention",
)
DEFAULT_OUTPUT_DIR = Path("artifacts/models/analysis/nail_token_residual_frozen")
DEFAULT_RESIDUAL_CACHE_DIR = Path("artifacts/models/analysis/nail_token_residual_inputs")
DEFAULT_STATE_THROUGH_SEASON = "2025-26"


@dataclass(frozen=True)
class NailTokenScaler:
    """Training-season moments for the strict eight-feature token contract."""

    means: np.ndarray
    scales: np.ndarray
    player_season_count: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] != len(NAIL_TOKEN_FEATURE_COLUMNS):
            raise ValueError("NAIL token values do not match the feature contract")
        return (values - self.means) / self.scales


@dataclass(frozen=True)
class NailSeasonTokenTable:
    """One compact standardized player-token table for a target season."""

    player_columns: dict[int, int]
    values: torch.Tensor


class NailResidualStintDataset(Dataset[dict[str, torch.Tensor]]):
    """Compact stint rows with vectorized player-season token lookup."""

    def __init__(
        self,
        frame: pd.DataFrame,
        token_tables: Mapping[str, NailSeasonTokenTable],
    ) -> None:
        if frame.empty:
            raise ValueError("NAIL residual stint dataset cannot be empty")
        required = {
            "season",
            "home_player_ids",
            "away_player_ids",
            "target_residual_net_rating",
            "possessions",
        }
        if missing := required - set(frame):
            raise ValueError(f"NAIL residual stint rows are missing: {sorted(missing)}")
        seasons = tuple(token_tables)
        season_to_index = {season: index for index, season in enumerate(seasons)}
        season_values = frame["season"].astype(str).to_numpy()
        if missing_seasons := sorted(set(season_values) - set(seasons)):
            raise ValueError(f"NAIL token tables are missing seasons: {missing_seasons}")
        self.season_indices = torch.as_tensor(
            [season_to_index[season] for season in season_values],
            dtype=torch.long,
        )
        self.home_indices = torch.empty((len(frame), 5), dtype=torch.long)
        self.away_indices = torch.empty((len(frame), 5), dtype=torch.long)
        tables = tuple(token_tables.values())
        for season, table_index in season_to_index.items():
            rows = np.flatnonzero(season_values == season)
            table = tables[table_index]
            self.home_indices[rows] = _encode_lineups(
                frame.iloc[rows]["home_player_ids"], table.player_columns
            )
            self.away_indices[rows] = _encode_lineups(
                frame.iloc[rows]["away_player_ids"], table.player_columns
            )
        max_players = max(table.values.shape[0] for table in tables)
        self.token_values = torch.zeros(
            (len(tables), max_players, len(NAIL_TOKEN_FEATURE_COLUMNS)),
            dtype=torch.float32,
        )
        for table_index, table in enumerate(tables):
            self.token_values[table_index, : table.values.shape[0]] = table.values
        self.target = torch.as_tensor(
            frame["target_residual_net_rating"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.possessions = torch.as_tensor(
            frame["possessions"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.batch([index])

    def batch(self, rows: list[int] | torch.Tensor) -> dict[str, torch.Tensor]:
        indices = torch.as_tensor(rows, dtype=torch.long)
        seasons = self.season_indices[indices]
        home = self.home_indices[indices]
        away = self.away_indices[indices]
        return {
            "home_profiles": self.token_values[seasons.unsqueeze(1), home],
            "away_profiles": self.token_values[seasons.unsqueeze(1), away],
            "target_residual": self.target[indices],
            "possessions": self.possessions[indices],
        }


class _EpochLogger(Callback):
    def __init__(self, target: str, architecture: str, epochs: int) -> None:
        super().__init__()
        self.target = target
        self.architecture = architecture
        self.epochs = epochs

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        value = trainer.callback_metrics.get("train_weighted_mse")
        mse = float(value.detach().cpu()) if value is not None else float("nan")
        print(
            f"target={self.target} architecture={self.architecture} "
            f"epoch={trainer.current_epoch + 1}/{self.epochs} "
            f"train_weighted_rmse={np.sqrt(mse):.4f}",
            flush=True,
        )


def fit_nail_token_scaler(
    tokens: pd.DataFrame,
    residual_stints: pd.DataFrame,
) -> NailTokenScaler:
    """Fit token normalization on player-seasons observed in training stints."""

    keys = {
        (str(season), int(player_id))
        for season, home, away in zip(
            residual_stints["season"],
            residual_stints["home_player_ids"],
            residual_stints["away_player_ids"],
            strict=True,
        )
        for player_id in (*home, *away)
    }
    indexed = tokens.set_index(["target_season", "player_id"], verify_integrity=True)
    if missing := sorted(keys - set(indexed.index)):
        raise ValueError(f"NAIL token mart is missing training keys: {missing[:10]}")
    ordered = sorted(keys)
    values = indexed.loc[ordered, list(NAIL_TOKEN_FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("NAIL token scaler cannot use non-finite values")
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0.0] = 1.0
    return NailTokenScaler(means=means, scales=scales, player_season_count=len(ordered))


def nail_token_tables(
    tokens: pd.DataFrame,
    scaler: NailTokenScaler,
    seasons: Sequence[str],
) -> dict[str, NailSeasonTokenTable]:
    """Build compact standardized lookup tables for selected seasons."""

    output: dict[str, NailSeasonTokenTable] = {}
    for season in seasons:
        rows = tokens.loc[tokens["target_season"].astype(str).eq(season)].copy()
        if rows.empty or rows["player_id"].duplicated().any():
            raise ValueError(f"Invalid NAIL profile tokens for {season}")
        values = scaler.transform(
            rows.loc[:, NAIL_TOKEN_FEATURE_COLUMNS].to_numpy(dtype=float)
        ).astype(np.float32)
        table_values = np.zeros(
            (len(rows) + 1, len(NAIL_TOKEN_FEATURE_COLUMNS)), dtype=np.float32
        )
        table_values[1:] = values
        output[season] = NailSeasonTokenTable(
            player_columns={
                int(player_id): index
                for index, player_id in enumerate(rows["player_id"], start=1)
            },
            values=torch.as_tensor(table_values),
        )
    return output


def build_frozen_residual_stints(
    target: str,
    *,
    state: object,
    profiles: pd.DataFrame,
    analytical_dir: Path | str = Path("data/analytical"),
) -> pd.DataFrame:
    """Aggregate target stints against the immediately prior additive-only state."""

    source = _previous_season(target)
    stints = read_rapm_stints(target, analytical_dir=analytical_dir)
    available_profiles = set(profiles["player_id"].astype(int))
    profile_complete = np.array(
        [
            all(int(player_id) in available_profiles for player_id in (*home, *away))
            for home, away in zip(
                stints["home_player_ids"], stints["away_player_ids"], strict=True
            )
        ],
        dtype=bool,
    )
    excluded_stints = int((~profile_complete).sum())
    stints = stints.loc[profile_complete].reset_index(drop=True)
    if stints.empty:
        raise ValueError(f"No profile-complete residual stints remain for {target}")
    priors = state.priors.loc[  # type: ignore[attr-defined]
        state.priors["season"].eq(target), ["player_id", "prior_rapm"]  # type: ignore[attr-defined]
    ]
    prior_map = dict(zip(priors["player_id"].astype(int), priors["prior_rapm"], strict=True))
    effects, unknown = _lineup_effects(stints, prior_map)
    source_coefficients = state.coefficients.loc[  # type: ignore[attr-defined]
        state.coefficients["season"].eq(source), ["player_id", "rapm"]  # type: ignore[attr-defined]
    ]
    home_intercept = _recover_home_intercept(
        read_rapm_stints(source, analytical_dir=analytical_dir), source_coefficients
    )
    context_model = state.context_models[source]  # type: ignore[attr-defined]
    home = [tuple(int(player_id) for player_id in ids) for ids in stints["home_player_ids"]]
    away = [tuple(int(player_id) for player_id in ids) for ids in stints["away_player_ids"]]
    pairs = pd.DataFrame({"home": home, "away": away}).drop_duplicates()
    context_values = context_model.predict_lineups(
        pairs["home"].tolist(), pairs["away"].tolist(), profiles
    )
    context_map = dict(
        zip(zip(pairs["home"], pairs["away"], strict=True), context_values, strict=True)
    )
    context = np.array(
        [context_map[(home_ids, away_ids)] for home_ids, away_ids in zip(home, away, strict=True)]
    )
    frame = pd.DataFrame(
        {
            "season": target,
            "home_player_ids": home,
            "away_player_ids": away,
            "possessions": stints["possessions"].to_numpy(dtype=float),
            "actual_home_margin": stints["home_margin"].to_numpy(dtype=float),
            "baseline_home_net_rating": effects + home_intercept + context,
            "unknown_player_slots": unknown,
            "excluded_profile_stints": excluded_stints,
        }
    )
    frame["weighted_residual"] = (
        100.0 * frame["actual_home_margin"]
        - frame["baseline_home_net_rating"] * frame["possessions"]
    )
    grouped = (
        frame.groupby(
            ["season", "home_player_ids", "away_player_ids"],
            as_index=False,
            sort=False,
        )
        .agg(
            possessions=("possessions", "sum"),
            weighted_residual=("weighted_residual", "sum"),
            stint_count=("possessions", "size"),
            unknown_player_slots=("unknown_player_slots", "sum"),
            excluded_profile_stints=("excluded_profile_stints", "max"),
        )
        .reset_index(drop=True)
    )
    grouped["target_residual_net_rating"] = (
        grouped["weighted_residual"] / grouped["possessions"]
    )
    return grouped.drop(columns="weighted_residual")


def run_nail_token_residual_frozen_backtest(
    *,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    architectures: Sequence[TokenResidualArchitecture] = DEFAULT_ARCHITECTURES,
    epochs: int = 5,
    batch_size: int = 4_096,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    state_through_season: str = DEFAULT_STATE_THROUGH_SEASON,
    training_start_season: str | None = None,
    residual_cache_dir: Path | str = DEFAULT_RESIDUAL_CACHE_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
) -> Path:
    """Fit the token-MLP/attention ladder and score three frozen seasons."""

    targets = tuple(str(season) for season in seasons)
    selected_architectures = tuple(architectures)
    if epochs < 1 or batch_size < 1:
        raise ValueError("NAIL token residual training dimensions must be positive")
    if not selected_architectures or any(
        value not in MODEL_NAMES for value in selected_architectures
    ):
        raise ValueError("NAIL token residual architectures are invalid")
    panel_path = Path(player_season_panel_path)
    panel = pd.read_parquet(panel_path)
    available_seasons = tuple(
        sorted(panel["season"].astype(str).unique(), key=lambda value: int(value[:4]))
    )
    candidate = BacktestModel(ADDITIVE_ONLY_MODEL_NAME, "NAIL additive-only context")
    run_dir = _latest_recursive_run(
        Path(artifacts_dir) / ADDITIVE_ONLY_MODEL_NAME / state_through_season
    )
    state = _load_state(run_dir, candidate=candidate, target_seasons=targets)
    prior_seasons = set(state.priors["season"].astype(str))
    coefficient_seasons = set(state.coefficients["season"].astype(str))
    required_seasons = tuple(
        season
        for season in available_seasons
        if season <= targets[-1]
        and (training_start_season is None or season >= training_start_season or season in targets)
        and season in prior_seasons
        and _previous_season(season) in coefficient_seasons
        and _previous_season(season) in state.context_models
    )
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(targets[-1])],
        through_season=targets[-1],
        analytical_dir=analytical_dir,
    )
    profiles = {
        season: _target_contextual_profiles(
            season,
            panel=panel,
            exposure_cohort=exposure_cohort,
            analytical_dir=Path(analytical_dir),
        )
        for season in required_seasons
    }
    tokens = pd.concat(
        [frame.assign(target_season=season) for season, frame in profiles.items()],
        ignore_index=True,
    )
    if missing := set(NAIL_TOKEN_FEATURE_COLUMNS) - set(tokens):
        raise ValueError(f"Current NAIL profiles are missing token features: {sorted(missing)}")
    profile_source_sha = _sha256_file(panel_path)
    cache_root = (
        Path(residual_cache_dir)
        / f"{state.run_id}-{profile_source_sha.removeprefix('sha256:')[:12]}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    residual_frames: dict[str, pd.DataFrame] = {}
    residual_seasons = tuple(season for season in required_seasons if season < targets[-1])
    for season in residual_seasons:
        cache_path = cache_root / f"{season}.parquet"
        if cache_path.is_file():
            print(f"season={season} loading cached additive-only residual stints", flush=True)
            residual_frames[season] = _read_residual_cache(cache_path)
        else:
            print(f"season={season} building frozen additive-only residual stints", flush=True)
            residual_frames[season] = build_frozen_residual_stints(
                season,
                state=state,
                profiles=profiles[season],
                analytical_dir=analytical_dir,
            )
            residual_frames[season].to_parquet(cache_path, index=False)

    tables: dict[str, list[pd.DataFrame]] = {
        "cohort_metrics": [],
        "possession_predictions": [],
        "game_predictions": [],
        "regular_game_predictions": [],
        "team_net_rating_predictions": [],
        "team_net_rating_metrics": [],
        "team_win_predictions": [],
        "team_win_metrics": [],
        "source_states": [],
    }
    checkpoints: dict[tuple[str, str], dict[str, object]] = {}
    for target in targets:
        training_seasons = tuple(season for season in required_seasons if season < target)
        training_frame = pd.concat(
            [residual_frames[season] for season in training_seasons], ignore_index=True
        )
        scaler = fit_nail_token_scaler(tokens, training_frame)
        training_tables = nail_token_tables(tokens, scaler, training_seasons)
        dataset = NailResidualStintDataset(training_frame, training_tables)
        for architecture in selected_architectures:
            model_name = MODEL_NAMES[architecture]
            label = MODEL_LABELS[architecture]
            print(
                f"target={target} architecture={architecture} "
                f"train_seasons={len(training_seasons)} rows={len(training_frame):,} "
                f"possessions={training_frame['possessions'].sum():,.0f}",
                flush=True,
            )
            module = _fit_module(
                dataset,
                target=target,
                architecture=architecture,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            evaluation = _evaluate_target(
                target,
                module=module,
                scaler=scaler,
                tokens=tokens,
                state=state,
                profiles=profiles[target],
                model_name=model_name,
                analytical_dir=Path(analytical_dir),
                curated_dir=Path(curated_dir),
                batch_size=batch_size,
            )
            _append_evaluation(
                tables,
                evaluation,
                target=target,
                model_name=model_name,
                label=label,
            )
            checkpoints[(target, architecture)] = {
                "state_dict": {key: value.cpu() for key, value in module.state_dict().items()},
                "feature_columns": NAIL_TOKEN_FEATURE_COLUMNS,
                "scaler_means": scaler.means,
                "scaler_scales": scaler.scales,
                "training_seasons": training_seasons,
                "architecture": architecture,
                "hidden_dim": 32,
                "attention_heads": 4,
                "attention_layers": 2,
                "feedforward_dim": 64,
            }
            print(f"target={target} architecture={architecture} scoring complete", flush=True)
    outputs = {name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()}
    outputs["aggregate_metrics"] = _aggregate_metrics(outputs)
    return _write_run(
        outputs,
        checkpoints=checkpoints,
        targets=targets,
        architectures=selected_architectures,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        source_run=run_dir,
        output_dir=Path(output_dir),
    )


def _fit_module(
    dataset: NailResidualStintDataset,
    *,
    target: str,
    architecture: TokenResidualArchitecture,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> NailTokenResidualModule:
    L.seed_everything(17, workers=True, verbose=False)
    module = NailTokenResidualModule(
        len(NAIL_TOKEN_FEATURE_COLUMNS),
        architecture=architecture,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    loader = DataLoader(
        range(len(dataset)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(17),
        collate_fn=dataset.batch,
    )
    trainer = _trainer(
        NeuralTrainingConfig(
            batch_size=batch_size,
            max_epochs=epochs,
            learning_rates=(learning_rate,),
            weight_decays=(weight_decay,),
        ),
        max_epochs=epochs,
        callbacks=[_EpochLogger(target, architecture, epochs)],
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, train_dataloaders=loader)
    return module


def _evaluate_target(
    target: str,
    *,
    module: NailTokenResidualModule,
    scaler: NailTokenScaler,
    tokens: pd.DataFrame,
    state: object,
    profiles: pd.DataFrame,
    model_name: str,
    analytical_dir: Path,
    curated_dir: Path,
    batch_size: int,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    source = _previous_season(target)
    context_model = state.context_models[source]  # type: ignore[attr-defined]

    def predictor(
        home_lineups: Sequence[Sequence[int]],
        away_lineups: Sequence[Sequence[int]],
        profile_frame: pd.DataFrame,
    ) -> np.ndarray:
        baseline = context_model.predict_lineups(home_lineups, away_lineups, profile_frame)
        correction = _predict_lineups(
            module,
            target=target,
            home_lineups=home_lineups,
            away_lineups=away_lineups,
            tokens=tokens,
            scaler=scaler,
            batch_size=batch_size,
        )
        return np.asarray(baseline, dtype=float) + correction

    prior_frame = state.priors.loc[  # type: ignore[attr-defined]
        state.priors["season"].eq(target), ["player_id", "prior_rapm"]  # type: ignore[attr-defined]
    ].rename(columns={"prior_rapm": "prior_rapm_mean"})
    source_coefficients = state.coefficients.loc[  # type: ignore[attr-defined]
        state.coefficients["season"].eq(source), ["player_id", "rapm"]  # type: ignore[attr-defined]
    ]
    source_home_intercept = _recover_home_intercept(
        read_rapm_stints(source, analytical_dir=analytical_dir), source_coefficients
    )
    source_possessions, _ = _read_regular_possessions(
        source, analytical_dir=analytical_dir, curated_dir=curated_dir
    )
    source_mean = float(source_possessions["target_offense_margin"].mean())
    regular = _score_possessions(
        read_neural_possessions(target, analytical_dir=analytical_dir),
        cohort="regular_season",
        profiles=profiles,
        context_predictor=predictor,
        priors=prior_frame,
        source_mean=source_mean,
        source_home_intercept=source_home_intercept,
    )
    predictions = [regular]
    metrics = [score_possession_cohort(regular, source_mean=source_mean, model=model_name)]
    playoff_path = curated_dir / "possession_segments" / target / "playoffs" / "_manifest.json"
    if playoff_path.is_file():
        playoffs, _ = _read_playoff_possessions(target, curated_dir)
        playoff_predictions = _score_possessions(
            playoffs,
            cohort="playoffs",
            profiles=profiles,
            context_predictor=predictor,
            priors=prior_frame,
            source_mean=source_mean,
            source_home_intercept=source_home_intercept,
        )
        predictions.append(playoff_predictions)
        metrics.append(
            score_possession_cohort(
                playoff_predictions,
                source_mean=source_mean,
                model=model_name,
            )
        )
    regular_games, teams = _contextual_stint_predictions(
        read_rapm_stints(target, analytical_dir=analytical_dir),
        profiles=profiles,
        context_predictor=predictor,
        priors=prior_frame,
        source_home_intercept=source_home_intercept,
    )
    pythagorean = fit_pythagorean_win_model(
        _historical_team_seasons(analytical_dir=analytical_dir, through_season=source)
    )
    team_wins, team_win_metrics = _team_win_evaluation(
        regular_games, teams, pythagorean, model=model_name
    )
    return {
        "source_state": {
            "target_season": target,
            "source_season": source,
            "source_additive_only_run_id": state.run_id,  # type: ignore[attr-defined]
            "target_regular_outcomes_used_for_fit": False,
            "target_playoff_outcomes_used_for_fit": False,
            "token_information_cutoff": "end_of_prior_regular_season",
        },
        "cohort_metrics": pd.concat(metrics, ignore_index=True),
        "possession_predictions": pd.concat(predictions, ignore_index=True),
        "game_predictions": _game_prediction_frame(pd.concat(predictions, ignore_index=True)),
        "regular_game_predictions": regular_games,
        "team_net_rating_predictions": teams,
        "team_net_rating_metrics": _team_net_rating_metrics(teams, model=model_name),
        "team_win_predictions": team_wins,
        "team_win_metrics": team_win_metrics,
    }


def _predict_lineups(
    module: NailTokenResidualModule,
    *,
    target: str,
    home_lineups: Sequence[Sequence[int]],
    away_lineups: Sequence[Sequence[int]],
    tokens: pd.DataFrame,
    scaler: NailTokenScaler,
    batch_size: int,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "season": target,
            "home_player_ids": [tuple(int(value) for value in row) for row in home_lineups],
            "away_player_ids": [tuple(int(value) for value in row) for row in away_lineups],
            "target_residual_net_rating": 0.0,
            "possessions": 1.0,
        }
    )
    dataset = NailResidualStintDataset(
        frame,
        nail_token_tables(tokens, scaler, (target,)),
    )
    loader = DataLoader(
        range(len(dataset)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.batch,
    )
    values: list[np.ndarray] = []
    module.eval()
    with torch.inference_mode():
        for batch in loader:
            values.append(
                module(
                    batch["home_profiles"].to(module.device),
                    batch["away_profiles"].to(module.device),
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(values)


def _append_evaluation(
    tables: dict[str, list[pd.DataFrame]],
    evaluation: dict[str, pd.DataFrame | dict[str, object]],
    *,
    target: str,
    model_name: str,
    label: str,
) -> None:
    for key in tables:
        value = evaluation["source_state"] if key == "source_states" else evaluation[key]
        frame = pd.DataFrame([value]) if isinstance(value, dict) else value.copy()
        frame = frame.drop(columns=["season", "model", "label"], errors="ignore")
        frame.insert(0, "season", target)
        frame.insert(1, "model", model_name)
        frame.insert(2, "label", label)
        tables[key].append(frame)


def _write_run(
    outputs: dict[str, pd.DataFrame],
    *,
    checkpoints: dict[tuple[str, str], dict[str, object]],
    targets: tuple[str, ...],
    architectures: tuple[TokenResidualArchitecture, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    source_run: Path,
    output_dir: Path,
) -> Path:
    now = datetime.now(UTC)
    run_id = (
        f"nail-token-residual-{targets[0]}-to-{targets[-1]}-"
        f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = output_dir / f"{targets[0]}_to_{targets[-1]}"
    run_dir = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for name, frame in outputs.items():
            frame.to_parquet(temporary / f"{name}.parquet", index=False)
        for (target, architecture), payload in checkpoints.items():
            torch.save(payload, temporary / f"model-{target}-{architecture}.pt")
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": now.isoformat(),
            "target_seasons": list(targets),
            "architectures": list(architectures),
            "token_feature_columns": list(NAIL_TOKEN_FEATURE_COLUMNS),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "source_additive_only_run": str(source_run),
            "training_contract": (
                "possession-weighted unique stint-lineup residuals reconstructed from each "
                "season's immediately prior additive-only NAIL state"
            ),
            "information_boundary": (
                "each frozen target excludes its regular-season and playoff outcomes from "
                "training; tokens contain only prior-season player profiles"
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
    print(f"wrote={run_dir}", flush=True)
    return run_dir


def _encode_lineups(
    lineups: pd.Series,
    player_columns: Mapping[int, int],
) -> torch.Tensor:
    values = np.empty((len(lineups), 5), dtype=np.int64)
    for row, lineup in enumerate(lineups):
        ids = tuple(int(player_id) for player_id in lineup)
        if len(ids) != 5 or len(set(ids)) != 5:
            raise ValueError("NAIL token residual requires five unique players per side")
        try:
            values[row] = [player_columns[player_id] for player_id in ids]
        except KeyError as error:
            raise ValueError(f"NAIL token table is missing player {error.args[0]}") from error
    return torch.as_tensor(values, dtype=torch.long)


def _read_residual_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for column in ("home_player_ids", "away_player_ids"):
        frame[column] = frame[column].map(lambda values: tuple(int(value) for value in values))
    return frame


def _previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start:04d}-{str(start + 1)[-2:]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit frozen NAIL token-MLP and within-unit Set Attention residuals"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=sorted(MODEL_NAMES),
        default=list(DEFAULT_ARCHITECTURES),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    run_nail_token_residual_frozen_backtest(
        seasons=tuple(args.seasons),
        architectures=tuple(args.architectures),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
