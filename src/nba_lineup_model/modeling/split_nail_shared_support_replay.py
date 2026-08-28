"""Replay persisted Split NAIL states on the shared frozen-possession support.

This module is deliberately a *no-refit* verifier.  It proves that Split NAIL
can be evaluated on the same valid possession and game support as the frozen
leaderboard before any new Split NAIL training run is considered.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    lineup_side_context_features,
)
from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    _previous_season,
)
from nba_lineup_model.modeling.forward_split_nail import (
    FROZEN_SEASONS,
    _attach_back_to_back_flags,
    _frozen_scoring_prediction,
)
from nba_lineup_model.modeling.frozen_game_outcomes import score_full_game_outcomes
from nba_lineup_model.modeling.frozen_prior_evaluation import (
    _historical_team_seasons,
    _read_playoff_possessions,
    _team_net_rating_metrics,
    _team_win_evaluation,
    _game_prediction_frame,
    fit_pythagorean_win_model,
    score_possession_cohort,
)
from nba_lineup_model.modeling.neural_data import read_neural_possessions
from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.modeling.split_nail import SplitNailSeasonFit
from nba_lineup_model.modeling.stints import read_rapm_stints

MODEL_NAME = "forward_split_nail_rapm"
BASELINE_MODEL = "forward_nail_rapm_v1212_back_to_back"
RUN_PREFIX = "split-nail-shared-support-replay"
DEFAULT_SPLIT_RUN_DIR = Path(
    "artifacts/models/forward_split_nail_rapm/2025-26/"
    "forward-split-nail-rapm-2025-26-20260826T134158Z-8fe83ba0"
)
DEFAULT_BASELINE_RUN_DIR = Path(
    "artifacts/models/nail_v1212_back_to_back_frozen_backtest/"
    "frozen_multiseason_backtest/2023-24_to_2025-26/"
    "frozen_multiseason_backtest-2023-24-to-2025-26-20260824T141701Z-1b7f1e15"
)
DEFAULT_AUDITS_DIR = Path("artifacts/audits")


def replay_shared_support(
    *,
    split_run_dir: Path | str = DEFAULT_SPLIT_RUN_DIR,
    baseline_run_dir: Path | str = DEFAULT_BASELINE_RUN_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    audits_dir: Path | str = DEFAULT_AUDITS_DIR,
    game_catalog_path: Path | str = Path("data/catalog/games.parquet"),
    seasons: tuple[str, ...] = FROZEN_SEASONS,
) -> Path:
    """Score existing Split states and assert exact leaderboard support parity."""

    split_dir = Path(split_run_dir)
    baseline_dir = Path(baseline_run_dir)
    prior_frame = pd.read_parquet(split_dir / "player_season_ratings.parquet")
    required_prior_columns = {"season", "player_id", "scalar_prior"}
    missing_prior_columns = required_prior_columns - set(prior_frame)
    if missing_prior_columns:
        raise ValueError(
            "Standalone Split NAIL artifact lacks persisted scalar prior columns: "
            f"{sorted(missing_prior_columns)}"
        )
    scalar_priors = {
        str(season): dict(
            zip(group["player_id"].astype(int), group["scalar_prior"], strict=True)
        )
        for season, group in prior_frame.groupby("season", sort=False)
    }
    loaded_fits: dict[str, SplitNailSeasonFit] = joblib.load(
        split_dir / "season_split_nail_models.joblib"
    )
    required_fit_seasons = {*seasons, *(_previous_season(season) for season in seasons)}
    fits = {season: loaded_fits[season] for season in required_fit_seasons if season in loaded_fits}
    # Each persisted design retains its sparse training matrix. The replay needs
    # only the source and target states for the frozen target seasons.
    del loaded_fits
    baseline_predictions = pd.read_parquet(baseline_dir / "possession_predictions.parquet")
    baseline_states = pd.read_parquet(baseline_dir / "source_states.parquet")
    baseline_regular_games = pd.read_parquet(baseline_dir / "regular_game_predictions.parquet")
    panel = pd.read_parquet(player_season_panel_path)
    schedule_features = build_back_to_back_game_features(pd.read_parquet(game_catalog_path))
    prediction_tables: list[pd.DataFrame] = []
    metric_tables: list[pd.DataFrame] = []
    full_game_tables: list[pd.DataFrame] = []
    team_tables: list[pd.DataFrame] = []
    win_tables: list[pd.DataFrame] = []
    support_rows: list[dict[str, object]] = []
    for season in seasons:
        source = _previous_season(season)
        print(f"Replaying persisted Split NAIL state {source} onto {season}", flush=True)
        if source not in fits:
            raise ValueError(f"Split run lacks persisted source state for {source}")
        target = _attach_back_to_back_flags(
            read_neural_possessions(season, analytical_dir=analytical_dir), schedule_features
        )
        playoff_target = _attach_back_to_back_flags(
            _read_playoff_possessions(season, curated_dir)[0], schedule_features
        )
        print(
            f"  Loaded shared possession support: {len(target):,} regular; "
            f"{len(playoff_target):,} playoff rows",
            flush=True,
        )
        if season not in fits:
            raise ValueError(f"Split run lacks persisted target feature state for {season}")
        target_stints = _attach_back_to_back_flags(
            read_rapm_stints(season, analytical_dir=analytical_dir),
            schedule_features,
        )
        target_features = _persisted_unit_features(
            target_stints,
            fits[season],
        )
        # The persisted target design contains regular-season stints only.  A
        # playoff unit can therefore be valid even when it has no persisted
        # target feature row.  Build one fallback profile state over both
        # cohorts so neither replay consults target outcomes.
        all_target_possessions = pd.concat([target, playoff_target], ignore_index=True)
        missing_lineups = _missing_unit_lineups(all_target_possessions, target_features)
        fallback_profiles = (
            _missing_unit_profiles(
                target_season=season,
                lineups=missing_lineups,
                split_run_dir=split_dir,
                panel=panel,
                analytical_dir=analytical_dir,
                curated_dir=curated_dir,
            )
            if missing_lineups
            else None
        )
        print(
            "  Recovered persisted target unit features: "
            f"{len(target_features):,} units; {len(missing_lineups):,} fallbacks",
            flush=True,
        )
        regular_predictions = _score_split_possessions_batched(
            target, fit=fits[source], scalar_priors=scalar_priors.get(season, {}),
            source_differences=_side_differences(fits[source]), unit_features=target_features,
            fallback_profiles=fallback_profiles, cohort="regular_season",
        )
        playoff_predictions = _score_split_possessions_batched(
            playoff_target, fit=fits[source], scalar_priors=scalar_priors.get(season, {}),
            source_differences=_side_differences(fits[source]), unit_features=target_features,
            fallback_profiles=fallback_profiles, cohort="playoffs",
        )
        print("  Scored regular and playoff shared-support states", flush=True)
        for predictions in (regular_predictions, playoff_predictions):
            predictions.insert(1, "model", MODEL_NAME)
            predictions.insert(2, "label", "Standalone constrained Split NAIL")

        baseline = _baseline_predictions(baseline_predictions, season, "regular_season")
        baseline_game = _game_prediction_frame(baseline)
        support_row = _assert_shared_support(regular_predictions, baseline, baseline_game, season)
        support_rows.append(support_row)
        if not support_row["support_matches"]:
            print(f"  Support mismatch: {support_row}", flush=True)
        else:
            print("  Verified exact possession and game support", flush=True)
        metric_tables.append(
            score_possession_cohort(
                regular_predictions.drop(columns=["season", "model", "label"]),
                source_mean=_baseline_source_mean(baseline_states, season),
                model=MODEL_NAME,
            ).assign(season=season)
        )
        playoff_baseline = _baseline_predictions(baseline_predictions, season, "playoffs")
        playoff_support = _assert_playoff_support(playoff_predictions, playoff_baseline, season)
        support_row.update(playoff_support)
        metric_tables.append(
            score_possession_cohort(
                playoff_predictions.drop(columns=["season", "model", "label"]),
                source_mean=_baseline_source_mean(baseline_states, season), model=MODEL_NAME,
            ).assign(season=season)
        )
        full_games, teams = _score_full_target_stints(
            season,
            target_stints,
            target_fit=fits[season],
            source_fit=fits[source],
            scalar_priors=scalar_priors.get(season, {}),
        )
        support_row.update(
            _assert_full_game_support(
                full_games,
                baseline_regular_games,
                season,
            )
        )
        pythagorean = fit_pythagorean_win_model(
            _historical_team_seasons(
                analytical_dir=Path(analytical_dir),
                through_season=source,
            )
        )
        wins, _ = _team_win_evaluation(
            full_games,
            teams,
            pythagorean,
            model=MODEL_NAME,
        )
        full_game_tables.append(full_games.assign(season=season, model=MODEL_NAME))
        team_tables.append(teams.assign(season=season, model=MODEL_NAME))
        win_tables.append(wins.assign(season=season, model=MODEL_NAME))
        prediction_tables.extend([regular_predictions, playoff_predictions])
        del target, playoff_target, all_target_possessions, target_stints, target_features, fallback_profiles
        gc.collect()

    support = pd.DataFrame(support_rows)
    if (
        not support["support_matches"].all()
        or not support["playoff_support_matches"].all()
        or not support["full_game_support_matches"].all()
    ):
        raise AssertionError("Split replay did not reproduce the shared frozen support")
    return _write_audit(
        split_dir=split_dir,
        baseline_dir=baseline_dir,
        predictions=pd.concat(prediction_tables, ignore_index=True),
        metrics=pd.concat(metric_tables, ignore_index=True),
        full_games=pd.concat(full_game_tables, ignore_index=True),
        teams=pd.concat(team_tables, ignore_index=True),
        wins=pd.concat(win_tables, ignore_index=True),
        support=support,
        audits_dir=Path(audits_dir),
    )


def _score_split_possessions(
    possessions: pd.DataFrame,
    *,
    fit: SplitNailSeasonFit,
    scalar_priors: dict[int, float],
    source_differences: dict[int, float],
    unit_features: dict[tuple[int, ...], np.ndarray],
    fallback_profiles: pd.DataFrame | None,
    cohort: str,
) -> pd.DataFrame:
    """Score target possessions from a persisted source Split NAIL state."""

    home_offense = possessions["home_offense"].to_numpy(dtype=bool)
    home_lineups = [
        tuple(int(player_id) for player_id in (offense if is_home else defense))
        for offense, defense, is_home in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            home_offense,
            strict=True,
        )
    ]
    away_lineups = [
        tuple(int(player_id) for player_id in (defense if is_home else offense))
        for offense, defense, is_home in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            home_offense,
            strict=True,
        )
    ]
    unique_lineups = list(dict.fromkeys([*home_lineups, *away_lineups]))
    lineup_index = {lineup: index for index, lineup in enumerate(unique_lineups)}
    missing_lineups = [lineup for lineup in unique_lineups if lineup not in unit_features]
    if missing_lineups:
        if fallback_profiles is None:
            raise ValueError(
                f"Persisted target feature state lacks {len(missing_lineups)} possession lineups"
            )
        fallback = lineup_side_context_features(
            missing_lineups,
            fallback_profiles,
            feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        )
        fallback = fallback.loc[
            :, (*fit.design.additive_features, *fit.design.nonadditive_features)
        ]
        unit_features = {
            **unit_features,
            **{
                lineup: row.to_numpy(dtype=float)
                for lineup, (_, row) in zip(missing_lineups, fallback.iterrows(), strict=True)
            },
        }
    source_features = fit.feature_coefficients.set_index("feature")
    offense_weights = np.array(
        [
            float(source_features.loc[feature, "offense_raw_coefficient"])
            for feature in (*fit.design.additive_features, *fit.design.nonadditive_features)
        ]
    )
    defense_weights = np.array(
        [
            float(source_features.loc[feature, "defense_raw_coefficient"])
            for feature in (*fit.design.additive_features, *fit.design.nonadditive_features)
        ]
    )
    feature_matrix = np.vstack([unit_features[lineup] for lineup in unique_lineups])
    unit_offense_profile = feature_matrix @ offense_weights
    unit_defense_profile = feature_matrix @ defense_weights
    unit_offense_player = np.array(
        [
            sum(
                0.5
                * (
                    float(scalar_priors.get(player, 0.0))
                    + float(source_differences.get(player, 0.0))
                )
                for player in lineup
            )
            for lineup in unique_lineups
        ],
        dtype=float,
    )
    unit_defense_player = np.array(
        [
            sum(
                0.5
                * (
                    float(scalar_priors.get(player, 0.0))
                    - float(source_differences.get(player, 0.0))
                )
                for player in lineup
            )
            for lineup in unique_lineups
        ],
        dtype=float,
    )
    unit_unknown = np.array(
        [sum(player not in scalar_priors for player in lineup) for lineup in unique_lineups],
        dtype=int,
    )
    home_index = np.fromiter((lineup_index[lineup] for lineup in home_lineups), dtype=int)
    away_index = np.fromiter((lineup_index[lineup] for lineup in away_lineups), dtype=int)
    home_ppp = (
        unit_offense_player[home_index]
        - unit_defense_player[away_index]
        + unit_offense_profile[home_index]
        - unit_defense_profile[away_index]
        + fit.model.coef_[fit.design.home_court_column(side="offense")]
    )
    away_ppp = (
        unit_offense_player[away_index]
        - unit_defense_player[home_index]
        + unit_offense_profile[away_index]
        - unit_defense_profile[home_index]
        - fit.model.coef_[fit.design.home_court_column(side="defense")]
    )
    if getattr(fit.design, "includes_back_to_back", False):
        schedule = fit.schedule_coefficients.set_index("schedule_control")
        if "back_to_back" not in schedule.index:
            raise ValueError("Split NAIL source state lacks back-to-back coefficients")
        home_b2b = possessions["home_back_to_back"].to_numpy(dtype=float)
        away_b2b = possessions["away_back_to_back"].to_numpy(dtype=float)
        offense_b2b = float(schedule.loc["back_to_back", "offense_raw_coefficient"])
        defense_b2b = float(schedule.loc["back_to_back", "defense_raw_coefficient"])
        home_ppp += offense_b2b * home_b2b - defense_b2b * away_b2b
        away_ppp += offense_b2b * away_b2b - defense_b2b * home_b2b

    # The source intercept is expected points per 100 scoring possessions.
    home_prediction = (fit.model.intercept_ + home_ppp) / 100.0
    away_prediction = (fit.model.intercept_ + away_ppp) / 100.0
    signs = possessions["home_offense_sign"].to_numpy(dtype=float)
    prediction_offense = np.where(home_offense, home_prediction, away_prediction)
    output = possessions.loc[
        :, [
            "game_id",
            "possession_id",
            "home_offense_sign",
            "target_offense_margin",
            "season",
            "season_type",
            "game_date",
            "game_time_utc",
            "possession_index",
            "offense_team_id",
            "defense_team_id",
            "offense_team_tricode",
            "defense_team_tricode",
        ]
    ].copy()
    output.insert(0, "cohort", cohort)
    output["prediction_offense_margin"] = prediction_offense
    output["target_home_margin"] = output["target_offense_margin"] * signs
    output["prediction_home_margin"] = prediction_offense * signs
    output["residual_offense_margin"] = (
        output["target_offense_margin"] - output["prediction_offense_margin"]
    )
    output["unknown_player_exposures"] = unit_unknown[home_index] + unit_unknown[away_index]
    return output


def _score_full_target_stints(
    season: str,
    stints: pd.DataFrame,
    *,
    target_fit: SplitNailSeasonFit,
    source_fit: SplitNailSeasonFit,
    scalar_priors: dict[int, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate every target regular-season stint into games and team totals."""

    source_differences = _side_differences(source_fit)
    predicted_ppp = _frozen_scoring_prediction(
        target_fit.design,
        source_fit,
        scalar_priors,
        source_differences,
    )
    if len(predicted_ppp) != len(target_fit.design.target):
        raise ValueError("Split NAIL target predictions do not align with its scoring rows")
    scoring = pd.DataFrame(
        {
            "game_id": target_fit.design.game_ids,
            "home_offense": target_fit.design.home_offense,
            "possessions": target_fit.design.weights,
            "predicted_points": predicted_ppp * target_fit.design.weights / 100.0,
            "actual_points": target_fit.design.target * target_fit.design.weights / 100.0,
        }
    )
    scoring = scoring.groupby(["game_id", "home_offense"], as_index=False, sort=False).agg(
        possessions=("possessions", "sum"),
        predicted_points=("predicted_points", "sum"),
        actual_points=("actual_points", "sum"),
    )
    wide = scoring.set_index(["game_id", "home_offense"]).unstack("home_offense")
    if True not in wide.columns.get_level_values("home_offense") or False not in wide.columns.get_level_values(
        "home_offense"
    ):
        raise ValueError("Every regular game must have home and away scoring exposure")
    metadata = stints.loc[
        :,
        [
            "game_id",
            "home_team_id",
            "away_team_id",
            "home_team_tricode",
            "away_team_tricode",
            "possessions",
            "home_margin",
        ],
    ].groupby(
        ["game_id", "home_team_id", "away_team_id", "home_team_tricode", "away_team_tricode"],
        as_index=False,
        sort=False,
    ).agg(
        possessions=("possessions", "sum"),
        actual_home_margin=("home_margin", "sum"),
    )
    games = metadata.merge(
        pd.DataFrame(
            {
                "game_id": wide.index,
                "predicted_home_margin": wide["predicted_points"][True].to_numpy()
                - wide["predicted_points"][False].to_numpy(),
            }
        ),
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    games["predicted_tie"] = np.isclose(games["predicted_home_margin"], 0.0)
    games["actual_home_win"] = games["actual_home_margin"].gt(0.0)
    games["predicted_home_win"] = games["predicted_home_margin"].gt(0.0)
    games["margin_error"] = games["predicted_home_margin"] - games["actual_home_margin"]

    base = games.loc[:, ["home_team_id", "away_team_id", "home_team_tricode", "away_team_tricode", "possessions", "actual_home_margin", "predicted_home_margin"]].copy()
    home = base.rename(
        columns={
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "actual_home_margin": "actual_margin",
            "predicted_home_margin": "predicted_margin",
        }
    ).loc[:, ["team_id", "team_tricode", "possessions", "actual_margin", "predicted_margin"]]
    away = base.rename(
        columns={
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "actual_home_margin": "actual_margin",
            "predicted_home_margin": "predicted_margin",
        }
    ).loc[:, ["team_id", "team_tricode", "possessions", "actual_margin", "predicted_margin"]]
    away[["actual_margin", "predicted_margin"]] *= -1.0
    teams = (
        pd.concat([home, away], ignore_index=True)
        .groupby(["team_id", "team_tricode"], as_index=False, sort=True)
        .agg(
            possessions=("possessions", "sum"),
            actual_total_margin=("actual_margin", "sum"),
            predicted_total_margin=("predicted_margin", "sum"),
        )
    )
    teams["actual_net_rating"] = 100.0 * teams["actual_total_margin"] / teams["possessions"]
    teams["predicted_net_rating"] = (
        100.0 * teams["predicted_total_margin"] / teams["possessions"]
    )
    teams["net_rating_error"] = teams["predicted_net_rating"] - teams["actual_net_rating"]
    return (
        games.sort_values("game_id", kind="stable").reset_index(drop=True),
        teams.sort_values("team_id", kind="stable").reset_index(drop=True),
    )


def _score_split_possessions_batched(
    possessions: pd.DataFrame,
    *,
    fit: SplitNailSeasonFit,
    scalar_priors: dict[int, float],
    source_differences: dict[int, float],
    unit_features: dict[tuple[int, ...], np.ndarray],
    fallback_profiles: pd.DataFrame | None,
    cohort: str,
    batch_size: int = 1_000,
) -> pd.DataFrame:
    """Score unique target units once, then map their values to possessions."""

    del batch_size
    return _score_split_possessions(
        possessions,
        fit=fit,
        scalar_priors=scalar_priors,
        source_differences=source_differences,
        unit_features=unit_features,
        fallback_profiles=fallback_profiles,
        cohort=cohort,
    )


def _side_differences(fit: SplitNailSeasonFit) -> dict[int, float]:
    rows = fit.player_coefficients
    return dict(
        zip(
            rows["player_id"].astype(int),
            rows["offense_base_rating"] - rows["defense_base_rating"],
            strict=True,
        )
    )


def _persisted_unit_features(
    stints: pd.DataFrame,
    fit: SplitNailSeasonFit,
) -> dict[tuple[int, ...], np.ndarray]:
    """Recover raw target-unit features from the completed Split design.

    The completed state stores the sparse scoring design used in its fit.  Its
    offense-side feature columns are the raw lineup-profile values divided by
    the persisted feature scale, so this recovers exactly the pre-target
    features without rebuilding profiles or touching outcomes.
    """

    features = (*fit.design.additive_features, *fit.design.nonadditive_features)
    start = 2 * fit.design.player_count
    columns = np.arange(start, start + len(features))
    raw_values = fit.design.features[:, columns].toarray() * np.asarray(
        fit.design.feature_scales, dtype=float
    )
    lookup: dict[tuple[int, ...], np.ndarray] = {}
    row = 0
    for stint in stints.itertuples(index=False):
        scoring_sides = (
            (stint.home_player_ids, float(stint.home_offensive_possessions)),
            (stint.away_player_ids, float(stint.away_offensive_possessions)),
        )
        for lineup, possessions in scoring_sides:
            if possessions <= 0.0:
                continue
            key = tuple(int(player_id) for player_id in lineup)
            value = raw_values[row]
            existing = lookup.get(key)
            if existing is not None and not np.allclose(existing, value, rtol=0.0, atol=1e-12):
                raise ValueError("Persisted Split NAIL features vary within a five-man unit")
            lookup[key] = value
            row += 1
    if row != len(raw_values):
        raise ValueError("Persisted Split NAIL scoring rows do not align with target stints")
    return lookup


def _missing_unit_lineups(
    possessions: pd.DataFrame,
    known_features: dict[tuple[int, ...], np.ndarray],
) -> list[tuple[int, ...]]:
    """Return valid possession units absent from the persisted stint design."""

    missing: dict[tuple[int, ...], None] = {}
    for offense, defense, home_offense in zip(
        possessions["offense_player_ids"],
        possessions["defense_player_ids"],
        possessions["home_offense"],
        strict=True,
    ):
        home = tuple(int(player_id) for player_id in (offense if home_offense else defense))
        away = tuple(int(player_id) for player_id in (defense if home_offense else offense))
        for lineup in (home, away):
            if lineup not in known_features:
                missing[lineup] = None
    return list(missing)


def _missing_unit_profiles(
    *,
    target_season: str,
    lineups: list[tuple[int, ...]],
    split_run_dir: Path,
    panel: pd.DataFrame,
    analytical_dir: Path | str,
    curated_dir: Path | str,
) -> pd.DataFrame:
    """Recover only the profile state necessary for valid but unstinted units."""

    player_ids = {player_id for lineup in lineups for player_id in lineup}
    persisted_path = split_run_dir / "target_player_profiles.parquet"
    if persisted_path.is_file():
        persisted = pd.read_parquet(persisted_path)
        if (
            "target_season" in persisted
            and persisted["target_season"].astype(str).eq(target_season).all()
            and player_ids.issubset(set(persisted["player_id"].astype(int)))
        ):
            return persisted.loc[persisted["player_id"].astype(int).isin(player_ids)].copy()
    return build_contextual_player_profiles(
        panel,
        target_season=target_season,
        target_player_ids=player_ids,
        analytical_dir=str(analytical_dir),
        curated_dir=str(curated_dir),
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )


def _baseline_predictions(predictions: pd.DataFrame, season: str, cohort: str) -> pd.DataFrame:
    output = predictions.loc[
        predictions["season"].eq(season)
        & predictions["model"].eq(BASELINE_MODEL)
        & predictions["cohort"].eq(cohort)
    ].copy()
    if output.empty:
        raise ValueError(f"Baseline artifact lacks {cohort} predictions for {season}")
    return output


def _baseline_source_mean(source_states: pd.DataFrame, season: str) -> float:
    rows = source_states.loc[
        source_states["season"].eq(season) & source_states["model"].eq(BASELINE_MODEL),
        "source_offense_margin_mean",
    ]
    if len(rows) != 1:
        raise ValueError(f"Baseline artifact lacks one source mean for {season}")
    return float(rows.iloc[0])


def _assert_shared_support(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_games: pd.DataFrame,
    season: str,
) -> dict[str, object]:
    keys = ["game_id", "possession_id"]
    actual_columns = ["target_offense_margin", "target_home_margin", "home_offense_sign"]
    candidate = predictions.sort_values(keys, kind="stable").reset_index(drop=True)
    reference = baseline.sort_values(keys, kind="stable").reset_index(drop=True)
    # Game and possession identifiers can arrive from parquet with different
    # physical dtypes while denoting the same source identifiers. Compare the
    # canonical lexical key rather than conflating storage dtype with support.
    same_keys = all(
        candidate[column].astype(str).equals(reference[column].astype(str)) for column in keys
    )
    same_actuals = all(
        np.array_equal(
            candidate[column].to_numpy(dtype=float), reference[column].to_numpy(dtype=float)
        )
        for column in actual_columns
    )
    candidate_games = _game_prediction_frame(
        predictions.drop(columns=["season", "model", "label"])
    ).sort_values("game_id", kind="stable").reset_index(drop=True)
    reference_games = baseline_games.sort_values("game_id", kind="stable").reset_index(drop=True)
    same_game_ids = candidate_games["game_id"].astype(str).equals(
        reference_games["game_id"].astype(str)
    )
    same_game_actuals = np.array_equal(
        candidate_games["actual_home_margin"].to_numpy(dtype=float),
        reference_games["actual_home_margin"].to_numpy(dtype=float),
    )
    return {
        "season": season,
        "candidate_possession_count": len(candidate),
        "baseline_possession_count": len(reference),
        "candidate_game_count": len(candidate_games),
        "baseline_game_count": len(reference_games),
        "same_possession_keys": same_keys,
        "same_possession_actuals": same_actuals,
        "same_game_ids": same_game_ids,
        "same_game_actual_margins": same_game_actuals,
        "support_matches": bool(same_keys and same_actuals and same_game_ids and same_game_actuals),
    }


def _assert_playoff_support(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    season: str,
) -> dict[str, object]:
    """Confirm exact playoff possession support without a regular-game contract."""

    keys = ["game_id", "possession_id"]
    actual_columns = ["target_offense_margin", "target_home_margin", "home_offense_sign"]
    candidate = predictions.sort_values(keys, kind="stable").reset_index(drop=True)
    reference = baseline.sort_values(keys, kind="stable").reset_index(drop=True)
    same_keys = all(
        candidate[column].astype(str).equals(reference[column].astype(str)) for column in keys
    )
    same_actuals = all(
        np.array_equal(
            candidate[column].to_numpy(dtype=float), reference[column].to_numpy(dtype=float)
        )
        for column in actual_columns
    )
    return {
        "candidate_playoff_possession_count": len(candidate),
        "baseline_playoff_possession_count": len(reference),
        "same_playoff_possession_keys": same_keys,
        "same_playoff_possession_actuals": same_actuals,
        "playoff_support_matches": bool(same_keys and same_actuals),
    }


def _assert_full_game_support(
    games: pd.DataFrame,
    baseline_games: pd.DataFrame,
    season: str,
) -> dict[str, object]:
    """Confirm the all-stint game aggregation shares production's outcomes."""

    baseline = baseline_games.loc[
        baseline_games["season"].eq(season) & baseline_games["model"].eq(BASELINE_MODEL)
    ].copy()
    candidate = games.copy()
    candidate["game_key"] = candidate["game_id"].astype(str)
    baseline["game_key"] = baseline["game_id"].astype(str)
    candidate = candidate.sort_values("game_key", kind="stable").reset_index(drop=True)
    baseline = baseline.sort_values("game_key", kind="stable").reset_index(drop=True)
    same_keys = candidate["game_key"].equals(baseline["game_key"])
    same_actuals = np.array_equal(
        candidate["actual_home_margin"].to_numpy(dtype=float),
        baseline["actual_home_margin"].to_numpy(dtype=float),
    )
    return {
        "candidate_full_game_count": len(candidate),
        "baseline_full_game_count": len(baseline),
        "same_full_game_ids": same_keys,
        "same_full_game_actual_margins": same_actuals,
        "full_game_support_matches": bool(same_keys and same_actuals),
    }


def _write_audit(
    *,
    split_dir: Path,
    baseline_dir: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    full_games: pd.DataFrame,
    teams: pd.DataFrame,
    wins: pd.DataFrame,
    support: pd.DataFrame,
    audits_dir: Path,
) -> Path:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = audits_dir / "split_nail_shared_support"
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        predictions.to_parquet(temporary / "possession_predictions.parquet", index=False)
        metrics.to_parquet(temporary / "cohort_metrics.parquet", index=False)
        full_games.to_parquet(temporary / "regular_game_predictions.parquet", index=False)
        teams.to_parquet(temporary / "team_net_rating_predictions.parquet", index=False)
        wins.to_parquet(temporary / "team_win_predictions.parquet", index=False)
        _aggregate_cohort_metrics(predictions, metrics).to_parquet(
            temporary / "aggregate_cohort_metrics.parquet", index=False
        )
        _full_metric_frame(full_games, teams, wins).to_parquet(
            temporary / "full_game_metrics.parquet", index=False
        )
        support.to_parquet(temporary / "support_verification.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": now.isoformat(),
            "model": MODEL_NAME,
            "mode": "persisted_state_replay_no_refit",
            "split_run_dir": str(split_dir),
            "baseline_run_dir": str(baseline_dir),
            "support_contract": (
                "exact possession keys, game keys, and realized regular-season outcomes "
                "must match the production frozen artifact"
            ),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return output
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _aggregate_cohort_metrics(
    predictions: pd.DataFrame,
    season_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Pool frozen possession and eligible-game metrics without cross-season key collisions."""

    rows: list[dict[str, object]] = []
    for cohort, frame in predictions.groupby("cohort", sort=False):
        actual = frame["target_offense_margin"].to_numpy(dtype=float)
        predicted = frame["prediction_offense_margin"].to_numpy(dtype=float)
        source = season_metrics.loc[season_metrics["cohort"].eq(cohort)]
        possession_weight = source["possession_count"].to_numpy(dtype=float)
        mean_mse = np.average(
            np.square(source["frozen_mean_reference_possession_rmse"]),
            weights=possession_weight,
        )
        keyed = frame.copy()
        keyed["game_id"] = keyed["season"].astype(str) + ":" + keyed["game_id"].astype(str)
        games = _game_prediction_frame(
            keyed.drop(columns=["season", "model", "label"])
        )
        game_weight = source["game_count"].to_numpy(dtype=float)
        mean_game_mse = np.average(
            np.square(source["frozen_mean_reference_game_margin_rmse"]),
            weights=game_weight,
        )
        residual = actual - predicted
        game_mse = float(np.mean(np.square(games["margin_error"].to_numpy(dtype=float))))
        rows.append(
            {
                "model": MODEL_NAME,
                "cohort": cohort,
                "season_count": int(frame["season"].nunique()),
                "game_count": len(games),
                "possession_count": len(frame),
                "possession_rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "possession_mae": float(np.mean(np.abs(residual))),
                "eligible_game_margin_rmse": float(np.sqrt(game_mse)),
                "possession_skill_vs_frozen_mean": float(
                    1.0 - np.mean(np.square(residual)) / mean_mse
                ),
                "eligible_game_skill_vs_frozen_mean": float(1.0 - game_mse / mean_game_mse),
            }
        )
    return pd.DataFrame(rows)


def _full_metric_frame(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    wins: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize pooled full-game and team metrics from all target stints."""

    unique_games = games.copy()
    unique_games["game_id"] = unique_games["season"].astype(str) + ":" + unique_games[
        "game_id"
    ].astype(str)
    outcome = score_full_game_outcomes(unique_games)
    team_error = teams["net_rating_error"].to_numpy(dtype=float)
    win_error = wins["pythagorean_win_error"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                **outcome,
                "team_net_rating_rmse": float(np.sqrt(np.mean(np.square(team_error)))),
                "pythagorean_win_rmse": float(np.sqrt(np.mean(np.square(win_error)))),
            }
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay persisted Split NAIL states on shared support"
    )
    parser.add_argument("--split-run-dir", default=str(DEFAULT_SPLIT_RUN_DIR))
    parser.add_argument("--baseline-run-dir", default=str(DEFAULT_BASELINE_RUN_DIR))
    args = parser.parse_args()
    audit = replay_shared_support(
        split_run_dir=args.split_run_dir,
        baseline_run_dir=args.baseline_run_dir,
    )
    print(f"Split NAIL shared-support replay: audit={audit}")


if __name__ == "__main__":
    main()
