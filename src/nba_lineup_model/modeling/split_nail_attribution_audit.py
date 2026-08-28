"""No-refit attribution audit for the standalone constrained Split NAIL state.

The audit deliberately separates two questions:

* how closely frozen Split NAIL predictions agree with the published scalar NAIL
  forecast on identical possessions; and
* how much of each Split prediction is carried by player offense/defense,
  additive profiles, retained non-additive lineup terms, and known schedule
  controls.

It never fits a player, profile, context, or schedule coefficient.  A target
season is always scored from its immediately preceding completed Split state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import matplotlib.pyplot as plt
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
)
from nba_lineup_model.modeling.frozen_prior_evaluation import _read_playoff_possessions
from nba_lineup_model.modeling.neural_data import read_neural_possessions
from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.modeling.split_nail import SplitNailSeasonFit
from nba_lineup_model.modeling.split_nail_shared_support_replay import (
    BASELINE_MODEL,
    DEFAULT_BASELINE_RUN_DIR,
    _missing_unit_lineups,
    _missing_unit_profiles,
    _persisted_unit_features,
    _side_differences,
)
from nba_lineup_model.modeling.stints import read_rapm_stints

MODEL_NAME = "forward_split_nail_rapm"
DEFAULT_SPLIT_ROOT = Path("artifacts/models/forward_split_nail_rapm/2025-26")
DEFAULT_REPLAY_ROOT = Path("artifacts/audits/split_nail_shared_support")
DEFAULT_AUDIT_ROOT = Path("artifacts/audits/split_nail_attribution")
DEFAULT_CHART_PATH = Path("docs/assets/images/split-nail/nonadditive-od-trajectories.svg")
DEFAULT_PRODUCTION_STATE_ROOT = Path(
    "artifacts/models/forward_nail_rapm_v1212_back_to_back/2025-26"
)


@dataclass(frozen=True)
class SplitNailAttributionAuditRun:
    """Persistent outputs from one immutable no-refit attribution audit."""

    run_dir: Path
    chart_path: Path


def build_split_nail_attribution_audit(
    *,
    split_run_dir: Path | str | None = None,
    shared_replay_dir: Path | str | None = None,
    baseline_run_dir: Path | str = DEFAULT_BASELINE_RUN_DIR,
    production_state_dir: Path | str | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    game_catalog_path: Path | str = Path("data/catalog/games.parquet"),
    audit_root: Path | str = DEFAULT_AUDIT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
    seasons: tuple[str, ...] = FROZEN_SEASONS,
) -> SplitNailAttributionAuditRun:
    """Materialize Split NAIL frozen-prediction and component diagnostics."""

    split_dir = _resolve_run_dir(Path(split_run_dir) if split_run_dir else DEFAULT_SPLIT_ROOT)
    replay_dir = _resolve_run_dir(
        Path(shared_replay_dir) if shared_replay_dir else DEFAULT_REPLAY_ROOT
    )
    baseline_dir = Path(baseline_run_dir)
    production_dir = _resolve_run_dir(
        Path(production_state_dir) if production_state_dir else DEFAULT_PRODUCTION_STATE_ROOT
    )
    metadata = json.loads((split_dir / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Split NAIL attribution audit requires a standalone Split NAIL artifact")

    ratings = pd.read_parquet(split_dir / "player_season_ratings.parquet")
    fits: dict[str, SplitNailSeasonFit] = joblib.load(split_dir / "season_split_nail_models.joblib")
    replay_predictions = pd.read_parquet(replay_dir / "possession_predictions.parquet")
    baseline_predictions = pd.read_parquet(baseline_dir / "possession_predictions.parquet")
    baseline_priors = pd.read_parquet(production_dir / "season_player_priors.parquet")
    panel = pd.read_parquet(player_season_panel_path)
    schedule_features = build_back_to_back_game_features(pd.read_parquet(game_catalog_path))

    scalar_priors = {
        str(season): dict(zip(group["player_id"].astype(int), group["scalar_prior"], strict=True))
        for season, group in ratings.groupby("season", sort=False)
    }
    component_rows: list[pd.DataFrame] = []
    reconstruction_rows: list[dict[str, object]] = []
    for season in seasons:
        source = _previous_season(season)
        if source not in fits or season not in fits:
            raise ValueError(f"Split NAIL run lacks source or target fit for {season}")
        print(f"Auditing frozen Split NAIL components: {source} -> {season}", flush=True)
        regular = _attach_back_to_back_flags(
            read_neural_possessions(season, analytical_dir=analytical_dir), schedule_features
        )
        playoffs = _attach_back_to_back_flags(
            _read_playoff_possessions(season, curated_dir)[0], schedule_features
        )
        target_stints = _attach_back_to_back_flags(
            read_rapm_stints(season, analytical_dir=analytical_dir), schedule_features
        )
        unit_features = _persisted_unit_features(target_stints, fits[season])
        missing = _missing_unit_lineups(
            pd.concat([regular, playoffs], ignore_index=True), unit_features
        )
        fallback_profiles = (
            _missing_unit_profiles(
                target_season=season,
                lineups=missing,
                split_run_dir=split_dir,
                panel=panel,
                analytical_dir=analytical_dir,
                curated_dir=curated_dir,
            )
            if missing
            else None
        )
        for cohort, possessions in (("regular_season", regular), ("playoffs", playoffs)):
            components = _split_components(
                possessions,
                source_fit=fits[source],
                scalar_priors=scalar_priors.get(season, {}),
                source_differences=_side_differences(fits[source]),
                unit_features=unit_features,
                fallback_profiles=fallback_profiles,
                season=season,
                cohort=cohort,
            )
            expected = replay_predictions.loc[
                (replay_predictions["season"].eq(season))
                & (replay_predictions["cohort"].eq(cohort)),
                ["game_id", "possession_id", "prediction_home_margin"],
            ].copy()
            check = components.merge(
                expected,
                on=["game_id", "possession_id"],
                how="inner",
                validate="one_to_one",
                suffixes=("", "_replay"),
            )
            if len(check) != len(components) or len(check) != len(expected):
                raise ValueError(f"Split component support does not match replay for {season} {cohort}")
            max_error = float(
                np.abs(
                    check["prediction_home_margin"].to_numpy(dtype=float)
                    - check["prediction_home_margin_replay"].to_numpy(dtype=float)
                ).max()
            )
            if max_error > 1e-12:
                raise ValueError(
                    f"Split component reconstruction differs from frozen replay for {season} "
                    f"{cohort}: {max_error:.3e}"
                )
            reconstruction_rows.append(
                {
                    "season": season,
                    "source_season": source,
                    "cohort": cohort,
                    "possession_count": len(components),
                    "max_prediction_error": max_error,
                    "source_state_only": True,
                }
            )
            component_rows.append(components)

    components = pd.concat(component_rows, ignore_index=True)
    agreement = _prediction_agreement(replay_predictions, baseline_predictions, seasons)
    player_agreement = _frozen_player_prior_agreement(ratings, baseline_priors, seasons)
    component_summary = _component_summary(components)
    coefficient_history = _coefficient_history(
        pd.read_parquet(split_dir / "season_feature_coefficients.parquet")
    )

    rendered_chart = Path(chart_path)
    rendered_chart.parent.mkdir(parents=True, exist_ok=True)
    _render_nonadditive_coefficients(coefficient_history, rendered_chart)

    now = datetime.now(UTC)
    run_id = f"split-nail-attribution-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = Path(audit_root)
    run_dir = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        # These are all trace outputs from immutable existing model states.
        components.to_parquet(temporary / "frozen_component_predictions.parquet", index=False)
        agreement.to_parquet(temporary / "frozen_prediction_agreement.parquet", index=False)
        player_agreement.to_parquet(temporary / "frozen_player_prior_agreement.parquet", index=False)
        component_summary.to_parquet(temporary / "frozen_component_summary.parquet", index=False)
        coefficient_history.to_parquet(temporary / "nonadditive_od_coefficient_history.parquet", index=False)
        pd.DataFrame(reconstruction_rows).to_parquet(
            temporary / "reconstruction_verification.parquet", index=False
        )
        run_metadata = {
            "run_id": run_id,
            "created_at": now.isoformat(),
            "split_run_dir": str(split_dir),
            "shared_replay_dir": str(replay_dir),
            "baseline_run_dir": str(baseline_dir),
            "production_state_dir": str(production_dir),
            "information_boundary": (
                "No coefficients or player values are fit. Every target season uses its "
                "immediately preceding completed Split NAIL state."
            ),
            "nonadditive_contract": {
                "features": ["top_two_assists", "usage_concentration"],
                "coefficient_series": 4,
                "sides": ["offense", "defense"],
            },
            "chart_path": str(rendered_chart),
        }
        (temporary / "metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
        manifest = [
            {
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**run_metadata, "artifacts": manifest}, indent=2) + "\n"
        )
        temporary.replace(run_dir)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return SplitNailAttributionAuditRun(run_dir=run_dir, chart_path=rendered_chart)


def _split_components(
    possessions: pd.DataFrame,
    *,
    source_fit: SplitNailSeasonFit,
    scalar_priors: dict[int, float],
    source_differences: dict[int, float],
    unit_features: dict[tuple[int, ...], np.ndarray],
    fallback_profiles: pd.DataFrame | None,
    season: str,
    cohort: str,
) -> pd.DataFrame:
    """Return each source-state contribution in home-net-rating units."""

    home_offense = possessions["home_offense"].to_numpy(dtype=bool)
    home_lineups = [
        tuple(int(player) for player in (offense if is_home else defense))
        for offense, defense, is_home in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            home_offense,
            strict=True,
        )
    ]
    away_lineups = [
        tuple(int(player) for player in (defense if is_home else offense))
        for offense, defense, is_home in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            home_offense,
            strict=True,
        )
    ]
    unique = list(dict.fromkeys([*home_lineups, *away_lineups]))
    features = _resolve_features(
        unique,
        unit_features=unit_features,
        fallback_profiles=fallback_profiles,
        source_fit=source_fit,
    )
    lookup = {lineup: index for index, lineup in enumerate(unique)}
    home_index = np.fromiter((lookup[lineup] for lineup in home_lineups), dtype=int)
    away_index = np.fromiter((lookup[lineup] for lineup in away_lineups), dtype=int)
    feature_matrix = np.vstack([features[lineup] for lineup in unique])
    source_features = source_fit.feature_coefficients.set_index("feature")
    all_features = (*source_fit.design.additive_features, *source_fit.design.nonadditive_features)
    offense_weights = source_features.loc[list(all_features), "offense_raw_coefficient"].to_numpy(
        dtype=float
    )
    defense_weights = source_features.loc[list(all_features), "defense_raw_coefficient"].to_numpy(
        dtype=float
    )
    additive_count = len(source_fit.design.additive_features)
    offense_additive = feature_matrix[:, :additive_count] @ offense_weights[:additive_count]
    defense_additive = feature_matrix[:, :additive_count] @ defense_weights[:additive_count]
    offense_nonadditive = feature_matrix[:, additive_count:] @ offense_weights[additive_count:]
    defense_nonadditive = feature_matrix[:, additive_count:] @ defense_weights[additive_count:]
    offense_players = np.asarray(
        [
            sum(
                0.5 * (float(scalar_priors.get(player, 0.0)) + float(source_differences.get(player, 0.0)))
                for player in lineup
            )
            for lineup in unique
        ],
        dtype=float,
    )
    defense_players = np.asarray(
        [
            sum(
                0.5 * (float(scalar_priors.get(player, 0.0)) - float(source_differences.get(player, 0.0)))
                for player in lineup
            )
            for lineup in unique
        ],
        dtype=float,
    )

    def home_net(offense_values: np.ndarray, defense_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.where(home_offense, offense_values[home_index], -offense_values[away_index]),
            np.where(home_offense, -defense_values[away_index], defense_values[home_index]),
        )

    player_offense, player_defense = home_net(offense_players, defense_players)
    additive_offense, additive_defense = home_net(offense_additive, defense_additive)
    nonadditive_offense, nonadditive_defense = home_net(
        offense_nonadditive, defense_nonadditive
    )
    hca_offense = float(source_fit.model.coef_[source_fit.design.home_court_column(side="offense")])
    hca_defense = float(source_fit.model.coef_[source_fit.design.home_court_column(side="defense")])
    home_court = np.where(home_offense, hca_offense, hca_defense)
    back_to_back = np.zeros(len(possessions), dtype=float)
    if source_fit.design.includes_back_to_back:
        schedule = source_fit.schedule_coefficients.set_index("schedule_control")
        offense_b2b = float(schedule.loc["back_to_back", "offense_raw_coefficient"])
        defense_b2b = float(schedule.loc["back_to_back", "defense_raw_coefficient"])
        home_b2b = possessions["home_back_to_back"].to_numpy(dtype=float)
        away_b2b = possessions["away_back_to_back"].to_numpy(dtype=float)
        back_to_back = np.where(
            home_offense,
            offense_b2b * home_b2b - defense_b2b * away_b2b,
            defense_b2b * home_b2b - offense_b2b * away_b2b,
        )
    intercept = np.where(home_offense, source_fit.model.intercept_, -source_fit.model.intercept_)
    output = possessions.loc[:, ["game_id", "possession_id"]].copy()
    output.insert(0, "cohort", cohort)
    output.insert(0, "season", season)
    output["player_offense_net_rating"] = player_offense
    output["player_defense_net_rating"] = player_defense
    output["additive_profile_offense_net_rating"] = additive_offense
    output["additive_profile_defense_net_rating"] = additive_defense
    output["nonadditive_offense_net_rating"] = nonadditive_offense
    output["nonadditive_defense_net_rating"] = nonadditive_defense
    output["home_court_net_rating"] = home_court
    output["back_to_back_net_rating"] = back_to_back
    output["scoring_intercept_net_rating"] = intercept
    output["player_total_net_rating"] = player_offense + player_defense
    output["additive_profile_total_net_rating"] = additive_offense + additive_defense
    output["nonadditive_total_net_rating"] = nonadditive_offense + nonadditive_defense
    for index, feature in enumerate(source_fit.design.nonadditive_features, start=additive_count):
        feature_offense, feature_defense = home_net(
            feature_matrix[:, index] * offense_weights[index],
            feature_matrix[:, index] * defense_weights[index],
        )
        output[f"{feature}_offense_net_rating"] = feature_offense
        output[f"{feature}_defense_net_rating"] = feature_defense
        output[f"{feature}_total_net_rating"] = feature_offense + feature_defense
    output["prediction_home_net_rating"] = (
        output[
            [
                "player_total_net_rating",
                "additive_profile_total_net_rating",
                "nonadditive_total_net_rating",
                "home_court_net_rating",
                "back_to_back_net_rating",
                "scoring_intercept_net_rating",
            ]
        ]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )
    output["prediction_home_margin"] = output["prediction_home_net_rating"] / 100.0
    return output


def _resolve_features(
    lineups: list[tuple[int, ...]],
    *,
    unit_features: dict[tuple[int, ...], np.ndarray],
    fallback_profiles: pd.DataFrame | None,
    source_fit: SplitNailSeasonFit,
) -> dict[tuple[int, ...], np.ndarray]:
    """Use persisted target features first, then forward-safe profile fallbacks."""

    missing = [lineup for lineup in lineups if lineup not in unit_features]
    if not missing:
        return unit_features
    if fallback_profiles is None:
        raise ValueError("Missing target lineup features without a fallback profile state")
    fallback = lineup_side_context_features(
        missing,
        fallback_profiles,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    ).loc[:, (*source_fit.design.additive_features, *source_fit.design.nonadditive_features)]
    return {
        **unit_features,
        **{lineup: row.to_numpy(dtype=float) for lineup, (_, row) in zip(missing, fallback.iterrows(), strict=True)},
    }


def _prediction_agreement(
    split: pd.DataFrame,
    baseline: pd.DataFrame,
    seasons: tuple[str, ...],
) -> pd.DataFrame:
    """Compare Split and published NAIL predictions on exactly matched possessions."""

    reference = baseline.loc[baseline["model"].eq(BASELINE_MODEL)].copy()
    rows = []
    for season in seasons:
        for cohort in ("regular_season", "playoffs"):
            left = split.loc[
                (split["season"].eq(season)) & (split["cohort"].eq(cohort)),
                ["game_id", "possession_id", "prediction_offense_margin"],
            ]
            right = reference.loc[
                (reference["season"].eq(season)) & (reference["cohort"].eq(cohort)),
                ["game_id", "possession_id", "prediction_offense_margin"],
            ]
            merged = left.merge(right, on=["game_id", "possession_id"], suffixes=("_split", "_nail"))
            if len(merged) != len(left) or len(merged) != len(right):
                raise ValueError(f"Prediction agreement lacks shared support for {season} {cohort}")
            delta = (
                merged["prediction_offense_margin_split"].to_numpy(dtype=float)
                - merged["prediction_offense_margin_nail"].to_numpy(dtype=float)
            )
            rows.append(
                {
                    "season": season,
                    "cohort": cohort,
                    "possession_count": len(merged),
                    "prediction_pearson_correlation": float(
                        merged["prediction_offense_margin_split"].corr(
                            merged["prediction_offense_margin_nail"]
                        )
                    ),
                    "delta_mean": float(delta.mean()),
                    "delta_standard_deviation": float(delta.std(ddof=0)),
                    "delta_rmse": float(np.sqrt(np.mean(np.square(delta)))),
                }
            )
    return pd.DataFrame(rows)


def _frozen_player_prior_agreement(
    split_ratings: pd.DataFrame,
    baseline_priors: pd.DataFrame,
    seasons: tuple[str, ...],
) -> pd.DataFrame:
    """Compare the prospectively available scalar states for each target season."""

    rows = []
    for season in seasons:
        left = split_ratings.loc[
            split_ratings["season"].eq(season), ["player_id", "scalar_prior"]
        ]
        right = baseline_priors.loc[
            baseline_priors["season"].eq(season), ["player_id", "prior_rapm"]
        ]
        merged = left.merge(right, on="player_id", how="inner", validate="one_to_one")
        delta = merged["scalar_prior"].to_numpy(dtype=float) - merged["prior_rapm"].to_numpy(dtype=float)
        rows.append(
            {
                "target_season": season,
                "source_season": _previous_season(season),
                "shared_player_count": len(merged),
                "prior_pearson_correlation": float(merged["scalar_prior"].corr(merged["prior_rapm"])),
                "prior_delta_mean": float(delta.mean()),
                "prior_delta_standard_deviation": float(delta.std(ddof=0)),
                "prior_delta_rmse": float(np.sqrt(np.mean(np.square(delta)))),
            }
        )
    return pd.DataFrame(rows)


def _component_summary(components: pd.DataFrame) -> pd.DataFrame:
    """Summarize scale, not a misleading additive variance attribution."""

    columns = (
        "player_total_net_rating",
        "additive_profile_total_net_rating",
        "nonadditive_total_net_rating",
        "top_two_assists_total_net_rating",
        "usage_concentration_total_net_rating",
        "home_court_net_rating",
        "back_to_back_net_rating",
    )
    rows = []
    for (season, cohort), group in components.groupby(["season", "cohort"], sort=True):
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            rows.append(
                {
                    "season": season,
                    "cohort": cohort,
                    "component": column.removesuffix("_net_rating"),
                    "mean_net_rating": float(values.mean()),
                    "mean_absolute_net_rating": float(np.abs(values).mean()),
                    "standard_deviation_net_rating": float(values.std(ddof=0)),
                    "root_mean_square_net_rating": float(np.sqrt(np.mean(np.square(values)))),
                }
            )
    return pd.DataFrame(rows)


def _coefficient_history(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Return the four completed-season O/D non-additive coefficient series."""

    retained = ("top_two_assists", "usage_concentration")
    rows = coefficients.loc[
        coefficients["feature"].isin(retained),
        [
            "season",
            "feature",
            "feature_standard_deviation",
            "offense_raw_coefficient",
            "defense_raw_coefficient",
        ],
    ].copy()
    if rows.empty or rows.duplicated(["season", "feature"]).any():
        raise ValueError("Split NAIL non-additive coefficient history is incomplete")
    long = rows.melt(
        id_vars=["season", "feature", "feature_standard_deviation"],
        value_vars=["offense_raw_coefficient", "defense_raw_coefficient"],
        var_name="side",
        value_name="raw_coefficient",
    )
    long["side"] = long["side"].str.replace("_raw_coefficient", "", regex=False)
    long["one_standard_deviation_effect"] = (
        long["raw_coefficient"] * long["feature_standard_deviation"]
    )
    return long.sort_values(["feature", "side", "season"], kind="stable").reset_index(drop=True)


def _render_nonadditive_coefficients(history: pd.DataFrame, path: Path) -> None:
    """Render offense/defense coefficient histories with a shared readable contract."""

    features = ("usage_concentration", "top_two_assists")
    labels = {
        "usage_concentration": "Usage concentration",
        "top_two_assists": "Top-two assists",
    }
    colors = {"offense": "#2d6da3", "defense": "#e07a35"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False, layout="constrained")
    for axis, feature in zip(axes, features, strict=True):
        frame = history.loc[history["feature"].eq(feature)]
        seasons = sorted(frame["season"].unique())
        for side in ("offense", "defense"):
            values = frame.loc[frame["side"].eq(side)].set_index("season").reindex(seasons)
            axis.plot(
                seasons,
                values["one_standard_deviation_effect"],
                color=colors[side],
                linewidth=2.2,
                marker="o",
                markersize=3.8,
                label=side.title(),
            )
        axis.axhline(0.0, color="#7c837c", linestyle="--", linewidth=1.0)
        axis.set_title(labels[feature])
        axis.tick_params(axis="x", rotation=45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel("Net-rating points per lineup SD")
    figure.suptitle(
        "Standalone constrained Split NAIL: completed non-additive O/D coefficients",
        fontsize=13,
        fontweight="bold",
    )
    figure.legend(loc="lower center", ncol=2, frameon=False)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _resolve_run_dir(path: Path) -> Path:
    """Resolve a model/audit root through its immutable latest pointer."""

    if (path / "metadata.json").is_file():
        return path
    latest = path / "latest.json"
    if not latest.is_file():
        raise ValueError(f"No immutable run or latest.json exists under {path}")
    run_id = json.loads(latest.read_text()).get("run_id")
    if not isinstance(run_id, str):
        raise ValueError(f"Invalid latest pointer under {path}")
    run_dir = path / run_id
    if not (run_dir / "metadata.json").is_file():
        raise ValueError(f"Latest pointer does not resolve to a run under {path}")
    return run_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Split NAIL attribution without refitting")
    parser.add_argument("--split-run-dir")
    parser.add_argument("--shared-replay-dir")
    parser.add_argument("--baseline-run-dir", default=str(DEFAULT_BASELINE_RUN_DIR))
    parser.add_argument("--production-state-dir")
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run = build_split_nail_attribution_audit(
        split_run_dir=args.split_run_dir,
        shared_replay_dir=args.shared_replay_dir,
        baseline_run_dir=args.baseline_run_dir,
        production_state_dir=args.production_state_dir,
        audit_root=args.audit_root,
        chart_path=args.chart_path,
    )
    print(f"Split NAIL attribution audit: run={run.run_dir}; chart={run.chart_path}")


if __name__ == "__main__":
    main()
