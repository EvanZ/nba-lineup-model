"""Forward-safe cross-season estimation of player-profile padding constants."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    ProfilePaddingContract,
    _rebound_percentage_frame,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
)
from nba_lineup_model.season.schema import validate_season

MODEL_NAME = "profile_padding_study"
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_THROUGH_TARGET_SEASON = "2022-23"
MAX_PSEUDO_SAMPLE_SIZE = 10_000.0


@dataclass(frozen=True)
class ProfilePaddingStudyRun:
    run_dir: Path
    run_id: str
    contract: ProfilePaddingContract


@dataclass(frozen=True)
class _MetricSpec:
    name: str
    value_column: str
    denominator_column: str


METRICS = (
    _MetricSpec("three_pa", "three_pa_per_100", "possessions"),
    _MetricSpec("three_point_pct", "three_point_pct", "three_pointers_attempted"),
    _MetricSpec("assists", "assists_per_100", "possessions"),
    _MetricSpec("turnovers", "turnovers_per_100", "possessions"),
    _MetricSpec("field_goals_attempted", "field_goals_attempted_per_100", "possessions"),
    _MetricSpec("free_throws_attempted", "free_throws_attempted_per_100", "possessions"),
    _MetricSpec("offensive_rebounds", "offensive_rebounds_per_100", "possessions"),
    _MetricSpec("defensive_rebounds", "defensive_rebounds_per_100", "possessions"),
    _MetricSpec("steals", "steals_per_100", "possessions"),
    _MetricSpec("blocks", "blocks_per_100", "possessions"),
    _MetricSpec("offensive_rebound_pct", "offensive_rebound_pct", "possessions"),
    _MetricSpec("defensive_rebound_pct", "defensive_rebound_pct", "possessions"),
)


def run_profile_padding_study(
    *,
    through_target_season: str = DEFAULT_THROUGH_TARGET_SEASON,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ProfilePaddingStudyRun:
    """Estimate one padding constant per primitive statistic and persist the audit."""

    through = validate_season(through_target_season)
    panel = pd.read_parquet(player_season_panel_path)
    observed = _observed_metric_frame(panel, through=through, curated_dir=Path(curated_dir))
    transitions = _transition_frame(observed, through=through)
    estimates, predictions = estimate_padding_constants(transitions)
    contract = contract_from_estimates(estimates, through_target_season=through)
    return _write_run(
        estimates=estimates,
        predictions=predictions,
        contract=contract,
        through_target_season=through,
        artifacts_dir=Path(artifacts_dir),
    )


def estimate_padding_constants(
    transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimize next-season weighted MSE independently for every primitive statistic."""

    estimate_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in METRICS:
        source_value = pd.to_numeric(
            transitions[f"source_{spec.value_column}"], errors="coerce"
        ).to_numpy(dtype=float)
        target_value = pd.to_numeric(
            transitions[f"target_{spec.value_column}"], errors="coerce"
        ).to_numpy(dtype=float)
        source_denominator = pd.to_numeric(
            transitions[f"source_{spec.denominator_column}"], errors="coerce"
        ).to_numpy(dtype=float)
        target_weight = pd.to_numeric(
            transitions[f"target_{spec.denominator_column}"], errors="coerce"
        ).to_numpy(dtype=float)
        reference = pd.to_numeric(
            transitions[f"source_reference_{spec.value_column}"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = (
            np.isfinite(source_value)
            & np.isfinite(target_value)
            & np.isfinite(source_denominator)
            & np.isfinite(target_weight)
            & np.isfinite(reference)
            & (source_denominator > 0)
            & (target_weight > 0)
        )
        if valid.sum() < 100:
            raise ValueError(f"Padding study has too few transitions for {spec.name}")
        source_value = source_value[valid]
        target_value = target_value[valid]
        source_denominator = source_denominator[valid]
        target_weight = target_weight[valid]
        reference = reference[valid]

        def objective(
            log1p_k: float,
            source_value: np.ndarray = source_value,
            target_value: np.ndarray = target_value,
            source_denominator: np.ndarray = source_denominator,
            target_weight: np.ndarray = target_weight,
            reference: np.ndarray = reference,
        ) -> float:
            k = float(np.expm1(log1p_k))
            predicted = (
                source_value * source_denominator + k * reference
            ) / (source_denominator + k)
            return float(np.average(np.square(target_value - predicted), weights=target_weight))

        result = minimize_scalar(
            objective,
            bounds=(0.0, float(np.log1p(MAX_PSEUDO_SAMPLE_SIZE))),
            method="bounded",
            options={"xatol": 1e-6},
        )
        if not result.success:
            raise RuntimeError(f"Padding optimization failed for {spec.name}: {result.message}")
        selected = float(np.expm1(result.x))
        selected_prediction = (
            source_value * source_denominator + selected * reference
        ) / (source_denominator + selected)
        uniform_prediction = (
            source_value * source_denominator + 300.0 * reference
        ) / (source_denominator + 300.0)
        published = _published_constant(spec.name)
        published_prediction = (
            source_value * source_denominator + published * reference
        ) / (source_denominator + published)
        estimate_rows.append(
            {
                "metric": spec.name,
                "value_column": spec.value_column,
                "denominator_column": spec.denominator_column,
                "selected_pseudo_sample_size": selected,
                "published_pseudo_sample_size": published,
                "transition_count": int(valid.sum()),
                "source_season_count": int(
                    transitions.loc[valid, "source_season"].nunique()
                ),
                "unshrunk_weighted_mse": objective(0.0),
                "uniform_300_weighted_mse": float(
                    np.average(np.square(target_value - uniform_prediction), weights=target_weight)
                ),
                "published_weighted_mse": float(
                    np.average(
                        np.square(target_value - published_prediction), weights=target_weight
                    )
                ),
                "selected_weighted_mse": float(
                    np.average(
                        np.square(target_value - selected_prediction), weights=target_weight
                    )
                ),
            }
        )
        keyed = transitions.loc[
            valid,
            ["source_season", "target_season", "player_id"],
        ].copy()
        keyed["metric"] = spec.name
        keyed["source_value"] = source_value
        keyed["target_value"] = target_value
        keyed["source_denominator"] = source_denominator
        keyed["target_weight"] = target_weight
        keyed["source_reference"] = reference
        keyed["selected_prediction"] = selected_prediction
        keyed["uniform_300_prediction"] = uniform_prediction
        keyed["published_prediction"] = published_prediction
        prediction_frames.append(keyed)
    return pd.DataFrame(estimate_rows), pd.concat(prediction_frames, ignore_index=True)


def contract_from_estimates(
    estimates: pd.DataFrame,
    *,
    through_target_season: str,
) -> ProfilePaddingContract:
    values = dict(
        zip(
            estimates["metric"].astype(str),
            estimates["selected_pseudo_sample_size"].astype(float),
            strict=True,
        )
    )
    return ProfilePaddingContract(
        name=f"cross_season_through_{through_target_season.replace('-', '_')}",
        rate_pseudo_possessions={
            "three_pa": values["three_pa"],
            "assists": values["assists"],
            "turnovers": values["turnovers"],
            "offensive_rebounds": values["offensive_rebounds"],
            "defensive_rebounds": values["defensive_rebounds"],
            "steals": values["steals"],
            "blocks": values["blocks"],
        },
        reference_mode="season",
        three_point_percentage_attempts=values["three_point_pct"],
        usage_component_pseudo_possessions={
            "field_goals_attempted": values["field_goals_attempted"],
            "free_throws_attempted": values["free_throws_attempted"],
            "turnovers": values["turnovers"],
        },
        rebound_percentage_pseudo_possessions={
            "offensive_rebound_pct": values["offensive_rebound_pct"],
            "defensive_rebound_pct": values["defensive_rebound_pct"],
        },
        source=(
            "Cross-season weighted-MSE fit using transitions whose target season ends by "
            f"{through_target_season}"
        ),
    )


def load_latest_padding_contract(
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> tuple[ProfilePaddingContract, dict[str, object]]:
    root = Path(artifacts_dir) / MODEL_NAME / DEFAULT_THROUGH_TARGET_SEASON
    latest = json.loads((root / "latest.json").read_text())
    run_dir = root / str(latest["run_id"])
    payload = json.loads((run_dir / "padding_contract.json").read_text())
    contract = ProfilePaddingContract(
        name=str(payload["name"]),
        rate_pseudo_possessions=payload["rate_pseudo_possessions"],
        reference_mode=str(payload["reference_mode"]),
        three_point_percentage_attempts=payload["three_point_percentage_attempts"],
        usage_component_pseudo_possessions=payload["usage_component_pseudo_possessions"],
        rebound_percentage_pseudo_possessions=payload[
            "rebound_percentage_pseudo_possessions"
        ],
        source=payload["source"],
    )
    return contract, {"study_run_id": run_dir.name, "study_run_dir": str(run_dir)}


def _observed_metric_frame(
    panel: pd.DataFrame,
    *,
    through: str,
    curated_dir: Path,
) -> pd.DataFrame:
    seasons = tuple(
        sorted(
            panel.loc[panel["season"].astype(str).le(through), "season"].astype(str).unique(),
            key=lambda value: int(value[:4]),
        )
    )
    columns = {
        "season",
        "player_id",
        "rapm_possessions",
        "three_pointers_attempted",
        "three_pointers_made",
        "assists",
        "turnovers",
        "field_goals_attempted",
        "free_throws_attempted",
        "rebounds_offensive",
        "rebounds_defensive",
        "steals",
        "blocks",
    }
    missing = columns - set(panel)
    if missing:
        raise ValueError(f"Player-season panel lacks padding fields: {sorted(missing)}")
    frame = panel.loc[panel["season"].isin(seasons), sorted(columns)].copy()
    frame = frame.rename(columns={"rapm_possessions": "possessions"})
    possessions = pd.to_numeric(frame["possessions"], errors="coerce").astype(float)
    for name, column in {
        "three_pa_per_100": "three_pointers_attempted",
        "assists_per_100": "assists",
        "turnovers_per_100": "turnovers",
        "field_goals_attempted_per_100": "field_goals_attempted",
        "free_throws_attempted_per_100": "free_throws_attempted",
        "offensive_rebounds_per_100": "rebounds_offensive",
        "defensive_rebounds_per_100": "rebounds_defensive",
        "steals_per_100": "steals",
        "blocks_per_100": "blocks",
    }.items():
        frame[name] = 100.0 * pd.to_numeric(frame[column], errors="coerce") / possessions
    attempts = pd.to_numeric(frame["three_pointers_attempted"], errors="coerce").astype(float)
    frame["three_point_pct"] = (
        pd.to_numeric(frame["three_pointers_made"], errors="coerce").astype(float) / attempts
    )
    rebounds = _rebound_percentage_frame(seasons, curated_dir=str(curated_dir))
    frame = frame.merge(rebounds, on=["season", "player_id"], how="left", validate="one_to_one")
    for spec in METRICS:
        frame[f"reference_{spec.value_column}"] = frame.groupby("season", sort=False)[
            spec.value_column
        ].transform(
            lambda values, denominator=spec.denominator_column: _weighted_reference(
                values.to_numpy(dtype=float),
                pd.to_numeric(
                    frame.loc[values.index, denominator], errors="coerce"
                ).to_numpy(dtype=float),
            )
        )
    return frame


def _weighted_reference(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _transition_frame(observed: pd.DataFrame, *, through: str) -> pd.DataFrame:
    seasons = sorted(observed["season"].astype(str).unique(), key=lambda value: int(value[:4]))
    next_season = dict(zip(seasons[:-1], seasons[1:], strict=True))
    rows: list[pd.DataFrame] = []
    value_columns = sorted(
        {
            spec.value_column for spec in METRICS
        }
        | {spec.denominator_column for spec in METRICS}
        | {f"reference_{spec.value_column}" for spec in METRICS}
    )
    for source, target in next_season.items():
        if target > through:
            continue
        left = observed.loc[observed["season"].eq(source), ["player_id", *value_columns]].copy()
        right = observed.loc[
            observed["season"].eq(target),
            [
                "player_id",
                *[
                    column
                    for column in value_columns
                    if not column.startswith("reference_")
                ],
            ],
        ].copy()
        pair = left.merge(right, on="player_id", how="inner", suffixes=("", "_target"))
        renamed: dict[str, str] = {}
        for column in value_columns:
            if column.startswith("reference_"):
                renamed[column] = f"source_{column}"
            else:
                renamed[column] = f"source_{column}"
                renamed[f"{column}_target"] = f"target_{column}"
        pair = pair.rename(columns=renamed)
        pair.insert(0, "source_season", source)
        pair.insert(1, "target_season", target)
        rows.append(pair)
    if not rows:
        raise ValueError("Padding study found no adjacent player-season transitions")
    return pd.concat(rows, ignore_index=True)


def _published_constant(metric: str) -> float:
    if metric == "three_point_pct":
        return float(MEDVEDOVSKY_2020_PROFILE_PADDING.three_point_percentage_attempts)
    if metric in {"field_goals_attempted", "free_throws_attempted"}:
        return float(
            MEDVEDOVSKY_2020_PROFILE_PADDING.usage_component_pseudo_possessions[metric]
        )
    if metric in {"offensive_rebound_pct", "defensive_rebound_pct"}:
        return float(
            MEDVEDOVSKY_2020_PROFILE_PADDING.rebound_percentage_pseudo_possessions[metric]
        )
    return float(MEDVEDOVSKY_2020_PROFILE_PADDING.rate_pseudo_possessions[metric])


def _write_run(
    *,
    estimates: pd.DataFrame,
    predictions: pd.DataFrame,
    contract: ProfilePaddingContract,
    through_target_season: str,
    artifacts_dir: Path,
) -> ProfilePaddingStudyRun:
    now = datetime.now(UTC)
    run_id = f"profile-padding-{through_target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / MODEL_NAME / through_target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        estimates.to_parquet(temporary / "padding_estimates.parquet", index=False)
        predictions.to_parquet(temporary / "transition_predictions.parquet", index=False)
        (temporary / "padding_contract.json").write_text(
            json.dumps(contract.metadata(), indent=2, sort_keys=True) + "\n"
        )
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "through_target_season": through_target_season,
            "first_frozen_evaluation_season": "2023-24",
            "objective": "target-denominator-weighted next-season squared error",
            "created_at": now.isoformat(),
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
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return ProfilePaddingStudyRun(output, run_id, contract)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate NAIL player-profile padding constants")
    parser.add_argument("--through-target-season", default=DEFAULT_THROUGH_TARGET_SEASON)
    args = parser.parse_args()
    run = run_profile_padding_study(through_target_season=args.through_target_season)
    print(f"Profile padding study: run={run.run_dir}")


if __name__ == "__main__":
    main()
