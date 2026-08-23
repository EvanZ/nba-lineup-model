"""Paired-bootstrap non-promotion gate for NAIL-RAPM v1.2.1."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.modeling.forward_nail_gap_returners import MODEL_NAME as INCUMBENT_MODEL_NAME
from nba_lineup_model.modeling.forward_nail_v121_pruned_nonadditive import (
    MODEL_NAME as CANDIDATE_MODEL_NAME,
)
from nba_lineup_model.modeling.frozen_model_tournament import _paired_metrics
from nba_lineup_model.modeling.frozen_multiseason_backtest import DEFAULT_SEASONS
from nba_lineup_model.modeling.nail_v13_additive_profiles_bootstrap import (
    DEFAULT_DRAWS,
    DEFAULT_SEED,
    PRIMARY_METRIC,
    RELATIVE_HARM_THRESHOLD,
    _filter_season,
    _latest_directory,
    _source,
)

DEFAULT_BACKTEST_ROOT = Path(
    "artifacts/models/nail_v121_pruned_nonadditive_frozen_backtest/"
    "frozen_multiseason_backtest/2023-24_to_2025-26"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/models/nail_v121_pruned_nonadditive_bootstrap/2023-24_to_2025-26"
)


def run_nail_v121_pruned_nonadditive_bootstrap(
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    backtest_root: Path | str = DEFAULT_BACKTEST_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Persist paired bootstrap evidence for the v1.2.1 feature ablation."""
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    source_run = _latest_directory(Path(backtest_root))
    games = pd.read_parquet(source_run / "regular_game_predictions.parquet")
    possessions = pd.read_parquet(source_run / "possession_predictions.parquet")
    sources = {
        model: _source(games, possessions, model)
        for model in (INCUMBENT_MODEL_NAME, CANDIDATE_MODEL_NAME)
    }
    scopes: list[tuple[str, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]] = [
        ("pooled", sources[INCUMBENT_MODEL_NAME], sources[CANDIDATE_MODEL_NAME])
    ]
    scopes.extend(
        (
            season,
            _filter_season(sources[INCUMBENT_MODEL_NAME], season),
            _filter_season(sources[CANDIDATE_MODEL_NAME], season),
        )
        for season in DEFAULT_SEASONS
    )
    metric_rows: list[pd.DataFrame] = []
    gate_rows: list[dict[str, object]] = []
    for scope_index, (scope, incumbent, candidate) in enumerate(scopes):
        result = _paired_metrics(
            incumbent, candidate, draws=draws, seed=seed + 10 * scope_index
        )
        result.insert(0, "scope", scope)
        result.insert(1, "incumbent_model", INCUMBENT_MODEL_NAME)
        result.insert(2, "challenger_model", CANDIDATE_MODEL_NAME)
        metric_rows.append(result)
        primary = result.loc[result["metric"].eq(PRIMARY_METRIC)].iloc[0]
        practical_harm = float(primary["incumbent_value"]) * RELATIVE_HARM_THRESHOLD
        gate_rows.append(
            {
                "scope": scope,
                "metric": PRIMARY_METRIC,
                "incumbent_value": float(primary["incumbent_value"]),
                "challenger_value": float(primary["challenger_value"]),
                "difference_candidate_minus_incumbent": float(
                    primary["difference_candidate_minus_incumbent"]
                ),
                "ci_lower": float(primary["ci_lower"]),
                "ci_upper": float(primary["ci_upper"]),
                "relative_harm_threshold": RELATIVE_HARM_THRESHOLD,
                "absolute_harm_threshold": practical_harm,
                "non_promotion_gate_passed": bool(
                    float(primary["ci_upper"]) <= practical_harm
                ),
            }
        )
    metrics = pd.concat(metric_rows, ignore_index=True)
    gate = pd.DataFrame(gate_rows)
    root = Path(output_root)
    run_id = (
        "nail-v121-pruned-nonadditive-bootstrap-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(run_dir / "paired_bootstrap_metrics.parquet", index=False)
    gate.to_parquet(run_dir / "non_promotion_gate.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source_run),
                "draws": draws,
                "seed": seed,
                "resampling_unit": "games stratified within frozen season",
                "incumbent_model": INCUMBENT_MODEL_NAME,
                "challenger_model": CANDIDATE_MODEL_NAME,
                "primary_metric": PRIMARY_METRIC,
                "relative_harm_threshold": RELATIVE_HARM_THRESHOLD,
                "non_promotion_gate_passed": bool(gate["non_promotion_gate_passed"].all()),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap NAIL-RAPM v1.2.1 non-additive ablation")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run_dir = run_nail_v121_pruned_nonadditive_bootstrap(draws=args.draws, seed=args.seed)
    print(f"NAIL-RAPM v1.2.1 bootstrap: run={run_dir}")


if __name__ == "__main__":
    main()
