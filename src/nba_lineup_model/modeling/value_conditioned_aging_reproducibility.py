"""Compare two deterministic Value-Conditioned Aging HPM artifact runs."""

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

from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.stints import modeling_code_fingerprint

DEFAULT_MODEL = (
    "forward_centered_value_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm"
)
DEFAULT_SEASON = "2025-26"
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_AUDITS_DIR = Path("artifacts/analysis/value_conditioned_aging_reproducibility")
DEFAULT_DOCS_PAGE = Path(
    "docs/models/forward-centered-value-conditioned-aging-"
    "bounded-hierarchical-portable-matchup-contextual-rapm.md"
)
SECTION_START = "<!-- value-conditioned-aging-reproducibility:start -->"
SECTION_END = "<!-- value-conditioned-aging-reproducibility:end -->"

_TABLES = (
    ("historical_player_coefficients.parquet", ("season", "player_id")),
    ("cohort_metrics.parquet", ("cohort",)),
    ("game_predictions.parquet", ("cohort", "game_id")),
    ("team_net_rating_predictions.parquet", ("team_id",)),
    ("team_win_predictions.parquet", ("team_id",)),
)


@dataclass(frozen=True)
class ValueConditionedAgingReproducibilityRun:
    """One immutable comparison of two deterministic model runs."""

    run_dir: Path
    run_id: str


def build_value_conditioned_aging_reproducibility_audit(
    *,
    reference_run_dir: Path | str,
    candidate_run_dir: Path | str | None = None,
    season: str = DEFAULT_SEASON,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    audits_dir: Path | str = DEFAULT_AUDITS_DIR,
    docs_page: Path | str = DEFAULT_DOCS_PAGE,
    tolerance: float = 1e-10,
) -> ValueConditionedAgingReproducibilityRun:
    """Compare a rerun with a prior immutable Value HPM artifact.

    This model has no random initialization, minibatch order, or random data
    split. The audit therefore treats any material difference as numerical or
    environment drift rather than a seed-based robustness result.
    """

    if tolerance < 0:
        raise ValueError("Reproducibility tolerance must be non-negative")
    reference = Path(reference_run_dir)
    if candidate_run_dir is None:
        candidate = _latest_run(Path(artifacts_dir) / DEFAULT_MODEL / season)
    else:
        candidate = Path(candidate_run_dir)
    if reference.resolve() == candidate.resolve():
        raise ValueError("Reference and candidate runs must be distinct")
    comparison = _compare_runs(reference, candidate, tolerance=tolerance)
    now = datetime.now(UTC)
    run_id = (
        f"value-conditioned-aging-reproducibility-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    root = Path(audits_dir) / season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        comparison.to_parquet(temporary / "comparison.parquet", index=False)
        summary = _summary(comparison)
        summary.to_parquet(temporary / "summary.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "season": season,
            "reference_run": str(reference),
            "candidate_run": str(candidate),
            "tolerance": tolerance,
            "randomness_contract": (
                "No stochastic training path: this is an exact deterministic rerun audit, "
                "not a seed-sensitivity experiment"
            ),
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
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
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    _update_docs_page(Path(docs_page), summary, output=output, tolerance=tolerance)
    return ValueConditionedAgingReproducibilityRun(run_dir=output, run_id=run_id)


def _compare_runs(reference: Path, candidate: Path, *, tolerance: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for filename, keys in _TABLES:
        left = pd.read_parquet(reference / filename)
        right = pd.read_parquet(candidate / filename)
        rows.extend(_compare_table(filename, keys, left, right, tolerance=tolerance))
    return pd.DataFrame(rows)


def _compare_table(
    filename: str,
    keys: tuple[str, ...],
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    tolerance: float,
) -> list[dict[str, object]]:
    if reference.duplicated(list(keys)).any() or candidate.duplicated(list(keys)).any():
        raise ValueError(f"Comparison keys are not unique for {filename}")
    common_columns = [
        column
        for column in reference.columns.intersection(candidate.columns)
        if column not in keys
        and pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(candidate[column])
    ]
    merged = reference.merge(
        candidate,
        on=list(keys),
        how="outer",
        suffixes=("_reference", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    shared = merged.loc[merged["_merge"].eq("both")]
    rows = []
    for column in common_columns:
        left = shared[f"{column}_reference"].to_numpy(dtype=float)
        right = shared[f"{column}_candidate"].to_numpy(dtype=float)
        difference = right - left
        rows.append(
            {
                "table": filename,
                "column": column,
                "reference_row_count": len(reference),
                "candidate_row_count": len(candidate),
                "shared_row_count": len(shared),
                "reference_only_row_count": int(merged["_merge"].eq("left_only").sum()),
                "candidate_only_row_count": int(merged["_merge"].eq("right_only").sum()),
                "max_absolute_difference": (
                    float(np.max(np.abs(difference))) if len(difference) else np.nan
                ),
                "rmse_difference": (
                    float(np.sqrt(np.mean(difference**2))) if len(difference) else np.nan
                ),
                "allclose": bool(
                    np.allclose(left, right, rtol=0.0, atol=tolerance, equal_nan=True)
                ),
            }
        )
    return rows


def _summary(comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        comparison.groupby("table", as_index=False)
        .agg(
            reference_row_count=("reference_row_count", "max"),
            candidate_row_count=("candidate_row_count", "max"),
            shared_row_count=("shared_row_count", "max"),
            reference_only_row_count=("reference_only_row_count", "max"),
            candidate_only_row_count=("candidate_only_row_count", "max"),
            max_absolute_difference=("max_absolute_difference", "max"),
            max_rmse_difference=("rmse_difference", "max"),
            allclose=("allclose", "all"),
        )
        .sort_values("table", kind="stable")
        .reset_index(drop=True)
    )


def _update_docs_page(
    path: Path,
    summary: pd.DataFrame,
    *,
    output: Path,
    tolerance: float,
) -> None:
    lines = [
        SECTION_START,
        "## Deterministic Reproducibility",
        "",
        "Value-Conditioned Aging HPM has no random initialization, sampling, or split",
        "selection. A second seed is therefore not a meaningful perturbation. This audit",
        "compares a full rerun against the previous immutable artifact with absolute",
        f"tolerance `{tolerance:.0e}`.",
        "",
        (
            "| Output | Reference rows | Rerun rows | Shared rows | "
            "Max absolute difference | All close |"
        ),
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.table}` | {row.reference_row_count:,} | {row.candidate_row_count:,} | "
            f"{row.shared_row_count:,} | {row.max_absolute_difference:.3e} | "
            f"{'yes' if row.allclose else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Audit artifact: `{output}`. It retains per-column numerical differences.",
            SECTION_END,
        ]
    )
    replacement = "\n".join(lines)
    original = path.read_text()
    if SECTION_START in original:
        start = original.index(SECTION_START)
        end = original.index(SECTION_END, start) + len(SECTION_END)
        updated = original[:start] + replacement + original[end:]
    else:
        updated = original.rstrip() + "\n\n" + replacement + "\n"
    path.write_text(updated)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two deterministic Value-Conditioned Aging HPM runs"
    )
    parser.add_argument("--reference-run-dir", required=True)
    parser.add_argument("--candidate-run-dir")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    run = build_value_conditioned_aging_reproducibility_audit(
        reference_run_dir=args.reference_run_dir,
        candidate_run_dir=args.candidate_run_dir,
        season=args.season,
        tolerance=args.tolerance,
    )
    print(f"Value-conditioned aging reproducibility audit: run={run.run_dir}")


if __name__ == "__main__":
    main()
