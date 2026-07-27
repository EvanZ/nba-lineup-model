from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba_lineup_model.audit.schema import AuditGameSpec, AuditManifest

_REQUIRED_COLUMNS = {"game_id", "season", "season_type"}
_STRATA_COLUMNS = ["season", "season_type", "sample_group"]


def sample_audit_manifest(
    catalog: pd.DataFrame,
    *,
    games_per_stratum: int,
    random_seed: int = 0,
) -> AuditManifest:
    """Deterministically sample games by season, season type, and sample group."""

    if games_per_stratum < 1:
        raise ValueError("games_per_stratum must be positive")
    missing = _REQUIRED_COLUMNS - set(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")
    if catalog["game_id"].duplicated().any():
        duplicates = sorted(catalog.loc[catalog["game_id"].duplicated(), "game_id"].unique())
        raise ValueError(f"Catalog contains duplicate game IDs: {duplicates}")

    candidates = catalog.copy()
    if "sample_group" not in candidates:
        candidates["sample_group"] = "default"
    null_columns = [
        column
        for column in [*_REQUIRED_COLUMNS, "sample_group"]
        if candidates[column].isna().any()
    ]
    if null_columns:
        raise ValueError(f"Catalog contains null values in: {sorted(null_columns)}")

    sampled_groups = []
    for group_number, (_, group) in enumerate(
        candidates.groupby(_STRATA_COLUMNS, dropna=False, sort=True)
    ):
        sample_size = min(games_per_stratum, len(group))
        sampled_groups.append(
            group.sample(
                n=sample_size,
                random_state=random_seed + group_number,
            )
        )
    sampled = pd.concat(sampled_groups, ignore_index=True)
    sampled = sampled.sort_values([*_STRATA_COLUMNS, "game_id"])

    games = []
    for row in sampled.to_dict(orient="records"):
        expected_overtime = row.get("expected_overtime")
        if pd.isna(expected_overtime) and "is_overtime" in row:
            expected_overtime = row["is_overtime"]
        if pd.isna(expected_overtime):
            expected_overtime = None
        elif expected_overtime is not None:
            expected_overtime = bool(expected_overtime)
        games.append(
            AuditGameSpec(
                game_id=str(row["game_id"]),
                season=str(row["season"]),
                season_type=str(row["season_type"]),
                sample_group=str(row["sample_group"]),
                expected_overtime=expected_overtime,
            )
        )
    return AuditManifest(games=games)


def _read_catalog(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"game_id": "string"})
    raise ValueError("Catalog must be a .parquet or .csv file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic stratified NBA audit manifest."
    )
    parser.add_argument("catalog", help="Game catalog in Parquet or CSV format")
    parser.add_argument(
        "--games-per-stratum",
        type=int,
        default=25,
        help="Maximum games per season/type/group stratum",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic sample seed")
    parser.add_argument(
        "--output",
        default="config/audit_manifest.json",
        help="Manifest JSON output path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = sample_audit_manifest(
        _read_catalog(Path(args.catalog)),
        games_per_stratum=args.games_per_stratum,
        random_seed=args.seed,
    )
    output_path = manifest.write(Path(args.output))
    print(f"{len(manifest.games)} games: {output_path}")


if __name__ == "__main__":
    main()
