"""Exact player-level accounting for the frozen portable HPM context state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import lineup_side_context_features
from nba_lineup_model.modeling.contextual_profiles import PROFILE_RATE_COLUMNS
from nba_lineup_model.modeling.forward_contextual_rapm import _previous_season
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_TARGET_SEASON = "2025-26"
DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_PAGE_PATH = Path("docs/models/portable-context-attribution-audit.md")
SOURCE_MODEL = (
    "forward_centered_value_conditioned_aging_"
    "bounded_hierarchical_portable_matchup_contextual_rapm"
)
MODEL = "portable_context_attribution_audit"
RUN_PREFIX = "portable-context-attribution-audit"


@dataclass(frozen=True)
class AuditCase:
    """One realized target-season unit matchup, oriented toward the focal unit."""

    slug: str
    title: str
    unit: tuple[str, ...]
    opponent: tuple[str, ...]


DEFAULT_CASES = (
    AuditCase(
        slug="warriors-at-spurs",
        title="Warriors unit at Spurs unit",
        unit=(
            "Stephen Curry",
            "Jimmy Butler III",
            "Draymond Green",
            "Moses Moody",
            "Will Richard",
        ),
        opponent=(
            "Harrison Barnes",
            "De'Aaron Fox",
            "Devin Vassell",
            "Victor Wembanyama",
            "Stephon Castle",
        ),
    ),
    AuditCase(
        slug="rockets-vs-clippers",
        title="Rockets unit vs Clippers unit",
        unit=(
            "Kevin Durant",
            "Alperen Sengun",
            "Jabari Smith Jr.",
            "Tari Eason",
            "Amen Thompson",
        ),
        opponent=(
            "Brook Lopez",
            "Kawhi Leonard",
            "Kris Dunn",
            "Derrick Jones Jr.",
            "John Collins",
        ),
    ),
    AuditCase(
        slug="knicks-vs-hawks",
        title="Knicks unit vs Hawks unit",
        unit=(
            "Karl-Anthony Towns",
            "OG Anunoby",
            "Josh Hart",
            "Mikal Bridges",
            "Jalen Brunson",
        ),
        opponent=(
            "CJ McCollum",
            "Nickeil Alexander-Walker",
            "Onyeka Okongwu",
            "Jalen Johnson",
            "Dyson Daniels",
        ),
    ),
)


def build_portable_context_attribution_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    target_season: str = DEFAULT_TARGET_SEASON,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = Path("data/analytical"),
) -> Path:
    """Build exact composition and matchup Shapley ledgers for named units."""

    root = Path(artifacts_dir)
    source = Path(source_run_dir) if source_run_dir is not None else _latest_run(
        root / SOURCE_MODEL / target_season
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != SOURCE_MODEL:
        raise ValueError("Attribution audit requires the published Value-Conditioned Aging HPM")
    source_season = _previous_season(target_season)
    models = joblib.load(source / "season_context_models.joblib")
    context_model = models.get(source_season)
    if not isinstance(context_model, MatchupContextualModel):
        raise ValueError(f"Source artifact has no {source_season} portable context state")
    profiles = pd.read_parquet(source / "target_player_profiles.parquet")
    coefficients = pd.read_parquet(source / "historical_player_coefficients.parquet")
    ratings = coefficients.loc[
        coefficients["season"].eq(target_season), ["player_id", "rapm"]
    ].copy()
    reference_profiles = _reference_player_profiles(
        profiles,
        panel_path=Path(panel_path),
        source_season=source_season,
    )
    augmented_profiles = pd.concat([profiles, reference_profiles], ignore_index=True)
    stints = read_rapm_stints(target_season, analytical_dir=analytical_dir)
    player_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for case in DEFAULT_CASES:
        unit_ids = _player_ids(case.unit, profiles)
        opponent_ids = _player_ids(case.opponent, profiles)
        attribution, summary = attribute_matchup_context(
            context_model,
            augmented_profiles,
            unit_ids=unit_ids,
            opponent_ids=opponent_ids,
            ratings=ratings,
        )
        exposure = _realized_exposure(stints, unit_ids, opponent_ids)
        attribution.insert(0, "case", case.slug)
        attribution.insert(1, "case_title", case.title)
        player_rows.append(attribution)
        summaries.append({"case": case.slug, "case_title": case.title, **exposure, **summary})
    return _write_run(
        source=source,
        target_season=target_season,
        source_season=source_season,
        players=pd.concat(player_rows, ignore_index=True),
        summary=pd.DataFrame(summaries),
        artifacts_dir=root,
    )


def attribute_matchup_context(
    context_model: MatchupContextualModel,
    profiles: pd.DataFrame,
    *,
    unit_ids: tuple[int, ...],
    opponent_ids: tuple[int, ...],
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Allocate one exact portable context score through Shapley values.

    The five-player portable score uses a synthetic league-average player as an
    accounting origin. The ten-player matchup ledger uses the same origin on
    both sides. This makes all player deltas sum exactly to the frozen model's
    composition edge plus matchup residual; it does not redefine the model's
    empirical reference-unit field.
    """

    _validate_lineup(unit_ids, "unit")
    _validate_lineup(opponent_ids, "opponent")
    reference_ids = tuple(-1_000_001 - index for index in range(5))
    unit_lineups = _subset_lineups(unit_ids, reference_ids)
    opponent_lineups = _subset_lineups(opponent_ids, reference_ids)
    unit_features = lineup_side_context_features(unit_lineups, profiles)
    opponent_features = lineup_side_context_features(opponent_lineups, profiles)
    unit_scores = context_model.predict_side_scores(unit_features)
    opponent_scores = context_model.predict_side_scores(opponent_features)
    unit_composition = _shapley_values(unit_scores, player_count=5)
    opponent_composition = _shapley_values(opponent_scores, player_count=5)

    pair_count = 1 << 10
    unit_masks = np.arange(pair_count, dtype=np.int64) & 31
    opponent_masks = np.arange(pair_count, dtype=np.int64) >> 5
    total_context = context_model.predict_lineups(
        [unit_lineups[mask] for mask in unit_masks],
        [opponent_lineups[mask] for mask in opponent_masks],
        profiles,
    )
    matchup_values = total_context - unit_scores[unit_masks] + opponent_scores[opponent_masks]
    matchup = _shapley_values(matchup_values, player_count=10)
    full_mask = (1 << 5) - 1
    composition_edge = float(unit_scores[full_mask] - opponent_scores[full_mask])
    matchup_edge = float(matchup_values[-1])
    total_edge = float(total_context[-1])
    if not np.isclose(unit_composition.sum() - opponent_composition.sum(), composition_edge):
        raise ValueError("Composition Shapley ledger does not reconcile to the portable edge")
    if not np.isclose(matchup.sum(), matchup_edge):
        raise ValueError("Matchup Shapley ledger does not reconcile to the matchup residual")
    if not np.isclose(
        unit_composition.sum() - opponent_composition.sum() + matchup.sum(), total_edge
    ):
        raise ValueError("Player context ledger does not reconcile to total context")

    rating_map = dict(zip(ratings["player_id"].astype(int), ratings["rapm"], strict=True))
    name_map = profiles.set_index("player_id")["player_name"].to_dict()
    rows: list[dict[str, object]] = []
    for side, player_ids, composition_values, sign, offset in (
        ("unit", unit_ids, unit_composition, 1.0, 0),
        ("opponent", opponent_ids, opponent_composition, -1.0, 5),
    ):
        for index, player_id in enumerate(player_ids):
            player_rating = float(rating_map[player_id])
            player_edge = sign * player_rating
            composition_edge_value = sign * float(composition_values[index])
            matchup_edge_value = float(matchup[offset + index])
            rows.append(
                {
                    "side": side,
                    "player_id": player_id,
                    "player_name": str(name_map[player_id]),
                    "hpm_player_rating": player_rating,
                    "player_rating_to_unit_edge": player_edge,
                    "composition_contribution_to_unit_edge": composition_edge_value,
                    "matchup_contribution_to_unit_edge": matchup_edge_value,
                    "context_contribution_to_unit_edge": (
                        composition_edge_value + matchup_edge_value
                    ),
                    "combined_contribution_to_unit_edge": (
                        player_edge + composition_edge_value + matchup_edge_value
                    ),
                }
            )
    output = pd.DataFrame(rows).sort_values(
        ["side", "context_contribution_to_unit_edge", "player_name"],
        ascending=[True, False, True],
        kind="stable",
    )
    return output, {
        "unit_composition_rating": float(unit_scores[full_mask]),
        "opponent_composition_rating": float(opponent_scores[full_mask]),
        "composition_edge": composition_edge,
        "matchup_edge": matchup_edge,
        "total_context_edge": total_edge,
        "additive_player_edge": float(
            sum(rating_map[player] for player in unit_ids)
            - sum(rating_map[player] for player in opponent_ids)
        ),
        "predicted_net_rating": float(
            total_edge
            + sum(rating_map[player] for player in unit_ids)
            - sum(rating_map[player] for player in opponent_ids)
        ),
        "synthetic_reference_unit_rating": float(unit_scores[0]),
    }


def _reference_player_profiles(
    profiles: pd.DataFrame,
    *,
    panel_path: Path,
    source_season: str,
) -> pd.DataFrame:
    """Create five distinct average-player rows for the Shapley accounting origin."""

    panel = pd.read_parquet(panel_path)
    weights = panel.loc[
        panel["season"].eq(source_season), ["player_id", "rapm_possessions"]
    ].rename(columns={"rapm_possessions": "weight"})
    weighted = profiles.merge(weights, on="player_id", how="left", validate="one_to_one")
    weights_array = weighted["weight"].fillna(0.0).to_numpy(dtype=float)
    if not np.any(weights_array > 0):
        weights_array = np.ones(len(weighted), dtype=float)
    row: dict[str, object] = {"player_name": "Synthetic reference player"}
    for column in (*PROFILE_RATE_COLUMNS, "profile_imputed", "profile_replacement_weight"):
        row[column] = float(np.average(weighted[column].to_numpy(dtype=float), weights=weights_array))
    return pd.DataFrame([{**row, "player_id": -1_000_001 - index} for index in range(5)])


def _subset_lineups(player_ids: tuple[int, ...], reference_ids: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [
        tuple(
            player_ids[index] if mask & (1 << index) else reference_ids[index]
            for index in range(5)
        )
        for mask in range(1 << 5)
    ]


def _shapley_values(values: np.ndarray, *, player_count: int) -> np.ndarray:
    """Return exact Shapley values from a complete Boolean-coalition value table."""

    expected = 1 << player_count
    if len(values) != expected:
        raise ValueError(f"Expected {expected} coalition values, received {len(values)}")
    output = np.zeros(player_count, dtype=float)
    denominator = math.factorial(player_count)
    for player in range(player_count):
        for coalition in range(expected):
            if coalition & (1 << player):
                continue
            size = coalition.bit_count()
            weight = math.factorial(size) * math.factorial(player_count - size - 1) / denominator
            output[player] += weight * (values[coalition | (1 << player)] - values[coalition])
    return output


def _player_ids(names: tuple[str, ...], profiles: pd.DataFrame) -> tuple[int, ...]:
    lookup = profiles.set_index("player_name")["player_id"]
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"Audit player profiles are missing: {missing}")
    return tuple(int(lookup[name]) for name in names)


def _realized_exposure(
    stints: pd.DataFrame, unit_ids: tuple[int, ...], opponent_ids: tuple[int, ...]
) -> dict[str, object]:
    unit = set(unit_ids)
    opponent = set(opponent_ids)
    mask = stints.apply(
        lambda row: (
            set(row["home_player_ids"]) == unit and set(row["away_player_ids"]) == opponent
        )
        or (
            set(row["home_player_ids"]) == opponent and set(row["away_player_ids"]) == unit
        ),
        axis=1,
    )
    matches = stints.loc[mask]
    return {
        "realized_matchup_possessions": float(matches["possessions"].sum()),
        "realized_matchup_games": int(matches["game_id"].nunique()),
    }


def _validate_lineup(player_ids: tuple[int, ...], label: str) -> None:
    if len(player_ids) != 5 or len(set(player_ids)) != 5:
        raise ValueError(f"{label} must contain five unique players")


def _write_run(
    *,
    source: Path,
    target_season: str,
    source_season: str,
    players: pd.DataFrame,
    summary: pd.DataFrame,
    artifacts_dir: Path,
) -> Path:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / MODEL / target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        players.to_parquet(temporary / "player_context_attribution.parquet", index=False)
        summary.to_parquet(temporary / "case_summary.parquet", index=False)
        metadata = {
            "run_id": run_id,
            "model": MODEL,
            "target_season": target_season,
            "source_season": source_season,
            "source_run_dir": str(source),
            "created_at": now.isoformat(),
            "method": "exact Shapley values over five-player composition and ten-player matchup coalitions",
            "composition_origin": "five synthetic possession-weighted average-player profiles",
            "portable_reference_contract": "HPM portable scores remain expectations over the frozen empirical reference-unit field",
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts = [
            {
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": artifacts}, indent=2) + "\n"
        )
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return output
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
    """Build the public portable-context attribution audit artifact."""

    parser = argparse.ArgumentParser(description="Build an HPM player context-attribution audit")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    args = parser.parse_args()
    run = build_portable_context_attribution_audit(target_season=args.through_season)
    print(f"Portable context attribution audit: run={run}")


if __name__ == "__main__":
    main()
