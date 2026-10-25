"""Possession-based, prior-centered in-season NAIL state updates.

The frozen preseason NAIL forecast is the prior mean.  Each completed game is
reduced to an immutable sparse contribution to the current-season normal
equations.  Nightly updates therefore add or replace a game's contribution;
they never refit or reread the historical NAIL training archive.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from nba_lineup_model.models.baselines import signed_entity_matrix, vocabulary_mapping
from nba_lineup_model.season.schema import validate_season

CONTRIBUTION_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
_SHA256_HEX_LENGTH = 64
_REQUIRED_STATE_COLUMNS = {
    "schema_version",
    "season",
    "core_model_version",
    "state_updater_version",
    "player_id",
    "team_at_cutoff",
    "cutoff_time",
    "cutoff_game_id",
    "frozen_preseason_nail",
    "current_season_residual",
    "updated_nail",
    "raw_on_court_possessions",
    "effective_information",
    "prior_equivalent_possessions",
    "prior_weight",
    "evidence_weight",
    "posterior_standard_error",
    "included_game_ids_json",
    "contribution_lineage_json",
}


@dataclass(frozen=True)
class InSeasonNailConfig:
    """Frozen parameters for one independently versioned state updater."""

    core_model_version: str
    updater_version: str
    prior_equivalent_possessions: float
    observation_variance: float = 1.0

    def validate(self) -> None:
        if not self.core_model_version.strip() or not self.updater_version.strip():
            raise ValueError("Core and updater model versions must be non-empty")
        if not np.isfinite(self.prior_equivalent_possessions):
            raise ValueError("Prior equivalent possessions must be finite")
        if self.prior_equivalent_possessions <= 0:
            raise ValueError("Prior equivalent possessions must be positive")
        if not np.isfinite(self.observation_variance) or self.observation_variance <= 0:
            raise ValueError("Observation variance must be finite and positive")


@dataclass(frozen=True)
class FrozenPreseasonNail:
    """Validated published forecast used as the complete in-season prior."""

    season: str
    model_artifact: str
    run_id: str
    contract: str
    rankings: pd.DataFrame
    rankings_sha256: str
    metadata_sha256: str

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(self.rankings["player_id"].astype(int))

    @property
    def ratings(self) -> dict[int, float]:
        return dict(
            zip(
                self.rankings["player_id"].astype(int),
                self.rankings["rapm"].astype(float),
                strict=True,
            )
        )

    @property
    def teams(self) -> dict[int, str]:
        return dict(
            zip(
                self.rankings["player_id"].astype(int),
                self.rankings["team"].astype(str),
                strict=True,
            )
        )

    @property
    def lineage_sha256(self) -> str:
        return hashlib.sha256(f"{self.rankings_sha256}:{self.metadata_sha256}".encode()).hexdigest()


@dataclass(frozen=True)
class GameContribution:
    """One game's immutable contribution to the residual normal equations."""

    season: str
    game_id: str
    game_time_utc: datetime
    finalized_at: datetime
    source_sha256: str
    source_uri: str
    core_model_version: str
    context_model_version: str
    player_ids: tuple[int, ...]
    xtwx: sparse.csr_matrix
    xtwy: np.ndarray
    on_court_possessions: np.ndarray
    game_possessions: float
    stint_count: int
    context_offset_sha256: str
    schema_version: int = CONTRIBUTION_SCHEMA_VERSION
    contribution_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        validate_season(self.season)
        if not self.game_id.strip():
            raise ValueError("Game ID must be non-empty")
        game_time = _as_utc(self.game_time_utc, "game_time_utc")
        finalized = _as_utc(self.finalized_at, "finalized_at")
        if finalized < game_time:
            raise ValueError("Finalized time cannot precede game time")
        _validate_sha256(self.source_sha256, "source_sha256")
        _validate_sha256(self.context_offset_sha256, "context_offset_sha256")
        if not self.source_uri.strip():
            raise ValueError("Source URI must be non-empty")
        if not self.core_model_version.strip() or not self.context_model_version.strip():
            raise ValueError("Model versions must be non-empty")
        player_ids = tuple(int(player_id) for player_id in self.player_ids)
        if not player_ids or len(player_ids) != len(set(player_ids)):
            raise ValueError("Player vocabulary must be non-empty and unique")
        if tuple(sorted(player_ids)) != player_ids:
            raise ValueError("Player vocabulary must be sorted")
        matrix = sparse.csr_matrix(self.xtwx, dtype=np.float64)
        matrix.sum_duplicates()
        matrix.sort_indices()
        player_count = len(player_ids)
        if matrix.shape != (player_count, player_count):
            raise ValueError("X'WX must be square and match the player vocabulary")
        if matrix.nnz and not np.isfinite(matrix.data).all():
            raise ValueError("X'WX must be finite")
        if not np.allclose(matrix.toarray(), matrix.toarray().T, atol=1e-10):
            raise ValueError("X'WX must be symmetric")
        xtwy = _finite_vector(self.xtwy, player_count, "X'Wy")
        exposure = _finite_vector(
            self.on_court_possessions,
            player_count,
            "on-court possessions",
        )
        if (exposure < 0).any():
            raise ValueError("On-court possessions cannot be negative")
        if not np.isfinite(self.game_possessions) or self.game_possessions <= 0:
            raise ValueError("Game possessions must be finite and positive")
        if self.stint_count <= 0:
            raise ValueError("Stint count must be positive")
        if self.schema_version != CONTRIBUTION_SCHEMA_VERSION:
            raise ValueError("Unsupported game-contribution schema version")

        object.__setattr__(self, "game_time_utc", game_time)
        object.__setattr__(self, "finalized_at", finalized)
        object.__setattr__(self, "player_ids", player_ids)
        object.__setattr__(self, "xtwx", matrix)
        object.__setattr__(self, "xtwy", xtwy)
        object.__setattr__(self, "on_court_possessions", exposure)
        object.__setattr__(self, "contribution_sha256", _contribution_sha256(self))
        matrix.data.setflags(write=False)
        matrix.indices.setflags(write=False)
        matrix.indptr.setflags(write=False)


@dataclass
class ContributionLedger:
    """Current immutable contribution selected for each game."""

    _by_game_id: dict[str, GameContribution] = field(default_factory=dict)

    def upsert(self, contribution: GameContribution) -> str:
        prior = self._by_game_id.get(contribution.game_id)
        if prior is not None and prior.contribution_sha256 == contribution.contribution_sha256:
            return "unchanged"
        self._by_game_id[contribution.game_id] = contribution
        return "revised" if prior is not None else "added"

    def values(self) -> tuple[GameContribution, ...]:
        return tuple(
            sorted(
                self._by_game_id.values(),
                key=lambda item: (item.finalized_at, item.game_time_utc, item.game_id),
            )
        )


def load_frozen_preseason_nail(
    rankings_path: Path | str,
    *,
    metadata_path: Path | str | None = None,
    expected_season: str | None = None,
    expected_model_artifact: str | None = None,
) -> FrozenPreseasonNail:
    """Load the exact published forecast and enforce its complete-prior contract."""

    parquet_path = Path(rankings_path)
    json_path = (
        Path(metadata_path) if metadata_path is not None else parquet_path.with_suffix(".json")
    )
    rankings = pd.read_parquet(parquet_path)
    metadata = json.loads(json_path.read_text())
    required_columns = {
        "season",
        "player_id",
        "player_name",
        "team",
        "rapm",
        "prior_rating",
        "additive_profile_adjustment",
        "forecast_source",
    }
    missing = required_columns - set(rankings.columns)
    if missing:
        raise ValueError(f"Preseason rankings missing columns: {sorted(missing)}")
    if rankings.empty or rankings["player_id"].duplicated().any():
        raise ValueError("Preseason rankings must contain one row per player")
    seasons = tuple(rankings["season"].astype(str).unique())
    if len(seasons) != 1:
        raise ValueError("Preseason rankings must contain exactly one season")
    season = validate_season(seasons[0])
    if metadata.get("target_season") != season:
        raise ValueError("Preseason ranking metadata season does not match the data")
    if expected_season is not None and validate_season(expected_season) != season:
        raise ValueError("Preseason rankings do not match the expected season")
    model_artifact = str(metadata.get("model_artifact", "")).strip()
    run_id = str(metadata.get("run_id", "")).strip()
    contract = str(metadata.get("contract", "")).strip()
    if not model_artifact or not run_id or not contract:
        raise ValueError("Preseason ranking metadata is incomplete")
    if expected_model_artifact is not None and model_artifact != expected_model_artifact:
        raise ValueError("Preseason rankings do not match the promoted model artifact")
    if int(metadata.get("player_count", -1)) != len(rankings):
        raise ValueError("Preseason ranking metadata player count does not match")

    numeric = rankings.loc[:, ["rapm", "prior_rating"]].apply(pd.to_numeric, errors="coerce")
    additive = pd.to_numeric(rankings["additive_profile_adjustment"], errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Published preseason ratings and scalar priors must be finite")
    cold_start = rankings["forecast_source"].astype(str).str.contains("cold_start")
    missing_additive = additive.isna()
    if (missing_additive & ~cold_start).any():
        raise ValueError("Only cold-start forecasts may omit additive profile adjustment")
    expected = numeric["prior_rating"] + additive.fillna(0.0)
    if not np.allclose(numeric["rapm"], expected, atol=1e-10):
        raise ValueError("Published preseason NAIL must equal scalar prior plus additive profile")
    normalized = rankings.copy()
    normalized["player_id"] = normalized["player_id"].astype(int)
    normalized.loc[:, numeric.columns] = numeric
    normalized["additive_profile_adjustment"] = additive
    normalized = normalized.sort_values("player_id", kind="stable").reset_index(drop=True)
    return FrozenPreseasonNail(
        season=season,
        model_artifact=model_artifact,
        run_id=run_id,
        contract=contract,
        rankings=normalized,
        rankings_sha256=_file_sha256(parquet_path),
        metadata_sha256=_file_sha256(json_path),
    )


def build_game_contribution(
    stints: pd.DataFrame,
    *,
    season: str,
    game_id: str,
    game_time_utc: datetime,
    finalized_at: datetime,
    source_sha256: str,
    source_uri: str,
    core_model_version: str,
    context_model_version: str,
    player_ids: Sequence[int],
    preseason_nail: Mapping[int, float] | Sequence[float] | np.ndarray,
    fixed_context_offset: float | Sequence[float] | np.ndarray,
) -> GameContribution:
    """Residualize one game's stints and form sparse sufficient statistics.

    ``fixed_context_offset`` must contain all frozen non-player effects for a
    stint (including HCA, back-to-back, and frozen lineup-context terms).
    """

    required = {
        "game_id",
        "home_player_ids",
        "away_player_ids",
        "target_home_net_rating",
        "possessions",
    }
    missing = required - set(stints.columns)
    if missing:
        raise ValueError(f"Stints missing required columns: {sorted(missing)}")
    if stints.empty:
        raise ValueError("A game contribution requires at least one stint")
    observed_game_ids = set(stints["game_id"].astype(str))
    if observed_game_ids != {str(game_id)}:
        raise ValueError("Stints must contain exactly the requested game")

    vocabulary = tuple(int(player_id) for player_id in player_ids)
    if tuple(sorted(vocabulary)) != vocabulary or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("Player vocabulary must be sorted and unique")
    player_to_column = vocabulary_mapping(vocabulary)
    _validate_lineups(stints, player_to_column)
    design = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_to_column,
        multiple=True,
    )
    weights = stints["possessions"].to_numpy(dtype=float)
    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Stint possessions must be finite and positive")
    if not np.isfinite(target).all():
        raise ValueError("Stint targets must be finite")
    prior = _prior_vector(preseason_nail, vocabulary)
    context = _broadcast_offset(fixed_context_offset, len(stints))
    residual = target - context - np.asarray(design @ prior).reshape(-1)
    weighted_design = design.multiply(weights[:, None])
    xtwx = (design.T @ weighted_design).tocsr()
    xtwy = np.asarray(design.T @ (weights * residual), dtype=float).reshape(-1)
    exposure = np.asarray(abs(design).T @ weights, dtype=float).reshape(-1)

    return GameContribution(
        season=season,
        game_id=str(game_id),
        game_time_utc=game_time_utc,
        finalized_at=finalized_at,
        source_sha256=source_sha256,
        source_uri=source_uri,
        core_model_version=core_model_version,
        context_model_version=context_model_version,
        player_ids=vocabulary,
        xtwx=xtwx,
        xtwy=xtwy,
        on_court_possessions=exposure,
        game_possessions=float(weights.sum()),
        stint_count=len(stints),
        context_offset_sha256=_array_sha256(context),
    )


def solve_in_season_state(
    *,
    season: str,
    cutoff_time: datetime,
    player_ids: Sequence[int],
    preseason_nail: Mapping[int, float] | Sequence[float] | np.ndarray,
    contributions: Sequence[GameContribution],
    config: InSeasonNailConfig,
    relative_prior_precision: Sequence[float] | np.ndarray | None = None,
    team_by_player: Mapping[int, str] | None = None,
    cutoff_game_id: str | None = None,
) -> pd.DataFrame:
    """Solve the exact cumulative player state strictly before ``cutoff_time``."""

    validate_season(season)
    config.validate()
    cutoff = _as_utc(cutoff_time, "cutoff_time")
    vocabulary = tuple(int(player_id) for player_id in player_ids)
    if tuple(sorted(vocabulary)) != vocabulary or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("Player vocabulary must be sorted and unique")
    prior = _prior_vector(preseason_nail, vocabulary)
    relative = (
        np.ones(len(vocabulary), dtype=float)
        if relative_prior_precision is None
        else _finite_vector(
            np.asarray(relative_prior_precision, dtype=float),
            len(vocabulary),
            "relative prior precision",
        )
    )
    if (relative <= 0).any():
        raise ValueError("Relative prior precision must be positive")

    selected = _selected_contributions(
        contributions,
        season=season,
        cutoff_time=cutoff,
        player_ids=vocabulary,
        core_model_version=config.core_model_version,
    )
    player_count = len(vocabulary)
    gram = sparse.csr_matrix((player_count, player_count), dtype=float)
    score = np.zeros(player_count, dtype=float)
    exposure = np.zeros(player_count, dtype=float)
    for contribution in selected:
        gram = gram + contribution.xtwx
        score += contribution.xtwy
        exposure += contribution.on_court_possessions

    prior_precision = config.prior_equivalent_possessions * relative
    posterior_precision = gram + sparse.diags(prior_precision, format="csr")
    adjustment = (
        np.zeros(player_count, dtype=float)
        if not selected
        else np.asarray(spsolve(posterior_precision, score), dtype=float)
    )
    if not np.isfinite(adjustment).all():
        raise ValueError("In-season state solve produced non-finite adjustments")
    information = np.asarray(gram.diagonal(), dtype=float)
    evidence_weight = information / (information + prior_precision)
    prior_weight = 1.0 - evidence_weight
    posterior_variance = config.observation_variance * np.diag(
        np.linalg.inv(posterior_precision.toarray())
    )
    standard_error = np.sqrt(np.maximum(posterior_variance, 0.0))
    included_game_ids = tuple(contribution.game_id for contribution in selected)
    lineage = tuple(contribution.contribution_sha256 for contribution in selected)
    teams = team_by_player or {}

    state = pd.DataFrame(
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "season": season,
            "core_model_version": config.core_model_version,
            "state_updater_version": config.updater_version,
            "player_id": vocabulary,
            "team_at_cutoff": [teams.get(player_id) for player_id in vocabulary],
            "cutoff_time": cutoff,
            "cutoff_game_id": cutoff_game_id,
            "frozen_preseason_nail": prior,
            "current_season_residual": adjustment,
            "updated_nail": prior + adjustment,
            "raw_on_court_possessions": exposure,
            "effective_information": information,
            "prior_equivalent_possessions": prior_precision,
            "prior_weight": prior_weight,
            "evidence_weight": evidence_weight,
            "posterior_standard_error": standard_error,
            "included_game_ids_json": json.dumps(included_game_ids),
            "contribution_lineage_json": json.dumps(lineage),
        }
    )
    validate_player_state(state)
    return state


def publish_game_contribution(
    contribution: GameContribution,
    root: Path | str,
) -> tuple[Path, bool]:
    """Atomically publish a content-addressed local contribution directory."""

    target = (
        Path(root)
        / f"season={contribution.season}"
        / f"game_id={contribution.game_id}"
        / f"contribution_sha256={contribution.contribution_sha256}"
    )
    if target.exists():
        loaded = read_game_contribution(target)
        if loaded.contribution_sha256 != contribution.contribution_sha256:
            raise ValueError("Existing immutable contribution does not match")
        return target, False

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        matrix_path = staging / "xtwx.npz"
        vectors_path = staging / "vectors.npz"
        sparse.save_npz(matrix_path, contribution.xtwx, compressed=True)
        np.savez_compressed(
            vectors_path,
            xtwy=contribution.xtwy,
            on_court_possessions=contribution.on_court_possessions,
        )
        manifest = _contribution_manifest(
            contribution,
            matrix_sha256=_file_sha256(matrix_path),
            vectors_sha256=_file_sha256(vectors_path),
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, True


def read_game_contribution(path: Path | str) -> GameContribution:
    """Read and fully validate one published game contribution."""

    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    matrix_path = root / "xtwx.npz"
    vectors_path = root / "vectors.npz"
    if _file_sha256(matrix_path) != manifest["files"]["xtwx.npz"]:
        raise ValueError("Game-contribution matrix checksum changed")
    if _file_sha256(vectors_path) != manifest["files"]["vectors.npz"]:
        raise ValueError("Game-contribution vector checksum changed")
    with np.load(vectors_path) as vectors:
        contribution = GameContribution(
            schema_version=int(manifest["schema_version"]),
            season=str(manifest["season"]),
            game_id=str(manifest["game_id"]),
            game_time_utc=datetime.fromisoformat(manifest["game_time_utc"]),
            finalized_at=datetime.fromisoformat(manifest["finalized_at"]),
            source_sha256=str(manifest["source_sha256"]),
            source_uri=str(manifest["source_uri"]),
            core_model_version=str(manifest["core_model_version"]),
            context_model_version=str(manifest["context_model_version"]),
            player_ids=tuple(int(value) for value in manifest["player_ids"]),
            xtwx=sparse.load_npz(matrix_path),
            xtwy=vectors["xtwy"],
            on_court_possessions=vectors["on_court_possessions"],
            game_possessions=float(manifest["game_possessions"]),
            stint_count=int(manifest["stint_count"]),
            context_offset_sha256=str(manifest["context_offset_sha256"]),
        )
    if contribution.contribution_sha256 != manifest["contribution_sha256"]:
        raise ValueError("Game-contribution semantic checksum changed")
    return contribution


def validate_player_state(state: pd.DataFrame) -> None:
    """Validate the published player-state contract and core arithmetic."""

    missing = _REQUIRED_STATE_COLUMNS - set(state.columns)
    if missing:
        raise ValueError(f"Player state missing required columns: {sorted(missing)}")
    if state.empty:
        raise ValueError("Player state cannot be empty")
    if state["player_id"].duplicated().any():
        raise ValueError("Player state must contain one row per player")
    for column in (
        "frozen_preseason_nail",
        "current_season_residual",
        "updated_nail",
        "raw_on_court_possessions",
        "effective_information",
        "prior_equivalent_possessions",
        "prior_weight",
        "evidence_weight",
        "posterior_standard_error",
    ):
        values = state[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Player-state column is not finite: {column}")
    if not np.allclose(
        state["updated_nail"],
        state["frozen_preseason_nail"] + state["current_season_residual"],
        atol=1e-12,
    ):
        raise ValueError("Updated NAIL must equal preseason NAIL plus the residual")
    if not np.allclose(
        state["prior_weight"] + state["evidence_weight"],
        1.0,
        atol=1e-12,
    ):
        raise ValueError("Prior and evidence weights must sum to one")
    if (state["raw_on_court_possessions"] < 0).any():
        raise ValueError("Raw on-court possessions cannot be negative")
    if (state["prior_equivalent_possessions"] <= 0).any():
        raise ValueError("Prior equivalent possessions must be positive")


def publish_player_state(
    state: pd.DataFrame,
    root: Path | str,
) -> tuple[Path, bool]:
    """Atomically publish an immutable, content-addressed player-state artifact."""

    validate_player_state(state)
    seasons = tuple(state["season"].astype(str).unique())
    core_versions = tuple(state["core_model_version"].astype(str).unique())
    updater_versions = tuple(state["state_updater_version"].astype(str).unique())
    cutoff_times = pd.to_datetime(state["cutoff_time"], utc=True).unique()
    if any(len(values) != 1 for values in (seasons, core_versions, updater_versions)):
        raise ValueError("A player-state artifact must contain one season and model version")
    if len(cutoff_times) != 1:
        raise ValueError("A player-state artifact must contain one cutoff time")
    season = validate_season(seasons[0])
    cutoff = pd.Timestamp(cutoff_times[0]).to_pydatetime().astimezone(UTC)

    base = Path(root)
    staging = base / f".player-state.tmp-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        state_path = staging / "player_state.parquet"
        ordered = state.sort_values("player_id", kind="stable").reset_index(drop=True)
        ordered.to_parquet(state_path, index=False)
        state_sha256 = _file_sha256(state_path)
        target = (
            base
            / f"season={season}"
            / f"cutoff={cutoff.strftime('%Y%m%dT%H%M%SZ')}"
            / f"state_sha256={state_sha256}"
        )
        manifest = {
            "schema_version": STATE_SCHEMA_VERSION,
            "season": season,
            "core_model_version": core_versions[0],
            "state_updater_version": updater_versions[0],
            "cutoff_time": cutoff.isoformat(),
            "player_count": len(ordered),
            "player_state_sha256": state_sha256,
            "included_game_ids": json.loads(ordered["included_game_ids_json"].iloc[0]),
            "contribution_lineage": json.loads(ordered["contribution_lineage_json"].iloc[0]),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _file_sha256(target / "player_state.parquet") != state_sha256:
                raise ValueError("Existing immutable player state does not match")
            shutil.rmtree(staging)
            return target, False
        staging.replace(target)
        return target, True
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _selected_contributions(
    contributions: Sequence[GameContribution],
    *,
    season: str,
    cutoff_time: datetime,
    player_ids: tuple[int, ...],
    core_model_version: str,
) -> tuple[GameContribution, ...]:
    selected: dict[str, GameContribution] = {}
    for contribution in contributions:
        if contribution.season != season:
            raise ValueError("Contribution season does not match the state season")
        if contribution.player_ids != player_ids:
            raise ValueError("Contribution vocabulary does not match the state vocabulary")
        if contribution.core_model_version != core_model_version:
            raise ValueError("Contribution core model does not match the frozen prior")
        if contribution.finalized_at >= cutoff_time:
            continue
        prior = selected.get(contribution.game_id)
        if prior is not None and prior.contribution_sha256 != contribution.contribution_sha256:
            raise ValueError(
                "Multiple revisions supplied for one game; select the current ledger revision"
            )
        selected[contribution.game_id] = contribution
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (item.finalized_at, item.game_time_utc, item.game_id),
        )
    )


def _validate_lineups(stints: pd.DataFrame, player_to_column: Mapping[int, int]) -> None:
    for home, away in zip(
        stints["home_player_ids"],
        stints["away_player_ids"],
        strict=True,
    ):
        home_ids = tuple(int(player_id) for player_id in home)
        away_ids = tuple(int(player_id) for player_id in away)
        if not home_ids or not away_ids:
            raise ValueError("Each stint must have players on both sides")
        if len(home_ids) != len(set(home_ids)) or len(away_ids) != len(set(away_ids)):
            raise ValueError("A lineup cannot contain duplicate player IDs")
        if set(home_ids) & set(away_ids):
            raise ValueError("A player cannot appear on both sides of a stint")
        unknown = (set(home_ids) | set(away_ids)) - set(player_to_column)
        if unknown:
            raise ValueError(f"Stint contains players outside the frozen vocabulary: {unknown}")


def _prior_vector(
    values: Mapping[int, float] | Sequence[float] | np.ndarray,
    player_ids: tuple[int, ...],
) -> np.ndarray:
    if isinstance(values, Mapping):
        missing = set(player_ids) - {int(player_id) for player_id in values}
        if missing:
            raise ValueError(f"Preseason prior is missing player IDs: {sorted(missing)}")
        vector = np.asarray([values[player_id] for player_id in player_ids], dtype=float)
    else:
        vector = np.asarray(values, dtype=float)
    return _finite_vector(vector, len(player_ids), "preseason NAIL")


def _broadcast_offset(values: float | Sequence[float] | np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(length, float(array), dtype=float)
    if array.ndim != 1 or len(array) != length or not np.isfinite(array).all():
        raise ValueError("Fixed context offset must be finite and match the stint rows")
    return array


def _finite_vector(values: np.ndarray, length: int, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1).copy()
    if len(vector) != length or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite vector matching the player vocabulary")
    vector.setflags(write=False)
    return vector


def _as_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _contribution_sha256(contribution: GameContribution) -> str:
    matrix = contribution.xtwx
    payload = {
        "schema_version": contribution.schema_version,
        "season": contribution.season,
        "game_id": contribution.game_id,
        "game_time_utc": contribution.game_time_utc.isoformat(),
        "finalized_at": contribution.finalized_at.isoformat(),
        "source_sha256": contribution.source_sha256,
        "source_uri": contribution.source_uri,
        "core_model_version": contribution.core_model_version,
        "context_model_version": contribution.context_model_version,
        "player_ids": contribution.player_ids,
        "game_possessions": contribution.game_possessions,
        "stint_count": contribution.stint_count,
        "context_offset_sha256": contribution.context_offset_sha256,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    for values in (
        matrix.indptr.astype("<i8", copy=False),
        matrix.indices.astype("<i8", copy=False),
        matrix.data.astype("<f8", copy=False),
        contribution.xtwy.astype("<f8", copy=False),
        contribution.on_court_possessions.astype("<f8", copy=False),
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _contribution_manifest(
    contribution: GameContribution,
    *,
    matrix_sha256: str,
    vectors_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": contribution.schema_version,
        "season": contribution.season,
        "game_id": contribution.game_id,
        "game_time_utc": contribution.game_time_utc.isoformat(),
        "finalized_at": contribution.finalized_at.isoformat(),
        "source_sha256": contribution.source_sha256,
        "source_uri": contribution.source_uri,
        "core_model_version": contribution.core_model_version,
        "context_model_version": contribution.context_model_version,
        "player_ids": contribution.player_ids,
        "game_possessions": contribution.game_possessions,
        "stint_count": contribution.stint_count,
        "context_offset_sha256": contribution.context_offset_sha256,
        "contribution_sha256": contribution.contribution_sha256,
        "files": {
            "xtwx.npz": matrix_sha256,
            "vectors.npz": vectors_sha256,
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
