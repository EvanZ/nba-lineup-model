"""Compact in-memory inference for the NBA GESTALT Lineup Lab."""

from __future__ import annotations

import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    contextual_feature_columns,
    lineup_context_features,
)
from nba_lineup_model.modeling.forward_contextual_rapm import MODEL_NAME
from nba_lineup_model.modeling.lineup_context_case_study import (
    _feature_contributions,
    _feature_label,
)

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
MODEL_ARTIFACT = "forward_contextual_rapm"
DISPLAY_SEASON = "2025-26"


class LineupEvaluationError(ValueError):
    """Raised when an input cannot be evaluated against the loaded model state."""


@dataclass(frozen=True)
class LineupEvaluator:
    """Serve one completed contextual RAPM state without reading raw possession data."""

    season: str
    run_id: str
    coefficients: pd.DataFrame
    profiles: pd.DataFrame
    players: pd.DataFrame
    context_model: object

    @classmethod
    def from_latest_artifact(
        cls,
        *,
        artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
        season: str = DISPLAY_SEASON,
        panel_path: Path | str = "data/analytical/player_season_panel/player_seasons.parquet",
    ) -> LineupEvaluator:
        """Load the completed seasonal model state used by the local Lineup Lab."""

        root = Path(artifacts_dir) / MODEL_ARTIFACT / season
        latest_path = root / "latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError(f"No published contextual artifact for {season}: {latest_path}")
        run_id = str(json.loads(latest_path.read_text())["run_id"])
        run_dir = root / run_id
        metadata = json.loads((run_dir / "metadata.json").read_text())
        if metadata.get("model") != MODEL_NAME:
            raise LineupEvaluationError("The selected artifact is not forward contextual RAPM")
        if str(metadata.get("target_season")) != season:
            raise LineupEvaluationError("The selected artifact has an unexpected target season")

        coefficients = pd.read_parquet(run_dir / "historical_player_coefficients.parquet")
        coefficients = coefficients.loc[
            coefficients["season"].eq(season), ["player_id", "rapm"]
        ].copy()
        if coefficients.empty or coefficients["player_id"].duplicated().any():
            raise LineupEvaluationError("The artifact has invalid completed player coefficients")
        profiles = pd.read_parquet(run_dir / "target_player_profiles.parquet")
        if profiles["player_id"].duplicated().any():
            raise LineupEvaluationError("The artifact has duplicate player profiles")
        players = _player_catalog(
            coefficients, profiles, panel_path=Path(panel_path), season=season
        )
        models = joblib.load(run_dir / "season_context_models.joblib")
        context_model = models.get(season)
        if context_model is None:
            raise LineupEvaluationError("The artifact has no completed contextual function")
        return cls(
            season=season,
            run_id=run_id,
            coefficients=coefficients,
            profiles=profiles,
            players=players,
            context_model=context_model,
        )

    def search_players(self, query: str, *, limit: int = 12) -> list[dict[str, Any]]:
        """Return a short, deterministic player-name search result list."""

        normalized = _normalize_search_text(query)
        if not normalized:
            return []
        names = self.players["player_name"].map(_normalize_search_text)
        matches = self.players.loc[names.str.contains(normalized, regex=False)]
        return _records(matches.head(max(1, min(limit, 25))))

    def player(self, player_id: int) -> dict[str, Any]:
        """Return the display profile for one available player."""

        row = self.players.loc[self.players["player_id"].eq(player_id)]
        if row.empty:
            raise LineupEvaluationError(f"Player {player_id} is not available in {self.season}")
        return _records(row)[0]

    def default_opponent(
        self, *, excluded_player_ids: set[int] | None = None
    ) -> list[dict[str, Any]]:
        """Sample a plausible baseline opponent from high-possession players."""

        excluded = excluded_player_ids or set()
        eligible = self.players.loc[
            self.players["possessions"].ge(self.players["possessions"].quantile(0.75))
            & ~self.players["player_id"].isin(excluded)
        ]
        if len(eligible) < 5:
            raise LineupEvaluationError(
                "The loaded player catalog has too few high-possession players"
            )
        sample = eligible.sample(
            n=5,
            replace=False,
            random_state=random.SystemRandom().randrange(2**32),
        )
        return _records(sample.sort_values("player_name", kind="stable"))

    def evaluate(
        self, unit_player_ids: list[int], opponent_player_ids: list[int]
    ) -> dict[str, Any]:
        """Score a neutral-court five-player unit against a five-player opponent."""

        _validate_lineup("unit", unit_player_ids)
        _validate_lineup("opponent", opponent_player_ids)
        overlap = set(unit_player_ids) & set(opponent_player_ids)
        if overlap:
            raise LineupEvaluationError("A player cannot appear on both sides of a matchup")
        available = set(self.players["player_id"].astype(int))
        missing = sorted((set(unit_player_ids) | set(opponent_player_ids)) - available)
        if missing:
            raise LineupEvaluationError(f"Players are unavailable in {self.season}: {missing}")

        coefficient_map = dict(
            zip(
                self.coefficients["player_id"].astype(int),
                self.coefficients["rapm"].astype(float),
                strict=True,
            )
        )
        features = lineup_context_features([unit_player_ids], [opponent_player_ids], self.profiles)
        contextual_adjustment = float(self.context_model.predict(features)[0])
        contributions = _feature_contributions(self.context_model, features)[0]
        contextual_intercept = _pipeline_intercept(self.context_model)
        additive_unit = float(sum(coefficient_map[player_id] for player_id in unit_player_ids))
        additive_opponent = float(
            sum(coefficient_map[player_id] for player_id in opponent_player_ids)
        )
        additive_margin = additive_unit - additive_opponent
        feature_rows = [
            {
                "id": column,
                "label": _feature_label(column),
                "value": float(features.iloc[0][column]),
                "contribution": float(contribution),
            }
            for column, contribution in zip(
                contextual_feature_columns(), contributions, strict=True
            )
        ]
        feature_rows.sort(key=lambda row: abs(float(row["contribution"])), reverse=True)
        return {
            "season": self.season,
            "run_id": self.run_id,
            "retrospective": True,
            "unit": self._side(unit_player_ids, coefficient_map),
            "opponent": self._side(opponent_player_ids, coefficient_map),
            "additive_margin": additive_margin,
            "contextual_adjustment": contextual_adjustment,
            "contextual_intercept": contextual_intercept,
            "predicted_net_rating": additive_margin + contextual_adjustment,
            "feature_contributions": feature_rows,
        }

    def _side(self, player_ids: list[int], coefficient_map: dict[int, float]) -> dict[str, Any]:
        rows = self.players.set_index("player_id").loc[player_ids].reset_index()
        players = _records(rows)
        for player in players:
            player["rapm"] = coefficient_map[int(player["player_id"])]
        return {"additive_rating": sum(player["rapm"] for player in players), "players": players}


def _player_catalog(
    coefficients: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    panel_path: Path,
    season: str,
) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    panel_columns = [
        "player_id",
        "player_name",
        "primary_team_tricode",
        "listed_position",
        "rapm_possessions",
        "games",
        "points",
    ]
    available_panel_columns = [column for column in panel_columns if column in panel]
    season_panel = panel.loc[panel["season"].eq(season), available_panel_columns].copy()
    if season_panel["player_id"].duplicated().any():
        season_panel = (
            season_panel.sort_values("rapm_possessions", ascending=False)
            .drop_duplicates("player_id", keep="first")
        )
    profile_columns = [
        "player_id",
        "player_name",
        "profile_source",
        "profile_imputed",
        "profile_replacement_weight",
        "three_pm_per_100",
        "assists_per_100",
        "usage_per_100",
        "offensive_rebounds_per_100",
        "defensive_rebounds_per_100",
    ]
    catalog = coefficients.merge(profiles.loc[:, profile_columns], on="player_id", how="inner")
    catalog = catalog.merge(
        season_panel.drop(columns=["player_name"], errors="ignore"), on="player_id", how="left"
    )
    catalog["team"] = catalog.get(
        "primary_team_tricode", pd.Series(index=catalog.index)
    ).fillna("-")
    catalog["position"] = catalog.get(
        "listed_position", pd.Series(index=catalog.index)
    ).fillna("-")
    catalog["possessions"] = catalog.get(
        "rapm_possessions", pd.Series(index=catalog.index)
    ).fillna(0.0)
    catalog["games"] = catalog.get("games", pd.Series(index=catalog.index)).fillna(0).astype(int)
    catalog = catalog.sort_values(["player_name", "player_id"], kind="stable").reset_index(
        drop=True
    )
    return catalog.loc[
        :,
        [
            "player_id",
            "player_name",
            "team",
            "position",
            "rapm",
            "possessions",
            "games",
            "profile_source",
            "profile_imputed",
            "profile_replacement_weight",
            "three_pm_per_100",
            "assists_per_100",
            "usage_per_100",
            "offensive_rebounds_per_100",
            "defensive_rebounds_per_100",
        ],
    ]


def _pipeline_intercept(context_model: object) -> float:
    try:
        return float(np.asarray(context_model.named_steps["ridge"].intercept_).item())
    except (AttributeError, KeyError, ValueError) as error:
        raise LineupEvaluationError("Contextual model does not expose a Ridge intercept") from error


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None})
    return [dict(row) for row in clean.to_dict(orient="records")]


def _normalize_search_text(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _validate_lineup(name: str, player_ids: list[int]) -> None:
    if len(player_ids) != 5:
        raise LineupEvaluationError(f"The {name} must contain exactly five players")
    if len(set(player_ids)) != 5:
        raise LineupEvaluationError(f"The {name} must contain five distinct players")
