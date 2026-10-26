"""Leakage-safe, prior-centered in-season minutes state updates.

The promoted preseason availability and conditional-minutes forecasts are the
complete prior. Completed games contribute compact player observations. Injury
and inactive-list absences are excluded from the available-player role update,
healthy coach DNPs contribute zero minutes, and unknown DNP rows update neither
state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

REGULATION_TEAM_MINUTES = 240.0
STATE_SCHEMA_VERSION = 2
CONTRIBUTION_SCHEMA_VERSION = 2

PLAYED = "played"
MEDICAL_ABSENCE = "medical_absence"
COACH_DNP = "coach_dnp"
INACTIVE = "inactive_or_assignment"
UNKNOWN = "unknown"

PARTICIPATION_OUTCOMES = frozenset({PLAYED, MEDICAL_ABSENCE, COACH_DNP, INACTIVE})
CONDITIONAL_OUTCOMES = frozenset({PLAYED, COACH_DNP})
HISTORICAL_AVAILABILITY_MAP = {
    "played": PLAYED,
    "available_dnp_coach": COACH_DNP,
    "available_dnp_active": COACH_DNP,
    "available_g_league_assignment": INACTIVE,
    "unavailable_injury_or_illness": MEDICAL_ABSENCE,
    "unavailable_rest": MEDICAL_ABSENCE,
    "unavailable_personal_or_not_with_team": MEDICAL_ABSENCE,
    "unavailable_suspension": INACTIVE,
    "unavailable_ineligible": INACTIVE,
    "unavailable_inactive_unspecified": INACTIVE,
    "unknown_dnp_reason": UNKNOWN,
}
STATUS_PROBABILITY = {
    "Out": 0.0,
    "Doubtful": 0.25,
    "Questionable": 0.5,
    "Probable": 0.75,
}


@dataclass(frozen=True)
class InSeasonMinutesConfig:
    """Calibrated shrinkage and allocation contract for one updater version."""

    updater_version: str
    participation_prior_games: float
    conditional_prior_games: float
    rotation_size: int = 15
    conditional_clip_delta: float = 12.0
    conditional_lookback_games: int | None = None

    def validate(self) -> None:
        if not self.updater_version.strip():
            raise ValueError("Updater version must be non-empty")
        if not np.isfinite(self.participation_prior_games) or self.participation_prior_games <= 0:
            raise ValueError("Participation prior games must be finite and positive")
        if not np.isfinite(self.conditional_prior_games) or self.conditional_prior_games <= 0:
            raise ValueError("Conditional prior games must be finite and positive")
        if self.rotation_size <= 0:
            raise ValueError("Rotation size must be positive")
        if not np.isfinite(self.conditional_clip_delta) or self.conditional_clip_delta <= 0:
            raise ValueError("Conditional clipping delta must be finite and positive")
        if self.conditional_lookback_games is not None and self.conditional_lookback_games <= 0:
            raise ValueError("Conditional lookback games must be positive or None")


def initialize_minutes_state(
    priors: pd.DataFrame,
    *,
    season: str,
    config: InSeasonMinutesConfig,
) -> pd.DataFrame:
    """Initialize state from the exact promoted preseason player payload."""

    config.validate()
    required = {
        "player_id",
        "player_name",
        "team",
        "availability_probability",
        "conditional_minutes_per_game",
        "baseline_minutes_per_game",
    }
    missing = sorted(required - set(priors))
    if missing:
        raise ValueError("Preseason minutes prior lacks: " + ", ".join(missing))
    if priors["player_id"].duplicated().any():
        raise ValueError("Preseason minutes prior has duplicate player IDs")
    state = priors.loc[:, sorted(required)].copy()
    state = state.rename(
        columns={
            "availability_probability": "preseason_participation_prior",
            "conditional_minutes_per_game": "preseason_conditional_minutes_prior",
            "baseline_minutes_per_game": "preseason_normalized_minutes",
        }
    )
    numeric = (
        "preseason_participation_prior",
        "preseason_conditional_minutes_prior",
        "preseason_normalized_minutes",
    )
    for column in numeric:
        state[column] = pd.to_numeric(state[column], errors="raise").astype(float)
    if not state["preseason_participation_prior"].between(0.0, 1.0).all():
        raise ValueError("Participation priors must lie in [0, 1]")
    if state["preseason_conditional_minutes_prior"].lt(0.0).any():
        raise ValueError("Conditional-minutes priors cannot be negative")
    state["schema_version"] = STATE_SCHEMA_VERSION
    state["season"] = season
    state["updater_version"] = config.updater_version
    state["current_participation_estimate"] = state["preseason_participation_prior"]
    state["current_conditional_minutes_estimate"] = state[
        "preseason_conditional_minutes_prior"
    ]
    state["participation_observations"] = 0
    state["participation_played"] = 0.0
    state["conditional_observations"] = 0
    state["conditional_minutes_weighted_sum"] = 0.0
    state["effective_conditional_weight"] = 0.0
    state["conditional_window_observations"] = 0
    state["conditional_history_json"] = "[]"
    state["included_game_ids_json"] = "[]"
    state["contribution_lineage_json"] = "[]"
    return state.sort_values("player_id", kind="stable").reset_index(drop=True)


def regulation_equivalent_minutes(
    minutes: pd.Series,
    *,
    team_total_minutes: float,
) -> pd.Series:
    """Scale any regulation or overtime team distribution to exactly 240."""

    total = float(team_total_minutes)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Team total minutes must be finite and positive")
    values = pd.to_numeric(minutes, errors="raise").astype(float)
    if values.lt(0).any():
        raise ValueError("Actual minutes cannot be negative")
    return values * (REGULATION_TEAM_MINUTES / total)


def advance_minutes_state(
    state: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    config: InSeasonMinutesConfig,
    contribution_sha256: str,
) -> pd.DataFrame:
    """Apply one or more completed-game observations without double counting."""

    config.validate()
    required = {
        "game_id",
        "player_id",
        "outcome_classification",
        "played",
        "regulation_equivalent_minutes",
        "conditional_observation_weight",
    }
    missing = sorted(required - set(observations))
    if missing:
        raise ValueError("Minutes observations lack: " + ", ".join(missing))
    output = state.copy().set_index("player_id", drop=False)
    included = set()
    lineage = set()
    if not output.empty:
        included = set(json.loads(str(output.iloc[0]["included_game_ids_json"])))
        lineage = set(json.loads(str(output.iloc[0]["contribution_lineage_json"])))
    game_ids = set(observations["game_id"].astype(str))
    if included.intersection(game_ids):
        raise ValueError("A completed game cannot be applied twice to one state")
    for row in observations.itertuples(index=False):
        player_id = int(row.player_id)
        if player_id not in output.index:
            continue
        classification = str(row.outcome_classification)
        if classification in PARTICIPATION_OUTCOMES:
            output.at[player_id, "participation_observations"] += 1
            output.at[player_id, "participation_played"] += float(bool(row.played))
        if classification in CONDITIONAL_OUTCOMES:
            weight = float(row.conditional_observation_weight)
            if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
                raise ValueError("Conditional observation weight must lie in [0, 1]")
            observed = float(
                getattr(
                    row,
                    "early_game_role_minutes",
                    getattr(
                        row,
                        "competitive_role_minutes",
                        row.regulation_equivalent_minutes,
                    ),
                )
            )
            prior = float(output.at[player_id, "preseason_conditional_minutes_prior"])
            clipped = (
                0.0
                if classification == COACH_DNP
                else float(
                    np.clip(
                        observed,
                        max(0.0, prior - config.conditional_clip_delta),
                        min(48.0, prior + config.conditional_clip_delta),
                    )
                )
            )
            history = json.loads(str(output.at[player_id, "conditional_history_json"]))
            history.append({"minutes": clipped, "weight": weight})
            if config.conditional_lookback_games is not None:
                history = history[-config.conditional_lookback_games :]
            output.at[player_id, "conditional_history_json"] = json.dumps(
                history, separators=(",", ":")
            )
            output.at[player_id, "conditional_observations"] += 1
            output.at[player_id, "conditional_window_observations"] = len(history)
            output.at[player_id, "conditional_minutes_weighted_sum"] = sum(
                float(item["weight"]) * float(item["minutes"]) for item in history
            )
            output.at[player_id, "effective_conditional_weight"] = sum(
                float(item["weight"]) for item in history
            )
    participation_denominator = (
        config.participation_prior_games + output["participation_observations"].astype(float)
    )
    output["current_participation_estimate"] = (
        config.participation_prior_games * output["preseason_participation_prior"]
        + output["participation_played"]
    ) / participation_denominator
    conditional_denominator = (
        config.conditional_prior_games + output["effective_conditional_weight"]
    )
    output["current_conditional_minutes_estimate"] = (
        config.conditional_prior_games * output["preseason_conditional_minutes_prior"]
        + output["conditional_minutes_weighted_sum"]
    ) / conditional_denominator
    included.update(game_ids)
    lineage.add(contribution_sha256)
    output["included_game_ids_json"] = json.dumps(sorted(included), separators=(",", ":"))
    output["contribution_lineage_json"] = json.dumps(
        sorted(lineage), separators=(",", ":")
    )
    return output.reset_index(drop=True).sort_values("player_id", kind="stable")


def forecast_team_minutes(
    state: pd.DataFrame,
    *,
    team: str,
    game_id: str,
    forecast_cutoff: str,
    injury_rows: list[dict[str, object]] | None,
    report_status: str,
    config: InSeasonMinutesConfig,
) -> pd.DataFrame:
    """Build one coherent shadow distribution from pre-tip state and status."""

    config.validate()
    players = state.loc[state["team"].astype(str).eq(team)].copy()
    if players.empty:
        raise ValueError(f"Minutes state has no roster for {team}")
    injury_by_id = {
        int(row["player_id"]): row
        for row in (injury_rows or [])
        if row.get("player_id") is not None
    }
    game_baseline_by_id = {
        int(row["player_id"]): float(row["projected_minutes"])
        for row in (injury_rows or [])
        if row.get("player_id") is not None and row.get("projected_minutes") is not None
    }
    has_game_baseline = bool(game_baseline_by_id)
    prior_p = players["preseason_participation_prior"].clip(lower=1e-6)
    current_conditional = players["current_conditional_minutes_estimate"].clip(
        lower=0.0, upper=48.0
    )
    prior_conditional = players["preseason_conditional_minutes_prior"].clip(lower=1e-6)
    forecast_probability: list[float] = []
    explicit_status: list[bool] = []
    statuses: list[str] = []
    reasons: list[object] = []
    for row in players.itertuples(index=False):
        injury = injury_by_id.get(int(row.player_id))
        if injury is not None and str(injury.get("status")) in STATUS_PROBABILITY:
            status = str(injury["status"])
            probability = STATUS_PROBABILITY[status]
            reason = injury.get("reason")
            is_explicit = True
        elif injury_rows and report_status == "submitted":
            # The submitted official report is authoritative at the game
            # cutoff. An unlisted player has no reported medical restriction;
            # the learned participation state remains an audited forecast but
            # does not override this game-specific evidence.
            status = "Unlisted"
            probability = 1.0
            reason = None
            is_explicit = False
        else:
            status = "Report pending" if report_status != "submitted" else "Unlisted"
            probability = float("nan") if report_status != "submitted" else 1.0
            reason = None
            is_explicit = False
        forecast_probability.append(probability)
        explicit_status.append(is_explicit)
        statuses.append(status)
        reasons.append(reason)
    players["game_participation_probability"] = forecast_probability
    players["has_explicit_injury_status"] = explicit_status
    players["injury_status"] = statuses
    players["injury_reason"] = reasons
    players["game_baseline_minutes"] = (
        players["player_id"].map(game_baseline_by_id).fillna(0.0)
        if has_game_baseline
        else players["preseason_normalized_minutes"]
    )
    players["pre_normalization_minutes"] = players["game_baseline_minutes"] * (
        current_conditional / prior_conditional
    )
    if not has_game_baseline:
        status_multiplier = pd.Series(1.0, index=players.index, dtype=float)
        status_multiplier.loc[players["has_explicit_injury_status"]] = (
            players.loc[
                players["has_explicit_injury_status"], "game_participation_probability"
            ]
            / prior_p.loc[players["has_explicit_injury_status"]]
        )
        players["pre_normalization_minutes"] *= status_multiplier
    # A player with a preseason zero can enter only after actual participation
    # evidence. This preserves exact zero-evidence identity while supporting
    # late signings and rotation changes.
    team_scale_rows = players.loc[
        players["game_baseline_minutes"].gt(0.0)
        & players["preseason_participation_prior"].gt(0.0)
        & players["preseason_conditional_minutes_prior"].gt(0.0)
    ]
    if team_scale_rows.empty:
        team_scale = 1.0
    else:
        team_scale = float(
            np.median(
                team_scale_rows["game_baseline_minutes"]
                / (
                    team_scale_rows["preseason_participation_prior"]
                    * team_scale_rows["preseason_conditional_minutes_prior"]
                )
            )
        )
    cold_entry = (
        players["game_baseline_minutes"].le(0.0)
        & players["participation_played"].gt(0.0)
        & players["game_participation_probability"].fillna(0.0).gt(0.0)
    )
    players.loc[cold_entry, "pre_normalization_minutes"] = (
        team_scale
        * players.loc[cold_entry, "game_participation_probability"]
        * players.loc[cold_entry, "current_conditional_minutes_estimate"]
    )
    players["selected_rotation"] = False
    selected = players.sort_values(
        ["pre_normalization_minutes", "player_name", "player_id"],
        ascending=[False, True, True],
        kind="stable",
    ).head(config.rotation_size).index
    players.loc[selected, "selected_rotation"] = True
    selected_total = float(players.loc[selected, "pre_normalization_minutes"].sum())
    if selected_total <= 0:
        raise ValueError(f"No positive shadow minutes for {team}/{game_id}")
    players["final_normalized_minutes"] = 0.0
    players.loc[selected, "final_normalized_minutes"] = (
        REGULATION_TEAM_MINUTES
        * players.loc[selected, "pre_normalization_minutes"]
        / selected_total
    )
    players["team_total_minutes"] = float(players["final_normalized_minutes"].sum())
    players["game_id"] = game_id
    players["forecast_cutoff"] = forecast_cutoff
    players["injury_report_submission_state"] = report_status
    return players.reset_index(drop=True)


def calibrate_prior_games(
    prior_season: pd.DataFrame,
    target_season: pd.DataFrame,
    *,
    participation_grid: tuple[float, ...] = (4.0, 8.0, 12.0, 20.0, 32.0, 48.0),
    conditional_grid: tuple[float, ...] = (2.0, 4.0, 8.0, 12.0, 20.0, 32.0),
    conditional_clip_delta: float = 12.0,
) -> dict[str, object]:
    """Select shrinkage strengths by chronological next-game replay."""

    required = {"game_id", "game_date", "team", "player_id", "minutes", "availability_state"}
    for label, frame in (("prior", prior_season), ("target", target_season)):
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"{label} calibration frame lacks: {', '.join(missing)}")
    prior = _historical_player_priors(_normalize_historical_availability(prior_season))
    target = _regulation_equivalent_history(
        _normalize_historical_availability(target_season)
    )
    target = target.merge(prior, on="player_id", how="inner", validate="many_to_one")
    target = target.sort_values(["game_date", "game_id", "player_id"], kind="stable")
    participation_scores = {
        strength: _sequential_participation_brier(target, strength)
        for strength in participation_grid
    }
    conditional_scores = {
        strength: _sequential_conditional_mae(
            target, strength, clip_delta=conditional_clip_delta
        )
        for strength in conditional_grid
    }
    selected_participation = min(participation_scores, key=participation_scores.get)
    selected_conditional = min(conditional_scores, key=conditional_scores.get)
    return {
        "participation_prior_games": float(selected_participation),
        "conditional_prior_games": float(selected_conditional),
        "participation_grid_brier": {
            str(key): value for key, value in participation_scores.items()
        },
        "conditional_grid_mae": {str(key): value for key, value in conditional_scores.items()},
        "target_player_games": int(len(target)),
    }


def calibrate_conditional_prior_games(
    preseason_priors: pd.DataFrame,
    target_season: pd.DataFrame,
    *,
    conditional_grid: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0),
    conditional_clip_delta: float = 12.0,
) -> dict[str, object]:
    """Calibrate role shrinkage against an exact preseason-prior analogue.

    The target includes appearances and healthy coach/active DNPs. Medical,
    inactive-list, assignment, and unknown outcomes are excluded from the role
    loss and update.
    """

    required_target = {
        "game_id",
        "game_date",
        "team",
        "player_id",
        "minutes",
        "availability_state",
    }
    missing = sorted(required_target - set(target_season))
    if missing:
        raise ValueError("target calibration frame lacks: " + ", ".join(missing))
    prior = preseason_priors.copy()
    if "preseason_conditional_minutes_prior" not in prior:
        source = "predicted_minutes_per_available_game"
        if source not in prior:
            raise ValueError(
                "preseason priors require preseason_conditional_minutes_prior "
                "or predicted_minutes_per_available_game"
            )
        prior = prior.rename(columns={source: "preseason_conditional_minutes_prior"})
    prior = prior[["player_id", "preseason_conditional_minutes_prior"]].copy()
    if prior["player_id"].duplicated().any():
        raise ValueError("preseason priors contain duplicate player IDs")
    prior["preseason_conditional_minutes_prior"] = pd.to_numeric(
        prior["preseason_conditional_minutes_prior"], errors="raise"
    ).astype(float)
    target = _regulation_equivalent_history(
        _normalize_historical_availability(target_season)
    )
    target = target.merge(prior, on="player_id", how="inner", validate="many_to_one")
    target = target.sort_values(["game_date", "game_id", "player_id"], kind="stable")
    scores = {
        strength: _sequential_conditional_mae(
            target, strength, clip_delta=conditional_clip_delta
        )
        for strength in conditional_grid
    }
    selected = min(scores, key=scores.get)
    available = target["availability_state"].isin(CONDITIONAL_OUTCOMES)
    return {
        "conditional_prior_games": float(selected),
        "conditional_grid_mae": {str(key): value for key, value in scores.items()},
        "target_player_games": int(len(target)),
        "target_available_player_games": int(available.sum()),
    }


def _normalize_historical_availability(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    states = output["availability_state"].astype("string")
    normalized = states.map(HISTORICAL_AVAILABILITY_MAP)
    output["availability_state"] = normalized.fillna(states).fillna(UNKNOWN)
    return output


def contribution_sha256(observations: pd.DataFrame, metadata: dict[str, object]) -> str:
    canonical_rows = observations.sort_values("player_id", kind="stable").to_dict("records")
    payload = json.dumps(
        {
            "schema_version": CONTRIBUTION_SCHEMA_VERSION,
            "metadata": metadata,
            "rows": canonical_rows,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def publish_minutes_contribution(
    observations: pd.DataFrame,
    metadata: dict[str, object],
    root: Path,
) -> tuple[Path, bool, str]:
    """Publish one immutable content-addressed player-game contribution."""

    digest = contribution_sha256(observations, metadata)
    game_id = str(metadata["game_id"])
    target = root / f"game_id={game_id}" / f"contribution_sha256={digest}"
    if target.is_dir():
        return target, False, digest
    staging = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    staging.mkdir(parents=True)
    observations.sort_values("player_id", kind="stable").to_parquet(
        staging / "observations.parquet", index=False
    )
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CONTRIBUTION_SCHEMA_VERSION,
                "contribution_sha256": digest,
                **metadata,
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.replace(target)
    except OSError:
        if not target.is_dir():
            raise
    return target, True, digest


def _historical_player_priors(frame: pd.DataFrame) -> pd.DataFrame:
    history = _regulation_equivalent_history(frame)
    history["played"] = history["minutes"].gt(0.0)
    known = history["availability_state"].ne(UNKNOWN)
    participation = (
        history.loc[known]
        .groupby("player_id", as_index=False)
        .agg(preseason_participation_prior=("played", "mean"))
    )
    available = history["availability_state"].isin(CONDITIONAL_OUTCOMES)
    conditional = (
        history.loc[available]
        .groupby("player_id", as_index=False)
        .agg(preseason_conditional_minutes_prior=("regulation_equivalent_minutes", "mean"))
    )
    return participation.merge(conditional, on="player_id", how="inner")


def _regulation_equivalent_history(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["minutes"] = pd.to_numeric(output["minutes"], errors="raise").astype(float)
    totals = output.groupby(["game_id", "team"], sort=False)["minutes"].transform("sum")
    output["regulation_equivalent_minutes"] = output["minutes"] * (
        REGULATION_TEAM_MINUTES / totals
    )
    output["played"] = output["minutes"].gt(0.0)
    return output


def _sequential_participation_brier(frame: pd.DataFrame, strength: float) -> float:
    played: dict[int, float] = {}
    count: dict[int, int] = {}
    errors: list[float] = []
    for _date, day in frame.groupby("game_date", sort=True):
        for row in day.itertuples(index=False):
            if str(row.availability_state) == UNKNOWN:
                continue
            player_id = int(row.player_id)
            prediction = (
                strength * float(row.preseason_participation_prior) + played.get(player_id, 0.0)
            ) / (strength + count.get(player_id, 0))
            actual = float(bool(row.played))
            errors.append((prediction - actual) ** 2)
        for row in day.itertuples(index=False):
            if str(row.availability_state) == UNKNOWN:
                continue
            player_id = int(row.player_id)
            played[player_id] = played.get(player_id, 0.0) + float(bool(row.played))
            count[player_id] = count.get(player_id, 0) + 1
    return float(np.mean(errors))


def _sequential_conditional_mae(
    frame: pd.DataFrame, strength: float, *, clip_delta: float
) -> float:
    target_column = (
        "competitive_role_minutes"
        if "competitive_role_minutes" in frame
        else "regulation_equivalent_minutes"
    )
    total: dict[int, float] = {}
    count: dict[int, int] = {}
    errors: list[float] = []
    for _date, day in frame.groupby("game_date", sort=True):
        available = day["availability_state"].isin(CONDITIONAL_OUTCOMES)
        for row in day.loc[available].itertuples(index=False):
            player_id = int(row.player_id)
            prediction = (
                strength * float(row.preseason_conditional_minutes_prior)
                + total.get(player_id, 0.0)
            ) / (strength + count.get(player_id, 0))
            target = float(getattr(row, target_column))
            errors.append(abs(prediction - target))
        for row in day.loc[available].itertuples(index=False):
            player_id = int(row.player_id)
            prior = float(row.preseason_conditional_minutes_prior)
            observed = (
                0.0
                if str(row.availability_state) == COACH_DNP
                else float(
                    np.clip(
                        getattr(row, target_column),
                        max(0.0, prior - clip_delta),
                        min(48.0, prior + clip_delta),
                    )
                )
            )
            total[player_id] = total.get(player_id, 0.0) + observed
            count[player_id] = count.get(player_id, 0) + 1
    return float(np.mean(errors))


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")
