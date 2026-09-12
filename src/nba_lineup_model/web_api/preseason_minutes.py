"""Build roster-wide preseason minute-allocation inputs for Win Projections."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    build_availability_season_summary,
    predict_availability_roster,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    PROMOTED_AVAILABILITY_CONFIG,
    ColdStartConditionalMinutesConfig,
    ForwardConditionalMinutesConfig,
    attach_cold_start_biographies,
    attach_expected_total_minutes,
    build_forward_conditional_roster_profile,
    fit_age_conditional_minutes_model,
    fit_cold_start_conditional_minutes_model,
    normalize_opening_roster_minutes,
    predict_conditional_minutes_roster,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_CURATED_DIR,
    read_regular_game_minutes,
)
from nba_lineup_model.rotation.l20_source_aware_preseason_minute_share import (
    DEFAULT_HISTORY_SEASONS,
    DEFAULT_PANEL_PATH,
    FittedSourceAwareMinuteModel,
    build_current_source_aware_role_state,
    predict_source_aware_minute_shares,
)
from nba_lineup_model.web_api.preseason_rankings_cache import append_target_bios

DEFAULT_ROSTER_PATH = Path("data/curated/team_rosters/2026-27/part-00000.parquet")
DEFAULT_TARGET_SEASON = "2026-27"
DEFAULT_COMPLETED_SEASON = "2025-26"
DEFAULT_EPSILON = 0.32
DEFAULT_NAIL_WEIGHT = 0.10
DEFAULT_INITIAL_ROTATION_SIZE = 15
REGULATION_TEAM_MINUTES = 240.0
DEFAULT_SOURCE_AWARE_ARTIFACTS_DIR = Path(
    "artifacts/rotation/l20_source_aware_preseason_minute_share"
)
PROMOTED_CONDITIONAL_MINUTES_CONFIG = ForwardConditionalMinutesConfig(
    persistence=1.0,
    update_strength=15.0,
    initial_strength=30.0,
)
PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG = ColdStartConditionalMinutesConfig(
    alpha=0.01
)


def build_preseason_minutes_payload(
    *,
    preseason_rankings: pd.DataFrame,
    roster_path: Path | str = DEFAULT_ROSTER_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    target_season: str = DEFAULT_TARGET_SEASON,
    completed_season: str = DEFAULT_COMPLETED_SEASON,
    epsilon: float = DEFAULT_EPSILON,
    nail_weight: float = DEFAULT_NAIL_WEIGHT,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
    prior_minutes: pd.Series | None = None,
) -> dict[str, object]:
    """Return immutable L20-NAIL-MSP baseline allocations for every roster.

    The payload deliberately contains no manual override state. Clients may
    layer explicit minutes judgments on top while retaining this baseline as the
    reproducible comparison point.
    """

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Preseason minutes epsilon must be between zero and one")
    if nail_weight < 0.0:
        raise ValueError("Preseason NAIL weight must be non-negative")
    if initial_rotation_size <= 0:
        raise ValueError("Initial rotation size must be positive")
    roster = pd.read_parquet(roster_path)
    _validate_roster(roster)
    _validate_rankings(preseason_rankings)
    historical_minutes = (
        read_regular_game_minutes(completed_season, curated_dir=curated_dir)
        .groupby("player_id", as_index=True)["minutes"]
        .sum()
        if prior_minutes is None
        else prior_minutes.astype(float)
    )
    ratings = preseason_rankings.loc[:, ["player_id", "rapm", "profile_source"]].copy()
    ratings["player_id"] = ratings["player_id"].astype(int)
    rating_by_player = ratings.set_index("player_id")["rapm"]
    rating_scale = float(np.std(rating_by_player.to_numpy(dtype=float), ddof=0))
    if not np.isfinite(rating_scale) or rating_scale <= 1e-12:
        rating_scale = 1.0
    rows: list[dict[str, object]] = []
    for team, group in roster.groupby("team_abbreviation", sort=True):
        players = group.copy()
        players["player_id"] = players["player_id"].astype(int)
        player_ids = players["player_id"].tolist()
        scores = pd.Series(
            [float(historical_minutes.get(player_id, 0.0)) for player_id in player_ids],
            index=players.index,
            dtype=float,
        )
        players["prior_season_minutes"] = scores
        players = players.sort_values(
            ["prior_season_minutes", "player_name", "player_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        rotation_index = players.index[:initial_rotation_size]
        players["is_baseline_rotation_candidate"] = players.index.isin(rotation_index)
        rotation_scores = players.loc[rotation_index, "prior_season_minutes"]
        rotation_count = len(rotation_scores)
        if rotation_scores.sum() > 0.0:
            rotation_shares = (
                (1.0 - epsilon) * (rotation_scores / rotation_scores.sum())
                + epsilon / rotation_count
            )
        else:
            rotation_shares = pd.Series(1.0 / rotation_count, index=rotation_index, dtype=float)
        rotation_ratings = (
            players.loc[rotation_index, "player_id"].map(rating_by_player).fillna(0.0)
        )
        rating_tilt = np.exp(
            np.clip(
                nail_weight * rotation_ratings.to_numpy(dtype=float) / rating_scale,
                -30.0,
                30.0,
            )
        )
        rotation_shares = rotation_shares * rating_tilt
        rotation_shares = rotation_shares / rotation_shares.sum()
        shares = pd.Series(0.0, index=players.index, dtype=float)
        shares.loc[rotation_index] = rotation_shares
        players["baseline_minute_share"] = shares
        players["baseline_minutes_per_game"] = REGULATION_TEAM_MINUTES * shares
        players = players.merge(ratings, on="player_id", how="left", validate="one_to_one")
        players["is_rating_fallback"] = players["rapm"].isna()
        # Late free-agent and two-way additions may arrive after the frozen
        # preseason ranking cache. A neutral rating keeps the planning tool
        # available without inventing a player-specific forecast.
        players["rapm"] = players["rapm"].fillna(0.0)
        players["profile_source"] = players["profile_source"].fillna("roster fallback")
        players = players.sort_values(
            ["baseline_minutes_per_game", "player_name", "player_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        rows.extend(
            {
                "player_id": int(row.player_id),
                "player_name": str(row.player_name),
                "position": str(row.listed_position),
                "age": None if pd.isna(row.age) else float(row.age),
                "team": str(team),
                "prior_season_minutes": float(row.prior_season_minutes),
                "is_baseline_rotation_candidate": bool(row.is_baseline_rotation_candidate),
                "is_rating_fallback": bool(row.is_rating_fallback),
                "baseline_minute_share": float(row.baseline_minute_share),
                "baseline_minutes_per_game": float(row.baseline_minutes_per_game),
                "projected_nail": float(row.rapm),
                "rating_source": str(row.profile_source),
            }
            for row in players.itertuples(index=False)
        )
    teams = sorted(roster["team_abbreviation"].astype(str).unique().tolist())
    return {
        "season": target_season,
        "completed_season": completed_season,
        "model": "L20-NAIL-MSP v0.2",
        "epsilon": epsilon,
        "nail_weight": nail_weight,
        "nail_rating_scale": rating_scale,
        "initial_rotation_size": initial_rotation_size,
        "regulation_team_minutes": REGULATION_TEAM_MINUTES,
        "contract": (
            "The current organization roster is reduced to an initial 15-player "
            "rotation by prior-season minutes. Those 15 shares are normalized, "
            "shrunk 32% toward equal shares, then tilted by the exact preseason "
            "NAIL forecast with beta=0.10; all other players begin at zero and can "
            "enter through a manual override. Late roster additions without a frozen "
            "NAIL forecast receive a neutral fallback rating."
        ),
        "teams": teams,
        "players": rows,
    }


def build_forward_conditional_preseason_minutes_payload(
    *,
    preseason_rankings: pd.DataFrame,
    roster_path: Path | str = DEFAULT_ROSTER_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    target_season: str = DEFAULT_TARGET_SEASON,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> dict[str, object]:
    """Build the Issue #9 availability-times-conditional-minutes baseline.

    The target roster supplies identity and static biography only. Both
    forecast components are fitted from completed availability seasons, then
    their raw expected totals are squashed within each roster.
    """

    roster = pd.read_parquet(roster_path)
    _validate_roster(roster)
    _validate_rankings(preseason_rankings)
    required = {"team_id", "team_abbreviation"}
    missing = sorted(required - set(roster))
    if missing:
        raise ValueError(f"Current roster is missing columns: {missing}")
    if initial_rotation_size <= 0:
        raise ValueError("Initial rotation size must be positive")
    roster = roster.copy()
    roster["team_id"] = pd.to_numeric(roster["team_id"], errors="raise").astype("int64")
    roster["player_id"] = pd.to_numeric(roster["player_id"], errors="raise").astype("int64")
    target_year = int(target_season[:4])
    history_seasons = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, target_year))
    summary = build_availability_season_summary(history_seasons, curated_dir=curated_dir)
    summary = attach_cold_start_biographies(summary)
    forecast_roster = build_forward_conditional_roster_profile(
        roster, target_season=target_season, curated_dir=curated_dir
    )
    availability, _availability_age, _availability_metadata = predict_availability_roster(
        summary,
        roster=forecast_roster,
        target_season=target_season,
        config=PROMOTED_AVAILABILITY_CONFIG,
    )
    conditional, _conditional_age = predict_conditional_minutes_roster(
        summary,
        roster=forecast_roster,
        target_season=target_season,
        config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
        cold_start_config=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    )
    combined = attach_expected_total_minutes(conditional, availability)
    opening_roster = roster.rename(columns={"team_abbreviation": "team"}).loc[
        :, ["team_id", "team", "player_id", "player_name"]
    ]
    combined = _apply_raw_expected_total_rotation_limit(
        combined,
        opening_roster=opening_roster,
        size=initial_rotation_size,
    )
    normalized = normalize_opening_roster_minutes(combined, opening_roster=opening_roster)
    normalized = normalized.merge(
        combined.loc[:, ["player_id", "is_baseline_rotation_candidate"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    prior_minutes = (
        summary.loc[summary["season_start_year"].eq(target_year - 1)]
        .set_index("player_id")["total_nba_minutes"]
        .astype(float)
    )
    ratings = preseason_rankings.loc[:, ["player_id", "rapm", "profile_source"]].copy()
    ratings["player_id"] = pd.to_numeric(ratings["player_id"], errors="raise").astype(int)
    players = (
        roster.rename(columns={"team_abbreviation": "team"})
        .merge(normalized, on=["team_id", "team", "player_id", "player_name"], how="inner")
        .merge(
            availability.loc[
                :, ["player_id", "predicted_available_share", "has_prior_availability_state"]
            ],
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            conditional.loc[
                :, [
                    "player_id",
                    "predicted_minutes_per_available_game",
                    "has_prior_minutes_state",
                ]
            ],
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        .merge(ratings, on="player_id", how="left", validate="one_to_one")
    )
    if len(players) != len(roster):
        raise ValueError("Preseason minutes forecast did not retain every rostered player")
    players["prior_season_minutes"] = players["player_id"].map(prior_minutes).fillna(0.0)
    players["is_rating_fallback"] = players["rapm"].isna()
    players["rapm"] = players["rapm"].fillna(0.0)
    players["profile_source"] = players["profile_source"].fillna("roster fallback")
    players["baseline_minutes_per_game"] = players["projected_total_minutes"] / 82.0
    players = players.sort_values(
        ["team", "baseline_minutes_per_game", "player_name", "player_id"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    rows = [
        {
            "player_id": int(row.player_id),
            "player_name": str(row.player_name),
            "position": str(row.listed_position),
            "age": None if pd.isna(row.age) else float(row.age),
            "team": str(row.team),
            "prior_season_minutes": float(row.prior_season_minutes),
            "is_rating_fallback": bool(row.is_rating_fallback),
            "is_baseline_rotation_candidate": bool(row.is_baseline_rotation_candidate),
            "has_prior_availability_state": bool(row.has_prior_availability_state),
            "has_prior_minutes_state": bool(row.has_prior_minutes_state),
            "availability_probability": float(row.predicted_available_share),
            "conditional_minutes_per_game": float(
                row.predicted_minutes_per_available_game
            ),
            "raw_expected_total_minutes": float(row.raw_expected_total_minutes),
            "baseline_minute_share": float(row.projected_minute_share),
            "baseline_minutes_per_game": float(row.baseline_minutes_per_game),
            "projected_nail": float(row.rapm),
            "rating_source": str(row.profile_source),
        }
        for row in players.itertuples(index=False)
    ]
    return {
        "season": target_season,
        "completed_season": history_seasons[-1],
        "model": "Forward Availability v0.2 + Forward Conditional Minutes v0.2",
        "initial_rotation_size": initial_rotation_size,
        "regulation_team_minutes": REGULATION_TEAM_MINUTES,
        "contract": (
            "Each player starts with a forward medical-availability probability and "
            "a forward conditional minutes-per-available-game forecast. Their raw "
            "expected season minutes are 82 times those two values, then every "
            f"opening roster is reduced to its top {initial_rotation_size} raw weights, then "
            "normalized to exactly 240 regulation minutes per game."
        ),
        "teams": sorted(roster["team_abbreviation"].astype(str).unique().tolist()),
        "players": rows,
    }


def build_incumbent_team_strength_minutes_payload(
    *,
    baseline_minutes_payload: dict[str, object],
    team_strength_history: pd.DataFrame,
    roster_path: Path | str = DEFAULT_ROSTER_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    target_season: str = DEFAULT_TARGET_SEASON,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> dict[str, object]:
    """Build the FCM v0.3 payload from the immutable baseline payload.

    Availability, preseason NAIL, and the published roster are read directly
    from ``baseline_minutes_payload``.  Only the rookie conditional-minutes
    prior is refit with the incumbent-team-strength feature; the raw expected
    minutes are then passed through the same top-15 gate and 240-minute
    normalization as the released payload.
    """

    baseline_players = pd.DataFrame(baseline_minutes_payload.get("players", []))
    required_player_columns = {
        "player_id",
        "player_name",
        "team",
        "availability_probability",
        "conditional_minutes_per_game",
        "has_prior_minutes_state",
        "projected_nail",
    }
    missing_player_columns = sorted(required_player_columns - set(baseline_players))
    if missing_player_columns:
        raise ValueError(
            "Baseline minutes payload lacks required player fields: "
            + ", ".join(missing_player_columns)
        )
    if baseline_players["player_id"].duplicated().any():
        raise ValueError("Baseline minutes payload has duplicate player IDs")
    if initial_rotation_size <= 0:
        raise ValueError("Initial rotation size must be positive")

    roster = pd.read_parquet(roster_path)
    _validate_roster(roster)
    roster = roster.copy()
    roster["team_id"] = pd.to_numeric(roster["team_id"], errors="raise").astype("int64")
    roster["player_id"] = pd.to_numeric(roster["player_id"], errors="raise").astype("int64")
    opening_roster = roster.rename(columns={"team_abbreviation": "team"}).loc[
        :, ["team_id", "team", "player_id", "player_name"]
    ]
    if opening_roster["player_id"].duplicated().any():
        raise ValueError("Opening roster contains duplicate player IDs")

    published = baseline_players.loc[
        :,
        [
            "player_id",
            "availability_probability",
            "conditional_minutes_per_game",
            "has_prior_minutes_state",
            "projected_nail",
        ],
    ].copy()
    published["player_id"] = pd.to_numeric(published["player_id"], errors="raise").astype(
        "int64"
    )
    roster_profile = build_forward_conditional_roster_profile(
        roster, target_season=target_season, curated_dir=curated_dir
    ).merge(
        opening_roster.loc[:, ["team_id", "team", "player_id"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    ).merge(published, on="player_id", how="left", validate="one_to_one")
    if roster_profile["availability_probability"].isna().any():
        missing = roster_profile.loc[
            roster_profile["availability_probability"].isna(), "player_name"
        ].tolist()
        raise ValueError("Production payload lacks rostered players: " + ", ".join(missing))

    incumbent = roster_profile.copy()
    incumbent["raw_incumbent_weight"] = (
        pd.to_numeric(incumbent["availability_probability"], errors="raise")
        * pd.to_numeric(incumbent["conditional_minutes_per_game"], errors="raise")
    )
    eligible = incumbent.loc[
        incumbent["has_prior_minutes_state"].astype(bool)
        & ~incumbent["is_rookie"].astype(bool)
        & incumbent["raw_incumbent_weight"].gt(0.0)
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible returning players for incumbent team strength")
    incumbent_player_ids = set(eligible["player_id"].astype(int).tolist())
    eligible["weighted_nail"] = eligible["raw_incumbent_weight"] * eligible["projected_nail"]
    team_strength = eligible.groupby(["team_id", "team"], as_index=False).agg(
        incumbent_weight=("raw_incumbent_weight", "sum"),
        weighted_nail=("weighted_nail", "sum"),
    )
    team_strength["incumbent_team_nail"] = (
        team_strength["weighted_nail"] / team_strength["incumbent_weight"]
    )
    league_strength_fallback = float(
        np.average(
            team_strength["incumbent_team_nail"],
            weights=team_strength["incumbent_weight"],
        )
    )
    roster_profile = roster_profile.merge(
        team_strength.loc[:, ["team_id", "incumbent_team_nail"]],
        on="team_id",
        how="left",
        validate="many_to_one",
    )
    roster_profile["incumbent_team_nail"] = roster_profile["incumbent_team_nail"].fillna(
        league_strength_fallback
    )

    candidate_config = ColdStartConditionalMinutesConfig(
        alpha=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG.alpha,
        draft_pick_half_life=None,
        include_incumbent_team_strength=True,
    )
    conditional, _age_model = predict_conditional_minutes_roster(
        team_strength_history,
        roster=roster_profile,
        target_season=target_season,
        config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
        cold_start_config=candidate_config,
    )
    history = team_strength_history.loc[
        pd.to_numeric(team_strength_history["season_start_year"], errors="raise").lt(
            int(target_season[:4])
        )
    ].copy()
    cold_start_model = fit_cold_start_conditional_minutes_model(
        history,
        age_model=fit_age_conditional_minutes_model(history),
        config=candidate_config,
    )
    strength_index = cold_start_model.feature_columns.index("incumbent_team_nail")
    strength_log_minutes_coefficient = float(
        cold_start_model.coefficients[strength_index]
        / cold_start_model.feature_scale[strength_index]
    )
    candidate = baseline_players.copy()
    candidate["player_id"] = pd.to_numeric(candidate["player_id"], errors="raise").astype(
        "int64"
    )
    candidate = candidate.merge(
        conditional.loc[
            :,
            [
                "player_id",
                "predicted_minutes_per_available_game",
                "uses_draft_cold_start_prior",
            ],
        ],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    if candidate["predicted_minutes_per_available_game"].isna().any():
        raise ValueError("Candidate conditional minutes lack a rostered player")
    candidate["conditional_minutes_per_game"] = candidate[
        "predicted_minutes_per_available_game"
    ].astype(float)
    candidate["uses_incumbent_team_strength"] = candidate[
        "uses_draft_cold_start_prior"
    ].astype(bool)
    candidate["is_incumbent_team_strength_incumbent"] = candidate["player_id"].isin(
        incumbent_player_ids
    )
    candidate["raw_expected_total_minutes"] = (
        82.0
        * candidate["availability_probability"].astype(float)
        * candidate["conditional_minutes_per_game"].astype(float)
    )
    candidate = candidate.drop(
        columns=[
            "predicted_minutes_per_available_game",
            "baseline_minute_share",
            "baseline_minutes_per_game",
        ],
        errors="ignore",
    )
    candidate = opening_roster.merge(
        candidate.drop(columns="team", errors="ignore"),
        on=["player_id", "player_name"],
        how="left",
        validate="one_to_one",
    )
    gated = _apply_raw_expected_total_rotation_limit(
        candidate,
        opening_roster=opening_roster,
        size=initial_rotation_size,
    )
    normalized = normalize_opening_roster_minutes(gated, opening_roster=opening_roster)
    candidate = (
        gated.drop(columns="team_id", errors="ignore")
        .merge(
            normalized.loc[
                :, ["player_id", "projected_minute_share", "projected_total_minutes"]
            ],
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "projected_minute_share": "baseline_minute_share",
                "projected_total_minutes": "_projected_total_minutes",
            }
        )
    )
    candidate["baseline_minutes_per_game"] = candidate["_projected_total_minutes"] / 82.0
    candidate = candidate.drop(columns="_projected_total_minutes")
    candidate = candidate.sort_values(
        ["team", "baseline_minutes_per_game", "player_name", "player_id"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    duplicate_columns = candidate.columns[candidate.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        raise ValueError(
            "Team-strength payload produced duplicate player fields: "
            + ", ".join(str(column) for column in duplicate_columns)
        )
    rows = candidate.to_dict(orient="records")
    return {
        **{key: value for key, value in baseline_minutes_payload.items() if key != "players"},
        "model": "Forward Availability v0.2 + Forward Conditional Minutes v0.3: Team-Strength Cold Starts",
        "conditional_minutes_version": "v0.3",
        "contract": (
            "Availability, the active-15 roster gate, and 240-minute normalization are "
            "unchanged. FCM v0.3 adds incumbent preseason team NAIL strength only to the "
            "rookie conditional-minutes prior."
        ),
        "players": rows,
        "incumbent_team_strength": {
            "league_strength_fallback": league_strength_fallback,
            "team_strength_log_minutes_coefficient": strength_log_minutes_coefficient,
            "teams": team_strength.sort_values("team", kind="stable").to_dict(orient="records"),
        },
    }


def _apply_raw_expected_total_rotation_limit(
    predictions: pd.DataFrame,
    *,
    opening_roster: pd.DataFrame,
    size: int,
) -> pd.DataFrame:
    """Keep each team's top raw expected-minute weights in the baseline rotation."""

    candidates = opening_roster.merge(
        predictions.loc[:, ["player_id", "raw_expected_total_minutes"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    selected_ids: set[int] = set()
    for _, team in candidates.groupby("team_id", sort=False):
        selected_ids.update(
            team.sort_values(
                ["raw_expected_total_minutes", "player_name", "player_id"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(size)["player_id"]
            .astype(int)
            .tolist()
        )
    output = predictions.copy()
    output["is_baseline_rotation_candidate"] = output["player_id"].isin(selected_ids)
    output.loc[
        ~output["is_baseline_rotation_candidate"], "raw_expected_total_minutes"
    ] = 0.0
    return output


def build_source_aware_preseason_minutes_payload(
    *,
    preseason_rankings: pd.DataFrame,
    roster_path: Path | str = DEFAULT_ROSTER_PATH,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    target_season: str = DEFAULT_TARGET_SEASON,
    completed_season: str = DEFAULT_COMPLETED_SEASON,
    production_model_path: Path | str | None = None,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> dict[str, object]:
    """Return frozen L20-NAIL-MSP v0.3 opening allocations for current rosters.

    The loaded model was fit only on completed seasons. The current target
    contributes its roster and static biography fields, never target minutes.
    """

    if initial_rotation_size <= 0:
        raise ValueError("Initial rotation size must be positive")
    roster = pd.read_parquet(roster_path)
    _validate_roster(roster)
    _validate_rankings(preseason_rankings)
    model_path = (
        Path(production_model_path)
        if production_model_path is not None
        else _latest_source_aware_model_path()
    )
    model = joblib.load(model_path)
    if not isinstance(model, FittedSourceAwareMinuteModel):
        raise ValueError(
            f"Source-aware production artifact is not a fitted v0.3 model: {model_path}"
        )
    panel = pd.read_parquet(panel_path)
    panel_with_target_bios = append_target_bios(panel, roster, target_season=target_season)
    ratings = preseason_rankings.loc[:, ["player_id", "rapm", "profile_source"]].copy()
    ratings["player_id"] = pd.to_numeric(ratings["player_id"], errors="raise").astype(int)
    rating_by_player = ratings.set_index("player_id")["rapm"]
    role_state = build_current_source_aware_role_state(
        target_season,
        roster=roster,
        panel=panel_with_target_bios,
        preseason_nail=rating_by_player,
        curated_dir=curated_dir,
        history_seasons=DEFAULT_HISTORY_SEASONS,
    )
    players = predict_source_aware_minute_shares(role_state, model=model).merge(
        ratings,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    players["is_rating_fallback"] = players["rapm"].isna()
    players["rapm"] = players["rapm"].fillna(0.0)
    players["profile_source"] = players["profile_source"].fillna("roster fallback")
    players = _apply_initial_rotation_limit(players, size=initial_rotation_size)
    players = players.sort_values(
        ["team", "baseline_minutes_per_game", "player_name", "player_id"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    rows = [
        {
            "player_id": int(row.player_id),
            "player_name": str(row.player_name),
            "position": str(row.listed_position),
            "age": None if pd.isna(row.age) else float(row.age),
            "team": str(row.team),
            "prior_season_minutes": float(row.source_minutes),
            "is_baseline_rotation_candidate": bool(row.is_baseline_rotation_candidate),
            "is_rating_fallback": bool(row.is_rating_fallback),
            "player_state": str(row.player_state),
            "baseline_minute_share": float(row.baseline_minute_share),
            "baseline_minutes_per_game": float(row.baseline_minutes_per_game),
            "projected_nail": float(row.rapm),
            "rating_source": str(row.profile_source),
        }
        for row in players.itertuples(index=False)
    ]
    return {
        "season": target_season,
        "completed_season": completed_season,
        "model": "L20-NAIL-MSP v0.3",
        "source_aware_model_path": str(model_path),
        "role_shrinkage_games": model.config.role_shrinkage_games,
        "gap_decay": model.config.gap_decay,
        "alpha": model.config.alpha,
        "initial_rotation_size": initial_rotation_size,
        "regulation_team_minutes": REGULATION_TEAM_MINUTES,
        "contract": (
            "A frozen roster-relative ridge-softmax model ranks each roster using "
            "last-observed role evidence, evidence-shrunken MPG/GP/GS, exact "
            "preseason NAIL, decayed role evidence for injury-gap returners, and a "
            "separate draft/age/position cold-start path. Its top 15 players are "
            "renormalized to 240 team minutes. The site re-ranks the full roster "
            "after any manual availability or minutes override."
        ),
        "teams": sorted(roster["team_abbreviation"].astype(str).unique().tolist()),
        "players": rows,
    }


def _apply_initial_rotation_limit(players: pd.DataFrame, *, size: int) -> pd.DataFrame:
    """Keep the top source-aware allocation candidates in each team baseline."""

    output = players.copy()
    output["is_baseline_rotation_candidate"] = False
    output["baseline_minute_share"] = 0.0
    for _, group in players.groupby("team", sort=True):
        ranked = group.sort_values(
            ["baseline_minute_share", "player_name", "player_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        rotation_index = ranked.index[:size]
        total_share = float(players.loc[rotation_index, "baseline_minute_share"].sum())
        if total_share <= 0.0:
            raise ValueError(f"Source-aware rotation has no positive allocation for {group.name}")
        output.loc[rotation_index, "is_baseline_rotation_candidate"] = True
        output.loc[rotation_index, "baseline_minute_share"] = (
            players.loc[rotation_index, "baseline_minute_share"] / total_share
        )
    output["baseline_minutes_per_game"] = (
        REGULATION_TEAM_MINUTES * output["baseline_minute_share"]
    )
    return output


def _latest_source_aware_model_path() -> Path:
    candidates = sorted(
        DEFAULT_SOURCE_AWARE_ARTIFACTS_DIR.glob(
            "l20-source-aware-production-*/production_model.joblib"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No L20-NAIL-MSP v0.3 production artifact found. "
            "Run scripts/fit_l20_source_aware_production.sh first."
        )
    return candidates[0]


def _validate_roster(roster: pd.DataFrame) -> None:
    required = {
        "team_abbreviation",
        "player_id",
        "player_name",
        "listed_position",
        "age",
    }
    missing = sorted(required - set(roster))
    if missing:
        raise ValueError(f"Current roster is missing columns: {missing}")
    if roster["player_id"].astype(int).duplicated().any():
        raise ValueError("Current roster contains duplicate player IDs")


def _validate_rankings(rankings: pd.DataFrame) -> None:
    required = {"player_id", "rapm", "profile_source"}
    missing = sorted(required - set(rankings))
    if missing:
        raise ValueError(f"Preseason rankings are missing columns: {missing}")
    if rankings["player_id"].astype(int).duplicated().any():
        raise ValueError("Preseason rankings contain duplicate player IDs")
