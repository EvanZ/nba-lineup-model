"""Leakage-safe player trait profiles for the contextual lineup prior.

The profile layer deliberately predicts *inputs* to a lineup-composition model,
not a second player value.  Returning players use prior-season possession rates.
First-season players receive a draft-cohort profile blended toward the same
low-exposure replacement population used by the forward cold-start RAPM.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.cold_start_exposure import (
    FEATURE_COLUMNS as EXPOSURE_FEATURE_COLUMNS,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    _feature_base as exposure_feature_base,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    _feature_reference as exposure_feature_reference,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    _prepare_features as prepare_exposure_features,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    enrich_exposure_cohort,
    fit_exposure_model,
)
from nba_lineup_model.modeling.cold_start_exposure import (
    select_regularization as select_exposure_regularization,
)
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.season.schema import validate_season

PROFILE_COUNTS = {
    "three_pa": "three_pointers_attempted",
    "three_pm": "three_pointers_made",
    "assists": "assists",
    "turnovers": "turnovers",
    "usage": ("field_goals_attempted", "free_throws_attempted", "turnovers"),
    "offensive_rebounds": "rebounds_offensive",
    "defensive_rebounds": "rebounds_defensive",
    "steals": "steals",
    "blocks": "blocks",
}
PROFILE_RATE_COLUMNS = tuple(f"{trait}_per_100" for trait in PROFILE_COUNTS)
PROFILE_REBOUND_PERCENT_COLUMNS = (
    "offensive_rebound_pct",
    "defensive_rebound_pct",
)
PROFILE_COLUMNS = (*PROFILE_RATE_COLUMNS, *PROFILE_REBOUND_PERCENT_COLUMNS)
PROFILE_PSEUDO_POSSESSIONS = 300.0
REPLACEMENT_SHARE_CUTOFF = 0.05


@dataclass(frozen=True)
class ProfilePaddingContract:
    """Versioned empirical-Bayes padding for lagged player profiles."""

    name: str
    rate_pseudo_possessions: Mapping[str, float]
    reference_mode: str = "expanding"
    three_point_percentage_attempts: float | None = None
    usage_component_pseudo_possessions: Mapping[str, float] | None = None
    rebound_percentage_pseudo_possessions: Mapping[str, float] | None = None
    source: str | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "reference_mode": self.reference_mode,
            "rate_pseudo_possessions": dict(self.rate_pseudo_possessions),
            "three_point_percentage_attempts": self.three_point_percentage_attempts,
            "usage_component_pseudo_possessions": (
                dict(self.usage_component_pseudo_possessions)
                if self.usage_component_pseudo_possessions is not None
                else None
            ),
            "rebound_percentage_pseudo_possessions": (
                dict(self.rebound_percentage_pseudo_possessions)
                if self.rebound_percentage_pseudo_possessions is not None
                else None
            ),
            "source": self.source,
        }


UNIFORM_300_PROFILE_PADDING = ProfilePaddingContract(
    name="uniform_300_expanding_reference",
    rate_pseudo_possessions={
        trait: PROFILE_PSEUDO_POSSESSIONS for trait in PROFILE_COUNTS
    },
)

UNIFORM_300_SEASON_PROFILE_PADDING = ProfilePaddingContract(
    name="uniform_300_source_season_reference",
    rate_pseudo_possessions={
        trait: PROFILE_PSEUDO_POSSESSIONS for trait in PROFILE_COUNTS
    },
    reference_mode="season",
)

MEDVEDOVSKY_2020_PROFILE_PADDING = ProfilePaddingContract(
    name="medvedovsky_2020_stat_specific",
    rate_pseudo_possessions={
        "three_pa": 29.29,
        "assists": 55.57,
        "turnovers": 425.69,
        "offensive_rebounds": 101.46,
        "defensive_rebounds": 109.0,
        "steals": 632.61,
        "blocks": 151.22,
    },
    reference_mode="season",
    three_point_percentage_attempts=242.61,
    usage_component_pseudo_possessions={
        "field_goals_attempted": 89.62,
        "free_throws_attempted": 197.58,
        "turnovers": 425.69,
    },
    rebound_percentage_pseudo_possessions={
        "offensive_rebound_pct": 98.55,
        "defensive_rebound_pct": 108.26,
    },
    source="https://kmedved.com/2020/08/06/nba-stabilization-rates-and-the-padding-approach/",
)


def build_contextual_player_profiles(
    panel: pd.DataFrame,
    *,
    target_season: str,
    target_player_ids: Iterable[int],
    analytical_dir: str = "data/analytical",
    curated_dir: str = "data/curated",
    exposure_cohort: pd.DataFrame | None = None,
    padding_contract: ProfilePaddingContract = UNIFORM_300_PROFILE_PADDING,
    use_last_observed_profile: bool = False,
) -> pd.DataFrame:
    """Build target player profiles using information available before a season.

    ``target_player_ids`` is an oracle roster/lineup universe, which is allowed
    under the frozen evaluation contract. No target-season box-score or RAPM
    outcome is read to form the returned traits.
    """

    target = validate_season(target_season)
    source = _previous_season(target)
    _validate_panel(panel)
    player_ids = np.array(sorted({int(value) for value in target_player_ids}), dtype=np.int64)
    if len(player_ids) == 0:
        raise ValueError("Contextual profiles require at least one target player")

    history = panel.loc[panel["season"].astype(str).lt(target)].copy()
    if history.empty:
        raise ValueError("Contextual profiles require earlier player-season history")
    target_bios = panel.loc[
        panel["season"].eq(target) & panel["player_id"].isin(player_ids),
        _bio_columns(),
    ].copy()
    if target_bios["player_id"].duplicated().any():
        raise ValueError("Target player-season bios must be unique by player")
    missing = sorted(set(player_ids) - set(target_bios["player_id"].astype(int)))
    if missing:
        # Early historical box-score archives omit a small number of stint
        # participants entirely. They receive the explicit replacement profile,
        # not an implicit zero-valued trait vector.
        placeholders = pd.DataFrame(
            {
                "season": target,
                "season_start_year": int(target[:4]),
                "player_id": missing,
                "player_name": [f"Unknown player {player_id}" for player_id in missing],
                "is_rookie": False,
                "draft_number": np.nan,
                "is_undrafted": False,
                "draft_year": np.nan,
                "draft_round": np.nan,
                "age": np.nan,
                "height_inches": np.nan,
                "weight_pounds": np.nan,
            }
        )
        target_bios = pd.concat([target_bios, placeholders], ignore_index=True)

    _validate_padding_contract(padding_contract)
    historical_rates = _rate_frame(history, padding_contract=padding_contract).merge(
        _rebound_percentage_frame(
            tuple(sorted(history["season"].astype(str).unique(), key=lambda value: int(value[:4]))),
            curated_dir=curated_dir,
        ),
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    historical_rates = _pad_rebound_percentages(
        historical_rates,
        padding_contract=padding_contract,
    )
    if historical_rates.loc[:, list(PROFILE_REBOUND_PERCENT_COLUMNS)].isna().any(axis=None):
        raise ValueError("Contextual rebound percentage profiles are incomplete")
    previous = historical_rates.loc[historical_rates["season"].eq(source)].copy()
    if use_last_observed_profile:
        previous = (
            historical_rates.sort_values(["player_id", "season"], kind="stable")
            .groupby("player_id", as_index=False, sort=False)
            .tail(1)
            .copy()
        )
    previous = previous.rename(columns={"season": "profile_source_season"})
    returners = target_bios.merge(
        previous.loc[:, ["player_id", "profile_source_season", *PROFILE_COLUMNS]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    has_prior_profile = returners.loc[:, list(PROFILE_COLUMNS)].notna().all(axis=1)

    cohort_profiles = _rookie_cohort_profiles(
        historical_rates,
        history.loc[:, _bio_columns()],
    )
    replacement_profile = _replacement_profile(
        panel,
        historical_rates,
        source_season=source,
        analytical_dir=analytical_dir,
        exposure_cohort=exposure_cohort,
    )
    output = returners.copy()
    output["profile_gap_seasons"] = (
        int(target[:4]) - output["profile_source_season"].astype("string").str[:4].astype(float) - 1
    ).where(has_prior_profile, pd.NA).astype("Int64")
    output["profile_source"] = np.where(
        has_prior_profile & output["profile_gap_seasons"].eq(0),
        "prior_season",
        np.where(has_prior_profile, "last_observed_season", "cold_start"),
    )
    output["profile_imputed"] = (~has_prior_profile).astype(int)
    output["profile_replacement_weight"] = 0.0

    cold = output.loc[~has_prior_profile].copy()
    if not cold.empty:
        probability = _cold_start_replacement_probability(
            panel,
            cold,
            source_season=source,
            analytical_dir=analytical_dir,
            exposure_cohort=exposure_cohort,
        )
        cold["profile_replacement_weight"] = probability
        cold_profiles = _cold_profiles(cold, cohort_profiles, replacement_profile)
        for column in PROFILE_COLUMNS:
            output.loc[~has_prior_profile, column] = cold_profiles[column].to_numpy(dtype=float)
        output.loc[~has_prior_profile, "profile_replacement_weight"] = probability
        output.loc[~has_prior_profile, "profile_source"] = np.where(
            cold["is_rookie"].to_numpy(dtype=bool),
            "exposure_gated_rookie",
            "replacement_profile",
        )

    if output.loc[:, list(PROFILE_COLUMNS)].isna().any(axis=None):
        raise ValueError("Contextual profile construction left missing trait values")
    output["player_id"] = output["player_id"].astype("int64")
    output["target_season"] = target
    return (
        output.loc[
            :,
            [
                "target_season",
                "player_id",
                "player_name",
                "is_rookie",
                "profile_source",
                "profile_imputed",
                "profile_replacement_weight",
                "profile_gap_seasons",
                *PROFILE_COLUMNS,
            ],
        ]
        .sort_values("player_id", kind="stable")
        .reset_index(drop=True)
    )


def _rate_frame(
    frame: pd.DataFrame,
    *,
    padding_contract: ProfilePaddingContract,
) -> pd.DataFrame:
    output = frame.loc[:, ["season", "player_id", "rapm_possessions", *_count_columns()]].copy()
    possessions = pd.to_numeric(output.pop("rapm_possessions"), errors="raise").astype(float)
    if not possessions.gt(0).all():
        raise ValueError("Contextual profiles require positive player possession totals")
    output["_possessions"] = possessions
    references = _reference_rate_frame(frame, padding_contract=padding_contract)
    for trait, counts in PROFILE_COUNTS.items():
        if trait in {"three_pm", "usage"}:
            continue
        raw = _trait_count(output, counts)
        output[f"{trait}_per_100"] = _pad_possession_rate(
            raw,
            possessions,
            references[trait],
            padding_contract.rate_pseudo_possessions[trait],
        )
    output["three_pm_per_100"] = _three_point_makes_rate(
        output,
        possessions,
        references,
        padding_contract=padding_contract,
    )
    output["usage_per_100"] = _usage_rate(
        output,
        possessions,
        references,
        padding_contract=padding_contract,
    )
    return output.loc[:, ["season", "player_id", "_possessions", *PROFILE_RATE_COLUMNS]]


def _pad_possession_rate(
    raw: pd.Series,
    possessions: pd.Series,
    reference_rate: pd.Series,
    pseudo_possessions: float,
) -> pd.Series:
    return (
        100.0
        * (raw.to_numpy(dtype=float) + pseudo_possessions * reference_rate / 100.0)
        / (possessions.to_numpy(dtype=float) + pseudo_possessions)
    )


def _three_point_makes_rate(
    frame: pd.DataFrame,
    possessions: pd.Series,
    references: pd.DataFrame,
    *,
    padding_contract: ProfilePaddingContract,
) -> pd.Series:
    attempts = pd.to_numeric(frame["three_pointers_attempted"], errors="raise").astype(float)
    makes = pd.to_numeric(frame["three_pointers_made"], errors="raise").astype(float)
    percentage_attempts = padding_contract.three_point_percentage_attempts
    if percentage_attempts is None:
        return _pad_possession_rate(
            makes,
            possessions,
            references["three_pm"],
            padding_contract.rate_pseudo_possessions["three_pm"],
        )
    attempt_rate = _pad_possession_rate(
        attempts,
        possessions,
        references["three_pa"],
        padding_contract.rate_pseudo_possessions["three_pa"],
    )
    percentage = (
        makes.to_numpy(dtype=float)
        + percentage_attempts * references["three_point_pct"]
    ) / (attempts.to_numpy(dtype=float) + percentage_attempts)
    return pd.Series(attempt_rate.to_numpy(dtype=float) * percentage, index=frame.index)


def _usage_rate(
    frame: pd.DataFrame,
    possessions: pd.Series,
    references: pd.DataFrame,
    *,
    padding_contract: ProfilePaddingContract,
) -> pd.Series:
    components = padding_contract.usage_component_pseudo_possessions
    if components is None:
        return _pad_possession_rate(
            _trait_count(frame, PROFILE_COUNTS["usage"]),
            possessions,
            references["usage"],
            padding_contract.rate_pseudo_possessions["usage"],
        )
    rates: dict[str, pd.Series] = {}
    for column, pseudo_possessions in components.items():
        rates[column] = _pad_possession_rate(
            pd.to_numeric(frame[column], errors="raise").astype(float),
            possessions,
            references[f"{column}_per_100"],
            pseudo_possessions,
        )
    return (
        rates["field_goals_attempted"]
        + 0.44 * rates["free_throws_attempted"]
        + rates["turnovers"]
    )


def _reference_rate_frame(
    frame: pd.DataFrame,
    *,
    padding_contract: ProfilePaddingContract,
) -> pd.DataFrame:
    if padding_contract.reference_mode == "expanding":
        reference = _league_reference_rates(frame)
        return pd.DataFrame(
            {key: np.repeat(value, len(frame)) for key, value in reference.items()},
            index=frame.index,
        )
    rows: list[pd.DataFrame] = []
    for _season, members in frame.groupby("season", sort=False):
        reference = _league_reference_rates(members)
        rows.append(
            pd.DataFrame(
                {key: np.repeat(value, len(members)) for key, value in reference.items()},
                index=members.index,
            )
        )
    return pd.concat(rows).loc[frame.index]


def _league_reference_rates(frame: pd.DataFrame) -> dict[str, float]:
    possessions = pd.to_numeric(frame["rapm_possessions"], errors="raise").astype(float)
    if not possessions.gt(0).all():
        raise ValueError("Contextual profile reference requires positive possessions")
    rates = {
        trait: 100.0 * float(_trait_count(frame, counts).sum()) / float(possessions.sum())
        for trait, counts in PROFILE_COUNTS.items()
    }
    attempts = float(pd.to_numeric(frame["three_pointers_attempted"], errors="raise").sum())
    makes = float(pd.to_numeric(frame["three_pointers_made"], errors="raise").sum())
    rates["three_point_pct"] = makes / attempts if attempts > 0 else 0.0
    for column in (
        "field_goals_attempted",
        "free_throws_attempted",
        "turnovers",
    ):
        rates[f"{column}_per_100"] = (
            100.0 * float(pd.to_numeric(frame[column], errors="raise").sum())
            / float(possessions.sum())
        )
    return rates


def _pad_rebound_percentages(
    rates: pd.DataFrame,
    *,
    padding_contract: ProfilePaddingContract,
) -> pd.DataFrame:
    paddings = padding_contract.rebound_percentage_pseudo_possessions
    if paddings is None:
        return rates
    output = rates.copy()
    if padding_contract.reference_mode == "expanding":
        groups = [(None, output)]
    else:
        groups = list(output.groupby("season", sort=False))
    for _, members in groups:
        weights = members["_possessions"].to_numpy(dtype=float)
        for column, pseudo_possessions in paddings.items():
            reference = float(np.average(members[column].to_numpy(dtype=float), weights=weights))
            values = members[column].to_numpy(dtype=float)
            output.loc[members.index, column] = (
                values * weights + pseudo_possessions * reference
            ) / (weights + pseudo_possessions)
    return output


def _validate_padding_contract(contract: ProfilePaddingContract) -> None:
    if contract.reference_mode not in {"expanding", "season"}:
        raise ValueError("Profile padding reference_mode must be 'expanding' or 'season'")
    required = set(PROFILE_COUNTS) - {"three_pm", "usage"}
    if contract.three_point_percentage_attempts is None:
        required.add("three_pm")
    if contract.usage_component_pseudo_possessions is None:
        required.add("usage")
    missing = required - set(contract.rate_pseudo_possessions)
    if missing:
        raise ValueError(f"Profile padding is missing traits: {sorted(missing)}")
    values = list(contract.rate_pseudo_possessions.values())
    if contract.three_point_percentage_attempts is not None:
        values.append(contract.three_point_percentage_attempts)
    if contract.usage_component_pseudo_possessions is not None:
        required_usage = {
            "field_goals_attempted",
            "free_throws_attempted",
            "turnovers",
        }
        missing_usage = required_usage - set(contract.usage_component_pseudo_possessions)
        if missing_usage:
            raise ValueError(
                f"Profile padding is missing usage components: {sorted(missing_usage)}"
            )
        values.extend(contract.usage_component_pseudo_possessions.values())
    if contract.rebound_percentage_pseudo_possessions is not None:
        missing_rebounds = set(PROFILE_REBOUND_PERCENT_COLUMNS) - set(
            contract.rebound_percentage_pseudo_possessions
        )
        if missing_rebounds:
            raise ValueError(
                f"Profile padding is missing rebound percentages: {sorted(missing_rebounds)}"
            )
        values.extend(contract.rebound_percentage_pseudo_possessions.values())
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("Profile padding pseudo-sample sizes must be finite and non-negative")


def _rookie_cohort_profiles(rates: pd.DataFrame, bios: pd.DataFrame) -> pd.DataFrame:
    """Aggregate historical rookie profiles by a coarse pre-season draft band."""

    profiles = rates.merge(
        bios,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    rookies = profiles.loc[profiles["is_rookie"].astype(bool)].copy()
    if rookies.empty:
        raise ValueError("Contextual profiles require historical rookie seasons")
    rookies["draft_profile_group"] = _draft_profile_group(rookies)
    rows: list[dict[str, object]] = []
    for group, members in rookies.groupby("draft_profile_group", sort=True):
        row: dict[str, object] = {"draft_profile_group": group}
        weights = members["_possessions"].to_numpy(dtype=float)
        for column in PROFILE_COLUMNS:
            row[column] = float(np.average(members[column], weights=weights))
        rows.append(row)
    fallback = {"draft_profile_group": "all_rookies"}
    weights = rookies["_possessions"].to_numpy(dtype=float)
    for column in PROFILE_COLUMNS:
        fallback[column] = float(np.average(rookies[column], weights=weights))
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([fallback])], ignore_index=True)


def _replacement_profile(
    panel: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    source_season: str,
    analytical_dir: str,
    exposure_cohort: pd.DataFrame | None,
) -> dict[str, float]:
    cohort = _exposure_through_season(
        panel,
        source_season=source_season,
        analytical_dir=analytical_dir,
        exposure_cohort=exposure_cohort,
    )
    candidates = cohort.loc[cohort["exposure_share"].lt(REPLACEMENT_SHARE_CUTOFF)]
    keyed = rates.merge(
        candidates.loc[:, ["season", "player_id"]],
        on=["season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    if keyed.empty:
        raise ValueError("Contextual profiles found no historical replacement candidates")
    weights = keyed["_possessions"].to_numpy(dtype=float)
    return {
        column: float(np.average(keyed[column], weights=weights)) for column in PROFILE_COLUMNS
    }


def _cold_profiles(
    cold: pd.DataFrame,
    cohort_profiles: pd.DataFrame,
    replacement_profile: dict[str, float],
) -> pd.DataFrame:
    groups = cohort_profiles.set_index("draft_profile_group")
    fallback = groups.loc["all_rookies"]
    output = pd.DataFrame(index=cold.index)
    profile_groups = _draft_profile_group(cold)
    for column in PROFILE_COLUMNS:
        draft = profile_groups.map(groups[column]).fillna(float(fallback[column]))
        output[column] = (
            1.0 - cold["profile_replacement_weight"].to_numpy(dtype=float)
        ) * draft.to_numpy(dtype=float) + cold["profile_replacement_weight"].to_numpy(
            dtype=float
        ) * replacement_profile[column]
    return output


def _cold_start_replacement_probability(
    panel: pd.DataFrame,
    cold: pd.DataFrame,
    *,
    source_season: str,
    analytical_dir: str,
    exposure_cohort: pd.DataFrame | None,
) -> np.ndarray:
    """Fit the existing gate strictly before target and score rookie profiles."""

    rookie_mask = cold["is_rookie"].astype(bool)
    probability = np.ones(len(cold), dtype=float)
    if not rookie_mask.any():
        return probability
    history = panel.loc[panel["season"].astype(str).le(source_season)].copy()
    exposure = _exposure_through_season(
        history,
        source_season=source_season,
        analytical_dir=analytical_dir,
        exposure_cohort=exposure_cohort,
    )
    enriched = enrich_exposure_cohort(exposure, history)
    training_raw = enriched.loc[enriched["is_rookie"].astype(bool)].copy()
    if training_raw["season"].nunique() < 8:
        return probability
    training = prepare_exposure_features(training_raw)
    training["is_replacement_candidate"] = training["exposure_share"].lt(REPLACEMENT_SHARE_CUTOFF)
    if training["is_replacement_candidate"].nunique() != 2:
        return probability
    reference = exposure_feature_reference(exposure_feature_base(training_raw))
    target = prepare_exposure_features(cold.loc[rookie_mask].copy(), reference=reference)
    selected_c, _ = select_exposure_regularization(training, c_grid=(0.03, 0.1, 0.3, 1.0, 3.0))
    model = fit_exposure_model(training, c=selected_c)
    probability[rookie_mask.to_numpy()] = model.predict_proba(
        target.loc[:, EXPOSURE_FEATURE_COLUMNS]
    )[:, 1]
    return probability


def _exposure_through_season(
    panel: pd.DataFrame,
    *,
    source_season: str,
    analytical_dir: str,
    exposure_cohort: pd.DataFrame | None,
) -> pd.DataFrame:
    if exposure_cohort is None:
        return prepare_player_exposure_cohort(
            panel.loc[panel["season"].astype(str).le(source_season)],
            through_season=source_season,
            analytical_dir=analytical_dir,
        )
    required = {"season", "player_id", "exposure_share"}
    if required - set(exposure_cohort):
        raise ValueError("Contextual exposure cohort is missing required columns")
    return exposure_cohort.loc[exposure_cohort["season"].astype(str).le(source_season)].copy()


def _draft_profile_group(frame: pd.DataFrame) -> pd.Series:
    pick = pd.to_numeric(frame["draft_number"], errors="coerce")
    undrafted = frame["is_undrafted"].fillna(False).astype(bool)
    return pd.Series(
        np.select(
            [
                pick.between(1, 14),
                pick.between(15, 30),
                pick.between(31, 60),
                undrafted,
            ],
            ["lottery", "first_round", "second_round", "undrafted_or_other"],
            default="undrafted_or_other",
        ),
        index=frame.index,
        dtype="string",
    )


def _trait_count(frame: pd.DataFrame, counts: str | tuple[str, str, str]) -> pd.Series:
    if isinstance(counts, str):
        return pd.to_numeric(frame[counts], errors="raise").astype(float)
    fga, fta, turnovers = counts
    return (
        pd.to_numeric(frame[fga], errors="raise").astype(float)
        + 0.44 * pd.to_numeric(frame[fta], errors="raise").astype(float)
        + pd.to_numeric(frame[turnovers], errors="raise").astype(float)
    )


def _count_columns() -> tuple[str, ...]:
    columns: set[str] = set()
    for value in PROFILE_COUNTS.values():
        if isinstance(value, tuple):
            columns.update(value)
        else:
            columns.add(value)
    return tuple(sorted(columns))


def _rebound_percentage_frame(seasons: tuple[str, ...], *, curated_dir: str) -> pd.DataFrame:
    frames = [_season_rebound_percentage(season, curated_dir) for season in seasons]
    return pd.concat(frames, ignore_index=True)


@cache
def _season_rebound_percentage(season: str, curated_dir: str) -> pd.DataFrame:
    """Calculate standard player ORB% and DRB% from game-level opportunities."""

    path = Path(curated_dir) / "players" / season / "regular"
    players = pd.read_parquet(path)
    required = {
        "game_id",
        "team_side",
        "personId",
        "statistics_minutes",
        "statistics_reboundsOffensive",
        "statistics_reboundsDefensive",
    }
    missing = required - set(players)
    if missing:
        raise ValueError(f"Player boxscores missing rebound percentage columns: {sorted(missing)}")
    players = players.copy()
    players["minutes"] = pd.to_timedelta(
        players["statistics_minutes"], errors="coerce"
    ).dt.total_seconds() / 60.0
    players = players.loc[players["minutes"].gt(0)].copy()
    players["player_id"] = pd.to_numeric(players["personId"], errors="raise").astype("int64")
    players["offensive_rebounds"] = pd.to_numeric(
        players["statistics_reboundsOffensive"], errors="raise"
    ).astype(float)
    players["defensive_rebounds"] = pd.to_numeric(
        players["statistics_reboundsDefensive"], errors="raise"
    ).astype(float)
    team = players.groupby(["game_id", "team_side"], as_index=False).agg(
        team_minutes=("minutes", "sum"),
        team_offensive_rebounds=("offensive_rebounds", "sum"),
        team_defensive_rebounds=("defensive_rebounds", "sum"),
    )
    other = team.rename(
        columns={
            "team_side": "opponent_side",
            "team_offensive_rebounds": "opponent_offensive_rebounds",
            "team_defensive_rebounds": "opponent_defensive_rebounds",
        }
    ).drop(columns=["team_minutes"])
    players["opponent_side"] = np.where(players["team_side"].eq("home"), "away", "home")
    players = players.merge(team, on=["game_id", "team_side"], how="left", validate="many_to_one")
    players = players.merge(
        other,
        on=["game_id", "opponent_side"],
        how="left",
        validate="many_to_one",
    )
    defensive_opportunities = (
        players["team_defensive_rebounds"] + players["opponent_offensive_rebounds"]
    )
    offensive_opportunities = (
        players["team_offensive_rebounds"] + players["opponent_defensive_rebounds"]
    )
    if defensive_opportunities.le(0).any() or offensive_opportunities.le(0).any():
        raise ValueError(f"Invalid rebound opportunities in {season}")
    player_minutes_share = players["team_minutes"] / (5.0 * players["minutes"])
    players["offensive_rebound_pct"] = (
        100.0 * players["offensive_rebounds"] * player_minutes_share / offensive_opportunities
    )
    players["defensive_rebound_pct"] = (
        100.0 * players["defensive_rebounds"] * player_minutes_share / defensive_opportunities
    )
    weighted = players.assign(
        weighted_orb=players["offensive_rebound_pct"] * players["minutes"],
        weighted_drb=players["defensive_rebound_pct"] * players["minutes"],
    ).groupby("player_id", as_index=False).agg(
        minutes=("minutes", "sum"),
        weighted_orb=("weighted_orb", "sum"),
        weighted_drb=("weighted_drb", "sum"),
    )
    weighted["offensive_rebound_pct"] = weighted["weighted_orb"] / weighted["minutes"]
    weighted["defensive_rebound_pct"] = weighted["weighted_drb"] / weighted["minutes"]
    return weighted.loc[:, ["player_id", *PROFILE_REBOUND_PERCENT_COLUMNS]].assign(season=season)


def _bio_columns() -> list[str]:
    return [
        "season",
        "season_start_year",
        "player_id",
        "player_name",
        "is_rookie",
        "draft_number",
        "is_undrafted",
        "draft_year",
        "draft_round",
        "age",
        "height_inches",
        "weight_pounds",
    ]


def _validate_panel(panel: pd.DataFrame) -> None:
    required = set(_bio_columns()) | {"rapm_possessions", *_count_columns()}
    missing = required - set(panel)
    if missing:
        raise ValueError(
            f"Player-season panel missing contextual profile columns: {sorted(missing)}"
        )
    if panel.duplicated(["season", "player_id"]).any():
        raise ValueError("Player-season panel must be unique by season and player")


def _previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"
