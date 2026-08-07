"""Publish frozen lineup-context examples from the forward contextual model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    contextual_feature_columns,
    lineup_context_features,
)
from nba_lineup_model.modeling.forward_contextual_rapm import MODEL_NAME, _previous_season
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints

DEFAULT_TARGET_SEASON = "2025-26"
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_PAGE_PATH = Path("docs/models/forward-contextual-rapm.md")
DEFAULT_DOCS_ASSETS_DIR = Path("docs/assets/images/forward-contextual-rapm")
DEFAULT_MINIMUM_POSSESSIONS = 250.0
DEFAULT_MINIMUM_GAMES = 20
MODEL = "forward_contextual_lineup_case_study"
RUN_PREFIX = "forward-contextual-case-study"
SECTION_START = "<!-- forward-contextual-case-study:start -->"
SECTION_END = "<!-- forward-contextual-case-study:end -->"


@dataclass(frozen=True)
class LineupContextCaseStudyRun:
    run_dir: Path
    run_id: str


def build_lineup_context_case_study(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    analytical_dir: Path | str = Path("data/analytical"),
    docs_assets_dir: Path | str = DEFAULT_DOCS_ASSETS_DIR,
    minimum_possessions: float = DEFAULT_MINIMUM_POSSESSIONS,
    minimum_games: int = DEFAULT_MINIMUM_GAMES,
) -> LineupContextCaseStudyRun:
    """Score established target-season units using the frozen prior-season state."""

    if minimum_possessions <= 0:
        raise ValueError("Minimum possessions must be positive")
    if minimum_games <= 0:
        raise ValueError("Minimum games must be positive")
    root = Path(artifacts_dir)
    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(root / "forward_contextual_rapm" / DEFAULT_TARGET_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Lineup context case study requires a forward contextual RAPM artifact")
    target = str(metadata["target_season"])
    source_season = _previous_season(target)
    models = joblib.load(source / "season_context_models.joblib")
    frozen_context_model = models.get(source_season)
    if frozen_context_model is None:
        raise ValueError(f"Source artifact does not contain g_{source_season}")
    completed_context_model = models.get(target)
    if completed_context_model is None:
        raise ValueError(f"Source artifact does not contain completed g_{target}")
    profiles = pd.read_parquet(source / "target_player_profiles.parquet")
    priors = pd.read_parquet(source / "season_player_priors.parquet")
    frozen_priors = priors.loc[
        priors["season"].eq(target), ["player_id", "prior_rapm"]
    ].copy()
    if frozen_priors["player_id"].duplicated().any():
        raise ValueError("Target player priors contain duplicate player IDs")
    coefficients = pd.read_parquet(source / "historical_player_coefficients.parquet")
    completed_values = coefficients.loc[
        coefficients["season"].eq(target), ["player_id", "rapm"]
    ].rename(columns={"rapm": "player_value"})
    if completed_values["player_id"].duplicated().any():
        raise ValueError("Completed player coefficients contain duplicate player IDs")
    context_metadata = pd.read_parquet(source / "season_context_metadata.parquet")
    context_intercepts = dict(
        zip(
            context_metadata["season"].astype(str),
            context_metadata["context_home_intercept"].astype(float),
            strict=True,
        )
    )
    if source_season not in context_intercepts or target not in context_intercepts:
        raise ValueError("Source artifact is missing context home-intercept metadata")
    stints = read_rapm_stints(target, analytical_dir=analytical_dir)
    units = summarize_lineup_context(
        stints,
        frozen_context_model=frozen_context_model,
        completed_context_model=completed_context_model,
        profiles=profiles,
        frozen_priors=frozen_priors,
        completed_values=completed_values,
        frozen_home_intercept=context_intercepts[source_season],
        completed_home_intercept=context_intercepts[target],
        minimum_possessions=minimum_possessions,
        minimum_games=minimum_games,
    )
    return _write_run(
        source_run_dir=source,
        source_metadata=metadata,
        target=target,
        source_season=source_season,
        units=units,
        minimum_possessions=minimum_possessions,
        minimum_games=minimum_games,
        artifacts_dir=root,
        docs_assets_dir=Path(docs_assets_dir),
    )


def summarize_lineup_context(
    stints: pd.DataFrame,
    *,
    frozen_context_model: object,
    completed_context_model: object,
    profiles: pd.DataFrame,
    frozen_priors: pd.DataFrame,
    completed_values: pd.DataFrame,
    frozen_home_intercept: float,
    completed_home_intercept: float,
    minimum_possessions: float,
    minimum_games: int,
) -> dict[str, pd.DataFrame]:
    """Compare frozen and completed contextual states on target-season units."""

    required = {
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
        "home_player_ids",
        "away_player_ids",
        "possessions",
        "target_home_net_rating",
    }
    missing = required - set(stints)
    if missing:
        raise ValueError(f"Lineup context stints missing columns: {sorted(missing)}")
    frozen_value_map = dict(
        zip(
            frozen_priors["player_id"].astype(int),
            frozen_priors["prior_rapm"].astype(float),
            strict=True,
        )
    )
    completed_value_map = dict(
        zip(
            completed_values["player_id"].astype(int),
            completed_values["player_value"].astype(float),
            strict=True,
        )
    )
    exposure = _unit_exposure(stints)
    eligible_exposure = exposure.loc[
        exposure["possessions"].ge(minimum_possessions)
        & exposure["games"].ge(minimum_games)
    ].copy()
    if len(eligible_exposure) < 20:
        raise ValueError("Exposure threshold leaves fewer than 20 eligible lineups")
    eligible_keys = {
        (int(row.team_id), tuple(int(value) for value in row.lineup_player_ids))
        for row in eligible_exposure.itertuples(index=False)
    }
    home_keys = [
        (int(team_id), _lineup_key(lineup))
        for team_id, lineup in zip(
            stints["home_team_id"], stints["home_player_ids"], strict=True
        )
    ]
    away_keys = [
        (int(team_id), _lineup_key(lineup))
        for team_id, lineup in zip(
            stints["away_team_id"], stints["away_player_ids"], strict=True
        )
    ]
    selected = np.fromiter(
        (
            home_key in eligible_keys or away_key in eligible_keys
            for home_key, away_key in zip(home_keys, away_keys, strict=True)
        ),
        dtype=bool,
        count=len(stints),
    )
    scored_stints = stints.loc[selected].reset_index(drop=True)
    home = [key[1] for key, include in zip(home_keys, selected, strict=True) if include]
    away = [key[1] for key, include in zip(away_keys, selected, strict=True) if include]
    frozen_correction = _context_prediction(frozen_context_model, home, away, profiles)
    completed_correction = _context_prediction(completed_context_model, home, away, profiles)
    frozen_additive = np.array(
        [
            _lineup_value(lineup, frozen_value_map)
            - _lineup_value(opponent, frozen_value_map)
            for lineup, opponent in zip(home, away, strict=True)
        ],
        dtype=float,
    )
    completed_additive = np.array(
        [
            _lineup_value(lineup, completed_value_map)
            - _lineup_value(opponent, completed_value_map)
            for lineup, opponent in zip(home, away, strict=True)
        ],
        dtype=float,
    )
    observed = _observed_unit_rows(
        scored_stints,
        home,
        away,
        frozen_correction=frozen_correction,
        frozen_additive=frozen_additive,
        completed_correction=completed_correction,
        completed_additive=completed_additive,
        frozen_home_intercept=frozen_home_intercept,
        completed_home_intercept=completed_home_intercept,
    )
    observed["eligible"] = [
        (int(team_id), lineup) in eligible_keys
        for team_id, lineup in zip(observed["team_id"], observed["lineup"], strict=True)
    ]
    eligible = _aggregate_units(observed.loc[observed["eligible"]].copy(), profiles)
    eligible["standardized_context_net_rating"] = _standardized_context(
        eligible, context_model=frozen_context_model, profiles=profiles
    )
    eligible = eligible.sort_values(
        ["standardized_context_net_rating", "possessions", "team_tricode"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    eligible["standardized_rank"] = np.arange(1, len(eligible) + 1)
    attribution, attribution_summary = _top_lineup_attribution(
        eligible,
        context_model=frozen_context_model,
        profiles=profiles,
    )
    application_features = lineup_context_features(
        stints["home_player_ids"].tolist(), stints["away_player_ids"].tolist(), profiles
    )
    response_curves = _response_curves(
        frozen_context_model,
        application_features=application_features,
        focal_attribution=attribution,
    )
    return {
        "all_units": exposure,
        "eligible_units": eligible,
        "retrospective_metrics": _retrospective_metrics(eligible),
        "top_lineup_attribution": attribution,
        "top_lineup_attribution_summary": attribution_summary,
        "response_curves": response_curves,
        "positive_examples": eligible.head(10).copy(),
        "negative_examples": eligible.tail(10)
        .sort_values("standardized_context_net_rating", kind="stable")
        .reset_index(drop=True),
    }


def _unit_exposure(stints: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "team_id": stints["home_team_id"].to_numpy(),
            "team_tricode": stints["home_team_tricode"].to_numpy(),
            "lineup_player_ids": [_lineup_key(lineup) for lineup in stints["home_player_ids"]],
            "game_id": stints["game_id"].to_numpy(),
            "possessions": stints["possessions"].to_numpy(dtype=float),
        }
    )
    away = pd.DataFrame(
        {
            "team_id": stints["away_team_id"].to_numpy(),
            "team_tricode": stints["away_team_tricode"].to_numpy(),
            "lineup_player_ids": [_lineup_key(lineup) for lineup in stints["away_player_ids"]],
            "game_id": stints["game_id"].to_numpy(),
            "possessions": stints["possessions"].to_numpy(dtype=float),
        }
    )
    output = (
        pd.concat([home, away], ignore_index=True)
        .groupby(["team_id", "team_tricode", "lineup_player_ids"], as_index=False, sort=False)
        .agg(possessions=("possessions", "sum"), games=("game_id", "nunique"))
    )
    output["lineup_player_ids"] = output["lineup_player_ids"].map(list)
    return output


def render_lineup_context_case_study_page(
    run_dir: Path | str,
    *,
    page_path: Path | str = DEFAULT_PAGE_PATH,
) -> Path:
    """Render the positive and negative frozen-context examples on the model page."""

    root = Path(run_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("model") != MODEL:
        raise ValueError("Lineup context page requires a matching case-study artifact")
    positive = pd.read_parquet(root / "positive_examples.parquet")
    negative = pd.read_parquet(root / "negative_examples.parquet")
    retrospective_metrics = pd.read_parquet(root / "retrospective_metrics.parquet")
    attribution = pd.read_parquet(root / "top_lineup_attribution.parquet")
    attribution_summary = pd.read_parquet(root / "top_lineup_attribution_summary.parquet")
    response_curves = pd.read_parquet(root / "response_curves.parquet")
    if set(response_curves["feature"]) != set(RESPONSE_CURVE_FEATURES):
        raise ValueError("Case-study response curves do not match the published feature set")
    response_analysis = _response_curve_analysis(response_curves)
    focal = attribution_summary.iloc[0]
    usage_p05_response = response_analysis["usage_p05_response"]
    usage_p05_difference = response_analysis["usage_p05_difference"]
    usage_p95_response = response_analysis["usage_p95_response"]
    usage_p95_difference = response_analysis["usage_p95_difference"]
    usage_focal_difference = response_analysis["usage_focal_difference"]
    usage_focal_response = response_analysis["usage_focal_response"]
    defensive_rebounds_peak_response = response_analysis["defensive_rebounds_peak_response"]
    defensive_rebounds_peak_difference = response_analysis["defensive_rebounds_peak_difference"]
    defensive_rebounds_p95_response = response_analysis["defensive_rebounds_p95_response"]
    defensive_rebounds_p95_difference = response_analysis["defensive_rebounds_p95_difference"]
    defensive_rebounds_focal_difference = response_analysis[
        "defensive_rebounds_focal_difference"
    ]
    defensive_rebounds_focal_response = response_analysis[
        "defensive_rebounds_focal_response"
    ]
    lines = [
        SECTION_START,
        "## 2025-26 Lineup Context Case Study",
        "",
        "These examples use the frozen `g_2024-25` context function and 2025-26 "
        "preseason profiles. The context scores do not use 2025-26 point outcomes, "
        "but this retrospective case study does use the five-man units, opponent "
        "matchups, games, and possession exposure that actually occurred in 2025-26. "
        "It is therefore a frozen score conditional on realized lineup allocation, "
        "not a preseason forecast of which units would play.",
        "",
        "`Actual-matchup context` is the possession-weighted correction in the unit's "
        "real 2025-26 opponents. `Standardized context` instead averages an "
        "orientation-symmetrized correction against the possession-weighted "
        f"distribution of the {int(metadata['eligible_lineup_count'])} established units "
        "in this study, excluding the unit's own team:",
        "",
        "\\[",
        "C_{\\mathrm{standard}}(L) =",
        "\\mathbb{E}_O\\left[\\frac{g(L,O)-g(O,L)}{2}\\right].",
        "\\]",
        "",
        "All values are net-rating points per 100 possessions. `Frozen additive` is "
        "the preseason player-prior difference; `frozen full` adds frozen matchup "
        "context and the realized home/road mix. Observed NetRtg is descriptive and is "
        "not used to select or fit the frozen state.",
        "",
        "### Retrospective State Update",
        "",
        "The same realized units can also be scored after 2025-26 concludes. "
        "`Completed additive` replaces the preseason player priors with the completed "
        "2025-26 player coefficients. `Completed full` additionally replaces "
        "`g_2024-25` with the completed `g_2025-26` contextual function. Both completed "
        "columns are in-sample descriptive estimates: they use 2025-26 outcomes during "
        "fitting and must not be interpreted as a second forecast.",
        "",
        "For a realized home/away stint \\(s\\), the two full estimates are:",
        "",
        "\\[",
        "\\hat y_s^{\\mathrm{frozen}} = a_{2024-25} + x_s^{\\mathsf T}\\mu_{2025-26} "
        "+ g_{2024-25}(z_s),",
        "\\qquad",
        "\\hat y_s^{\\mathrm{completed}} = a_{2025-26} + x_s^{\\mathsf T}\\hat\\beta_{2025-26} "
        "+ g_{2025-26}(z_s).",
        "\\]",
        "",
        "The tables possession-weight these signed stint estimates over each unit's "
        "realized 2025-26 matchups. Thus the difference between the two full columns "
        "is the season's combined player-state and contextual-state revision.",
        "",
        "The error summary is possession-weighted across the established units in these "
        "tables; it is a compact accounting of how the fitted season revised the frozen "
        "expectation, rather than an independent validation result.",
        "",
        *_metric_table_lines(retrospective_metrics),
        "",
        "### Worked Context Decomposition",
        "",
        f"The highest-ranked unit is the {focal.team_tricode} lineup of {focal.players} "
        f"({focal.possessions:,.0f} possessions, {int(focal.games)} games). Its frozen "
        f"standardized context effect is {focal.standardized_context_net_rating:+.2f} "
        "points per 100 possessions.",
        "",
        r"For each original feature \(k\), the spline-Ridge pipeline produces five "
        r"basis contributions. They are summed into \(q_k(L,O)\), then the displayed "
        "component is the possession-weighted orientation-symmetrized attribution:",
        "",
        "\\[",
        r"C_k(L) = \mathbb{E}_O\left[\frac{q_k(L,O)-q_k(O,L)}{2}\right],",
        "\\qquad",
        r"C_{\mathrm{standard}}(L)=\sum_k C_k(L).",
        "\\]",
        "",
        "`Focal minus reference` is the focal lineup's possession-weighted raw feature "
        "difference against the same realized 2025-26 reference units. `Contribution` "
        "is the model's nonlinear net-rating attribution, not a causal or individual-"
        "player credit. The total row exactly equals the unit's standardized context score.",
        "",
        *_attribution_table_lines(attribution),
        "",
        "### Response Curves For Diminishing-Return Candidates",
        "",
        "These curves isolate the frozen orientation-symmetrized spline component for "
        "relative usage events and relative defensive rebounds. Zero means equal focal "
        "and opponent feature values, and all other contextual features are held at zero. "
        "The blue band "
        "marks the 5th-to-95th percentile range observed when applying the model to "
        "2025-26 stints; the orange line marks the Clippers unit's focal-minus-reference "
        "contrast. A flattening curve within the blue band is evidence that the fitted "
        "contextual residual is saturating for that feature.",
        "",
        r"For a single relative feature value \(z\), the plotted response is "
        r"\(r_k(z)=[q_k(z)-q_k(-z)]/2\). This is the same orientation convention "
        r"used by the standardized-context attribution, evaluated with every other "
        r"contextual feature held at zero.",
        "",
        "<figure class=\"case-study-figure\" markdown>",
        "  ![Frozen contextual response curves for relative usage and defensive rebounds]"
        "(../assets/images/forward-contextual-rapm/context-response-curves.svg)",
        "  <figcaption>",
        "    Frozen 2024-25 orientation-symmetrized spline components. These are fitted "
        "model components, not causal partial effects.",
        "  </figcaption>",
        "</figure>",
        "",
        "#### Interpretation",
        "",
        f"For usage events, the frozen component is approximately linear across the "
        f"observed application band: it moves from "
        f"{usage_p05_response:+.2f} at the 5th-percentile difference "
        f"({usage_p05_difference:+.2f}) to {usage_p95_response:+.2f} at the "
        f"95th percentile ({usage_p95_difference:+.2f}). The Clippers unit's relative "
        f"usage contrast of {usage_focal_difference:+.2f} maps to an isolated response "
        f"of {usage_focal_response:+.2f}. "
        "This frozen model does not show strong usage saturation in its observed "
        "application range.",
        "",
        f"Defensive rebounding is more nonlinear. Within the observed band, its largest "
        f"positive component is {defensive_rebounds_peak_response:+.2f} near a relative "
        f"difference of {defensive_rebounds_peak_difference:+.2f}, then it falls to "
        f"{defensive_rebounds_p95_response:+.2f} at the 95th-percentile difference "
        f"({defensive_rebounds_p95_difference:+.2f}). The Clippers contrast is "
        f"{defensive_rebounds_focal_difference:+.2f}, with an isolated response of "
        f"{defensive_rebounds_focal_response:+.2f}. "
        "That shape is consistent with diminishing marginal contextual value, but it is "
        "not a causal estimate of rebounding value.",
        "",
        "The attribution table averages this response over each actual reference lineup, "
        "whereas the orange line evaluates it at the average raw feature contrast. With "
        r"a nonlinear spline, \(\mathbb{E}[r_k(Z)]\) need not equal "
        r"\(r_k(\mathbb{E}[Z])\), so those two displayed values need not match.",
        "",
        "### Table Definitions",
        "",
        "- `Rank`: rank by frozen standardized context, used only to select the positive "
        "and negative examples.",
        "- `Poss.` and `Games`: realized 2025-26 shared exposure for the five-player unit; "
        "both are eligibility filters, not model inputs.",
        "- `Standardized context`: frozen `g_2024-25` averaged over the established-unit "
        "opponent distribution of established lineups that actually appeared in "
        "2025-26. It answers how favorable the unit's composition is against a shared "
        "realized reference schedule, not against a synthetic average lineup.",
        "- `Frozen matchup context`: frozen `g_2024-25` averaged only over the opponents "
        "the unit actually faced in 2025-26. It is the context component of `Frozen full`.",
        "- `Frozen additive`: the signed sum of the five preseason player priors minus the "
        "five opposing preseason player priors, averaged over actual matchups.",
        "- `Frozen full`: frozen additive value, frozen matchup context, and the unit's "
        "realized home/road mix. This is the frozen forecast for its realized allocation.",
        "- `Completed additive`: the same additive calculation after replacing preseason "
        "priors with completed 2025-26 player coefficients.",
        "- `Completed matchup context`: completed `g_2025-26` averaged over the unit's "
        "actual 2025-26 opponents.",
        "- `Completed full`: completed additive value, completed matchup context, and the "
        "same realized home/road mix. It is an in-sample retrospective estimate.",
        "- `Observed NetRtg`: the unit's possession-weighted realized net rating in those "
        "same matchups. It is an outcome, never an input to the frozen forecast.",
        "",
        f"Eligibility: at least {metadata['minimum_possessions']:.0f} shared possessions "
        f"and {int(metadata['minimum_games'])} games. The tables are sortable.",
        "",
        f"Immutable case-study artifact: `{root}`.",
        "",
        "The tables are ranked **only** by `Standardized context`: the frozen, "
        "opponent-standardized contextual effect. `Frozen matchup context`, the full "
        "model estimates, and Observed NetRtg do not determine rank.",
        "",
        "### Largest Positive Frozen Context Effects",
        "",
        *_table_lines(positive),
        "",
        "### Largest Negative Frozen Context Effects",
        "",
        *_table_lines(negative),
        SECTION_END,
        "",
    ]
    page = Path(page_path)
    source = page.read_text()
    if SECTION_START not in source or SECTION_END not in source:
        raise ValueError("Forward contextual model page is missing case-study section markers")
    before, remainder = source.split(SECTION_START, maxsplit=1)
    _, after = remainder.split(SECTION_END, maxsplit=1)
    page.write_text(before + "\n".join(lines) + after)
    return page


def _observed_unit_rows(
    stints: pd.DataFrame,
    home: list[tuple[int, ...]],
    away: list[tuple[int, ...]],
    *,
    frozen_correction: np.ndarray,
    frozen_additive: np.ndarray,
    completed_correction: np.ndarray,
    completed_additive: np.ndarray,
    frozen_home_intercept: float,
    completed_home_intercept: float,
) -> pd.DataFrame:
    shared = stints.loc[:, ["game_id", "possessions", "target_home_net_rating"]].copy()
    shared["home_lineup"] = home
    shared["away_lineup"] = away
    shared["frozen_context_home"] = frozen_correction
    shared["frozen_additive_home"] = frozen_additive
    shared["completed_context_home"] = completed_correction
    shared["completed_additive_home"] = completed_additive
    shared["frozen_full_home"] = frozen_additive + frozen_correction + frozen_home_intercept
    shared["completed_full_home"] = (
        completed_additive + completed_correction + completed_home_intercept
    )
    home_rows = pd.DataFrame(
        {
            "team_id": stints["home_team_id"].to_numpy(),
            "team_tricode": stints["home_team_tricode"].to_numpy(),
            "lineup": shared["home_lineup"].to_numpy(),
            "game_id": shared["game_id"].to_numpy(),
            "possessions": shared["possessions"].to_numpy(dtype=float),
            "frozen_actual_matchup_context_net_rating": shared[
                "frozen_context_home"
            ].to_numpy(dtype=float),
            "frozen_additive_prediction_net_rating": shared[
                "frozen_additive_home"
            ].to_numpy(dtype=float),
            "frozen_full_prediction_net_rating": shared["frozen_full_home"].to_numpy(
                dtype=float
            ),
            "retrospective_actual_matchup_context_net_rating": shared[
                "completed_context_home"
            ].to_numpy(dtype=float),
            "retrospective_additive_prediction_net_rating": shared[
                "completed_additive_home"
            ].to_numpy(dtype=float),
            "retrospective_full_prediction_net_rating": shared[
                "completed_full_home"
            ].to_numpy(dtype=float),
            "observed_net_rating": shared["target_home_net_rating"].to_numpy(dtype=float),
        }
    )
    away_rows = pd.DataFrame(
        {
            "team_id": stints["away_team_id"].to_numpy(),
            "team_tricode": stints["away_team_tricode"].to_numpy(),
            "lineup": shared["away_lineup"].to_numpy(),
            "game_id": shared["game_id"].to_numpy(),
            "possessions": shared["possessions"].to_numpy(dtype=float),
            "frozen_actual_matchup_context_net_rating": -shared[
                "frozen_context_home"
            ].to_numpy(dtype=float),
            "frozen_additive_prediction_net_rating": -shared[
                "frozen_additive_home"
            ].to_numpy(dtype=float),
            "frozen_full_prediction_net_rating": -shared["frozen_full_home"].to_numpy(
                dtype=float
            ),
            "retrospective_actual_matchup_context_net_rating": -shared[
                "completed_context_home"
            ].to_numpy(dtype=float),
            "retrospective_additive_prediction_net_rating": -shared[
                "completed_additive_home"
            ].to_numpy(dtype=float),
            "retrospective_full_prediction_net_rating": -shared[
                "completed_full_home"
            ].to_numpy(dtype=float),
            "observed_net_rating": -shared["target_home_net_rating"].to_numpy(dtype=float),
        }
    )
    return pd.concat([home_rows, away_rows], ignore_index=True)


def _aggregate_units(rows: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    names = dict(zip(profiles["player_id"].astype(int), profiles["player_name"], strict=True))
    output: list[dict[str, object]] = []
    for (team_id, tricode, lineup), group in rows.groupby(
        ["team_id", "team_tricode", "lineup"], sort=False
    ):
        weights = group["possessions"].to_numpy(dtype=float)
        row: dict[str, object] = {
            "team_id": int(team_id),
            "team_tricode": str(tricode),
            "lineup_player_ids": list(lineup),
            "players": " / ".join(
                str(names.get(player_id, f"Unknown {player_id}")) for player_id in lineup
            ),
            "possessions": float(weights.sum()),
            "games": int(group["game_id"].nunique()),
        }
        for column in _UNIT_VALUE_COLUMNS:
            row[column] = float(np.average(group[column].to_numpy(dtype=float), weights=weights))
        output.append(row)
    return pd.DataFrame(output)


_UNIT_VALUE_COLUMNS = (
    "frozen_actual_matchup_context_net_rating",
    "frozen_additive_prediction_net_rating",
    "frozen_full_prediction_net_rating",
    "retrospective_actual_matchup_context_net_rating",
    "retrospective_additive_prediction_net_rating",
    "retrospective_full_prediction_net_rating",
    "observed_net_rating",
)


def _retrospective_metrics(eligible: pd.DataFrame) -> pd.DataFrame:
    """Score frozen and completed estimates on the aggregate units shown here."""

    weights = eligible["possessions"].to_numpy(dtype=float)
    target = eligible["observed_net_rating"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for label, column in (
        ("Frozen additive player state", "frozen_additive_prediction_net_rating"),
        ("Frozen full contextual state", "frozen_full_prediction_net_rating"),
        ("Completed additive player state", "retrospective_additive_prediction_net_rating"),
        ("Completed full contextual state", "retrospective_full_prediction_net_rating"),
    ):
        residual = eligible[column].to_numpy(dtype=float) - target
        rows.append(
            {
                "estimate": label,
                "weighted_rmse": float(np.sqrt(np.average(np.square(residual), weights=weights))),
                "weighted_mae": float(np.average(np.abs(residual), weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def _standardized_context(
    eligible: pd.DataFrame,
    *,
    context_model: object,
    profiles: pd.DataFrame,
) -> np.ndarray:
    lineups = [tuple(int(value) for value in lineup) for lineup in eligible["lineup_player_ids"]]
    team_ids = eligible["team_id"].to_numpy(dtype=int)
    weights = eligible["possessions"].to_numpy(dtype=float)
    scores = np.empty(len(eligible), dtype=float)
    for index, (lineup, team_id) in enumerate(zip(lineups, team_ids, strict=True)):
        opponent_mask = team_ids != team_id
        opponents = [
            candidate
            for candidate, include in zip(lineups, opponent_mask, strict=True)
            if include
        ]
        opponent_weights = weights[opponent_mask]
        forward = _context_prediction(context_model, [lineup] * len(opponents), opponents, profiles)
        reverse = _context_prediction(context_model, opponents, [lineup] * len(opponents), profiles)
        scores[index] = float(np.average((forward - reverse) / 2.0, weights=opponent_weights))
    return scores


def _top_lineup_attribution(
    eligible: pd.DataFrame,
    *,
    context_model: object,
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose the top frozen standardized-context unit into feature groups."""

    focal = eligible.iloc[0]
    focal_lineup = tuple(int(value) for value in focal["lineup_player_ids"])
    opponent_mask = eligible["team_id"].ne(int(focal["team_id"]))
    opponents = [
        tuple(int(value) for value in lineup)
        for lineup in eligible.loc[opponent_mask, "lineup_player_ids"]
    ]
    weights = eligible.loc[opponent_mask, "possessions"].to_numpy(dtype=float)
    forward_features = lineup_context_features(
        [focal_lineup] * len(opponents), opponents, profiles
    )
    reverse_features = lineup_context_features(
        opponents, [focal_lineup] * len(opponents), profiles
    )
    forward = _feature_contributions(context_model, forward_features)
    reverse = _feature_contributions(context_model, reverse_features)
    contribution = np.average((forward - reverse) / 2.0, axis=0, weights=weights)
    expected_context = float(np.sum(contribution))
    standardized = float(focal["standardized_context_net_rating"])
    if not np.isclose(expected_context, standardized, atol=1e-9):
        raise ValueError("Feature attribution does not sum to the standardized context score")
    frame = pd.DataFrame(
        {
            "feature": contextual_feature_columns(),
            "label": [_feature_label(column) for column in contextual_feature_columns()],
            "focal_minus_reference": np.average(
                forward_features.to_numpy(dtype=float), axis=0, weights=weights
            ),
            "context_contribution_net_rating": contribution,
        }
    ).sort_values("context_contribution_net_rating", ascending=False, kind="stable")
    total = pd.DataFrame(
        {
            "feature": ["total"],
            "label": ["Total standardized context"],
            "focal_minus_reference": [np.nan],
            "context_contribution_net_rating": [expected_context],
        }
    )
    summary = pd.DataFrame(
        {
            "standardized_rank": [int(focal["standardized_rank"])],
            "team_tricode": [str(focal["team_tricode"])],
            "players": [str(focal["players"])],
            "possessions": [float(focal["possessions"])],
            "games": [int(focal["games"])],
            "standardized_context_net_rating": [standardized],
        }
    )
    return pd.concat([frame, total], ignore_index=True), summary


def _feature_contributions(context_model: object, features: pd.DataFrame) -> np.ndarray:
    """Return exact per-original-feature spline Ridge contributions, excluding intercept."""

    try:
        spline = context_model.named_steps["spline"]
        scale = context_model.named_steps["scale"]
        ridge = context_model.named_steps["ridge"]
    except AttributeError as error:
        raise TypeError(
            "Context attribution requires the published spline Ridge pipeline"
        ) from error
    basis = spline.transform(features.loc[:, contextual_feature_columns()])
    scaled = scale.transform(basis)
    coefficients = np.asarray(ridge.coef_, dtype=float)
    feature_count = len(contextual_feature_columns())
    if scaled.shape[1] % feature_count or len(coefficients) != scaled.shape[1]:
        raise ValueError("Spline feature layout is incompatible with contextual attribution")
    basis_count = scaled.shape[1] // feature_count
    return (scaled * coefficients).reshape(len(features), feature_count, basis_count).sum(axis=2)


def _feature_label(column: str) -> str:
    labels = {
        "home_minus_away_three_pa_per_100": "Three-point attempt volume",
        "home_minus_away_three_pm_per_100": "Three-point makes",
        "home_minus_away_assists_per_100": "Assists",
        "home_minus_away_turnovers_per_100": "Turnovers",
        "home_minus_away_usage_per_100": "Usage events",
        "home_minus_away_offensive_rebounds_per_100": "Offensive rebounds",
        "home_minus_away_defensive_rebounds_per_100": "Defensive rebounds",
        "home_minus_away_steals_per_100": "Steals",
        "home_minus_away_blocks_per_100": "Blocks",
        "home_minus_away_bottom_two_three_pm": "Bottom-two three-point makes",
        "home_minus_away_credible_shooter_count": "Credible-shooter count",
        "home_minus_away_top_two_assists": "Top-two assists",
        "home_minus_away_usage_concentration": "Usage concentration",
        "home_minus_away_sqrt_offensive_rebounds": "Diminishing offensive rebounding",
        "home_minus_away_sqrt_defensive_rebounds": "Diminishing defensive rebounding",
        "home_minus_away_imputed_count": "Imputed-profile count",
        "home_minus_away_replacement_weight": "Replacement-profile weight",
        "home_minus_away_shooting_usage_interaction": "Shooting-by-usage",
        "home_minus_away_shooter_passing_interaction": "Shooter-by-passing",
        "home_minus_away_rebounding_usage_interaction": "Rebounding-by-usage",
    }
    try:
        return labels[column]
    except KeyError as error:
        raise ValueError(f"No display label is defined for contextual feature {column}") from error


RESPONSE_CURVE_FEATURES = (
    "home_minus_away_usage_per_100",
    "home_minus_away_defensive_rebounds_per_100",
)


def _response_curves(
    context_model: object,
    *,
    application_features: pd.DataFrame,
    focal_attribution: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate selected spline components across their observed application range."""

    columns = contextual_feature_columns()
    focal_differences = dict(
        zip(
            focal_attribution.loc[focal_attribution["feature"].ne("total"), "feature"],
            focal_attribution.loc[
                focal_attribution["feature"].ne("total"), "focal_minus_reference"
            ],
            strict=True,
        )
    )
    curves: list[pd.DataFrame] = []
    for feature in RESPONSE_CURVE_FEATURES:
        index = columns.index(feature)
        observed = application_features[feature].to_numpy(dtype=float)
        lower, upper = np.quantile(observed, [0.01, 0.99])
        span = max(float(upper - lower), 1.0)
        grid = np.linspace(min(lower, 0.0) - 0.05 * span, max(upper, 0.0) + 0.05 * span, 241)
        frame = pd.DataFrame(np.zeros((len(grid), len(columns))), columns=columns)
        frame[feature] = grid
        reverse = frame.copy()
        reverse[feature] = -grid
        contribution = (
            _feature_contributions(context_model, frame)[:, index]
            - _feature_contributions(context_model, reverse)[:, index]
        ) / 2.0
        curves.append(
            pd.DataFrame(
                {
                    "feature": feature,
                    "label": _feature_label(feature),
                    "feature_difference": grid,
                    "orientation_symmetrized_contribution_net_rating": contribution,
                    "application_q05": float(np.quantile(observed, 0.05)),
                    "application_q95": float(np.quantile(observed, 0.95)),
                    "focal_minus_reference": float(focal_differences[feature]),
                }
            )
        )
    return pd.concat(curves, ignore_index=True)


def _response_curve_analysis(curves: pd.DataFrame) -> dict[str, float]:
    """Extract the quantitative interpretation published below the response-curve chart."""

    usage = curves.loc[
        curves["feature"].eq("home_minus_away_usage_per_100")
    ].sort_values("feature_difference")
    defensive_rebounds = curves.loc[
        curves["feature"].eq("home_minus_away_defensive_rebounds_per_100")
    ].sort_values("feature_difference")
    support = defensive_rebounds.loc[
        defensive_rebounds["feature_difference"].between(
            float(defensive_rebounds["application_q05"].iloc[0]),
            float(defensive_rebounds["application_q95"].iloc[0]),
        )
    ]
    peak = support.loc[support["orientation_symmetrized_contribution_net_rating"].idxmax()]
    return {
        "usage_p05_difference": float(usage["application_q05"].iloc[0]),
        "usage_p05_response": _interpolate_response(usage, float(usage["application_q05"].iloc[0])),
        "usage_p95_difference": float(usage["application_q95"].iloc[0]),
        "usage_p95_response": _interpolate_response(usage, float(usage["application_q95"].iloc[0])),
        "usage_focal_difference": float(usage["focal_minus_reference"].iloc[0]),
        "usage_focal_response": _interpolate_response(
            usage, float(usage["focal_minus_reference"].iloc[0])
        ),
        "defensive_rebounds_peak_difference": float(peak["feature_difference"]),
        "defensive_rebounds_peak_response": float(
            peak["orientation_symmetrized_contribution_net_rating"]
        ),
        "defensive_rebounds_p95_difference": float(
            defensive_rebounds["application_q95"].iloc[0]
        ),
        "defensive_rebounds_p95_response": _interpolate_response(
            defensive_rebounds, float(defensive_rebounds["application_q95"].iloc[0])
        ),
        "defensive_rebounds_focal_difference": float(
            defensive_rebounds["focal_minus_reference"].iloc[0]
        ),
        "defensive_rebounds_focal_response": _interpolate_response(
            defensive_rebounds, float(defensive_rebounds["focal_minus_reference"].iloc[0])
        ),
    }


def _interpolate_response(frame: pd.DataFrame, value: float) -> float:
    return float(
        np.interp(
            value,
            frame["feature_difference"].to_numpy(dtype=float),
            frame["orientation_symmetrized_contribution_net_rating"].to_numpy(dtype=float),
        )
    )


def _context_prediction(
    model: object,
    home: list[tuple[int, ...]],
    away: list[tuple[int, ...]],
    profiles: pd.DataFrame,
) -> np.ndarray:
    pairs = pd.DataFrame({"home": home, "away": away}).drop_duplicates()
    values = np.asarray(
        model.predict(
            lineup_context_features(pairs["home"].tolist(), pairs["away"].tolist(), profiles)
        ),
        dtype=float,
    )
    lookup = dict(zip(zip(pairs["home"], pairs["away"], strict=True), values, strict=True))
    return np.array(
        [
            lookup[(home_lineup, away_lineup)]
            for home_lineup, away_lineup in zip(home, away, strict=True)
        ]
    )


def _lineup_key(values: object) -> tuple[int, ...]:
    lineup = tuple(int(value) for value in values)  # type: ignore[union-attr]
    if len(lineup) != 5 or len(set(lineup)) != 5:
        raise ValueError("Case study requires five unique players in every lineup")
    return lineup


def _lineup_value(lineup: tuple[int, ...], values: dict[int, float]) -> float:
    return float(sum(values.get(player_id, 0.0) for player_id in lineup))


def _table_lines(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Rank | Team | Players | Poss. | Games | Standardized context | Frozen matchup context | "
        "Frozen additive | Frozen full | Completed additive | Completed matchup context | "
        "Completed full | Observed NetRtg |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.standardized_rank} | {row.team_tricode} | {row.players} | "
            f"{row.possessions:,.0f} | {row.games} | "
            f"{row.standardized_context_net_rating:+.2f} | "
            f"{row.frozen_actual_matchup_context_net_rating:+.2f} | "
            f"{row.frozen_additive_prediction_net_rating:+.2f} | "
            f"{row.frozen_full_prediction_net_rating:+.2f} | "
            f"{row.retrospective_additive_prediction_net_rating:+.2f} | "
            f"{row.retrospective_actual_matchup_context_net_rating:+.2f} | "
            f"{row.retrospective_full_prediction_net_rating:+.2f} | "
            f"{row.observed_net_rating:+.2f} |"
        )
    return lines


def _metric_table_lines(metrics: pd.DataFrame) -> list[str]:
    lines = [
        "| Estimate | Possession-weighted RMSE | Possession-weighted MAE |",
        "| --- | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.estimate} | {row.weighted_rmse:.2f} | {row.weighted_mae:.2f} |"
        )
    return lines


def _attribution_table_lines(attribution: pd.DataFrame) -> list[str]:
    lines = [
        "| Context feature | Focal minus reference | Contribution (NetRtg / 100) |",
        "| --- | ---: | ---: |",
    ]
    for row in attribution.itertuples(index=False):
        difference = (
            "-"
            if pd.isna(row.focal_minus_reference)
            else f"{row.focal_minus_reference:+.2f}"
        )
        lines.append(
            f"| {row.label} | {difference} | {row.context_contribution_net_rating:+.2f} |"
        )
    return lines


def _render_response_curve_chart(curves: pd.DataFrame, path: Path) -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, feature in zip(axes, RESPONSE_CURVE_FEATURES, strict=True):
        frame = curves.loc[curves["feature"].eq(feature)].sort_values("feature_difference")
        if frame.empty:
            raise ValueError(f"Response curve is missing feature {feature}")
        first = frame.iloc[0]
        axis.axvspan(
            float(first.application_q05),
            float(first.application_q95),
            color="#dce9f2",
            alpha=0.8,
            label="2025-26 application P5-P95",
        )
        axis.plot(
            frame["feature_difference"],
            frame["orientation_symmetrized_contribution_net_rating"],
            color="#1e628f",
            linewidth=2.2,
            label="Frozen spline component",
        )
        axis.axhline(0.0, color="#697786", linewidth=0.9, linestyle="--")
        axis.axvline(0.0, color="#697786", linewidth=0.9, linestyle="--")
        axis.axvline(
            float(first.focal_minus_reference),
            color="#e66a25",
            linewidth=1.7,
            label="Clippers reference contrast",
        )
        axis.set(
            title=str(first.label),
            xlabel="Focal lineup minus opponent lineup",
            ylabel="Orientation-symmetrized contribution\n(NetRtg / 100)",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "nba-lineup-model", "Date": None},
    )
    plt.close(figure)


def _pyplot():
    cache_root = Path(tempfile.gettempdir()) / "nba-lineup-model-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.hashsalt"] = "nba-lineup-model"
    matplotlib.rcParams["svg.fonttype"] = "none"
    from matplotlib import pyplot as plt

    return plt


def _write_run(
    *,
    source_run_dir: Path,
    source_metadata: dict[str, object],
    target: str,
    source_season: str,
    units: dict[str, pd.DataFrame],
    minimum_possessions: float,
    minimum_games: int,
    artifacts_dir: Path,
    docs_assets_dir: Path,
) -> LineupContextCaseStudyRun:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{target}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "forward_contextual_case_study" / target
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for name, frame in units.items():
            frame.to_parquet(temporary / f"{name}.parquet", index=False)
        response_curve_chart = temporary / "context-response-curves.svg"
        _render_response_curve_chart(units["response_curves"], response_curve_chart)
        metadata = {
            "schema_version": 3,
            "run_id": run_id,
            "model": MODEL,
            "source_model": MODEL_NAME,
            "source_run_id": source_metadata["run_id"],
            "source_run_dir": str(source_run_dir),
            "target_season": target,
            "context_source_season": source_season,
            "completed_context_season": target,
            "retrospective_contract": (
                "completed target-season player coefficients and g_t score the same "
                "realized target-season matchups in-sample"
            ),
            "minimum_possessions": minimum_possessions,
            "minimum_games": minimum_games,
            "eligible_lineup_count": len(units["eligible_units"]),
            "standardization": (
                "possession-weighted established opponent units, excluding the focal team; "
                "orientation-symmetrized"
            ),
            "response_curve_contract": (
                "individual frozen orientation-symmetrized spline components; support "
                "from target-season application stints"
            ),
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        docs_assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            output / response_curve_chart.name,
            docs_assets_dir / response_curve_chart.name,
        )
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return LineupContextCaseStudyRun(output, run_id)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen forward contextual lineup case study"
    )
    parser.add_argument("--source-run-dir")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--docs-assets-dir", default=str(DEFAULT_DOCS_ASSETS_DIR))
    parser.add_argument("--minimum-possessions", type=float, default=DEFAULT_MINIMUM_POSSESSIONS)
    parser.add_argument("--minimum-games", type=int, default=DEFAULT_MINIMUM_GAMES)
    parser.add_argument("--page-path", default=str(DEFAULT_PAGE_PATH))
    args = parser.parse_args()
    run = build_lineup_context_case_study(
        source_run_dir=args.source_run_dir,
        artifacts_dir=args.artifacts_dir,
        analytical_dir=args.analytical_dir,
        docs_assets_dir=args.docs_assets_dir,
        minimum_possessions=args.minimum_possessions,
        minimum_games=args.minimum_games,
    )
    page = render_lineup_context_case_study_page(run.run_dir, page_path=args.page_path)
    print(f"Forward contextual case study: run={run.run_dir} page={page}")


if __name__ == "__main__":
    main()
