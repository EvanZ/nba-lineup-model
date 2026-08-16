"""Forward RAPM with HPM's additive profile signals moved into the player prior."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba_lineup_model.modeling.box_score_prior import build_box_score_prior_features
from nba_lineup_model.modeling.box_score_rapm import DEFAULT_REGULARIZATION_GRID
from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.modeling.forward_box_score_hpm import ForwardBoxScoreResidualPriorBuilder
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_TARGET_SEASON,
)
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    ForwardPortableMatchupContextualRapmRun,
    train_forward_portable_matchup_contextual_rapm,
)

MODEL_NAME = "forward_additive_profile_prior_rapm"
RUN_PREFIX = "forward-additive-profile-prior-rapm"

# This is deliberately the additive portion of the HPM x3 lineup contract.
# The prior receives individual player equivalents of the former lineup sums;
# depth, concentration, interactions, imputation counts, and replacement
# weights are excluded because they are not additive player properties.
ADDITIVE_PROFILE_COLUMNS = (
    "three_pa_per_100",
    "three_pm_per_100",
    "assists_per_100",
    "turnovers_per_100",
    "usage_per_100",
    "offensive_rebound_pct",
    "steals_per_100",
    "blocks_per_100",
)
ADDITIVE_PROFILE_PRIOR_COLUMNS = tuple(
    f"prior_context_{column}" for column in ADDITIVE_PROFILE_COLUMNS
)


def build_additive_profile_prior_features(
    panel: pd.DataFrame,
    *,
    analytical_dir: Path | str,
    curated_dir: Path | str,
) -> pd.DataFrame:
    """Create leakage-safe, HPM-identical additive player-profile features.

    The profile builder uses only information preceding each target season and
    applies the same rate shrinkage and rebound-claim construction used by the
    HPM x3 context contract. It is evaluated for every player in the target
    player panel, not merely realized target-season stints.
    """

    base, _, _ = build_box_score_prior_features(panel)
    profiles: list[pd.DataFrame] = []
    for season, target in base.groupby("target_season", sort=False):
        player_ids = target["player_id"].astype(int).tolist()
        profile = build_contextual_player_profiles(
            panel,
            target_season=str(season),
            target_player_ids=player_ids,
            analytical_dir=str(analytical_dir),
            curated_dir=str(curated_dir),
        )
        missing = set(ADDITIVE_PROFILE_COLUMNS) - set(profile)
        if missing:
            raise ValueError(f"Contextual profile is missing additive columns: {sorted(missing)}")
        renamed_columns = {
            column: f"prior_context_{column}" for column in ADDITIVE_PROFILE_COLUMNS
        }
        profiles.append(
            profile.loc[:, ["player_id", *ADDITIVE_PROFILE_COLUMNS]]
            .rename(columns=renamed_columns)
            .assign(target_season=str(season))
        )
    profile_features = pd.concat(profiles, ignore_index=True)
    output = base.merge(
        profile_features,
        on=["target_season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    if output.loc[:, ADDITIVE_PROFILE_PRIOR_COLUMNS].isna().any().any():
        raise ValueError("Additive prior profiles must cover every target player")
    return output


def train_forward_additive_profile_prior_rapm(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Fit the additive-profile prior as a controlled no-context RAPM model."""

    panel = pd.read_parquet(player_season_panel_path)
    features = build_additive_profile_prior_features(
        panel,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    builder = ForwardBoxScoreResidualPriorBuilder(
        features=features,
        regularization_grid=regularization_grid,
        feature_columns=ADDITIVE_PROFILE_PRIOR_COLUMNS,
        prior_method_suffix="lagged_hpm_additive_profile_residual",
    )
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        use_context=False,
        player_prior_builder=builder,
        player_prior_description=(
            "prior-season centered value-conditioned aging/exposure-gated RAPM prior plus "
            "a strictly lagged residual model using HPM x3's additive player profiles; "
            "controlled no-context ablation"
        ),
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train no-context RAPM with additive HPM profiles in the player prior"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    args = parser.parse_args()
    run = train_forward_additive_profile_prior_rapm(through_season=args.through_season)
    print(f"Additive-profile prior RAPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
