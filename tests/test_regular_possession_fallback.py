from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from nba_lineup_model.modeling.frozen_prior_evaluation import _read_regular_possessions
from nba_lineup_model.modeling.neural_data import neural_possessions_frame
from nba_lineup_model.modeling.single_lineup_possessions import (
    single_lineup_possessions_frame,
)


def test_regular_possession_reader_uses_curated_history() -> None:
    frame, manifest = _read_regular_possessions(
        "2022-23",
        analytical_dir=Path("data/analytical"),
        curated_dir=Path("data/curated"),
    )

    assert not frame.empty
    assert "possession_id" in frame
    assert "data/curated/possession_segments/2022-23/regular" in str(manifest)


def test_canonical_converter_matches_neural_converter_on_curated_segments() -> None:
    partition_dir = Path("data/curated/possession_segments/2022-23/regular")
    segments = pd.read_parquet(partition_dir)
    game_ids = segments["game_id"].drop_duplicates().head(2)
    sample = segments.loc[segments["game_id"].isin(game_ids)].copy()

    assert_frame_equal(
        single_lineup_possessions_frame(sample),
        neural_possessions_frame(sample),
    )
