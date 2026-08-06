from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.replacement_token import (
    REPLACEMENT_TOKEN_ID,
    fit_replacement_token_season,
    tokenize_replacement_lineups,
)


def test_tokenization_retains_the_number_of_replacement_players_in_a_lineup() -> None:
    stints = _stints()

    tokenized = tokenize_replacement_lineups(stints, {1, 2, 9})

    assert tokenized.loc[0, "home_player_ids"].count(REPLACEMENT_TOKEN_ID) == 2
    assert tokenized.loc[0, "away_player_ids"].count(REPLACEMENT_TOKEN_ID) == 1


def test_pooled_token_recovers_a_shared_player_effect() -> None:
    stints = pd.DataFrame(
        {
            "home_player_ids": [[1, 10, 11, 12, 13], [10, 11, 12, 13, 14]],
            "away_player_ids": [[20, 21, 22, 23, 24], [2, 20, 21, 22, 23]],
            "possessions": [100.0, 100.0],
            "target_home_net_rating": [-2.0, 2.0],
        }
    )

    coefficient, _ = fit_replacement_token_season(
        stints,
        replacement_player_ids={1, 2},
        regularization=0.0,
    )

    assert coefficient == pytest.approx(-2.0)


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_player_ids": [[1, 2, 3, 4, 5]],
            "away_player_ids": [[9, 10, 11, 12, 13]],
            "possessions": [100.0],
            "target_home_net_rating": [0.0],
        }
    )
