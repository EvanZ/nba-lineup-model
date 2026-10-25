from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.web_api.preseason_rankings_cache import (
    extend_frozen_preseason_catalog,
)


def test_extension_preserves_frozen_rows_and_appends_new_players():
    frozen = pd.DataFrame(
        {
            "player_id": [1, 2],
            "rapm": [3.1, -0.4],
            "team": ["GSW", "DAL"],
        }
    )
    candidate = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "rapm": [1.5, 0.2, -1.9],
            "team": ["GSW", "MEM", "DAL"],
        }
    )

    result = extend_frozen_preseason_catalog(frozen, candidate)

    assert result.to_dict("records") == [
        {"player_id": 1, "rapm": 3.1, "team": "GSW"},
        {"player_id": 2, "rapm": -0.4, "team": "DAL"},
        {"player_id": 3, "rapm": -1.9, "team": "DAL"},
    ]


def test_extension_rejects_schema_drift():
    frozen = pd.DataFrame({"player_id": [1], "rapm": [1.0]})
    candidate = pd.DataFrame({"player_id": [1], "rating": [1.0]})

    with pytest.raises(ValueError, match="different schemas"):
        extend_frozen_preseason_catalog(frozen, candidate)
