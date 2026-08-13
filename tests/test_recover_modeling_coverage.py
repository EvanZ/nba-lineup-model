from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from nba_lineup_model.flows.recover_modeling_coverage import (
    unmodeled_regular_game_ids,
)


def test_selects_only_catalog_games_absent_from_rapm_stints(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "nba_lineup_model.flows.recover_modeling_coverage.read_game_catalog",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        "nba_lineup_model.flows.recover_modeling_coverage.select_catalog_games",
        lambda *_args, **_kwargs: [
            SimpleNamespace(game_id="0021700001"),
            SimpleNamespace(game_id="0021700002"),
            SimpleNamespace(game_id="0021700003"),
        ],
    )
    stint_dir = tmp_path / "rapm_stints" / "2017-18" / "regular"
    stint_dir.mkdir(parents=True)
    pd.DataFrame({"game_id": ["0021700001", "0021700003"]}).to_parquet(
        stint_dir / "part-00000.parquet",
        index=False,
    )

    actual = unmodeled_regular_game_ids(
        "2017-18",
        catalog_path=tmp_path / "games.parquet",
        analytical_dir=tmp_path,
    )

    assert actual == ["0021700002"]
