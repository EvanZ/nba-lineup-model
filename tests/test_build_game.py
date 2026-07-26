from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nba_lineup_model.build_game import process_game_payloads

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_process_game_payloads_writes_parquet_tables(tmp_path: Path):
    result = process_game_payloads(
        load_fixture("playbyplay_lineup_scenario.json"),
        load_fixture("boxscore_lineup_scenario.json"),
        output_root=tmp_path,
    )

    assert result.event_count == 13
    assert result.stint_count == 3
    assert result.possession_count == 2
    assert result.possession_segment_count == 3
    assert result.issue_count == 0
    assert set(result.output_paths) == {
        "events",
        "players",
        "event_lineups",
        "lineup_stints",
        "possessions",
        "possession_segments",
    }
    assert all(path.exists() for path in result.output_paths.values())
    events = pd.read_parquet(result.output_paths["events"])
    assert len(events) == 13
    assert str(events["team_id"].dtype) == "Int64"
    assert events.loc[1, "clock"] == "10:01.00"
    assert events.loc[1, "source_clock"] == "PT10M01.00S"
    assert len(pd.read_parquet(result.output_paths["lineup_stints"])) == 3
    possessions = pd.read_parquet(result.output_paths["possessions"])
    segments = pd.read_parquet(result.output_paths["possession_segments"])
    assert len(possessions) == 2
    assert possessions.loc[0, "lineup_segment_count"] == 2
    assert len(segments) == 3
    assert segments.groupby("possession_index")["points_home"].sum().tolist() == [2, 0]

    schema = pq.read_schema(result.output_paths["events"])
    for column in (
        "event_index",
        "source_action_number",
        "source_order_number",
        "team_id",
        "player_id",
        "source_possession_team_id",
    ):
        assert schema.field(column).type == pa.int64()

    possession_schema = pq.read_schema(result.output_paths["possessions"])
    for column in (
        "possession_index",
        "offense_team_id",
        "defense_team_id",
        "start_order_number",
        "end_order_number",
    ):
        assert possession_schema.field(column).type == pa.int64()
