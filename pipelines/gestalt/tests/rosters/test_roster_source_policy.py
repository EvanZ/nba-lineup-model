from datetime import date

from gestalt_pipelines.rosters.utils import NBA_TEAM_IDS, build_rosters, opening_ids, box_roster_ids


def test_ids_without_names_are_valid_roster_members():
    opening = {
        "date": date(2026, 11, 7),
        "teams": {team: [index + 1] for index, team in enumerate(NBA_TEAM_IDS)},
    }
    result = build_rosters(
        season="2026-27", day=date(2026, 11, 8), opening=opening,
        transactions=[{
            "Transaction_Type": "Signing", "TRANSACTION_DATE": "2026-11-08T00:00:00",
            "PLAYER_ID": 999999, "TEAM_ID": NBA_TEAM_IDS[0],
        }], boxes=[], inputs={},
    )
    assert result["teams"][str(NBA_TEAM_IDS[0])] == [1, 999999]
    assert len(result["teams"]) == 30


def test_source_parsers_require_ids_but_not_name_fields():
    raw = b'{"resultSets":[{"name":"CommonTeamRoster","headers":["TeamID","PLAYER_ID"],"rowSet":[[1,123]]}]}'
    assert opening_ids(raw, 1) == [123]
    game = {"game_id": "0022600200", "home_team_id": 1, "away_team_id": 2}
    raw = b'{"game":{"gameId":"0022600200","gameStatus":3,"homeTeam":{"teamId":1,"players":[{"personId":123}]},"awayTeam":{"teamId":2,"players":[{"personId":456}]}}}'
    assert box_roster_ids(raw, game) == {1: [123], 2: [456]}
