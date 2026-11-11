"""Validate official NBA scoreboard responses."""

import json

def _official_fields(pairs: list[tuple]) -> dict:
    for key, value in pairs:
        field = key.casefold()
        if "espn" in field or (
            field in {"source", "source_url", "provider"}
            and isinstance(value, str)
            and "espn" in value.casefold()
        ):
            raise ValueError("Prohibited basketball source in NBA scoreboard.")
    return dict(pairs)


def validate_games(games: object, scheduled: list[dict]) -> list[dict]:
    if not isinstance(games, list) or any(
        not isinstance(game, dict) or not isinstance(game.get("gameId"), str) for game in games
    ):
        raise ValueError("NBA scoreboard has invalid game rows.")

    expected = {game["game_id"]: game for game in scheduled}
    actual = {game["gameId"]: game for game in games}
    if len(actual) != len(games):
        raise ValueError("NBA scoreboard contains duplicate game IDs.")
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"NBA scoreboard differs from the Pacific slate. Missing games {missing}. Unexpected games {extra}.")

    for game_id, game in actual.items():
        if (
            type(game.get("gameStatus")) is not int or game["gameStatus"] not in {1, 2, 3}
            or type(game.get("period")) is not int or game["period"] < 0
            or not isinstance(game.get("gameClock"), str)
        ):
            raise ValueError(f"Invalid scoreboard status, clock, or period for {game_id}.")
        for side in ("home", "away"):
            team = game.get(f"{side}Team")
            if not isinstance(team, dict) or team.get("teamId") != expected[game_id][f"{side}_team_id"]:
                raise ValueError(f"Scoreboard {side} team differs from the official schedule for {game_id}.")
            if type(team.get("score")) is not int or team["score"] < 0:
                raise ValueError(f"Invalid {side} score for {game_id}.")
    return games


def parse_scoreboard(payload: bytes) -> dict:
    document = json.loads(payload, object_pairs_hook=_official_fields)
    board = document.get("scoreboard") if isinstance(document, dict) else None
    if not isinstance(board, dict) or not isinstance(board.get("gameDate"), str):
        raise ValueError("NBA scoreboard is missing its game date.")
    return board
