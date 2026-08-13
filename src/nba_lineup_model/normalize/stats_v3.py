from __future__ import annotations

import copy
import math
import re
import unicodedata
from collections.abc import Mapping
from itertools import combinations
from typing import Any

SUBSTITUTION_RE = re.compile(
    r"^SUB:\s*(?P<incoming>.+?)\s+FOR\s+(?P<outgoing>.+?)$",
    re.IGNORECASE,
)
JUMP_BALL_TIP_RE = re.compile(r"\btip to\s+(?P<recipient>.+?)\s*$", re.IGNORECASE)
FREE_THROW_SEQUENCE_RE = re.compile(
    r"(?P<attempt>\d+)\s+of\s+(?P<total>\d+)",
    re.IGNORECASE,
)
# The box score uses the later legal name while older substitutions retain Kanter.
HISTORICAL_NAME_PERSON_IDS = {
    "kanter": 202683,
}


def adapt_stats_v3_game(
    play_by_play_payload: Mapping[str, Any],
    boxscore_payload: Mapping[str, Any],
    game_rotation_payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt NBA Stats V3 documents to the existing liveData-shaped boundary."""

    boxscore = adapt_stats_v3_boxscore(boxscore_payload)
    play_by_play = adapt_stats_v3_play_by_play(
        play_by_play_payload,
        boxscore,
        game_rotation_payload=game_rotation_payload,
    )
    return play_by_play, boxscore


def adapt_stats_v3_boxscore(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ``boxscoretraditionalv3`` to a liveData-shaped box score."""

    source = payload.get("boxScoreTraditional")
    if not isinstance(source, Mapping):
        raise ValueError("Expected payload['boxScoreTraditional'] to be an object")

    game_id = source.get("gameId")
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("Expected a non-empty Stats V3 box score gameId")

    game: dict[str, Any] = {"gameId": game_id, "gameStatus": 3}
    for side in ("homeTeam", "awayTeam"):
        source_team = source.get(side)
        if not isinstance(source_team, Mapping):
            raise ValueError(f"Expected Stats V3 box score {side} to be an object")
        game[side] = _adapt_team(source_team)
    return {"game": game}


def adapt_stats_v3_play_by_play(
    payload: Mapping[str, Any],
    boxscore_payload: Mapping[str, Any],
    *,
    game_rotation_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert ``playbyplayv3`` actions to the liveData action vocabulary."""

    source_game = payload.get("game")
    if not isinstance(source_game, Mapping):
        raise ValueError("Expected payload['game'] to be an object")
    game_id = source_game.get("gameId")
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("Expected a non-empty Stats V3 play-by-play gameId")

    box_game = boxscore_payload.get("game")
    if not isinstance(box_game, Mapping):
        raise ValueError("Expected adapted boxscore payload['game'] to be an object")
    if box_game.get("gameId") != game_id:
        raise ValueError("Stats V3 play-by-play and box score game IDs differ")

    source_actions = source_game.get("actions")
    if not isinstance(source_actions, list):
        raise ValueError("Expected Stats V3 game actions to be a list")
    source_actions = _with_missing_period_starts(source_actions)
    ordered_source_actions = sorted(
        source_actions,
        key=_source_action_sort_key,
    )
    ordered_source_actions = _forward_fill_placeholder_scores(ordered_source_actions)

    rosters, lineups, team_tricodes = _roster_context(box_game)
    source_aliases = _source_player_aliases(ordered_source_actions, rosters)
    rotation_lineups = _rotation_period_lineups(
        game_rotation_payload,
        periods={
            _required_int(action, "period")
            for action in ordered_source_actions
            if isinstance(action, Mapping)
        },
        rosters=rosters,
    )
    period_lineups = _infer_period_lineups(
        ordered_source_actions,
        rosters,
        lineups,
        source_aliases,
        rotation_lineups=rotation_lineups,
    )
    actions: list[dict[str, Any]] = []
    pending_miss_team_id: int | None = None

    for source_action in ordered_source_actions:
        if not isinstance(source_action, Mapping):
            raise ValueError("Expected every Stats V3 action to be an object")
        action_type = str(source_action.get("actionType") or "")

        if (
            action_type.casefold() == "period"
            and str(source_action.get("subType") or "").casefold() == "start"
        ):
            action, pending_miss_team_id = _adapt_regular_action(
                source_action,
                rosters=rosters,
                team_tricodes=team_tricodes,
                pending_miss_team_id=pending_miss_team_id,
            )
            actions.append(action)
            period = _required_int(source_action, "period")
            if period > 1:
                actions.extend(
                    _period_boundary_substitutions(
                        source_action,
                        current_lineups=lineups,
                        target_lineups=period_lineups[period],
                        rosters=rosters,
                        team_tricodes=team_tricodes,
                    )
                )
            continue

        if action_type.casefold() == "substitution":
            out_action, in_action = _adapt_substitution(
                source_action,
                rosters=rosters,
                lineups=lineups,
                team_tricodes=team_tricodes,
                source_aliases=source_aliases,
            )
            actions.extend((out_action, in_action))
            continue

        action, pending_miss_team_id = _adapt_regular_action(
            source_action,
            rosters=rosters,
            team_tricodes=team_tricodes,
            pending_miss_team_id=pending_miss_team_id,
        )
        actions.append(action)

    return {
        "game": {
            "gameId": game_id,
            "actions": actions,
        }
    }


def _forward_fill_placeholder_scores(
    source_actions: list[Any],
) -> list[dict[str, Any]]:
    """Normalize known historical score placeholders to a monotone scoreboard.

    Some older Stats V3 games attach a score from a later administrative replay
    record to an earlier clock position. Other non-scoring records revert one
    side to zero. Neither is a real scoreboard transition. Keeping the last
    nondecreasing score prevents those administrative rows from manufacturing
    negative score deltas and corrupting the subsequent possession sequence.
    """

    scores = {"scoreHome": 0, "scoreAway": 0}
    adapted_actions: list[dict[str, Any]] = []
    for action in source_actions:
        if not isinstance(action, Mapping):
            raise ValueError("Expected every Stats V3 action to be an object")
        adapted = copy.deepcopy(dict(action))
        for field, prior_score in scores.items():
            raw_score = _nonnegative_score(adapted.get(field), field)
            if raw_score is None or raw_score < prior_score:
                adapted[field] = prior_score
            else:
                scores[field] = raw_score
        adapted_actions.append(adapted)
    return adapted_actions


def _with_missing_period_starts(source_actions: list[Any]) -> list[Any]:
    """Supply the period-start record omitted by a small historical feed subset."""

    periods = sorted(
        {
            _required_int(action, "period")
            for action in source_actions
            if isinstance(action, Mapping)
        }
    )
    present_starts = {
        _required_int(action, "period")
        for action in source_actions
        if isinstance(action, Mapping)
        and str(action.get("actionType") or "").casefold() == "period"
        and str(action.get("subType") or "").casefold() == "start"
    }
    synthetic_starts = [
        {
            "actionNumber": 0,
            "actionId": 0,
            "clock": "PT12M00.00S" if period <= 4 else "PT05M00.00S",
            "period": period,
            "teamId": 0,
            "teamTricode": "",
            "personId": 0,
            "playerName": "",
            "playerNameI": "",
            "description": f"Synthetic start of period {period}",
            "actionType": "period",
            "subType": "start",
            "isFieldGoal": 0,
            "shotValue": 0,
            "scoreHome": "",
            "scoreAway": "",
        }
        for period in periods
        if period not in present_starts
    ]
    return [*source_actions, *synthetic_starts]


def _nonnegative_score(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Expected non-negative integer {field}, got {value!r}")
    if isinstance(value, int):
        score = value
    elif isinstance(value, str) and value.isdigit():
        score = int(value)
    else:
        raise ValueError(f"Expected non-negative integer {field}, got {value!r}")
    if score < 0:
        raise ValueError(f"Expected non-negative integer {field}, got {value!r}")
    return score


def _adapt_regular_action(
    source_action: Mapping[str, Any],
    *,
    rosters: Mapping[int, Mapping[int, Mapping[str, Any]]],
    team_tricodes: Mapping[int, str],
    pending_miss_team_id: int | None,
) -> tuple[dict[str, Any], int | None]:
    source_type = str(source_action.get("actionType") or "")
    description = str(source_action.get("description") or "")
    action = _base_action(source_action, order_number=_source_order_base(source_action))
    normalized_type, subtype, descriptor = _action_classification(
        source_action,
        description=description,
    )
    action["actionType"] = normalized_type
    action["subType"] = subtype
    action["descriptor"] = descriptor

    team_id = _team_id(source_action, rosters)
    if normalized_type == "jumpball":
        team_id = _jump_ball_recovery_team_id(
            description,
            rosters=rosters,
            fallback_team_id=team_id,
        )
    action["teamId"] = team_id
    if team_id and not action.get("teamTricode"):
        action["teamTricode"] = team_tricodes.get(team_id, "")
    if _required_int(source_action, "personId") in rosters:
        action["personId"] = 0
        action["personIdsFilter"] = []

    action["possession"] = _possession_team_id(
        normalized_type,
        subtype=action["subType"],
        team_id=team_id,
        team_ids=set(rosters),
    )

    if normalized_type in {"2pt", "3pt"}:
        action["shotResult"] = "Made" if source_type.casefold() == "made shot" else "Missed"
        pending_miss_team_id = (team_id or None) if action["shotResult"] == "Missed" else None
    elif normalized_type == "freethrow":
        action["shotResult"] = "Missed" if description.upper().startswith("MISS ") else "Made"
        if action["shotResult"] == "Missed":
            pending_miss_team_id = team_id or None
    elif normalized_type == "rebound":
        action["subType"] = (
            "offensive"
            if pending_miss_team_id is not None and team_id == pending_miss_team_id
            else "defensive"
        )
        pending_miss_team_id = None
    elif normalized_type in {"period", "turnover"}:
        pending_miss_team_id = None

    if normalized_type == "jumpball":
        action["subType"] = "recovered"
        action["possession"] = team_id
    return action, pending_miss_team_id


def _adapt_team(source_team: Mapping[str, Any]) -> dict[str, Any]:
    players = source_team.get("players")
    if not isinstance(players, list):
        raise ValueError("Expected Stats V3 team players to be a list")

    starter_ids = _stats_v3_starter_ids(players)
    adapted_players: list[dict[str, Any]] = []
    for source_player in players:
        if not isinstance(source_player, Mapping):
            raise ValueError("Expected every Stats V3 player to be an object")
        player = copy.deepcopy(dict(source_player))
        player_id = _required_int(player, "personId")
        player["starter"] = "1" if player_id in starter_ids else "0"
        statistics = player.get("statistics")
        if isinstance(statistics, Mapping):
            player["statistics"] = _adapt_statistics(statistics)
        adapted_players.append(player)

    team = copy.deepcopy(dict(source_team))
    team["players"] = adapted_players
    statistics = team.get("statistics")
    if isinstance(statistics, Mapping):
        team["statistics"] = _adapt_statistics(statistics)
        points = statistics.get("points")
        if isinstance(points, int) and not isinstance(points, bool):
            team["score"] = points
    return team


def _stats_v3_starter_ids(players: list[Any]) -> set[int]:
    """Resolve the two starter encodings used by the Stats V3 box archive."""

    position_starters = {
        _required_int(player, "personId")
        for player in players
        if isinstance(player, Mapping) and str(player.get("position") or "")
    }
    if len(position_starters) == 5:
        return position_starters

    # Historical boxes list a position for many reserve players, but retain the
    # traditional box-score ordering of five starters followed by the bench.
    player_ids = [
        _required_int(player, "personId") for player in players if isinstance(player, Mapping)
    ]
    if len(player_ids) < 5:
        raise ValueError("Stats V3 box score has fewer than five listed players")
    return set(player_ids[:5])


def _adapt_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(dict(statistics))
    minutes = adapted.get("minutes")
    if isinstance(minutes, str) and minutes:
        adapted["minutes"] = _minutes_to_iso(minutes)
    return adapted


def _minutes_to_iso(value: str) -> str:
    if value.startswith("PT"):
        return value
    if value.isdigit():
        return f"PT{int(value)}M00.00S"
    match = re.fullmatch(r"(?P<minutes>\d+):(?P<seconds>\d{2})", value)
    if match is None:
        raise ValueError(f"Unexpected Stats V3 minutes value: {value!r}")
    return f"PT{int(match.group('minutes'))}M{int(match.group('seconds')):02d}.00S"


def _roster_context(
    box_game: Mapping[str, Any],
) -> tuple[
    dict[int, dict[int, Mapping[str, Any]]],
    dict[int, set[int]],
    dict[int, str],
]:
    rosters: dict[int, dict[int, Mapping[str, Any]]] = {}
    lineups: dict[int, set[int]] = {}
    team_tricodes: dict[int, str] = {}
    for side in ("homeTeam", "awayTeam"):
        team = box_game.get(side)
        if not isinstance(team, Mapping):
            raise ValueError(f"Expected adapted box score {side} to be an object")
        team_id = _required_int(team, "teamId")
        players = team.get("players")
        if not isinstance(players, list):
            raise ValueError(f"Expected adapted box score {side} players to be a list")
        roster: dict[int, Mapping[str, Any]] = {}
        starters: set[int] = set()
        for player in players:
            if not isinstance(player, Mapping):
                continue
            player_id = _required_int(player, "personId")
            roster[player_id] = player
            if str(player.get("starter")) == "1":
                starters.add(player_id)
        if len(starters) != 5:
            raise ValueError(f"Expected exactly five Stats V3 starters for team {team_id}")
        rosters[team_id] = roster
        lineups[team_id] = starters
        team_tricodes[team_id] = str(team.get("teamTricode") or "")
    return rosters, lineups, team_tricodes


def _source_player_aliases(
    source_actions: list[Any],
    rosters: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[int, dict[str, set[int]]]:
    aliases: dict[int, dict[str, set[int]]] = {team_id: {} for team_id in rosters}
    for action in source_actions:
        if not isinstance(action, Mapping):
            continue
        team_id = action.get("teamId")
        player_id = action.get("personId")
        if (
            isinstance(team_id, bool)
            or not isinstance(team_id, int)
            or isinstance(player_id, bool)
            or not isinstance(player_id, int)
            or team_id not in rosters
            or player_id not in rosters[team_id]
        ):
            continue
        source_names = [
            str(action.get("playerName") or ""),
            str(action.get("playerNameI") or ""),
        ]
        if str(action.get("actionType") or "").casefold() == "substitution":
            source_names.append(_substitution_outgoing_name(action))
        for source_name in source_names:
            normalized = _normalize_name(source_name)
            if normalized:
                aliases[team_id].setdefault(normalized, set()).add(player_id)
    return aliases


def _infer_period_lineups(
    source_actions: list[Any],
    rosters: Mapping[int, Mapping[int, Mapping[str, Any]]],
    first_period_lineups: Mapping[int, set[int]],
    source_aliases: Mapping[int, Mapping[str, set[int]]],
    *,
    rotation_lineups: Mapping[int, Mapping[int, set[int]]] | None = None,
) -> dict[int, dict[int, set[int]]]:
    periods = sorted(
        {
            _required_int(action, "period")
            for action in source_actions
            if isinstance(action, Mapping)
        }
    )
    if not periods or periods[0] != 1:
        raise ValueError("Stats V3 play-by-play does not begin in period 1")

    result: dict[int, dict[int, set[int]]] = {}
    prior_end = {team_id: set(lineup) for team_id, lineup in first_period_lineups.items()}
    original_starters = {team_id: set(lineup) for team_id, lineup in first_period_lineups.items()}

    for period in periods:
        period_actions = [
            action
            for action in source_actions
            if isinstance(action, Mapping) and _required_int(action, "period") == period
        ]
        starts: dict[int, set[int]] = {}
        for team_id, roster in rosters.items():
            starts[team_id] = (
                set(original_starters[team_id])
                if period == 1
                else _infer_or_solve_period_lineup(
                    period_actions,
                    team_id=team_id,
                    roster=roster,
                    prior_lineup=prior_end[team_id],
                    original_starters=original_starters[team_id],
                    source_aliases=source_aliases[team_id],
                    rotation_lineup=(rotation_lineups or {})
                    .get(period, {})
                    .get(team_id),
                )
            )

        result[period] = starts
        prior_end = {team_id: set(lineup) for team_id, lineup in starts.items()}
        for action in period_actions:
            if str(action.get("actionType") or "").casefold() != "substitution":
                continue
            team_id = _required_int(action, "teamId")
            if team_id not in rosters:
                continue
            outgoing_id = _required_int(action, "personId")
            incoming_name = _substitution_incoming_name(action)
            incoming_id = _resolve_player_name(
                incoming_name,
                roster=rosters[team_id],
                current_lineup=prior_end[team_id],
                source_aliases=source_aliases[team_id],
            )
            prior_end[team_id].discard(outgoing_id)
            prior_end[team_id].add(incoming_id)
    return result


def _infer_or_solve_period_lineup(
    period_actions: list[Mapping[str, Any]],
    *,
    team_id: int,
    roster: Mapping[int, Mapping[str, Any]],
    prior_lineup: set[int],
    original_starters: set[int],
    source_aliases: Mapping[str, set[int]],
    rotation_lineup: set[int] | None = None,
) -> set[int]:
    """Use the fast inference when valid, then solve ambiguous starts exactly."""

    if rotation_lineup is not None:
        if _period_lineup_matches_evidence(
            rotation_lineup,
            period_actions,
            team_id=team_id,
            roster=roster,
            source_aliases=source_aliases,
        ):
            return set(rotation_lineup)
        raise ValueError(
            f"Game Rotation period lineup contradicts play-by-play evidence for team "
            f"{team_id}"
        )

    try:
        inferred = _infer_team_period_lineup(
            period_actions,
            team_id=team_id,
            roster=roster,
            prior_lineup=prior_lineup,
            original_starters=original_starters,
            source_aliases=source_aliases,
        )
    except ValueError:
        inferred = None
    if inferred is not None and _period_lineup_matches_evidence(
        inferred,
        period_actions,
        team_id=team_id,
        roster=roster,
        source_aliases=source_aliases,
    ):
        return inferred

    candidates = [
        set(lineup)
        for lineup in combinations(sorted(roster), 5)
        if _period_lineup_matches_evidence(
            set(lineup),
            period_actions,
            team_id=team_id,
            roster=roster,
            source_aliases=source_aliases,
        )
    ]
    if not candidates:
        raise ValueError(f"No legal period lineup can be inferred for team {team_id}")
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda lineup: (len(lineup ^ prior_lineup), sorted(lineup)))
    if len(candidates) > 1 and len(candidates[0] ^ prior_lineup) == len(
        candidates[1] ^ prior_lineup
    ):
        raise ValueError(
            f"Period lineup remains ambiguous for team {team_id}: {len(candidates)} legal states"
        )
    return candidates[0]


def _period_lineup_matches_evidence(
    starting_lineup: set[int],
    period_actions: list[Mapping[str, Any]],
    *,
    team_id: int,
    roster: Mapping[int, Mapping[str, Any]],
    source_aliases: Mapping[str, set[int]],
) -> bool:
    lineup = set(starting_lineup)
    first_substitution_index = next(
        (
            index
            for index, action in enumerate(period_actions)
            if str(action.get("actionType") or "").casefold() == "substitution"
            and _required_int(action, "teamId") == team_id
        ),
        len(period_actions),
    )
    for index, action in enumerate(period_actions):
        action_team_id = _team_id(action, {team_id: roster})
        if action_team_id != team_id:
            continue
        if str(action.get("actionType") or "").casefold() == "substitution":
            outgoing_id = _required_int(action, "personId")
            if outgoing_id not in lineup:
                return False
            try:
                incoming_id = _resolve_player_name(
                    _substitution_incoming_name(action),
                    roster=roster,
                    current_lineup=lineup,
                    source_aliases=source_aliases,
                )
            except ValueError:
                return False
            if incoming_id in lineup:
                return False
            lineup.remove(outgoing_id)
            lineup.add(incoming_id)
            continue
        person_id = _required_int(action, "personId")
        if (
            person_id in roster
            and _actor_implies_on_court(action)
            and index < first_substitution_index
            and person_id not in lineup
        ):
            return False
    return True


def _infer_team_period_lineup(
    period_actions: list[Mapping[str, Any]],
    *,
    team_id: int,
    roster: Mapping[int, Mapping[str, Any]],
    prior_lineup: set[int],
    original_starters: set[int],
    source_aliases: Mapping[str, set[int]],
) -> set[int]:
    first_substitution_index = next(
        (
            index
            for index, action in enumerate(period_actions)
            if str(action.get("actionType") or "").casefold() == "substitution"
            and _required_int(action, "teamId") == team_id
        ),
        None,
    )
    first_substitution_clock = (
        str(period_actions[first_substitution_index].get("clock") or "")
        if first_substitution_index is not None
        else None
    )
    first_transactions: dict[int, tuple[int, str]] = {}
    later_incoming_ids: set[int] = set()
    first_actor_index: dict[int, int] = {}

    for index, action in enumerate(period_actions):
        action_team_id = _team_id(action, {team_id: roster})
        person_id = _required_int(action, "personId")
        if (
            action_team_id == team_id
            and person_id in roster
            and _actor_implies_on_court(action)
            and (first_substitution_index is None or index < first_substitution_index)
        ):
            first_actor_index.setdefault(person_id, index)
        if (
            str(action.get("actionType") or "").casefold() != "substitution"
            or _required_int(action, "teamId") != team_id
        ):
            continue
        outgoing_id = person_id
        incoming_id = _resolve_player_name(
            _substitution_incoming_name(action),
            roster=roster,
            current_lineup={outgoing_id},
            source_aliases=source_aliases,
        )
        later_incoming_ids.add(incoming_id)
        if (
            first_substitution_clock is None
            or str(action.get("clock") or "") != first_substitution_clock
        ):
            continue
        first_transactions.setdefault(outgoing_id, (index, "out"))
        first_transactions.setdefault(incoming_id, (index, "in"))

    required_on = {
        player_id for player_id, (_, direction) in first_transactions.items() if direction == "out"
    }
    required_on.update(
        player_id
        for player_id, actor_index in first_actor_index.items()
        if player_id not in first_transactions or actor_index < first_transactions[player_id][0]
    )
    required_off = {
        player_id
        for player_id, (_, direction) in first_transactions.items()
        if direction == "in" and player_id not in required_on
    }
    required_off.update(later_incoming_ids - required_on)
    if len(required_on) > 5:
        raise ValueError(
            f"Period lineup evidence requires more than five players for team {team_id}: "
            f"{sorted(required_on)}"
        )

    lineup = set(required_on)
    actor_order = [
        player_id for player_id, _ in sorted(first_actor_index.items(), key=lambda item: item[1])
    ]
    for candidates in (
        sorted(prior_lineup),
        sorted(original_starters),
        actor_order,
        sorted(roster),
    ):
        for player_id in candidates:
            if len(lineup) == 5:
                break
            if player_id not in required_off:
                lineup.add(player_id)
    if len(lineup) != 5:
        raise ValueError(f"Could not infer five period starters for team {team_id}")
    return lineup


def _rotation_period_lineups(
    payload: Mapping[str, Any] | None,
    *,
    periods: set[int],
    rosters: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[int, dict[int, set[int]]]:
    """Return exact-five period starters supported by cached Game Rotation data.

    ``IN_TIME_REAL`` and ``OUT_TIME_REAL`` are elapsed-game tenths of a second.
    Incomplete, malformed, or non-five-player intervals are intentionally ignored
    so the existing play-by-play inference remains the fallback.
    """

    if payload is None:
        return {}
    result_sets = payload.get("resultSets")
    if not isinstance(result_sets, list):
        return {}

    intervals_by_team: dict[int, list[tuple[int, int, int]]] = {}
    for result_set in result_sets:
        if not isinstance(result_set, Mapping):
            continue
        headers = result_set.get("headers")
        rows = result_set.get("rowSet")
        if not isinstance(headers, list) or not isinstance(rows, list):
            continue
        required = {"TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"}
        if not required.issubset(headers):
            continue
        indexes = {field: headers.index(field) for field in required}
        for row in rows:
            if not isinstance(row, list) or len(row) < len(headers):
                continue
            team_id = _positive_rotation_int(row[indexes["TEAM_ID"]])
            player_id = _positive_rotation_int(row[indexes["PERSON_ID"]])
            in_time = _rotation_tenths(row[indexes["IN_TIME_REAL"]])
            out_time = _rotation_tenths(row[indexes["OUT_TIME_REAL"]])
            if (
                team_id is None
                or player_id is None
                or in_time is None
                or out_time is None
                or out_time <= in_time
                or team_id not in rosters
                or player_id not in rosters[team_id]
            ):
                continue
            intervals_by_team.setdefault(team_id, []).append(
                (player_id, in_time, out_time)
            )

    result: dict[int, dict[int, set[int]]] = {}
    for period in sorted(period for period in periods if period > 1):
        start_tenths = _completed_period_seconds(period) * 10
        team_lineups: dict[int, set[int]] = {}
        for team_id, intervals in intervals_by_team.items():
            active = {
                player_id
                for player_id, in_time, out_time in intervals
                if in_time <= start_tenths < out_time
            }
            if len(active) == 5:
                team_lineups[team_id] = active
        if team_lineups:
            result[period] = team_lineups
    return result


def _positive_rotation_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _rotation_tenths(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _actor_implies_on_court(action: Mapping[str, Any]) -> bool:
    action_type = str(action.get("actionType") or "").casefold()
    subtype = str(action.get("subType") or "").casefold()
    description = str(action.get("description") or "").casefold()
    return not (
        action_type == "ejection"
        or (action_type == "foul" and ("technical" in subtype or "t.foul" in description))
    )


def _period_boundary_substitutions(
    period_action: Mapping[str, Any],
    *,
    current_lineups: dict[int, set[int]],
    target_lineups: Mapping[int, set[int]],
    rosters: Mapping[int, Mapping[int, Mapping[str, Any]]],
    team_tricodes: Mapping[int, str],
) -> list[dict[str, Any]]:
    order_number = _source_order_base(period_action) + 1
    actions: list[dict[str, Any]] = []

    for team_id in sorted(current_lineups):
        outgoing_ids = sorted(current_lineups[team_id] - target_lineups[team_id])
        incoming_ids = sorted(target_lineups[team_id] - current_lineups[team_id])
        if len(outgoing_ids) != len(incoming_ids):
            raise ValueError(f"Unbalanced period lineup change for team {team_id}")
        for subtype, player_ids in (("out", outgoing_ids), ("in", incoming_ids)):
            for player_id in player_ids:
                action = _base_action(period_action, order_number=order_number)
                action.update(
                    {
                        "actionType": "substitution",
                        "subType": subtype,
                        "descriptor": "stats_v3_period_lineup",
                        "teamId": team_id,
                        "teamTricode": team_tricodes.get(team_id, ""),
                        "personId": player_id,
                        "personIdsFilter": [player_id],
                        "description": (
                            f"SUB {subtype}: {_player_name(rosters[team_id][player_id])}"
                        ),
                    }
                )
                actions.append(action)
                order_number += 1
        current_lineups[team_id] = set(target_lineups[team_id])
    return actions


def _adapt_substitution(
    source_action: Mapping[str, Any],
    *,
    rosters: Mapping[int, Mapping[int, Mapping[str, Any]]],
    lineups: dict[int, set[int]],
    team_tricodes: Mapping[int, str],
    source_aliases: Mapping[int, Mapping[str, set[int]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    description = str(source_action.get("description") or "")
    match = SUBSTITUTION_RE.fullmatch(description)
    if match is None:
        raise ValueError(f"Unexpected Stats V3 substitution description: {description!r}")

    team_id = _required_int(source_action, "teamId")
    if team_id not in rosters:
        raise ValueError(f"Substitution has unknown team ID {team_id}")
    outgoing_id = _required_int(source_action, "personId")
    if outgoing_id not in rosters[team_id]:
        raise ValueError(f"Substitution outgoing player {outgoing_id} is not on team {team_id}")

    incoming_id = _resolve_player_name(
        match.group("incoming"),
        roster=rosters[team_id],
        current_lineup=lineups[team_id],
        source_aliases=source_aliases[team_id],
    )
    if outgoing_id not in lineups[team_id]:
        raise ValueError(
            f"Substitution outgoing player {outgoing_id} is not in team {team_id}'s lineup"
        )
    if incoming_id in lineups[team_id]:
        raise ValueError(
            f"Substitution incoming player {incoming_id} is already in team {team_id}'s lineup"
        )
    lineups[team_id].remove(outgoing_id)
    lineups[team_id].add(incoming_id)

    common = _base_action(
        source_action,
        order_number=_source_order_base(source_action),
    )
    common["actionType"] = "substitution"
    common["teamId"] = team_id
    common["teamTricode"] = team_tricodes.get(team_id, "")

    outgoing = common | {
        "subType": "out",
        "personId": outgoing_id,
        "personIdsFilter": [outgoing_id],
        "description": f"SUB out: {_player_name(rosters[team_id][outgoing_id])}",
    }
    incoming = common | {
        "orderNumber": common["orderNumber"] + 1,
        "subType": "in",
        "personId": incoming_id,
        "personIdsFilter": [incoming_id],
        "description": f"SUB in: {_player_name(rosters[team_id][incoming_id])}",
    }
    return outgoing, incoming


def _substitution_incoming_name(source_action: Mapping[str, Any]) -> str:
    description = str(source_action.get("description") or "")
    match = SUBSTITUTION_RE.fullmatch(description)
    if match is None:
        raise ValueError(f"Unexpected Stats V3 substitution description: {description!r}")
    return match.group("incoming")


def _substitution_outgoing_name(source_action: Mapping[str, Any]) -> str:
    description = str(source_action.get("description") or "")
    match = SUBSTITUTION_RE.fullmatch(description)
    if match is None:
        raise ValueError(f"Unexpected Stats V3 substitution description: {description!r}")
    return match.group("outgoing")


def _resolve_player_name(
    source_name: str,
    *,
    roster: Mapping[int, Mapping[str, Any]],
    current_lineup: set[int],
    source_aliases: Mapping[str, set[int]] | None = None,
) -> int:
    target = _normalize_name(source_name)
    suffixless_target = re.sub(r"\s+(?:jr|sr|ii|iii|iv)$", "", target)
    source_candidates = (
        set(source_aliases.get(target, set())) if source_aliases is not None else set()
    )
    resolved = _unique_name_candidate(source_candidates, current_lineup)
    if resolved is not None:
        return resolved

    exact_roster_candidates = {
        player_id for player_id, player in roster.items() if target in _player_aliases(player)
    }
    resolved = _unique_name_candidate(exact_roster_candidates, current_lineup)
    if resolved is not None:
        return resolved

    candidates = set(exact_roster_candidates)
    if suffixless_target != target:
        if source_aliases is not None:
            candidates.update(source_aliases.get(suffixless_target, set()))
        candidates.update(
            player_id
            for player_id, player in roster.items()
            if suffixless_target in _player_aliases(player)
        )
    known_person_id = HISTORICAL_NAME_PERSON_IDS.get(target)
    if known_person_id in roster:
        candidates.add(known_person_id)
    if not candidates:
        candidates.update(
            player_id
            for player_id, player in roster.items()
            if _fuzzy_player_name_match(target, player)
        )
    resolved = _unique_name_candidate(candidates, current_lineup)
    if resolved is not None:
        return resolved
    raise ValueError(
        f"Could not uniquely resolve Stats V3 substitution player {source_name!r}; "
        f"candidate IDs={candidates}"
    )


def _unique_name_candidate(
    candidates: set[int],
    current_lineup: set[int],
) -> int | None:
    bench_candidates = candidates - current_lineup
    if len(bench_candidates) == 1:
        return next(iter(bench_candidates))
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _player_aliases(player: Mapping[str, Any]) -> set[str]:
    first = str(player.get("firstName") or "")
    family = str(player.get("familyName") or "")
    aliases = {
        alias
        for alias in (
            _normalize_name(family),
            _normalize_name(f"{first} {family}"),
            _normalize_name(str(player.get("nameI") or "")),
            _normalize_name(str(player.get("playerSlug") or "").replace("-", " ")),
        )
        if alias
    }
    suffixless_family = re.sub(r"\s+(?:jr|sr|ii|iii|iv)$", "", _normalize_name(family))
    if suffixless_family:
        aliases.add(suffixless_family)
    return aliases


def _fuzzy_player_name_match(
    normalized_source_name: str,
    player: Mapping[str, Any],
) -> bool:
    source_parts = normalized_source_name.split()
    if len(source_parts) < 2:
        return False
    source_first = source_parts[0]
    source_family = " ".join(source_parts[1:])
    player_first = _normalize_name(str(player.get("firstName") or ""))
    player_family = re.sub(
        r"\s+(?:jr|sr|ii|iii|iv)$",
        "",
        _normalize_name(str(player.get("familyName") or "")),
    )
    return (
        source_family == player_family
        and len(source_first) >= 2
        and player_first.startswith(source_first)
    )


def _player_name(player: Mapping[str, Any]) -> str:
    name_i = str(player.get("nameI") or "")
    if name_i:
        return name_i
    return f"{player.get('firstName', '')} {player.get('familyName', '')}".strip()


def _normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.casefold()).strip()


def _base_action(
    source_action: Mapping[str, Any],
    *,
    order_number: int,
) -> dict[str, Any]:
    period = _required_int(source_action, "period")
    return {
        **copy.deepcopy(dict(source_action)),
        "orderNumber": order_number,
        "periodType": "REGULAR" if period <= 4 else "OVERTIME",
        "qualifiers": [],
        "personIdsFilter": _related_player_ids(source_action),
        "possession": 0,
    }


def _action_classification(
    source_action: Mapping[str, Any],
    *,
    description: str,
) -> tuple[str, str, str]:
    action_type = str(source_action.get("actionType") or "")
    subtype = str(source_action.get("subType") or "")
    folded_type = action_type.casefold()
    folded_subtype = subtype.casefold()
    folded_description = description.casefold()

    if folded_type in {"made shot", "missed shot"}:
        shot_value = _required_int(source_action, "shotValue")
        return ("3pt" if shot_value == 3 else "2pt"), subtype, ""
    if folded_type == "free throw":
        sequence = FREE_THROW_SEQUENCE_RE.search(subtype)
        normalized_subtype = (
            f"{sequence.group('attempt')} of {sequence.group('total')}"
            if sequence is not None
            else subtype
        )
        descriptor = _retaining_free_throw_descriptor(
            folded_subtype,
            folded_description,
        )
        return "freethrow", normalized_subtype, descriptor
    if folded_type == "rebound":
        return "rebound", "", ""
    if folded_type == "turnover":
        return "turnover", subtype, ""
    if folded_type == "foul":
        return _foul_classification(folded_subtype, folded_description)
    if folded_type == "jump ball":
        return "jumpball", "recovered", "startperiod"
    if folded_type == "instant replay":
        return "instantreplay", subtype.casefold().replace(" ", ""), ""
    if folded_type in {"timeout", "period"}:
        return folded_type, subtype.casefold(), ""
    if folded_type in {"violation", "ejection"}:
        return "stoppage", subtype.casefold(), folded_type
    if not folded_type and " steal " in f" {folded_description} ":
        return "steal", "", ""
    if not folded_type and " block " in f" {folded_description} ":
        return "block", "", ""
    if folded_type:
        return re.sub(r"[^a-z0-9]+", "", folded_type), subtype, ""
    raise ValueError(f"Unrecognized blank Stats V3 action: {description!r}")


def _foul_classification(
    subtype: str,
    description: str,
) -> tuple[str, str, str]:
    if "offensive" in subtype:
        descriptor = "charge" if "charge" in subtype else ""
        return "foul", "offensive", descriptor
    descriptor = ""
    for needle, value in (
        ("shooting", "shooting"),
        ("loose ball", "loose ball"),
        ("technical", "technical"),
        ("flagrant", "flagrant"),
        ("clear path", "clear-path"),
        ("away from play", "away-from-play"),
        ("personal take", "transition take"),
    ):
        if needle in subtype or needle in description:
            descriptor = value
            break
    return "foul", ("technical" if descriptor == "technical" else "personal"), descriptor


def _retaining_free_throw_descriptor(subtype: str, description: str) -> str:
    for needle, value in (
        ("technical", "technical"),
        ("flagrant", "flagrant"),
        ("clear path", "clear-path"),
        ("away from play", "away-from-play"),
        ("take", "transition take"),
    ):
        if needle in subtype or needle in description:
            return value
    return ""


def _team_id(
    source_action: Mapping[str, Any],
    rosters: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> int:
    team_id = _required_int(source_action, "teamId")
    if team_id:
        return team_id
    person_id = _required_int(source_action, "personId")
    if person_id in rosters:
        return person_id
    location = str(source_action.get("location") or "").casefold()
    team_ids = list(rosters)
    if len(team_ids) == 2 and location == "h":
        return team_ids[0]
    if len(team_ids) == 2 and location == "v":
        return team_ids[1]
    return 0


def _jump_ball_recovery_team_id(
    description: str,
    *,
    rosters: Mapping[int, Mapping[int, Mapping[str, Any]]],
    fallback_team_id: int,
) -> int:
    """Use the jump-ball tip recipient, rather than the listed jumper, as owner."""

    match = JUMP_BALL_TIP_RE.search(description)
    if match is None:
        return fallback_team_id
    recipient = match.group("recipient")
    matches: list[int] = []
    for team_id, roster in rosters.items():
        try:
            _resolve_player_name(recipient, roster=roster, current_lineup=set())
        except ValueError:
            continue
        matches.append(team_id)
    return matches[0] if len(matches) == 1 else fallback_team_id


def _possession_team_id(
    event_type: str,
    *,
    subtype: str,
    team_id: int,
    team_ids: set[int],
) -> int:
    if not team_id or team_id not in team_ids:
        return 0
    if event_type == "foul":
        if subtype == "offensive":
            return team_id
        if subtype == "technical":
            return 0
        return next(iter(team_ids - {team_id}))
    if event_type == "block":
        return next(iter(team_ids - {team_id}))
    if event_type in {
        "2pt",
        "3pt",
        "freethrow",
        "heave",
        "jumpball",
        "rebound",
        "steal",
        "turnover",
    }:
        return team_id
    return 0


def _related_player_ids(source_action: Mapping[str, Any]) -> list[int]:
    person_id = source_action.get("personId")
    if isinstance(person_id, int) and not isinstance(person_id, bool) and person_id > 0:
        return [person_id]
    return []


def _source_order_base(source_action: Mapping[str, Any]) -> int:
    period = _required_int(source_action, "period")
    elapsed_centiseconds = (
        _completed_period_seconds(period) * 100
        + _period_duration_seconds(period) * 100
        - _clock_centiseconds(str(source_action.get("clock") or ""))
    )
    return (
        elapsed_centiseconds * 10_000_000_000_000
        + period * 100_000_000_000
        + _period_boundary_priority(source_action) * 10_000_000_000
        + _required_int(source_action, "actionNumber") * 1_000_000
        + _required_int(source_action, "actionId") * 1_000
    )


def _source_action_sort_key(source_action: Any) -> tuple[int, int, int, int, int]:
    if not isinstance(source_action, Mapping):
        raise ValueError("Expected every Stats V3 action to be an object")
    return (
        _required_int(source_action, "period"),
        -_clock_centiseconds(str(source_action.get("clock") or "")),
        _period_boundary_priority(source_action),
        _required_int(source_action, "actionNumber"),
        _required_int(source_action, "actionId"),
    )


def _period_boundary_priority(source_action: Mapping[str, Any]) -> int:
    if str(source_action.get("actionType") or "").casefold() != "period":
        return 1
    subtype = str(source_action.get("subType") or "").casefold()
    if subtype == "start":
        return 0
    if subtype == "end":
        return 2
    return 1


def _clock_centiseconds(clock: str) -> int:
    match = re.fullmatch(
        r"PT(?:(?P<minutes>\d+)M)?(?P<seconds>\d+)(?:\.(?P<fraction>\d+))?S",
        clock,
    )
    if match is None:
        raise ValueError(f"Unexpected Stats V3 clock value: {clock!r}")
    fraction = (match.group("fraction") or "").ljust(2, "0")[:2]
    return (
        int(match.group("minutes") or 0) * 60 * 100
        + int(match.group("seconds")) * 100
        + int(fraction or 0)
    )


def _period_duration_seconds(period: int) -> int:
    return 12 * 60 if period <= 4 else 5 * 60


def _completed_period_seconds(period: int) -> int:
    return min(period - 1, 4) * 12 * 60 + max(period - 5, 0) * 5 * 60


def _required_int(value: Mapping[str, Any], field: str) -> int:
    raw_value = value.get(field)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"Expected integer {field}, got {raw_value!r}")
    return raw_value
