"""Model-independent, full-timeline NBA stint reconstruction."""

from bisect import bisect_right
from collections import Counter
from math import isclose

from . import official_json

PROCESSING_VERSION = "nba_stints_v1"


class StintsError(ValueError):
    def __init__(self, message: str, details=None):
        super().__init__(message)
        self.details = details or {}


def parse_stints(play_by_play: bytes, box_score: bytes, scheduled: dict) -> tuple[list[dict], dict]:
    from .reconstruction import reconstruct_game_payloads

    pbp, box = official_json(play_by_play), official_json(box_score)
    game_id = scheduled["game_id"]
    if pbp.get("game", {}).get("gameId") != game_id or box.get("game", {}).get("gameId") != game_id:
        raise StintsError("Source game IDs differ from the official schedule.")
    game = box["game"]
    if game.get("gameStatus") != 3:
        raise StintsError("A final box score is required.")
    for side in ("home", "away"):
        if game[side+"Team"]["teamId"] != scheduled[side+"_team_id"]:
            raise StintsError("Source team IDs differ from the official schedule.")
    result = reconstruct_game_payloads(pbp, box)
    issues = [
        issue.model_dump(mode="json")
        for group in (result.lineups, result.possessions, result.possession_segments)
        for issue in group.issues
    ]
    if any(issue["severity"] == "error" for issue in issues):
        raise StintsError("NBA reconstruction reported errors.", {"issues": issues})
    stints = sorted(result.lineups.stints, key=lambda s: (s.start_event_index, s.stint_index))
    if not stints:
        raise StintsError("No lineup intervals were reconstructed.", {"issues": issues})
    starts = [s.start_event_index for s in stints]
    segments = result.possession_segments.segments
    counts = Counter(s.possession_id for s in segments)
    exposure = {s.stint_index: {"home": 0.0, "away": 0.0, "points_home": 0, "points_away": 0} for s in stints}
    for segment in segments:
        pos = bisect_right(starts, segment.start_event_index)-1
        if pos < 0:
            raise StintsError("A possession segment precedes the first stint.")
        stint = stints[pos]
        if (
            segment.end_event_index > stint.end_event_index
            or segment.home_player_ids != stint.home_player_ids
            or segment.away_player_ids != stint.away_player_ids
        ):
            raise StintsError("A possession segment crosses or disagrees with its lineup stint.")
        if segment.offense_team_id == scheduled["home_team_id"]:
            side = "home"
        elif segment.offense_team_id == scheduled["away_team_id"]:
            side = "away"
        else:
            raise StintsError("Possession offense is outside the scheduled matchup.")
        bucket = exposure[stint.stint_index]
        bucket[side] += 1.0/counts[segment.possession_id]
        bucket["points_home"] += segment.points_home
        bucket["points_away"] += segment.points_away

    rows = []
    end = 0.0
    for stint in stints:
        if not isclose(stint.start_elapsed_game_seconds, end, abs_tol=1e-6, rel_tol=0):
            raise StintsError("The lineup timeline has a gap or overlap.", {
                "stint_index": stint.stint_index, "previous_end": end,
                "next_start": stint.start_elapsed_game_seconds, "issues": issues,
            })
        end = stint.end_elapsed_game_seconds
        bucket = exposure[stint.stint_index]
        if (stint.points_home, stint.points_away) != (bucket["points_home"], bucket["points_away"]):
            raise StintsError("Possession segments do not conserve stint scoring.", {"stint_index": stint.stint_index})
        if (
            len(set(stint.home_player_ids)) != 5 or len(set(stint.away_player_ids)) != 5
            or set(stint.home_player_ids) & set(stint.away_player_ids)
        ):
            raise StintsError("Each interval must have five distinct players per team.")
        possessions = (bucket["home"]+bucket["away"])/2.0
        margin = stint.points_home-stint.points_away
        rows.append({
            "game_id": game_id, "game_time_utc": scheduled["game_time_utc"].isoformat(),
            "stint_index": stint.stint_index, "period": stint.period,
            "home_team_id": scheduled["home_team_id"], "away_team_id": scheduled["away_team_id"],
            "home_player_ids": list(stint.home_player_ids), "away_player_ids": list(stint.away_player_ids),
            "start_elapsed_game_seconds": stint.start_elapsed_game_seconds,
            "end_elapsed_game_seconds": stint.end_elapsed_game_seconds,
            "duration_seconds": stint.duration_seconds,
            "start_event_index": stint.start_event_index, "end_event_index": stint.end_event_index,
            "points_home": stint.points_home, "points_away": stint.points_away,
            "home_margin": margin, "home_offensive_possessions": bucket["home"],
            "away_offensive_possessions": bucket["away"], "possessions": possessions,
            "target_home_net_rating": 100.0*margin/possessions if possessions else None,
        })
    expected_seconds = 2880.0+max(0, int(game["period"])-4)*300.0
    if not isclose(end, expected_seconds, abs_tol=1e-6, rel_tol=0):
        raise StintsError("The reconstructed timeline does not cover the completed game.", {
            "reconstructed_seconds": end, "expected_seconds": expected_seconds, "issues": issues,
        })
    for side in ("home", "away"):
        if sum(row["points_"+side] for row in rows) != game[side+"Team"]["score"]:
            raise StintsError("Reconstructed scoring differs from the final box score.")
    possession_count = len(result.possessions.possessions)
    if not isclose(sum(row["possessions"] for row in rows)*2, possession_count, abs_tol=1e-6, rel_tol=0):
        raise StintsError("Stint possession shares do not conserve reconstructed possessions.")
    return rows, {
        "stint_count": len(rows), "zero_possession_stint_count": sum(not r["possessions"] for r in rows),
        "duration_seconds": end, "team_possession_count": possession_count, "issues": issues,
    }


__all__ = ["PROCESSING_VERSION", "StintsError", "parse_stints"]
