from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nba_lineup_model.ingest.nba_cdn import NbaCdnEndpoint, RawJsonCache
from nba_lineup_model.ingest.nba_stats import (
    NbaStatsEndpoint,
    NbaStatsRawCache,
)
from nba_lineup_model.normalize.stats_v3 import (
    adapt_stats_v3_boxscore,
    adapt_stats_v3_play_by_play,
)
from nba_lineup_model.season.fetch import RawArtifactEvidence, artifact_evidence
from nba_lineup_model.season.stats import stats_artifact_evidence

GameSource = Literal["live_data", "stats_v3"]


class GameSourceError(RuntimeError):
    """Raised when no processable raw document exists for a game endpoint."""


@dataclass(frozen=True)
class SelectedRawArtifact:
    """One selected raw artifact and its processing-boundary payload."""

    source: GameSource
    payload: dict[str, Any]
    sha256: str
    byte_count: int
    path: Path


@dataclass(frozen=True)
class GameSourceDocuments:
    """Selected play-by-play and box-score documents for one game."""

    game_id: str
    play_by_play: SelectedRawArtifact
    boxscore: SelectedRawArtifact


def load_game_source_documents(
    game_id: str,
    *,
    raw_dir: Path | str = Path("data/raw"),
) -> GameSourceDocuments:
    """Prefer Stats V3 per endpoint and fall back to liveData.

    Historical liveData can represent substitutions with only the outgoing
    player. Stats V3 retains both sides of each transaction.
    """

    root = Path(raw_dir)
    live_cache = RawJsonCache(root)
    stats_cache = NbaStatsRawCache(root / "stats")

    stats_boxscore = stats_artifact_evidence(
        stats_cache,
        NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3,
        game_id,
    )
    if stats_boxscore is not None:
        boxscore = SelectedRawArtifact(
            source="stats_v3",
            payload=adapt_stats_v3_boxscore(stats_boxscore.response.payload),
            sha256=stats_boxscore.sha256,
            byte_count=stats_boxscore.byte_count,
            path=stats_cache.path_for(
                NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3,
                game_id,
            ),
        )
    else:
        live_boxscore = _live_artifact_evidence(
            live_cache, NbaCdnEndpoint.BOXSCORE, game_id
        )
        if live_boxscore is None:
            raise GameSourceError(f"No Stats V3 or liveData box score is cached for {game_id}")
        boxscore = SelectedRawArtifact(
            source="live_data",
            payload=live_boxscore.response.payload,
            sha256=live_boxscore.sha256,
            byte_count=live_boxscore.byte_count,
            path=live_cache.path_for(NbaCdnEndpoint.BOXSCORE, game_id),
        )

    stats_play_by_play = stats_artifact_evidence(
        stats_cache,
        NbaStatsEndpoint.PLAY_BY_PLAY_V3,
        game_id,
    )
    if stats_play_by_play is not None:
        play_by_play = SelectedRawArtifact(
            source="stats_v3",
            payload=adapt_stats_v3_play_by_play(
                stats_play_by_play.response.payload, boxscore.payload
            ),
            sha256=stats_play_by_play.sha256,
            byte_count=stats_play_by_play.byte_count,
            path=stats_cache.path_for(NbaStatsEndpoint.PLAY_BY_PLAY_V3, game_id),
        )
    else:
        live_play_by_play = _live_artifact_evidence(
            live_cache, NbaCdnEndpoint.PLAY_BY_PLAY, game_id
        )
        if live_play_by_play is None:
            raise GameSourceError(
                f"No Stats V3 or liveData play-by-play is cached for {game_id}"
            )
        play_by_play = SelectedRawArtifact(
            source="live_data",
            payload=_adapt_live_data_substitutions(live_play_by_play.response.payload),
            sha256=live_play_by_play.sha256,
            byte_count=live_play_by_play.byte_count,
            path=live_cache.path_for(NbaCdnEndpoint.PLAY_BY_PLAY, game_id),
        )

    return GameSourceDocuments(
        game_id=game_id,
        play_by_play=play_by_play,
        boxscore=boxscore,
    )


def _adapt_live_data_substitutions(payload: dict[str, Any]) -> dict[str, Any]:
    """Expand legacy one-record substitutions into explicit out/in actions.

    Older cached liveData responses encode the incoming player only in
    ``personIdsFilter``. The canonical lineup stream requires one action for
    each direction.  Raw cache files remain unchanged.
    """

    game = payload.get("game")
    if not isinstance(game, dict):
        return payload
    source_actions = game.get("actions")
    if not isinstance(source_actions, list):
        return payload

    actions: list[dict[str, Any]] = []
    for action in source_actions:
        if not isinstance(action, dict):
            actions.append(action)
            continue
        actions.append(dict(action))
        if str(action.get("actionType") or "").casefold() != "substitution":
            continue
        if str(action.get("subType") or "").casefold() != "out":
            continue
        outgoing_id = action.get("personId")
        player_ids = action.get("personIdsFilter")
        if (
            isinstance(outgoing_id, bool)
            or not isinstance(outgoing_id, int)
            or not isinstance(player_ids, list)
        ):
            continue
        incoming_ids = [
            player_id
            for player_id in player_ids
            if isinstance(player_id, int) and not isinstance(player_id, bool)
            and player_id != outgoing_id
        ]
        if len(incoming_ids) != 1:
            continue
        incoming = dict(action)
        incoming["orderNumber"] = int(action["orderNumber"]) + 1
        incoming["subType"] = "in"
        incoming["personId"] = incoming_ids[0]
        incoming["personIdsFilter"] = [incoming_ids[0]]
        actions.append(incoming)

    if len(actions) == len(source_actions):
        return payload
    return payload | {"game": game | {"actions": actions}}


def _live_artifact_evidence(
    cache: RawJsonCache,
    endpoint: NbaCdnEndpoint,
    game_id: str,
) -> RawArtifactEvidence | None:
    if not cache.path_for(endpoint, game_id).exists():
        return None
    return artifact_evidence(cache, endpoint, game_id)
