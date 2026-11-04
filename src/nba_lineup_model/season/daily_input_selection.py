"""Build a daily data partition from explicit, immutable NBA revisions only."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from nba_lineup_model.orchestration.daily_work import verify_reference
from nba_lineup_model.season.daily_data_replay import (
    OfficialBoxscore, OfficialInactiveList, OfficialInjuryReport,
    _build_date, _load_official_game_catalog,
)
from nba_lineup_model.season.observed_revisions import read_revision

REQUIRED = {"nba_box_score", "nba_play_by_play", "nba_injury_report", "nba_inactive_list"}


def build_selected_daily_data(*, selection: dict, catalog_reference: dict,
                              season: str, game_date: date, output_root: Path):
    catalog = verify_reference(catalog_reference)
    games = _load_official_game_catalog(catalog, season=season,
                                      start_date=game_date, end_date=game_date).get(game_date, [])
    expected = {str(game["game_id"]) for game in games}
    if set(selection) != expected:
        raise ValueError("Selected observations do not cover the exact official slate")
    loaded = {}
    for game_id, feeds in selection.items():
        if set(feeds) != REQUIRED:
            raise ValueError(f"{game_id}: required observation revisions are missing")
        loaded[game_id] = {}
        for feed, reference in feeds.items():
            revision = read_revision(reference)["identity"]
            if revision["game_id"] != game_id or revision["feed"] != feed:
                raise ValueError("Selected revision game/feed mismatch")
            loaded[game_id][feed] = revision
    injuries = {entry["nba_injury_report"]["raw_capture"]["sha256"] for entry in loaded.values()}
    if len(injuries) > 1:
        raise ValueError("A daily data partition must explicitly select one archived injury report")
    def observation(game_id, feed, cls):
        item = loaded[game_id][feed]
        return cls(payload=item["normalized"], sha256=item["raw_capture"]["sha256"],
                   immutable_uri=Path(item["raw_capture"]["path"]).as_uri(),
                   source_url=item["source_url"])
    def injury(_day):
        if _day != game_date:
            raise ValueError("Injury selection date mismatch")
        return observation(next(iter(loaded)), "nba_injury_report", OfficialInjuryReport)
    return _build_date(
        game_date_pacific=game_date, games=games,
        coverage={gid: [entry["nba_play_by_play"]["raw_capture"]["sha256"]]
                  for gid, entry in loaded.items()},
        game_catalog_path=catalog, output_root=output_root,
        play_by_play_s3_prefix="unused-explicit-revisions", season=season,
        boxscore_loader=lambda gid: observation(gid, "nba_box_score", OfficialBoxscore),
        injury_report_loader=injury,
        inactive_list_loader=lambda game: observation(str(game["game_id"]), "nba_inactive_list", OfficialInactiveList),
        play_by_play_loader=lambda gid: (
            loaded[gid]["nba_play_by_play"]["normalized"],
            Path(loaded[gid]["nba_play_by_play"]["raw_capture"]["path"]).as_uri(),
        ),
    )
