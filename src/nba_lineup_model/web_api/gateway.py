"""Route migrated reads independently of the legacy model-backed application."""

from __future__ import annotations

import logging
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from nba_lineup_model.web_api.compare import router as compare_router
from nba_lineup_model.web_api.current_season_records import router as records_router
from nba_lineup_model.web_api.historical_lineups import router as lineups_router
from nba_lineup_model.web_api.market_performance import router
from nba_lineup_model.web_api.player_discovery import router as discovery_router
from nba_lineup_model.web_api.player_history import router as history_router
from nba_lineup_model.web_api.player_rotation_history_api import router as rotation_history_router
from nba_lineup_model.web_api.player_season_history import router as season_history_router
from nba_lineup_model.web_api.player_summary import router as player_router
from nba_lineup_model.web_api.rankings import router as rankings_router
from nba_lineup_model.web_api.recent_team_ratings import router as recent_ratings_router
from nba_lineup_model.web_api.saved_win_projections import router as saved_wins_router
from nba_lineup_model.web_api.schedule import router as schedule_router
from nba_lineup_model.web_api.team_navigation import router as team_navigation_router
from nba_lineup_model.web_api.team_rating_history import router as team_history_router
from nba_lineup_model.web_api.team_rotation import router as team_rotation_router
from nba_lineup_model.web_api.team_rotation_history import router as team_history_animation_router


class MarketGateway:
    def __init__(self, legacy_factory):
        self.market = FastAPI(title="NBA GESTALT saved-data API")
        self.market.include_router(router)
        self.market.include_router(schedule_router)
        self.market.include_router(team_navigation_router)
        self.market.include_router(team_history_router)
        self.market.include_router(team_rotation_router)
        self.market.include_router(team_history_animation_router)
        self.market.include_router(rankings_router)
        self.market.include_router(lineups_router)
        self.market.include_router(saved_wins_router)
        self.market.include_router(records_router)
        self.market.include_router(recent_ratings_router)
        self.market.include_router(player_router)
        self.market.include_router(history_router)
        self.market.include_router(season_history_router)
        self.market.include_router(rotation_history_router)
        self.market.include_router(discovery_router)
        self.market.include_router(compare_router)
        self.legacy_factory = legacy_factory
        self.legacy = None
        self.legacy_failed = False
        self.lock = threading.Lock()

    def _legacy(self):
        if self.legacy_factory is None:
            return None
        with self.lock:
            if self.legacy is None and not self.legacy_failed:
                try:
                    self.legacy = self.legacy_factory()
                except Exception:
                    self.legacy_failed = True
                    logging.getLogger(__name__).exception("Legacy API initialization failed")
            return self.legacy

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "").rstrip("/")
        is_player = (
            path.startswith("/api/players/") and path.removeprefix("/api/players/").isdigit()
        )
        if (
            scope["type"] == "lifespan"
            or is_player
            or (path.startswith("/api/teams/") and path.removeprefix("/api/teams/").isalnum())
            or (
                path.startswith("/api/players/")
                and path.endswith(("/rating-history", "/season-history", "/rotation-history"))
            )
            or (
                path.startswith("/api/teams/")
                and path.endswith(("/rating-history", "/rotation", "/rotation-history"))
            )
            or (path.startswith("/api/team-logos/") and path.endswith(".svg"))
            or (path.startswith("/api/headshots/") and path.endswith(".png"))
            or path
            in {
                "/api/market-performance",
                "/api/search",
                "/api/compare/players",
                "/api/teams",
                "/api/rankings",
                "/api/lineups",
                "/api/win-projections",
                "/api/current-season-records",
                "/api/recent-team-ratings",
                "/api/schedule-view",
                "/api/schedule",
            }
        ):
            await self.market(scope, receive, send)
            return
        legacy = await run_in_threadpool(self._legacy)
        if legacy is None:
            response = JSONResponse(
                {"detail": "This legacy API is currently unavailable."},
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return
        await legacy(scope, receive, send)
