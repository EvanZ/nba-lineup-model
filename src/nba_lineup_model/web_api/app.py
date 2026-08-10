"""FastAPI application for the local NBA GESTALT Lineup Lab."""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nba_lineup_model.modeling.forward_bounded_hierarchical_portable_matchup_contextual_rapm import (
    MODEL_NAME,
)
from nba_lineup_model.web_api.inference import LineupEvaluationError, LineupEvaluator


class MatchupRequest(BaseModel):
    """One neutral-court matchup entered in the Lineup Lab."""

    unit_player_ids: list[int] = Field(min_length=5, max_length=5)
    opponent_player_ids: list[int] = Field(min_length=5, max_length=5)
    include_response_curves: bool = False
    response_curve_feature_id: str | None = None
    response_curve_kind: Literal["composition", "matchup"] | None = None


def create_app(evaluator: LineupEvaluator | None = None) -> FastAPI:
    """Create the API, loading the published model state only once per process."""

    app = FastAPI(title="NBA GESTALT API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @lru_cache(maxsize=1)
    def get_evaluator() -> LineupEvaluator:
        return evaluator or LineupEvaluator.from_latest_artifact()

    @app.get("/api/health")
    def health() -> dict[str, object]:
        state = get_evaluator()
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "season": state.season,
            "run_id": state.run_id,
            "player_count": len(state.players),
        }

    @app.get("/api/players")
    def search_players(
        q: str = Query(min_length=1), limit: int = Query(default=12, ge=1, le=25)
    ) -> dict[str, object]:
        return {"players": get_evaluator().search_players(q, limit=limit)}

    @app.get("/api/players/{player_id}")
    def player(player_id: int) -> dict[str, object]:
        try:
            return get_evaluator().player(player_id)
        except LineupEvaluationError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/default-opponent")
    def default_opponent(
        exclude_player_id: Annotated[tuple[int, ...], Query()] = (),
    ) -> dict[str, object]:
        try:
            return {
                "players": get_evaluator().default_opponent(
                    excluded_player_ids=set(exclude_player_id)
                )
            }
        except LineupEvaluationError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.post("/api/matchups")
    def matchup(request: MatchupRequest) -> dict[str, object]:
        try:
            return get_evaluator().evaluate(
                request.unit_player_ids,
                request.opponent_player_ids,
                include_response_curves=request.include_response_curves,
                response_curve_feature_id=request.response_curve_feature_id,
                response_curve_kind=request.response_curve_kind,
            )
        except LineupEvaluationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


app = create_app()


def main() -> None:
    """Run the local API on the NBA GESTALT development port."""

    parser = argparse.ArgumentParser(description="Run the NBA GESTALT Lineup Lab API")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("nba_lineup_model.web_api.app:app", host="127.0.0.1", port=args.port, reload=True)


if __name__ == "__main__":
    main()
