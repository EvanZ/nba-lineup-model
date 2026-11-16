"""Dev browsing must not reach legacy forecast publication through API reads."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.web_api import app as web_app

BLOCKED_PATHS = (
    "/api/schedule?date=2026-11-07",
    "/api/schedule-view?date=2026-11-07",
    "/api/schedule-performance",
    "/api/current-season-records",
    "/api/win-projections",
    "/api/schedule/?date=2026-11-07",
)


@pytest.mark.parametrize("path", BLOCKED_PATHS)
@pytest.mark.parametrize("bound_release", [False, True])
def test_disabled_calls_stop_before_observations_and_handlers(
    monkeypatch, tmp_path, path, bound_release
):
    monkeypatch.setenv("GESTALT_DISABLE_LEGACY_SCHEDULE_API", "1")
    monkeypatch.delenv("GESTALT_ACTIVE_RELEASE_ROOT", raising=False)
    observations = Mock(side_effect=AssertionError("observation reader was reached"))
    reader = Mock(side_effect=AssertionError("legacy handler was reached"))
    serving = SimpleNamespace(
        path=lambda name: tmp_path / name,
        observations=observations,
    ) if bound_release else None
    monkeypatch.setattr(web_app.ReleaseServing, "load", lambda root: serving)
    for name in (
        "read_schedule_day",
        "read_schedule_performance",
        "read_current_season_records",
        "build_current_season_win_projection_payload",
        "read_win_projection_cache",
    ):
        monkeypatch.setattr(web_app, name, reader)

    with TestClient(web_app.create_app(object())) as client:
        response = client.get(path, headers={"X-Gestalt-Release": "stale"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Schedule data is temporarily unavailable."}
    assert response.headers["cache-control"] == "no-store"
    observations.assert_not_called()
    reader.assert_not_called()


@pytest.mark.parametrize("setting", [None, "0"])
def test_other_environments_keep_existing_schedule_behavior(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("GESTALT_DISABLE_LEGACY_SCHEDULE_API", raising=False)
    else:
        monkeypatch.setenv("GESTALT_DISABLE_LEGACY_SCHEDULE_API", setting)
    monkeypatch.delenv("GESTALT_ACTIVE_RELEASE_ROOT", raising=False)
    monkeypatch.delenv("GESTALT_OBSERVATION_S3_URI", raising=False)
    monkeypatch.setattr(web_app.ReleaseServing, "load", lambda root: None)
    reader = Mock(return_value={"date": "2026-11-07", "games": []})
    monkeypatch.setattr(web_app, "read_schedule_day", reader)

    with TestClient(web_app.create_app(object())) as client:
        response = client.get("/api/schedule?date=2026-11-07")

    assert response.status_code == 200
    reader.assert_called_once()


def test_unrelated_api_remains_available(monkeypatch):
    monkeypatch.setenv("GESTALT_DISABLE_LEGACY_SCHEDULE_API", "1")
    monkeypatch.delenv("GESTALT_ACTIVE_RELEASE_ROOT", raising=False)
    monkeypatch.setattr(web_app.ReleaseServing, "load", lambda root: None)

    with TestClient(web_app.create_app(object())) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/players" in response.json()["paths"]
