"""Generate open forecasts from one verified dated model; never rewrite history."""

from __future__ import annotations

import copy
import shutil
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from nba_lineup_model.season.daily_artifact_common import file_sha256
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.forecast_publication import ForecastRegistry
from nba_lineup_model.season.release_artifact_contracts import (
    ArtifactReference,
    ForecastVersionManifest,
    SchedulingInstance,
    SealedCodeIdentity,
    ValidationRecord,
)
from nba_lineup_model.season.serving_selection import (
    INDEX_FILENAME,
    MARKET_FIELDS,
    _write,
    digest,
    forecast_inputs,
    read_object,
    verify_schedule_selection,
)


def require_forecast_model(response: dict, model_bundle: Path, forecast_date: date, initialization: dict | None = None) -> dict:
    """The API and the forecast must use exactly the selected model, not a label."""
    manifest = verify_daily_model_artifact(model_bundle)
    identity = manifest["identity"]
    if date.fromisoformat(identity["game_date_pacific"]) >= forecast_date:
        raise ValueError("Forecast cannot use same-day or future completed model evidence")
    minutes = response.get("minutes", response)
    forecast = response.get("win_projection", response)
    provenance = minutes.get("projection_provenance", {})
    model_id = manifest["bundle_sha256"]
    suffix = ''
    expected_rating_sha = identity['components']['player_ratings']['sha256']
    if initialization:
        from nba_lineup_model.season.current_serving import resolve_latest_bundle, verify_serving_bundle
        from nba_lineup_model.players.reconciliation_artifact import reference, verify
        root = Path(initialization['serving_root'])
        receipt = read_object(root / 'forecast-initialization.json')
        selected = reference(verify(Path(initialization['reconciliation_root']))).as_dict()
        if receipt['reference'] != selected or date.fromisoformat(receipt['through_date']) != forecast_date:
            raise ValueError('Forecast initialization selection/date mismatch')
        serving = verify_serving_bundle(resolve_latest_bundle(root))
        suffix = '#reconciliation=' + selected['content_id']
        sources = serving['identity']['sources']
        if (sources['rating_state']['uri'] != f'artifact://daily-model/{model_id}/player_ratings' + suffix
            or sources['minutes_state']['uri'] != f'artifact://daily-model/{model_id}/minutes_state#serving-v1' + suffix):
            raise ValueError('Forecast serving does not bind the selected model and introduction history')
        expected_rating_sha = sources['rating_state']['sha256']

    if (
        provenance.get("rating_as_of") != identity["game_date_pacific"]
        or provenance.get("rating_source_uri")
        != f"artifact://daily-model/{model_id}/player_ratings" + suffix
        or provenance.get("minutes_source_uri")
        != f"artifact://daily-model/{model_id}/minutes_state#serving-v1" + suffix
        or provenance.get("rating_state_sha256")
        != expected_rating_sha
    ):
        raise ValueError("Forecast ratings/minutes do not use the selected completed model")
    hca_ref = identity["components"]["home_court_state"]
    hca = read_object(model_bundle / hca_ref["path"])["home_court_estimate"]
    if forecast.get("home_court") != hca:
        raise ValueError("Forecast HCA does not use the selected completed model")
    return manifest


def refresh_open_selection(**kwargs) -> dict:
    """Atomic and retryable: a failed run cannot leave a partial selection."""
    source, target = kwargs["source"], kwargs["target"]
    identity_directory = kwargs.pop("identity_directory", False)
    from nba_lineup_model.season.official_schedule_runtime import select_captured_injuries
    now = (kwargs.get("now") or datetime.now(UTC)).astimezone(UTC)
    captures = kwargs.get("injury_captures")
    if captures is None:
        captures = select_captured_injuries(
            schedule_root=source, observation_root=kwargs["observation_root"].resolve(strict=True),
            start_date=kwargs["start_date"].isoformat(), days_ahead=kwargs["days_ahead"], now=now,
        )
    from nba_lineup_model.season.market_selection import select_captured_markets
    markets = kwargs.get("market_captures")
    if markets is None:
        markets = select_captured_markets(schedule_root=source,
            observation_root=kwargs["observation_root"].resolve(strict=True),
            start_date=kwargs["start_date"].isoformat(), days_ahead=kwargs["days_ahead"], now=now)
    kwargs["market_captures"] = markets
    kwargs = {k: v for k, v in kwargs.items() if k != "observation_root"}
    kwargs.update(now=now, injury_captures=captures)
    model = require_forecast_model(
        kwargs["forecast_response"], kwargs["model_bundle"], kwargs["start_date"], kwargs.get("initialization")
    )
    request_identity = {
            "source": verify_schedule_selection(source)["selection_map_sha256"],
            "injury_captures": captures,
            "model": model["bundle_sha256"],
            "market_captures": markets,
            "response": kwargs["forecast_response"],
            "start_date": str(kwargs["start_date"]),
            "days_ahead": kwargs["days_ahead"],
            "code": kwargs["code"],
            "environment": kwargs["environment"],
            "catalog_sha256": file_sha256(kwargs["catalog"]),
        }
    request_id = digest(request_identity)
    if identity_directory:
        target = target / request_id
        kwargs["target"] = target
    def output(result):
        return {**result, "selection_path": str(target)} if identity_directory else result
    if target.exists():
        index = verify_schedule_selection(target)
        if index.get("refresh_request_id") != request_id:
            raise ValueError("Existing forecast output has different inputs")
        return output(index["refresh_result"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".forecast-", dir=target.parent) as temporary:
        staged = Path(temporary) / "selection"
        result = _refresh_open_selection(**{**kwargs, "target": staged})
        index = verify_schedule_selection(staged)
        index.update(refresh_request_id=request_id, refresh_request=request_identity, refresh_result=result)
        index.pop("selection_map_sha256")
        index["selection_map_sha256"] = digest(index)
        _write(staged / INDEX_FILENAME, index)
        verify_schedule_selection(staged)
        staged.rename(target)
    return output(result)


def _refresh_open_selection(
    *,
    source: Path,
    target: Path,
    model_bundle: Path,
    forecast_response: dict,
    start_date: date,
    days_ahead: int,
    registry_root: Path,
    catalog: Path,
    code: dict,
    environment: str,
    injury_captures: dict[str, dict],
    now: datetime | None = None,
    market_captures: dict[str, dict],
    initialization: dict | None = None,
) -> dict:
    """Publish a new selection bundle, preserving every locked selection exactly.

    New versions go through the existing forecast registry and real publication
    receipts. No receipt is invented here and no source artifact is modified.
    """
    from nba_lineup_model.season.official_schedule_runtime import reproject_captured_schedule_window

    now = (now or datetime.now(UTC)).astimezone(UTC)
    model = require_forecast_model(forecast_response, model_bundle, start_date, initialization)
    original = verify_schedule_selection(source)
    if target.exists():
        raise ValueError("Upcoming forecast output must be a new bundle")
    shutil.copytree(source, target)
    result = reproject_captured_schedule_window(
        start_date=start_date.isoformat(),
        days_ahead=days_ahead,
        state_dir=target,
        projection_url="captured://selected-model",
        forecast_response=forecast_response,
        injury_captures=injury_captures,
        now=now,
    )
    index = copy.deepcopy(original)
    registry = ForecastRegistry(registry_root)
    if index.get("forecast_registry_root") not in (None, str(registry_root.resolve())):
        raise ValueError("Upcoming refresh must retain the existing forecast registry")
    minutes = forecast_response.get("minutes", forecast_response)
    roster = minutes["players"]
    from nba_lineup_model.orchestration.daily_work import atomic_json

    roster_path = registry_root / "inputs" / f"roster-{digest(roster)}.json"
    atomic_json(roster_path, {"players": roster}, immutable=True)

    def ref(kind, checksum, uri):
        return ArtifactReference(
            artifact_kind=kind, schema_version=1, content_sha256=checksum, uri=uri
        )

    versions = []
    refreshed = []
    protected = []
    hca_ref = model["identity"]["components"]["home_court_state"]
    for day, day_ref in index["dates"].items():
        path = target / day_ref["path"]
        before = read_object(source / day_ref["path"])
        board = read_object(path)
        old_games = {str(g["game_id"]): g for g in before["games"]}
        changed = False
        for game in board["games"]:
            gid = str(game["game_id"])
            old = old_games[gid]
            market_capture = market_captures.get(gid)
            if market_capture:
                game["market"] = copy.deepcopy(market_capture["market"])
                game["market_selection"] = copy.deepcopy(market_capture["capture"])
            if forecast_inputs(game) == forecast_inputs(old) and game.get("market") == old.get("market"):
                protected.append(gid)
                continue
            cutoff = datetime.fromisoformat(str(old["start_time"]).replace("Z", "+00:00"))
            if (
                cutoff <= now
                or old.get("pregame")
                or old.get("final_audit")
                or old.get("status", {}).get("state", "pre") != "pre"
            ):
                raise ValueError(f"Upcoming refresh attempted to alter locked game {gid}")
            projection = game["projection"]
            provenance = projection["input_provenance"]
            if (
                provenance["rating_source_uri"]
                != minutes["projection_provenance"]["rating_source_uri"]
            ):
                raise ValueError(f"Regenerated game uses another model: {gid}")
            ledger = projection["ledger"]
            provenance.update(
                {
                    "home_court_source_uri": f"artifact://daily-model/{model['bundle_sha256']}/home_court_state",
                    "home_court_state_sha256": hca_ref["sha256"],
                }
            )
            if initialization:
                provenance['player_introductions'] = read_object(Path(initialization['serving_root']) / 'forecast-initialization.json')
            ledger["provenance"].update(provenance)
            components = {
                "projected_minutes": game["projected_minutes"],
                "player_ratings": {
                    str(row["player_id"]): row["frozen_nail"]
                    for side in ("home", "away")
                    for row in ledger[side]["players"]
                },
                "contributions": {side: ledger[side]["players"] for side in ("home", "away")},
                "ledger": ledger,
            }
            version_id = digest(
                {
                    "forecast": forecast_inputs(game),
                    "market": {field:game.get(field) for field in MARKET_FIELDS},
                    "computed_at": now.isoformat(),
                    "code": code,
                    "edition": index["edition_id"],
                }
            )
            scheduling = SchedulingInstance(
                scheduling_instance_id=index["games"][gid]["scheduling_instance_id"],
                game_id=gid,
                official_schedule_revision=ref(
                    "schedule_revision", file_sha256(catalog), catalog.as_uri()
                ),
                scheduled_tipoff_at=cutoff,
                cutoff_at=cutoff,
            )
            injury = game["injury_report"]
            version = ForecastVersionManifest(
                forecast_id=version_id,
                season=model["identity"]["season"],
                game_id=gid,
                scheduling=scheduling,
                edition_id=index["edition_id"],
                model_state=ref("daily_model_state", model["bundle_sha256"], model_bundle.as_uri()),
                injury_revision=ref(
                    "injury_revision", injury["artifact_sha256"], injury["source_url"]
                ),
                roster_revision=ref(
                    "roster_revision", file_sha256(roster_path), roster_path.as_uri()
                ),
                information_cutoff_at=cutoff,
                computed_at=now,
                components={
                    key: ref(key, digest(value), f"artifact://forecast/{version_id}/{key}")
                    for key, value in components.items()
                },
                predicted_home_margin=projection["home_margin"],
                predicted_home_win_probability=projection["home_win_probability"],
                ledger_home_margin=ledger["home_margin"],
                numeric_tolerance=1e-9,
                validation=ValidationRecord(status="clean"),
                code=SealedCodeIdentity(**code),
            )
            registry.stage(version, game)
            _, open_selection = registry.resolve(
                edition_id=index["edition_id"],
                forecast_id=version_id,
                environment=environment,
                now=now,
            )
            selected = {
                key: value
                for key, value in index["games"][gid].items()
                if key not in ("game_date", "selection_id")
            }
            selected.update(
                forecast_id=digest(forecast_inputs(game)),
                market_selection_id=digest({field:game.get(field) for field in MARKET_FIELDS}),
                provenance="original_live",
                registry_selection=open_selection,
                version_reference=version_id,
                original_publication_receipt=None,
            )
            selected["selection_id"] = digest(selected)
            game["forecast_selection"] = selected
            index["games"][gid] = {**selected, "game_date": day}
            if game.get("prediction_market") != old.get("prediction_market"):
                raise ValueError(f"Model refresh changed Polymarket selection: {gid}")
            if game.get("market") != (market_capture["market"] if market_capture else old.get("market")):
                raise ValueError(f"Forecast did not select its captured market: {gid}")
            versions.append(version_id)
            refreshed.append(gid)
            changed = True
        if changed:
            _write(path, board)
        day_ref["sha256"] = file_sha256(path)
    if len(refreshed) != result["reprojected_game_count"]:
        raise ValueError("Regenerated forecast coverage is incomplete")
    if versions:
        index.update(
            forecast_registry_root=str(registry_root.resolve()),
            publication_forecast_ids=sorted(versions),
            environment=environment,
        )
    index["files"] = {
        str(p.relative_to(target)): file_sha256(p)
        for p in sorted(target.rglob("*"))
        if p.is_file() and p.name != INDEX_FILENAME
    }
    index.pop("selection_map_sha256", None)
    index["selection_map_sha256"] = digest(index)
    _write(target / INDEX_FILENAME, index)
    verify_schedule_selection(target)
    return {
        **result,
        "model_date": model["identity"]["game_date_pacific"],
        "model_sha256": model["bundle_sha256"],
        "refreshed_game_ids": sorted(refreshed),
        "unchanged_game_ids": sorted(protected),
    }
