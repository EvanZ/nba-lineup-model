from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta

import pytest
from test_official_schedule_runtime import GAME_ID, START, _board

from nba_lineup_model.season.forecast_publication import (
    ForecastRegistry,
    record_verified_release_forecasts,
)
from nba_lineup_model.season.release_artifact_contracts import (
    ArtifactReference,
    ForecastVersionManifest,
    MarketSnapshotManifest,
    PublicationReceipt,
    SchedulingInstance,
    SealedCodeIdentity,
    ValidationRecord,
)
from nba_lineup_model.season.serving_selection import digest

TIP = datetime.fromisoformat(START.replace("Z", "+00:00"))
BEFORE = TIP - timedelta(minutes=30)


def ref(kind, value="a"):
    return ArtifactReference(
        artifact_kind=kind,
        schema_version=1,
        content_sha256=value * 64,
        uri=f"artifact://{kind}/{value * 64}",
    )


def candidate(registry, name, *, edition="edition", scheduling=None, rating=2.0):
    payload = _board(collected_at=BEFORE, home_star_rating=rating)["games"][0]
    scheduling = scheduling or SchedulingInstance(
        scheduling_instance_id="original",
        game_id=GAME_ID,
        official_schedule_revision=ref("schedule_revision"),
        scheduled_tipoff_at=TIP,
        cutoff_at=TIP,
    )
    ledger = payload["projection"]["ledger"]
    parts = {
        "projected_minutes": payload["projected_minutes"],
        "player_ratings": {
            str(row["player_id"]): row["frozen_nail"]
            for side in ("home", "away")
            for row in ledger[side]["players"]
        },
        "contributions": {side: ledger[side]["players"] for side in ("home", "away")},
        "ledger": ledger,
    }
    manifest = ForecastVersionManifest(
        forecast_id=name,
        season="2026-27",
        game_id=GAME_ID,
        scheduling=scheduling,
        model_state=ref("daily_model_state"),
        edition_id=edition,
        injury_revision=ref("injury_revision"),
        roster_revision=ref("roster_revision"),
        information_cutoff_at=scheduling.cutoff_at,
        computed_at=BEFORE,
        components={
            key: ArtifactReference(
                artifact_kind=key,
                schema_version=1,
                content_sha256=digest(value),
                uri=f"artifact://{key}/{digest(value)}",
            )
            for key, value in parts.items()
        },
        predicted_home_margin=payload["projection"]["home_margin"],
        predicted_home_win_probability=payload["projection"]["home_win_probability"],
        ledger_home_margin=ledger["home_margin"],
        numeric_tolerance=1e-9,
        validation=ValidationRecord(status="clean"),
        code=SealedCodeIdentity(
            package_sha256="a" * 64,
            dependency_lock_sha256="b" * 64,
            git_revision="c" * 40,
            sealed=True,
        ),
    )
    registry.stage(manifest, payload)
    return manifest, payload


def receipt(registry, name, at, *, outcome="successful"):
    value = PublicationReceipt(
        receipt_id=f"receipt-{name}",
        forecast_id=name,
        environment="dev",
        published_at=at,
        publisher_generation=1,
        outcome=outcome,
        error="failed" if outcome == "failed" else None,
    )
    registry.record_publication(value, loaded_forecast_id=name)


def test_latest_successful_publication_not_latest_computation_and_independent_market(tmp_path):
    registry = ForecastRegistry(tmp_path)
    first, _ = candidate(registry, "first")
    assert not (tmp_path / "scheduling").exists()
    candidate(registry, "last-good", rating=3.0)
    candidate(registry, "unpublished", rating=4.0)
    candidate(registry, "failed", rating=5.0)
    candidate(registry, "late", rating=6.0)
    candidate(registry, "other-edition", edition="other", rating=7.0)
    registry.validate_scheduling_transition(first.scheduling)
    receipt(registry, "first", TIP - timedelta(minutes=20))
    receipt(registry, "last-good", TIP - timedelta(minutes=5))
    receipt(registry, "failed", TIP - timedelta(minutes=4), outcome="failed")
    receipt(registry, "late", TIP)
    receipt(registry, "other-edition", TIP - timedelta(minutes=1))
    for name, offset, spread, source in (
        ("older", -10, -1.0, "bookmaker_consensus"),
        ("closing", -1, -2.0, "bookmaker_consensus"),
        ("live", 1, -8.0, "bookmaker_consensus"),
        ("poly", -0.5, None, "polymarket"),
    ):
        registry.record_market(
            MarketSnapshotManifest(
                snapshot_id=name,
                game_id=GAME_ID,
                scheduling_instance_id="original",
                source=source,
                captured_at=TIP + timedelta(minutes=offset),
                line_id=name,
                valid=True,
            ),
            {"home_spread": spread},
        )
    selection = registry.lock(
        edition_id="edition",
        scheduling=first.scheduling,
        environment="dev",
        now=TIP + timedelta(hours=3),
    )
    assert selection["forecast"]["selected_forecast_id"] == "last-good"
    assert selection["market"]["selected_snapshot_id"] == "closing"
    payload = registry.selected_payload(selection)
    assert payload["market"]["home_spread"] == -2.0
    assert payload["projection"] == registry.version("last-good")[1]["projection"]
    assert (
        registry.lock(
            edition_id="edition",
            scheduling=first.scheduling,
            environment="dev",
            now=TIP + timedelta(days=1),
        )
        == selection
    )


def test_unpublished_missing_original_is_durable_and_no_replacement(tmp_path):
    registry = ForecastRegistry(tmp_path)
    first, _ = candidate(registry, "only")
    registry.validate_scheduling_transition(first.scheduling)
    selection = registry.lock(
        edition_id="edition", scheduling=first.scheduling, environment="dev", now=TIP
    )
    assert selection["forecast"]["state"] == "missing_original"
    assert registry.selected_payload(selection) is None
    assert len(list((tmp_path / "failures").glob("*.json"))) == 1
    receipt(registry, "only", TIP + timedelta(seconds=1))
    assert (
        registry.lock(edition_id="edition", scheduling=first.scheduling, environment="dev", now=TIP)
        == selection
    )


def test_open_forecasts_change_but_cannot_lock_before_tip(tmp_path):
    registry = ForecastRegistry(tmp_path)
    first, first_payload = candidate(registry, "first")
    _, second_payload = candidate(registry, "second", rating=3.0)
    registry.validate_scheduling_transition(first.scheduling)
    assert (
        registry.resolve(edition_id="edition", forecast_id="first", environment="dev", now=BEFORE)[
            0
        ]
        == first_payload
    )
    assert (
        registry.resolve(edition_id="edition", forecast_id="second", environment="dev", now=BEFORE)[
            0
        ]
        == second_payload
    )
    with pytest.raises(ValueError, match="before"):
        registry.lock(
            edition_id="edition", scheduling=first.scheduling, environment="dev", now=BEFORE
        )


def test_official_reschedule_and_suspension_have_separate_rules(tmp_path):
    registry = ForecastRegistry(tmp_path)
    first, _ = candidate(registry, "first")
    registry.validate_scheduling_transition(first.scheduling)
    receipt(registry, "first", BEFORE)
    selected = registry.lock(
        edition_id="edition", scheduling=first.scheduling, environment="dev", now=TIP
    )
    later = first.scheduling.model_copy(
        update={
            "scheduling_instance_id": "rescheduled",
            "event": "rescheduled",
            "supersedes_scheduling_instance_id": "original",
            "official_schedule_revision": ref("schedule_revision", "b"),
            "cutoff_at": TIP + timedelta(days=1),
            "scheduled_tipoff_at": TIP + timedelta(days=1),
        }
    )
    with pytest.raises(ValueError, match="started"):
        registry.validate_scheduling_transition(later, previous=first.scheduling, play_has_started=True)
    registry.validate_scheduling_transition(later, previous=first.scheduling)
    candidate(registry, "new-instance", scheduling=later)
    receipt(registry, "new-instance", TIP + timedelta(hours=1))
    rescheduled = registry.lock(
        edition_id="edition", scheduling=later, environment="dev", now=later.cutoff_at
    )
    assert rescheduled["forecast"]["selected_forecast_id"] == "new-instance"
    assert selected["forecast"]["selected_forecast_id"] == "first"
    suspended = later.model_copy(
        update={
            "scheduling_instance_id": "suspended",
            "event": "suspended",
            "retains_original_selection_id": selected["forecast"]["selection_id"],
        }
    )
    registry.validate_scheduling_transition(suspended, previous=first.scheduling, play_has_started=True)
    retained = registry.lock(
        edition_id="edition", scheduling=suspended, environment="dev", now=suspended.cutoff_at
    )
    assert retained["forecast"] == selected["forecast"]
    delayed = later.model_copy(update={"event": "original"})
    with pytest.raises(ValueError, match="late starts"):
        registry.validate_scheduling_transition(delayed, previous=first.scheduling)


def test_immutable_versions_and_component_integrity(tmp_path):
    registry = ForecastRegistry(tmp_path)
    manifest, payload = candidate(registry, "first")
    registry.stage(manifest, payload)
    invalid = copy.deepcopy(payload)
    invalid["projected_minutes"]["home"][0]["projected_minutes"] += 1
    with pytest.raises(ValueError, match="checksum"):
        registry.stage(manifest, invalid)
    path = tmp_path / "versions/first.json"
    record = json.loads(path.read_text())
    record["payload"]["projection"]["home_margin"] = 100
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        registry.version("first")


def test_receipt_requires_verified_forecast_and_valid_chronology(tmp_path):
    registry = ForecastRegistry(tmp_path)
    candidate(registry, "first")
    value = PublicationReceipt(
        receipt_id="r",
        forecast_id="first",
        environment="dev",
        published_at=BEFORE,
        publisher_generation=1,
        outcome="successful",
    )
    with pytest.raises(ValueError, match="verified"):
        registry.record_publication(value, loaded_forecast_id="other")
    with pytest.raises(ValueError, match="precede"):
        registry.record_publication(
            value.model_copy(update={"published_at": BEFORE - timedelta(seconds=1)}),
            loaded_forecast_id="first",
        )


def test_replay_does_not_create_live_receipts_for_replaced_forecasts(tmp_path):
    registry = ForecastRegistry(tmp_path / "registry")
    candidate(registry, "old")
    release = tmp_path / "release"
    (release / "selected-schedule").mkdir(parents=True)
    (release / "selected-schedule/forecast-selections.json").write_text(json.dumps({
        "forecast_registry_root": str(registry.root),
        "publication_forecast_ids": ["old"],
        "games": {"one": {"provenance": "retrospective_replay", "forecast_id": "corrected"}},
    }))
    record_verified_release_forecasts(release, environment="dev", generation=3,
        operation_id="corrected-activation", published_at=TIP + timedelta(hours=3))
    assert not list((registry.root / "receipts").glob("*.json"))


def test_verified_activation_receipts_are_idempotent_and_never_backdated(tmp_path):
    registry = ForecastRegistry(tmp_path / "registry")
    candidate(registry, "first")
    release = tmp_path / "release"
    (release / "selected-schedule").mkdir(parents=True)
    (release / "selected-schedule/forecast-selections.json").write_text(
        json.dumps(
            {
                "forecast_registry_root": str(registry.root),
                "publication_forecast_ids": ["first"],
                "games": {"one": {"provenance": "original_live", "version_reference": "first"}},
            }
        )
    )
    record_verified_release_forecasts(
        release,
        environment="dev",
        generation=3,
        operation_id="activation",
        published_at=TIP + timedelta(seconds=1),
    )
    record_verified_release_forecasts(
        release,
        environment="dev",
        generation=3,
        operation_id="activation",
        published_at=TIP + timedelta(seconds=2),
    )
    receipts = list((registry.root / "receipts").glob("*.json"))
    assert len(receipts) == 1
    assert PublicationReceipt.model_validate_json(
        receipts[0].read_bytes()
    ).published_at == TIP + timedelta(seconds=1)


@pytest.mark.parametrize("source", ["bookmaker_consensus", "polymarket"])
def test_selected_market_integrity_is_checked_without_substituting_another_line(tmp_path, source):
    registry = ForecastRegistry(tmp_path)
    forecast, _ = candidate(registry, "first")
    registry.validate_scheduling_transition(forecast.scheduling)
    receipt(registry, "first", BEFORE)
    registry.record_market(
        MarketSnapshotManifest(
            snapshot_id="locked-market",
            game_id=GAME_ID,
            scheduling_instance_id="original",
            source=source,
            captured_at=BEFORE,
            line_id="line",
            valid=True,
        ),
        {"home_spread": -1.5},
    )
    selected = registry.lock(
        edition_id="edition",
        scheduling=forecast.scheduling,
        environment="dev",
        now=TIP,
    )
    field = "market" if source == "bookmaker_consensus" else "prediction_market"
    assert registry.selected_payload(selected)[field]["home_spread"] == -1.5
    path = tmp_path / "markets/locked-market.json"
    record = json.loads(path.read_text())
    record["payload"]["home_spread"] = -20.0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        registry.selected_payload(selected)
