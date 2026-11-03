from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from nba_lineup_model.season.release_artifact_contracts import (
    ActivationReceipt,
    ArtifactReference,
    CoordinatedReleaseManifest,
    DailyDataPartitionManifest,
    ForecastVersionManifest,
    MarketSnapshotManifest,
    ObservationSelection,
    PublicationReceipt,
    RawCaptureManifest,
    ReplayForecastLineage,
    RevisionImpactAssessment,
    SchedulingInstance,
    SealedCodeIdentity,
    SeasonEditionManifest,
    SurfaceBinding,
    ValidationRecord,
    metric_eligibility,
    require_generation_fence,
    select_authoritative_forecast,
    select_evaluation_market,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
TIP = datetime(2026, 11, 3, 3, 0, tzinfo=UTC)


def _ref(kind: str, suffix: str = "a") -> ArtifactReference:
    digest = suffix * 64
    return ArtifactReference(
        artifact_kind=kind,
        schema_version=1,
        content_sha256=digest,
        uri=f"artifact://{kind}/{digest}",
    )


def _code() -> SealedCodeIdentity:
    return SealedCodeIdentity(
        package_sha256=SHA_A,
        dependency_lock_sha256=SHA_B,
        git_revision="c" * 40,
        sealed=True,
    )


def _scheduling() -> SchedulingInstance:
    return SchedulingInstance(
        scheduling_instance_id="schedule-instance-1",
        game_id="0022600100",
        official_schedule_revision=_ref("schedule_revision"),
        scheduled_tipoff_at=TIP,
        cutoff_at=TIP,
    )


def _forecast(forecast_id: str) -> ForecastVersionManifest:
    components = {
        "projected_minutes": _ref("projected_minutes", "a"),
        "player_ratings": _ref("player_ratings", "b"),
        "contributions": _ref("player_contributions", "c"),
        "ledger": _ref("game_ledger", "d"),
    }
    return ForecastVersionManifest(
        forecast_id=forecast_id,
        season="2026-27",
        game_id="0022600100",
        scheduling=_scheduling(),
        model_state=_ref("daily_model_state", "e"),
        edition_id="edition-1",
        injury_revision=_ref("injury_revision", "f"),
        market_snapshots=(),
        roster_revision=_ref("roster_revision", "1"),
        information_cutoff_at=TIP,
        computed_at=TIP,
        components=components,
        predicted_home_margin=2.5,
        predicted_home_win_probability=0.57,
        ledger_home_margin=2.5,
        numeric_tolerance=1e-9,
        validation=ValidationRecord(status="clean"),
        code=_code(),
    )


def _receipt(
    forecast_id: str,
    receipt_id: str,
    published_at: datetime,
    *,
    outcome: str = "successful",
) -> PublicationReceipt:
    return PublicationReceipt(
        receipt_id=receipt_id,
        forecast_id=forecast_id,
        environment="prod",
        published_at=published_at,
        publisher_generation=12,
        outcome=outcome,
        error=None if outcome == "successful" else "publication failed",
    )


def test_strict_tipoff_boundary_selects_latest_successful_publication() -> None:
    forecasts = (_forecast("F1"), _forecast("F2"), _forecast("F3"))
    receipts = (
        _receipt("F1", "R1", datetime(2026, 11, 3, 2, 40, tzinfo=UTC)),
        _receipt("F2", "R2", datetime(2026, 11, 3, 2, 59, tzinfo=UTC)),
        _receipt("F3", "R3", TIP),
    )

    selected = select_authoritative_forecast(
        selection_id="S1",
        edition_id="edition-1",
        scheduling=_scheduling(),
        forecasts=forecasts,
        receipts=receipts,
        selected_at=TIP,
        environment="prod",
    )

    assert selected.state == "selected"
    assert selected.selected_forecast_id == "F2"
    assert selected.publication_receipt_id == "R2"


def test_computation_before_tip_does_not_rescue_late_publication() -> None:
    forecast = _forecast("F2").model_copy(
        update={"computed_at": datetime(2026, 11, 3, 2, 59, tzinfo=UTC)}
    )
    receipt = _receipt("F2", "R2", datetime(2026, 11, 3, 3, 1, tzinfo=UTC))

    selected = select_authoritative_forecast(
        selection_id="S1",
        edition_id="edition-1",
        scheduling=_scheduling(),
        forecasts=(forecast,),
        receipts=(receipt,),
        selected_at=TIP,
        environment="prod",
    )

    assert selected.state == "missing_original"
    assert selected.selected_forecast_id is None


def test_failed_publication_is_not_eligible() -> None:
    selected = select_authoritative_forecast(
        selection_id="S1",
        edition_id="edition-1",
        scheduling=_scheduling(),
        forecasts=(_forecast("F1"),),
        receipts=(
            _receipt(
                "F1",
                "R1",
                datetime(2026, 11, 3, 2, 59, tzinfo=UTC),
                outcome="failed",
            ),
        ),
        selected_at=TIP,
        environment="prod",
    )
    assert selected.state == "missing_original"


def test_market_selection_is_independent_and_uses_latest_pre_tip_snapshot() -> None:
    snapshots = (
        MarketSnapshotManifest(
            snapshot_id="L1-snapshot",
            game_id="0022600100",
            scheduling_instance_id="schedule-instance-1",
            source="bookmaker_consensus",
            captured_at=datetime(2026, 11, 3, 2, 55, tzinfo=UTC),
            line_id="L1",
            valid=True,
        ),
        MarketSnapshotManifest(
            snapshot_id="L2-snapshot",
            game_id="0022600100",
            scheduling_instance_id="schedule-instance-1",
            source="bookmaker_consensus",
            captured_at=datetime(2026, 11, 3, 2, 59, tzinfo=UTC),
            line_id="L2",
            valid=True,
        ),
        MarketSnapshotManifest(
            snapshot_id="L3-snapshot",
            game_id="0022600100",
            scheduling_instance_id="schedule-instance-1",
            source="bookmaker_consensus",
            captured_at=TIP,
            line_id="L3",
            valid=True,
        ),
    )

    selected = select_evaluation_market(
        selection_id="MS1", scheduling=_scheduling(), snapshots=snapshots
    )

    assert selected.state == "selected"
    assert selected.selected_line_id == "L2"


def test_missing_forecast_and_market_have_different_metric_denominators() -> None:
    missing_forecast = select_authoritative_forecast(
        selection_id="S1",
        edition_id="edition-1",
        scheduling=_scheduling(),
        forecasts=(),
        receipts=(),
        selected_at=TIP,
        environment="prod",
    )
    missing_market = select_evaluation_market(
        selection_id="MS1", scheduling=_scheduling(), snapshots=()
    )
    forecast_eligibility = metric_eligibility(missing_forecast, missing_market)
    assert forecast_eligibility.model_dump() == {
        "winner_accuracy": False,
        "brier": False,
        "ats": False,
        "model_vs_market": False,
        "reasons": ("missing_original_forecast",),
    }

    selected_forecast = select_authoritative_forecast(
        selection_id="S2",
        edition_id="edition-1",
        scheduling=_scheduling(),
        forecasts=(_forecast("F1"),),
        receipts=(_receipt("F1", "R1", datetime(2026, 11, 3, 2, 59, tzinfo=UTC)),),
        selected_at=TIP,
        environment="prod",
    )
    market_eligibility = metric_eligibility(selected_forecast, missing_market)
    assert market_eligibility.winner_accuracy is True
    assert market_eligibility.brier is True
    assert market_eligibility.ats is False
    assert market_eligibility.model_vs_market is False


def test_zero_game_day_is_a_clean_no_op_partition() -> None:
    partition = DailyDataPartitionManifest(
        season="2026-27",
        game_date_pacific=date(2026, 11, 3),
        scheduled_game_ids=(),
        required_feeds=(),
        selected_revisions={},
        zero_game_day=True,
        validation=ValidationRecord(status="clean"),
    )
    assert partition.zero_game_day is True


def test_partial_slate_cannot_publish_as_clean_by_dropping_game() -> None:
    with pytest.raises(ValidationError, match="silently drop"):
        DailyDataPartitionManifest(
            season="2026-27",
            game_date_pacific=date(2026, 11, 2),
            scheduled_game_ids=("game-1", "game-2"),
            required_feeds=("finals",),
            selected_revisions={
                "game-1": {"finals": _ref("observed_revision")},
            },
            zero_game_day=False,
            validation=ValidationRecord(status="clean"),
        )


def test_replay_rejects_target_day_and_same_slate_evidence() -> None:
    with pytest.raises(ValidationError, match="earlier complete Pacific slate"):
        ReplayForecastLineage(
            target_game_id="game-2",
            target_game_date_pacific=date(2026, 11, 2),
            evidence_through_date_pacific=date(2026, 11, 2),
            parent_model=_ref("daily_model_state"),
            input_partitions=(_ref("daily_data_partition"),),
        )


def _surface_binding(surface: str, model_release_id: str = "model-12") -> SurfaceBinding:
    keys = {
        "schedule": ("forecast_selection", "schedule_projection", "metric_inputs"),
        "player_pages": ("model_state", "player_trajectory", "game_log"),
        "rankings": ("model_state", "rankings"),
        "teams": ("model_state", "roster", "team_view"),
        "lineups": ("model_state", "roster", "lineup_values"),
        "wins": ("model_state", "win_simulation", "observed_record"),
    }[surface]
    return SurfaceBinding(
        surface=surface,
        artifacts={key: _ref(key) for key in keys},
        model_release_id=model_release_id,
        forecast_selection_generation=7,
        observation_generation=11,
    )


def _release_bindings() -> dict[str, SurfaceBinding]:
    return {
        surface: _surface_binding(surface)
        for surface in (
            "schedule",
            "player_pages",
            "rankings",
            "teams",
            "lineups",
            "wins",
        )
    }


def test_release_requires_every_surface_binding() -> None:
    bindings = _release_bindings()
    del bindings["wins"]
    with pytest.raises(ValidationError, match="must be complete"):
        CoordinatedReleaseManifest(
            release_id="release-1",
            edition=_ref("season_edition"),
            code=_code(),
            frontend=_ref("frontend_bundle"),
            bindings=bindings,
            compatibility_policy_version="surface-compatibility-v1",
            tested_api_ui_contract_version="ui-contract-v1",
            validation_evidence=(_ref("validation_report"),),
        )


def test_current_model_surfaces_cannot_split_brain() -> None:
    bindings = _release_bindings()
    bindings["player_pages"] = _surface_binding("player_pages", "model-11")
    with pytest.raises(ValidationError, match="coordinated release"):
        CoordinatedReleaseManifest(
            release_id="release-1",
            edition=_ref("season_edition"),
            code=_code(),
            frontend=_ref("frontend_bundle"),
            bindings=bindings,
            compatibility_policy_version="surface-compatibility-v1",
            tested_api_ui_contract_version="ui-contract-v1",
            validation_evidence=(_ref("validation_report"),),
        )


def test_mutable_artifact_alias_is_rejected() -> None:
    with pytest.raises(ValidationError, match="latest/current"):
        ArtifactReference(
            artifact_kind="daily_model_state",
            schema_version=1,
            content_sha256=SHA_A,
            uri="s3://nba-gestalt-data/lake/model/latest/manifest.json",
        )


def test_promotable_code_must_be_sealed() -> None:
    with pytest.raises(ValidationError):
        SealedCodeIdentity(
            package_sha256=SHA_A,
            dependency_lock_sha256=SHA_B,
            git_revision="c" * 40,
            sealed=False,
        )


def test_basketball_capture_must_use_official_nba_source() -> None:
    with pytest.raises(ValidationError, match="official NBA"):
        RawCaptureManifest(
            capture_id="capture-1",
            feed="scores",
            source_provider="the_odds_api",
            endpoint="https://example.test/scores",
            game_id="0022600100",
            captured_at=TIP,
            response_status=200,
            payload_sha256=SHA_A,
            payload_uri=f"artifact://raw_payload/{SHA_A}",
            content_type="application/json",
            outcome="success",
        )


def test_stale_observation_requires_explicit_retained_revision() -> None:
    with pytest.raises(ValidationError, match="retain an explicit valid revision"):
        ObservationSelection(
            feed="injuries",
            generation=8,
            state="stale",
            selected_revision=None,
            observed_at=TIP,
            error="latest report failed validation",
        )


def test_historical_model_input_correction_requires_approval() -> None:
    with pytest.raises(ValidationError, match="requires approval"):
        RevisionImpactAssessment(
            model_input_changed=True,
            classification="model_input",
        )


def test_retrospective_edition_requires_explicit_approval() -> None:
    with pytest.raises(ValidationError, match="requires explicit approval"):
        SeasonEditionManifest(
            edition_id="edition-replay-2",
            season="2026-27",
            classification="retrospective_replay",
            model_configuration=_ref("model_configuration"),
            prior_bundle=_ref("prior_bundle"),
            data_collection=_ref("daily_data_collection"),
            covered_game_ids=("0022600100",),
            selection_map=_ref("forecast_selection_map"),
            built_at=TIP,
            historical_information_policy="earlier complete Pacific slates only",
            outcome_tuning_disclosure="no target-game outcomes used",
        )


def test_stale_generation_is_rejected() -> None:
    receipt = ActivationReceipt(
        operation_id="activate-12",
        environment="prod",
        expected_previous_generation=10,
        next_generation=11,
        release_id="release-1",
        edition_id="edition-1",
        binding_map_sha256=SHA_C,
        forecast_selection_generation=7,
        observation_generation=11,
        code=_code(),
        approval_reference="user-approved-2026-11-03",
        outcome="successful",
        activated_at=TIP,
    )
    with pytest.raises(ValueError, match="stale publisher"):
        require_generation_fence(receipt, current_generation=11)


def test_official_reschedule_requires_superseded_instance() -> None:
    with pytest.raises(ValidationError, match="superseded instance"):
        SchedulingInstance(
            scheduling_instance_id="schedule-instance-2",
            game_id="0022600100",
            official_schedule_revision=_ref("schedule_revision"),
            scheduled_tipoff_at=TIP,
            cutoff_at=TIP,
            event="rescheduled",
        )


def test_spread_ledger_must_reconcile() -> None:
    payload = _forecast("F1").model_dump()
    payload["predicted_home_margin"] = 3.0
    with pytest.raises(ValidationError, match="does not reproduce"):
        ForecastVersionManifest.model_validate(payload)
