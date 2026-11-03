"""Executable contracts for versioned observations, forecasts, and releases.

The models in this module describe immutable identities and validate the
cross-artifact invariants approved in GitHub issue 53.  They do not publish,
activate, rebuild, or migrate any runtime artifact.  Later release slices must
use these contracts rather than creating another source of truth.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]
ArtifactKind = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
Feed = Literal["injuries", "odds", "scores", "finals", "schedule"]
Environment = Literal["dev", "prod"]
Surface = Literal[
    "schedule",
    "player_pages",
    "rankings",
    "teams",
    "lineups",
    "wins",
]

_MUTABLE_ALIAS = re.compile(r"(?:^|/)(?:latest|current)(?:/|$)", re.IGNORECASE)
_REQUIRED_SURFACES = {
    "schedule",
    "player_pages",
    "rankings",
    "teams",
    "lineups",
    "wins",
}
_SURFACE_REQUIREMENTS = {
    "schedule": {"forecast_selection", "schedule_projection", "metric_inputs"},
    "player_pages": {"model_state", "player_trajectory", "game_log"},
    "rankings": {"model_state", "rankings"},
    "teams": {"model_state", "roster", "team_view"},
    "lineups": {"model_state", "roster", "lineup_values"},
    "wins": {"model_state", "win_simulation", "observed_record"},
}


def _require_aware(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value


class ContractModel(BaseModel):
    """Strict base for serialized contract records."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ArtifactReference(ContractModel):
    """Content identity plus a retrievable, non-authoritative location."""

    artifact_kind: ArtifactKind
    schema_version: int = Field(ge=1)
    content_sha256: Sha256
    uri: NonEmpty

    @field_validator("uri")
    @classmethod
    def reject_mutable_aliases(cls, value: str) -> str:
        path = value.split("?", 1)[0].rstrip("/")
        if _MUTABLE_ALIAS.search(path):
            raise ValueError("artifact references cannot resolve through latest/current")
        return value


class SealedCodeIdentity(ContractModel):
    package_sha256: Sha256
    dependency_lock_sha256: Sha256
    git_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    sealed: Literal[True]


class ValidationRecord(ContractModel):
    status: Literal["clean", "error"]
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def status_matches_errors(self) -> Self:
        if self.status == "clean" and self.errors:
            raise ValueError("clean validation cannot contain errors")
        if self.status == "error" and not self.errors:
            raise ValueError("error validation must explain the failure")
        return self


class RawCaptureManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["raw_capture"] = "raw_capture"
    capture_id: NonEmpty
    feed: Feed
    source_provider: Literal["nba_official", "the_odds_api", "polymarket"]
    endpoint: NonEmpty
    game_id: str | None = None
    player_id: int | None = None
    source_published_at: datetime | None = None
    captured_at: datetime
    response_status: int | None = Field(default=None, ge=100, le=599)
    payload_sha256: Sha256 | None = None
    payload_uri: str | None = None
    content_type: str | None = None
    outcome: Literal["success", "failure"]
    failure_reason: str | None = None

    @field_validator("endpoint")
    @classmethod
    def endpoint_has_no_secret(cls, value: str) -> str:
        lowered = value.casefold()
        if "espn" in lowered:
            raise ValueError("ESPN basketball sources and identifiers are forbidden")
        if any(token in lowered for token in ("apikey=", "api_key=", "token=")):
            raise ValueError("capture endpoints must not contain secrets")
        return value

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        _require_aware(self.captured_at, "captured_at")
        if self.source_published_at is not None:
            _require_aware(self.source_published_at, "source_published_at")
        if self.feed in {"injuries", "scores", "finals", "schedule"}:
            if self.source_provider != "nba_official":
                raise ValueError("basketball observation feeds must use official NBA sources")
        if self.outcome == "success":
            if self.payload_sha256 is None or not self.payload_uri:
                raise ValueError("successful capture requires payload identity and location")
            if self.failure_reason is not None:
                raise ValueError("successful capture cannot have a failure reason")
        elif not self.failure_reason:
            raise ValueError("failed capture must explain the failure")
        return self


class RevisionImpactAssessment(ContractModel):
    presentation_artifacts: tuple[ArtifactReference, ...] = ()
    grade_artifacts: tuple[ArtifactReference, ...] = ()
    model_input_changed: bool
    model_descendants: tuple[ArtifactReference, ...] = ()
    classification: Literal["presentation_only", "grade_only", "model_input"]
    approval_reference: str | None = None

    @model_validator(mode="after")
    def causal_changes_require_approval(self) -> Self:
        if self.classification == "model_input":
            if not self.model_input_changed:
                raise ValueError("model_input impact must mark the model input changed")
            if not self.approval_reference:
                raise ValueError("historical model-input correction requires approval")
        elif self.model_input_changed:
            raise ValueError("model input changes must use model_input classification")
        return self


class ObservedRevisionManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["observed_revision"] = "observed_revision"
    revision_id: NonEmpty
    feed: Feed
    game_id: NonEmpty
    capture_ids: tuple[NonEmpty, ...] = Field(min_length=1)
    parser_code: SealedCodeIdentity
    components: dict[NonEmpty, ArtifactReference]
    validation: ValidationRecord
    corrected_reason: str | None = None
    supersedes_revision_id: str | None = None
    impact: RevisionImpactAssessment | None = None
    processed_at: datetime

    @model_validator(mode="after")
    def correction_fields_are_complete(self) -> Self:
        _require_aware(self.processed_at, "processed_at")
        correction = self.supersedes_revision_id is not None
        if correction != (self.corrected_reason is not None):
            raise ValueError("correction reason and superseded revision must travel together")
        if correction and self.impact is None:
            raise ValueError("corrected revision requires an impact assessment")
        return self


class ObservationSelection(ContractModel):
    feed: Feed
    generation: int = Field(ge=1)
    state: Literal["current", "stale", "error"]
    selected_revision: ArtifactReference | None
    observed_at: datetime
    error: str | None = None

    @model_validator(mode="after")
    def selection_retains_last_valid(self) -> Self:
        _require_aware(self.observed_at, "observed_at")
        if self.selected_revision is None:
            raise ValueError("observation selection must retain an explicit valid revision")
        if self.state == "current" and self.error is not None:
            raise ValueError("current observation cannot carry an error")
        if self.state in {"stale", "error"} and not self.error:
            raise ValueError("stale/error observation must explain its state")
        return self


class DailyDataPartitionManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["daily_data_partition"] = "daily_data_partition"
    season: NonEmpty
    game_date_pacific: date
    scheduled_game_ids: tuple[NonEmpty, ...]
    required_feeds: tuple[Feed, ...]
    selected_revisions: dict[NonEmpty, dict[Feed, ArtifactReference]]
    zero_game_day: bool
    validation: ValidationRecord

    @model_validator(mode="after")
    def exact_slate_coverage(self) -> Self:
        expected = set(self.scheduled_game_ids)
        if len(expected) != len(self.scheduled_game_ids):
            raise ValueError("scheduled game IDs must be unique")
        if self.zero_game_day:
            if expected or self.selected_revisions:
                raise ValueError("zero-game partition must have an empty expected set")
            return self
        if not expected:
            raise ValueError("non-zero partition requires scheduled games")
        if set(self.selected_revisions) != expected:
            raise ValueError("daily partition cannot add or silently drop a scheduled game")
        if self.validation.status == "clean":
            required = set(self.required_feeds)
            for game_id, feeds in self.selected_revisions.items():
                missing = required - set(feeds)
                if missing:
                    raise ValueError(f"{game_id} is missing required feeds: {sorted(missing)}")
        return self


class DailyDataCollectionManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["daily_data_collection"] = "daily_data_collection"
    season: NonEmpty
    through_date_pacific: date
    partitions: dict[date, ArtifactReference]

    @model_validator(mode="after")
    def through_date_matches_selection(self) -> Self:
        if not self.partitions:
            raise ValueError("daily data collection requires partitions")
        if max(self.partitions) != self.through_date_pacific:
            raise ValueError("collection through-date must match its selected partitions")
        return self


class DailyModelStateManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["daily_model_state"] = "daily_model_state"
    season: NonEmpty
    evidence_through_date_pacific: date
    prior_bundle: ArtifactReference
    parent_model: ArtifactReference | None = None
    daily_data: tuple[ArtifactReference, ...] = Field(min_length=1)
    algorithm_id: NonEmpty
    hyperparameters_sha256: Sha256
    components: dict[NonEmpty, ArtifactReference]
    cumulative_game_ids: tuple[NonEmpty, ...]
    validation: ValidationRecord
    code: SealedCodeIdentity

    @model_validator(mode="after")
    def coupled_components_are_present(self) -> Self:
        required = {"ratings", "minutes", "context"}
        missing = required - set(self.components)
        if missing:
            raise ValueError(f"coupled model state is missing: {sorted(missing)}")
        if len(set(self.cumulative_game_ids)) != len(self.cumulative_game_ids):
            raise ValueError("cumulative game IDs must be unique")
        return self


class SchedulingInstance(ContractModel):
    scheduling_instance_id: NonEmpty
    game_id: NonEmpty
    official_schedule_revision: ArtifactReference
    scheduled_tipoff_at: datetime
    cutoff_at: datetime
    event: Literal["original", "rescheduled", "suspended"] = "original"
    supersedes_scheduling_instance_id: str | None = None
    retains_original_selection_id: str | None = None

    @model_validator(mode="after")
    def validate_scheduling_event(self) -> Self:
        _require_aware(self.scheduled_tipoff_at, "scheduled_tipoff_at")
        _require_aware(self.cutoff_at, "cutoff_at")
        if self.cutoff_at != self.scheduled_tipoff_at:
            raise ValueError("forecast cutoff is the official scheduled tipoff")
        if self.event == "rescheduled" and not self.supersedes_scheduling_instance_id:
            raise ValueError("official reschedule requires the superseded instance")
        if self.event == "suspended" and not self.retains_original_selection_id:
            raise ValueError("suspended game must retain its original selection")
        return self


class ForecastVersionManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["forecast_version"] = "forecast_version"
    forecast_id: NonEmpty
    season: NonEmpty
    game_id: NonEmpty
    scheduling: SchedulingInstance
    model_state: ArtifactReference
    edition_id: NonEmpty
    injury_revision: ArtifactReference
    market_snapshots: tuple[ArtifactReference, ...] = ()
    roster_revision: ArtifactReference
    information_cutoff_at: datetime
    computed_at: datetime
    components: dict[NonEmpty, ArtifactReference]
    predicted_home_margin: float
    predicted_home_win_probability: float = Field(ge=0.0, le=1.0)
    ledger_home_margin: float
    numeric_tolerance: float = Field(gt=0.0)
    validation: ValidationRecord
    code: SealedCodeIdentity

    @model_validator(mode="after")
    def forecast_is_auditable(self) -> Self:
        _require_aware(self.information_cutoff_at, "information_cutoff_at")
        _require_aware(self.computed_at, "computed_at")
        if self.game_id != self.scheduling.game_id:
            raise ValueError("forecast game and scheduling instance disagree")
        if self.information_cutoff_at != self.scheduling.cutoff_at:
            raise ValueError("forecast information cutoff must match scheduling instance")
        required = {"projected_minutes", "player_ratings", "contributions", "ledger"}
        missing = required - set(self.components)
        if missing:
            raise ValueError(f"forecast components are missing: {sorted(missing)}")
        if abs(self.predicted_home_margin - self.ledger_home_margin) > self.numeric_tolerance:
            raise ValueError("forecast ledger does not reproduce the predicted spread")
        return self


class PublicationReceipt(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["forecast_publication_receipt"] = "forecast_publication_receipt"
    receipt_id: NonEmpty
    forecast_id: NonEmpty
    environment: Environment
    published_at: datetime
    publisher_generation: int = Field(ge=1)
    outcome: Literal["successful", "failed"]
    error: str | None = None

    @model_validator(mode="after")
    def receipt_is_consistent(self) -> Self:
        _require_aware(self.published_at, "published_at")
        if self.outcome == "successful" and self.error is not None:
            raise ValueError("successful publication cannot carry an error")
        if self.outcome == "failed" and not self.error:
            raise ValueError("failed publication must explain the failure")
        return self


class ForecastSelectionManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["forecast_selection"] = "forecast_selection"
    selection_id: NonEmpty
    edition_id: NonEmpty
    game_id: NonEmpty
    scheduling_instance_id: NonEmpty
    rule_version: NonEmpty
    cutoff_at: datetime
    state: Literal["selected", "missing_original"]
    selected_forecast_id: str | None = None
    publication_receipt_id: str | None = None
    selected_at: datetime
    supersedes_selection_id: str | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def selected_or_explicitly_missing(self) -> Self:
        _require_aware(self.cutoff_at, "cutoff_at")
        _require_aware(self.selected_at, "selected_at")
        if self.state == "selected":
            if not self.selected_forecast_id or not self.publication_receipt_id:
                raise ValueError("selected forecast requires its publication receipt")
            if self.failure_reason is not None:
                raise ValueError("selected forecast cannot have a missing reason")
        else:
            if self.selected_forecast_id or self.publication_receipt_id:
                raise ValueError("missing-original state cannot substitute a forecast")
            if not self.failure_reason:
                raise ValueError("missing-original state must explain the failure")
        return self


class MarketSnapshotManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["market_snapshot"] = "market_snapshot"
    snapshot_id: NonEmpty
    game_id: NonEmpty
    scheduling_instance_id: NonEmpty
    source: Literal["bookmaker_consensus", "polymarket"]
    captured_at: datetime
    line_id: NonEmpty
    valid: bool
    validation_error: str | None = None

    @model_validator(mode="after")
    def snapshot_is_consistent(self) -> Self:
        _require_aware(self.captured_at, "captured_at")
        if self.valid and self.validation_error is not None:
            raise ValueError("valid market snapshot cannot carry an error")
        if not self.valid and not self.validation_error:
            raise ValueError("invalid market snapshot must explain the failure")
        return self


class MarketSelectionManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["market_selection"] = "market_selection"
    selection_id: NonEmpty
    game_id: NonEmpty
    scheduling_instance_id: NonEmpty
    cutoff_at: datetime
    state: Literal["selected", "missing_market"]
    selected_snapshot_id: str | None = None
    selected_line_id: str | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def selected_or_missing(self) -> Self:
        _require_aware(self.cutoff_at, "cutoff_at")
        if self.state == "selected":
            if not self.selected_snapshot_id or not self.selected_line_id:
                raise ValueError("selected market requires snapshot and line IDs")
            if self.failure_reason is not None:
                raise ValueError("selected market cannot have a missing reason")
        else:
            if self.selected_snapshot_id or self.selected_line_id:
                raise ValueError("missing market cannot contain a substitute line")
            if not self.failure_reason:
                raise ValueError("missing market must explain the failure")
        return self


class MetricEligibility(ContractModel):
    winner_accuracy: bool
    brier: bool
    ats: bool
    model_vs_market: bool
    reasons: tuple[str, ...] = ()


class ReplayForecastLineage(ContractModel):
    target_game_id: NonEmpty
    target_game_date_pacific: date
    evidence_through_date_pacific: date
    parent_model: ArtifactReference
    input_partitions: tuple[ArtifactReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def only_earlier_complete_slates(self) -> Self:
        if self.evidence_through_date_pacific >= self.target_game_date_pacific:
            raise ValueError("forecast evidence must stop at an earlier complete Pacific slate")
        return self


class SeasonEditionManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["season_edition"] = "season_edition"
    edition_id: NonEmpty
    season: NonEmpty
    classification: Literal["original_live", "retrospective_replay"]
    model_configuration: ArtifactReference
    prior_bundle: ArtifactReference
    data_collection: ArtifactReference
    covered_game_ids: tuple[NonEmpty, ...]
    selection_map: ArtifactReference
    built_at: datetime
    historical_information_policy: NonEmpty
    outcome_tuning_disclosure: NonEmpty
    approval_reference: str | None = None

    @model_validator(mode="after")
    def replay_is_disclosed_and_approved(self) -> Self:
        _require_aware(self.built_at, "built_at")
        if len(set(self.covered_game_ids)) != len(self.covered_game_ids):
            raise ValueError("edition game coverage must be unique")
        if self.classification == "retrospective_replay" and not self.approval_reference:
            raise ValueError("retrospective edition requires explicit approval")
        return self


class SurfaceBinding(ContractModel):
    surface: Surface
    artifacts: dict[NonEmpty, ArtifactReference]
    model_release_id: NonEmpty
    forecast_selection_generation: int = Field(ge=1)
    observation_generation: int = Field(ge=1)

    @model_validator(mode="after")
    def required_artifacts_are_present(self) -> Self:
        missing = _SURFACE_REQUIREMENTS[self.surface] - set(self.artifacts)
        if missing:
            raise ValueError(f"{self.surface} binding is missing: {sorted(missing)}")
        return self


class CoordinatedReleaseManifest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["coordinated_release"] = "coordinated_release"
    release_id: NonEmpty
    edition: ArtifactReference
    code: SealedCodeIdentity
    frontend: ArtifactReference
    bindings: dict[Surface, SurfaceBinding]
    compatibility_policy_version: NonEmpty
    tested_api_ui_contract_version: NonEmpty
    validation_evidence: tuple[ArtifactReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_and_compatible(self) -> Self:
        if set(self.bindings) != _REQUIRED_SURFACES:
            missing = _REQUIRED_SURFACES - set(self.bindings)
            extra = set(self.bindings) - _REQUIRED_SURFACES
            raise ValueError(
                f"release binding map must be complete; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        for name, binding in self.bindings.items():
            if binding.surface != name:
                raise ValueError(f"binding key and surface disagree for {name}")
        current_model_surfaces = ("player_pages", "rankings", "teams", "lineups", "wins")
        model_ids = {self.bindings[name].model_release_id for name in current_model_surfaces}
        if len(model_ids) != 1:
            raise ValueError("current-model surfaces must advance as one coordinated release")
        return self


class ActivationReceipt(ContractModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["activation_receipt"] = "activation_receipt"
    operation_id: NonEmpty
    environment: Environment
    expected_previous_generation: int = Field(ge=0)
    next_generation: int = Field(ge=1)
    release_id: NonEmpty
    edition_id: NonEmpty
    binding_map_sha256: Sha256
    forecast_selection_generation: int = Field(ge=1)
    observation_generation: int = Field(ge=1)
    code: SealedCodeIdentity
    approval_reference: str | None = None
    outcome: Literal["successful", "rejected", "failed"]
    activated_at: datetime
    error: str | None = None

    @model_validator(mode="after")
    def activation_is_monotonic(self) -> Self:
        _require_aware(self.activated_at, "activated_at")
        if self.next_generation != self.expected_previous_generation + 1:
            raise ValueError("activation generation must advance exactly once")
        if self.outcome == "successful" and self.error is not None:
            raise ValueError("successful activation cannot carry an error")
        if self.outcome != "successful" and not self.error:
            raise ValueError("rejected/failed activation must explain the failure")
        if self.environment == "prod" and not self.approval_reference:
            raise ValueError("production activation requires explicit approval")
        return self


def select_authoritative_forecast(
    *,
    selection_id: str,
    edition_id: str,
    scheduling: SchedulingInstance,
    forecasts: tuple[ForecastVersionManifest, ...],
    receipts: tuple[PublicationReceipt, ...],
    selected_at: datetime,
    environment: Environment,
    rule_version: str = "latest-successful-publication-strictly-before-tip-v1",
) -> ForecastSelectionManifest:
    """Select by actual successful publication, never computation/file time."""

    receipt_by_forecast: dict[str, list[PublicationReceipt]] = {}
    for receipt in receipts:
        if receipt.environment == environment and receipt.outcome == "successful":
            receipt_by_forecast.setdefault(receipt.forecast_id, []).append(receipt)
    candidates: list[tuple[datetime, str, ForecastVersionManifest, PublicationReceipt]] = []
    for forecast in forecasts:
        if (
            forecast.game_id != scheduling.game_id
            or forecast.scheduling.scheduling_instance_id != scheduling.scheduling_instance_id
            or forecast.validation.status != "clean"
        ):
            continue
        for receipt in receipt_by_forecast.get(forecast.forecast_id, []):
            if receipt.published_at < scheduling.cutoff_at:
                candidates.append((receipt.published_at, receipt.receipt_id, forecast, receipt))
    if not candidates:
        return ForecastSelectionManifest(
            selection_id=selection_id,
            edition_id=edition_id,
            game_id=scheduling.game_id,
            scheduling_instance_id=scheduling.scheduling_instance_id,
            rule_version=rule_version,
            cutoff_at=scheduling.cutoff_at,
            state="missing_original",
            selected_at=selected_at,
            failure_reason="no successful forecast publication strictly before cutoff",
        )
    _, _, forecast, receipt = max(candidates, key=lambda row: (row[0], row[1]))
    return ForecastSelectionManifest(
        selection_id=selection_id,
        edition_id=edition_id,
        game_id=scheduling.game_id,
        scheduling_instance_id=scheduling.scheduling_instance_id,
        rule_version=rule_version,
        cutoff_at=scheduling.cutoff_at,
        state="selected",
        selected_forecast_id=forecast.forecast_id,
        publication_receipt_id=receipt.receipt_id,
        selected_at=selected_at,
    )


def select_evaluation_market(
    *,
    selection_id: str,
    scheduling: SchedulingInstance,
    snapshots: tuple[MarketSnapshotManifest, ...],
) -> MarketSelectionManifest:
    """Select the market independently from the forecast version."""

    candidates = [
        snapshot
        for snapshot in snapshots
        if snapshot.game_id == scheduling.game_id
        and snapshot.scheduling_instance_id == scheduling.scheduling_instance_id
        and snapshot.source == "bookmaker_consensus"
        and snapshot.valid
        and snapshot.captured_at < scheduling.cutoff_at
    ]
    if not candidates:
        return MarketSelectionManifest(
            selection_id=selection_id,
            game_id=scheduling.game_id,
            scheduling_instance_id=scheduling.scheduling_instance_id,
            cutoff_at=scheduling.cutoff_at,
            state="missing_market",
            failure_reason="no valid bookmaker consensus captured strictly before cutoff",
        )
    selected = max(candidates, key=lambda item: (item.captured_at, item.snapshot_id))
    return MarketSelectionManifest(
        selection_id=selection_id,
        game_id=scheduling.game_id,
        scheduling_instance_id=scheduling.scheduling_instance_id,
        cutoff_at=scheduling.cutoff_at,
        state="selected",
        selected_snapshot_id=selected.snapshot_id,
        selected_line_id=selected.line_id,
    )


def metric_eligibility(
    forecast: ForecastSelectionManifest,
    market: MarketSelectionManifest,
) -> MetricEligibility:
    if forecast.state == "missing_original":
        return MetricEligibility(
            winner_accuracy=False,
            brier=False,
            ats=False,
            model_vs_market=False,
            reasons=("missing_original_forecast",),
        )
    if market.state == "missing_market":
        return MetricEligibility(
            winner_accuracy=True,
            brier=True,
            ats=False,
            model_vs_market=False,
            reasons=("missing_evaluation_market",),
        )
    return MetricEligibility(
        winner_accuracy=True,
        brier=True,
        ats=True,
        model_vs_market=True,
    )


def require_generation_fence(receipt: ActivationReceipt, *, current_generation: int) -> None:
    if receipt.expected_previous_generation != current_generation:
        raise ValueError(
            "stale publisher: expected generation "
            f"{receipt.expected_previous_generation}, active generation {current_generation}"
        )
