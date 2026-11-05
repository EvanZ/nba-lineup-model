"""Versioned forecasts and immutable cutoff decisions.

This module is the separate forecast/publication job boundary. It consumes
already captured inputs. It does not poll providers or train player models.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nba_lineup_model.season.daily_artifact_common import canonical_json
from nba_lineup_model.season.release_artifact_contracts import (
    ForecastVersionManifest,
    MarketSnapshotManifest,
    PublicationReceipt,
    SchedulingInstance,
    select_authoritative_forecast,
    select_evaluation_market,
)
from nba_lineup_model.season.release_publisher import publisher_lock
from nba_lineup_model.season.serving_selection import digest, read_object


class ForecastRegistry:
    """Immutable inputs/receipts/selections, serialized by the publisher lock."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _put(self, category: str, identity: str, value: dict[str, Any]) -> Path:
        if not identity or Path(identity).name != identity or identity in {".", ".."}:
            raise ValueError("Invalid registry identity")
        path = self.root / category / f"{identity}.json"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.parent.chmod(0o755)
        data = canonical_json(value) + b"\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                # The privileged publisher and unprivileged API share these
                # non-secret immutable records. NamedTemporaryFile defaults to 0600.
                os.fchmod(handle.fileno(), 0o644)
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(
                        f"Immutable registry entry changed: {category}:{identity}"
                    ) from None
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def stage(self, forecast: ForecastVersionManifest, payload: dict[str, Any]) -> str:
        """Stage a forecast. Staging never supplies publication eligibility."""
        if forecast.validation.status != "clean":
            raise ValueError("Cannot stage an invalid forecast")
        if str(payload.get("game_id")) != forecast.game_id:
            raise ValueError("Forecast payload game identity mismatch")
        from nba_lineup_model.season.game_prediction_artifact import (
            require_projection_reconciliation,
        )

        require_projection_reconciliation(payload["projection"], game_id=forecast.game_id)
        for key, expected in (
            ("home_margin", forecast.predicted_home_margin),
            ("home_win_probability", forecast.predicted_home_win_probability),
        ):
            if abs(float(payload["projection"][key]) - expected) > forecast.numeric_tolerance:
                raise ValueError(f"Forecast manifest/payload mismatch: {key}")
        component_values = {
            "projected_minutes": payload["projected_minutes"],
            "player_ratings": {
                str(row["player_id"]): row["frozen_nail"]
                for side in ("home", "away")
                for row in payload["projection"]["ledger"][side]["players"]
            },
            "contributions": {
                side: payload["projection"]["ledger"][side]["players"] for side in ("home", "away")
            },
            "ledger": payload["projection"]["ledger"],
        }
        for key, value in component_values.items():
            if forecast.components[key].content_sha256 != digest(value):
                raise ValueError(f"Forecast component checksum mismatch: {key}")
        record = {"manifest": forecast.model_dump(mode="json"), "payload": payload}
        record["record_sha256"] = digest(record)
        with publisher_lock(self.root):
            self._put("versions", forecast.forecast_id, record)
        return forecast.forecast_id

    def ensure_receipts_readable(self) -> int:
        """Migrate old root-only receipts without changing a single content byte."""
        changed = 0
        with publisher_lock(self.root):
            for path in sorted((self.root / "receipts").glob("*.json")):
                if path.is_symlink():
                    raise ValueError("Publication receipt must not be a symlink")
                before = path.read_bytes()
                receipt = PublicationReceipt.model_validate_json(before)
                if receipt.receipt_id != path.stem:
                    raise ValueError("Publication receipt filename differs from identity")
                self.version(receipt.forecast_id)
                if path.stat().st_mode & 0o777 != 0o644:
                    path.chmod(0o644)
                    changed += 1
                if path.read_bytes() != before:
                    raise ValueError("Publication receipt changed during permission migration")
        return changed

    def record_publication(
        self,
        receipt: PublicationReceipt,
        *,
        loaded_forecast_id: str,
    ) -> None:
        """Called only after the publisher verifies what the API loaded."""
        version = self.version(receipt.forecast_id)[0]
        if receipt.forecast_id != loaded_forecast_id:
            raise ValueError("Publication receipt does not match the verified loaded forecast")
        if receipt.published_at < version.computed_at:
            raise ValueError("Publication receipt cannot precede computation")
        with publisher_lock(self.root):
            self._put("receipts", receipt.receipt_id, receipt.model_dump(mode="json"))

    def record_market(self, snapshot: MarketSnapshotManifest, payload: dict[str, Any]) -> None:
        with publisher_lock(self.root):
            record = {"manifest": snapshot.model_dump(mode="json"), "payload": payload}
            record["record_sha256"] = digest(record)
            self._put("markets", snapshot.snapshot_id, record)

    @staticmethod
    def validate_scheduling_transition(
        scheduling: SchedulingInstance,
        *,
        previous: SchedulingInstance | None = None,
        play_has_started: bool = False,
    ) -> None:
        """Validate schedule changes without creating a second source of truth."""
        if previous is not None:
            if previous.game_id != scheduling.game_id:
                raise ValueError("Rescheduling cannot change NBA game identity")
            if scheduling.event == "rescheduled":
                if play_has_started:
                    raise ValueError("A started game retains its original selection")
                if scheduling.supersedes_scheduling_instance_id != previous.scheduling_instance_id:
                    raise ValueError("Reschedule must explicitly replace the prior instance")
                if scheduling.official_schedule_revision == previous.official_schedule_revision:
                    raise ValueError("Rescheduling requires a new official NBA schedule revision")
            elif scheduling.event == "suspended":
                if not play_has_started:
                    raise ValueError("A suspension requires that play has started")
            elif scheduling.cutoff_at != previous.cutoff_at:
                raise ValueError("Ordinary late starts cannot move the cutoff")
        elif scheduling.event != "original":
            raise ValueError("Rescheduling or suspension requires the previous instance")

    def lock(
        self,
        *,
        edition_id: str,
        scheduling: SchedulingInstance,
        environment: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        if now < scheduling.cutoff_at:
            raise ValueError("Cannot lock before scheduled tipoff")
        key = digest(
            {
                "edition": edition_id,
                "game": scheduling.game_id,
                "instance": scheduling.scheduling_instance_id,
                "environment": environment,
            }
        )
        with publisher_lock(self.root):
            existing = self.root / "selections" / f"{key}.json"
            if existing.is_file():
                return read_object(existing)
            if scheduling.event == "suspended":
                matches = [read_object(p) for p in (self.root / "selections").glob("*.json")]
                prior = next(
                    (
                        v
                        for v in matches
                        if v["forecast"]["selection_id"] == scheduling.retains_original_selection_id
                    ),
                    None,
                )
                if (
                    prior is None
                    or prior["edition_id"] != edition_id
                    or prior["environment"] != environment
                ):
                    raise ValueError("Suspended game has no matching original selection")
                if prior["forecast"]["game_id"] != scheduling.game_id:
                    raise ValueError("Suspended game identity differs")
                result = {
                    **prior,
                    "retained_for_scheduling_instance": scheduling.scheduling_instance_id,
                }
            else:
                forecasts = tuple(
                    ForecastVersionManifest.model_validate_json(
                        json.dumps(read_object(p)["manifest"])
                    )
                    for p in (self.root / "versions").glob("*.json")
                )
                receipts = tuple(
                    PublicationReceipt.model_validate_json(p.read_bytes())
                    for p in (self.root / "receipts").glob("*.json")
                )
                snapshots = tuple(
                    self.market(p.stem)[0] for p in (self.root / "markets").glob("*.json")
                )
                forecast = select_authoritative_forecast(
                    selection_id=key,
                    edition_id=edition_id,
                    scheduling=scheduling,
                    forecasts=forecasts,
                    receipts=receipts,
                    selected_at=now,
                    environment=environment,
                )
                market = select_evaluation_market(
                    selection_id=key + "-market",
                    scheduling=scheduling,
                    snapshots=snapshots,
                )
                prediction_market = select_evaluation_market(
                    selection_id=key + "-polymarket",
                    scheduling=scheduling,
                    snapshots=snapshots,
                    source="polymarket",
                )
                result = {
                    "edition_id": edition_id,
                    "environment": environment,
                    "forecast": forecast.model_dump(mode="json"),
                    "market": market.model_dump(mode="json"),
                    "prediction_market": prediction_market.model_dump(mode="json"),
                    "supersedes_scheduling_instance_id": (
                        scheduling.supersedes_scheduling_instance_id
                    ),
                }
            self._put("selections", key, result)
            if (
                result["forecast"]["state"] == "missing_original"
                or result["market"]["state"] == "missing_market"
            ):
                # A durable failure record is consumed by the daily DQ/alert job.
                self._put("failures", key, result)
            return result

    def selected_payload(self, selection: dict[str, Any]) -> dict[str, Any] | None:
        """Use exact references. No latest-file or archive search fallback."""
        forecast = selection["forecast"]
        if forecast["state"] == "missing_original":
            return None
        manifest, source_payload = self.version(forecast["selected_forecast_id"])
        if (
            manifest.edition_id != selection["edition_id"]
            or manifest.game_id != forecast["game_id"]
        ):
            raise ValueError("Selected forecast belongs to another edition/game")
        payload = json.loads(json.dumps(source_payload))
        for field, source in (
            ("market", "bookmaker_consensus"),
            ("prediction_market", "polymarket"),
        ):
            chosen = selection.get(field, {})
            payload[field] = None
            if chosen.get("state") != "selected":
                continue
            market, market_payload = self.market(chosen["selected_snapshot_id"])
            if (
                market.game_id != manifest.game_id
                or market.source != source
                or market.scheduling_instance_id != chosen["scheduling_instance_id"]
                or market.line_id != chosen["selected_line_id"]
                or not market.valid
                or market.captured_at >= datetime.fromisoformat(chosen["cutoff_at"])
            ):
                raise ValueError("Selected market reference does not match its snapshot")
            payload[field] = market_payload
        payload["forecast_selection"] = selection
        return payload

    def market(self, snapshot_id: str) -> tuple[MarketSnapshotManifest, dict[str, Any]]:
        if Path(snapshot_id).name != snapshot_id or snapshot_id in {".", ".."}:
            raise ValueError("Invalid market reference")
        record = read_object(self.root / "markets" / f"{snapshot_id}.json")
        if (
            digest({k: v for k, v in record.items() if k != "record_sha256"})
            != record["record_sha256"]
        ):
            raise ValueError("Market snapshot checksum mismatch")
        manifest = MarketSnapshotManifest.model_validate_json(json.dumps(record["manifest"]))
        if manifest.snapshot_id != snapshot_id:
            raise ValueError("Market snapshot identity mismatch")
        return manifest, record["payload"]

    def version(self, forecast_id: str) -> tuple[ForecastVersionManifest, dict[str, Any]]:
        if Path(forecast_id).name != forecast_id or forecast_id in {".", ".."}:
            raise ValueError("Invalid forecast reference")
        record = read_object(self.root / "versions" / f"{forecast_id}.json")
        if (
            digest({k: v for k, v in record.items() if k != "record_sha256"})
            != record["record_sha256"]
        ):
            raise ValueError("Staged forecast checksum mismatch")
        manifest = ForecastVersionManifest.model_validate_json(json.dumps(record["manifest"]))
        if manifest.forecast_id != forecast_id:
            raise ValueError("Staged forecast identity mismatch")
        return manifest, record["payload"]

    def resolve(
        self,
        *,
        edition_id: str,
        forecast_id: str,
        environment: str,
        now: datetime | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """One resolver for assembly, cards, ledgers, grading and metrics.

        Before cutoff a candidate can be displayed by an activation, but only
        the publisher's verified receipt makes it eligible for the original.
        After cutoff even a late worker must use the locked historical choice.
        """
        now = now or datetime.now(UTC)
        manifest, payload = self.version(forecast_id)
        if manifest.edition_id != edition_id:
            raise ValueError("Forecast belongs to another season edition")
        if now < manifest.scheduling.cutoff_at:
            return payload, {
                "state": "open",
                "edition_id": edition_id,
                "game_id": manifest.game_id,
                "forecast_id": forecast_id,
                "scheduling_instance_id": manifest.scheduling.scheduling_instance_id,
                "cutoff_at": manifest.scheduling.cutoff_at.isoformat(),
            }
        selection = self.lock(
            edition_id=edition_id,
            scheduling=manifest.scheduling,
            environment=environment,
            now=now,
        )
        return self.selected_payload(selection), selection


def record_verified_release_forecasts(
    release: Path,
    *,
    environment: str,
    generation: int,
    operation_id: str,
    published_at: datetime,
) -> None:
    """Publisher hook. Called only after API loaded-identity verification.

    It never backdates a receipt. Recovered activation receives a receipt at
    recovery time, so an uncertain pre-tip activation cannot become eligible
    through a post-tip recovery.
    """
    index_path = release / "selected-schedule/forecast-selections.json"
    if not index_path.is_file():
        return
    index = read_object(index_path)
    registry_root = index.get("forecast_registry_root")
    if not registry_root:
        return  # Existing history is not retroactively given live receipts.
    registry = ForecastRegistry(registry_root)
    # A corrected selection may retain older version IDs for audit. Only
    # versions actually selected by this release received a live publication.
    selected_versions = {
        selection["version_reference"] for selection in index["games"].values()
        if selection.get("provenance") == "original_live" and selection.get("version_reference")
    }
    for forecast_id in index.get("publication_forecast_ids", []):
        if forecast_id not in selected_versions:
            continue
        receipt_id = digest(
            {"operation": operation_id, "forecast": forecast_id, "environment": environment}
        )
        existing = registry.root / "receipts" / f"{receipt_id}.json"
        if existing.is_file():
            receipt = PublicationReceipt.model_validate_json(existing.read_bytes())
            if (
                receipt.forecast_id != forecast_id
                or receipt.publisher_generation != generation
                or receipt.environment != environment
            ):
                raise ValueError("Publication retry differs from its original receipt")
            continue
        registry.record_publication(
            PublicationReceipt(
                receipt_id=receipt_id,
                forecast_id=forecast_id,
                environment=environment,
                published_at=published_at,
                publisher_generation=generation,
                outcome="successful",
            ),
            loaded_forecast_id=forecast_id,
        )
