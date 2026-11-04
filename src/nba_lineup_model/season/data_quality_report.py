"""Immutable daily data-quality reports and alert-state transitions.

The report describes what an environment is actually serving.  It does not
activate a release, rebuild a model, or mutate any basketball artifact.
Runtime probes provide plain mappings so this module remains usable from the
small release environment without importing Dagster.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256

SCHEMA_VERSION = 1
ARTIFACT_KIND = "daily_data_quality_report"
REPORT_JSON = "report.json"
REPORT_HTML = "report.html"
MANIFEST_JSON = "manifest.json"
VALID_CATEGORIES = {"freshness", "completeness", "correctness", "consistency"}
VALID_SEVERITIES = {"info", "warning", "error"}
RUNNING_STATUSES = {"QUEUED", "STARTED", "STARTING", "CANCELING"}
FAILED_STATUSES = {"FAILURE", "FAILED"}


def utc_timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _finding(
    *,
    category: str,
    code: str,
    severity: str,
    scope: str,
    message: str,
    expected: object = None,
    observed: object = None,
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unsupported finding category: {category}")
    if severity not in VALID_SEVERITIES:
        raise ValueError(f"Unsupported finding severity: {severity}")
    return {
        "category": category,
        "code": str(code),
        "severity": severity,
        "scope": str(scope),
        "message": str(message),
        "expected": expected,
        "observed": observed,
        "evidence": dict(evidence or {}),
    }


def _age_minutes(now: datetime, value: object) -> float | None:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds() / 60.0)


def _normalize_coverage(
    payload: object,
) -> dict[str, dict[str, int | None]]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("Artifact coverage must be a mapping")
    normalized: dict[str, dict[str, int | None]] = {}
    for name, values in payload.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"Coverage row {name!r} must be a mapping")
        expected = values.get("expected")
        observed = values.get("observed")
        normalized[str(name)] = {
            "expected": None if expected is None else int(expected),
            "observed": None if observed is None else int(observed),
        }
    return normalized


def _normalize_states(payload: object) -> dict[str, bool | None]:
    source = payload if isinstance(payload, Mapping) else {}
    return {
        name: (None if source.get(name) is None else bool(source.get(name)))
        for name in ("materialized", "validated", "activated", "served")
    }


def _normalize_artifact(payload: Mapping[str, object]) -> dict[str, object]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Artifact name is required")
    return {
        "name": name,
        "layer": str(payload.get("layer") or "unknown"),
        "artifact_id": str(payload.get("artifact_id") or ""),
        "checksum": str(payload.get("checksum") or ""),
        "evidence_through_date": str(payload.get("evidence_through_date") or ""),
        "observed_data_revision": str(payload.get("observed_data_revision") or ""),
        "model_edition": str(payload.get("model_edition") or ""),
        "coverage": _normalize_coverage(payload.get("coverage")),
        "states": _normalize_states(payload.get("states")),
        "required": bool(payload.get("required", True)),
        "required_states": sorted(
            {
                str(value)
                for value in payload.get("required_states", ())
                if str(value) in {"materialized", "validated", "activated", "served"}
            }
        ),
        "details": dict(payload.get("details") or {}),
    }


def _normalize_collector(payload: Mapping[str, object]) -> dict[str, object]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Collector name is required")
    return {
        "name": name,
        "job_name": str(payload.get("job_name") or ""),
        "automation_name": str(payload.get("automation_name") or ""),
        "automation_status": str(payload.get("automation_status") or "unknown"),
        "automation_status_source": str(payload.get("automation_status_source") or "unverified"),
        "automation_error": str(payload.get("automation_error") or ""),
        "last_failure_run_id": str(payload.get("last_failure_run_id") or ""),
        "last_failure_at": str(payload.get("last_failure_at") or ""),
        "latest_run_id": str(payload.get("latest_run_id") or ""),
        "unresolved_failures": payload.get("unresolved_failures"),
        "unfinished_runs": payload.get("unfinished_runs", []),
        "latest_run_status": str(payload.get("latest_run_status") or "missing").upper(),
        "latest_run_created_at": str(payload.get("latest_run_created_at") or ""),
        "latest_run_started_at": str(payload.get("latest_run_started_at") or ""),
        "latest_run_finished_at": str(payload.get("latest_run_finished_at") or ""),
        "last_success_at": str(payload.get("last_success_at") or ""),
        "artifact_observed_at": str(payload.get("artifact_observed_at") or ""),
        "required_since_at": str(payload.get("required_since_at") or ""),
        "expected_by": str(payload.get("expected_by") or ""),
        "required_now": bool(payload.get("required_now", True)),
        "quiet_reason": str(payload.get("quiet_reason") or ""),
        "stale_after_minutes": (
            None
            if payload.get("stale_after_minutes") is None
            else float(payload["stale_after_minutes"])
        ),
        "stall_after_minutes": (
            None
            if payload.get("stall_after_minutes") is None
            else float(payload["stall_after_minutes"])
        ),
        "details": dict(payload.get("details") or {}),
    }


def build_data_quality_report(
    *,
    report_date: str,
    environment: str,
    expected: Mapping[str, object],
    activation: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
    collectors: Sequence[Mapping[str, object]],
    pipeline_failures: Sequence[Mapping[str, object]] = (),
    generated_at: datetime | None = None,
    notes: Sequence[str] = (),
) -> dict[str, object]:
    """Evaluate one environment snapshot without mutating any source."""

    now = generated_at or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    now = now.astimezone(UTC)
    expected_date = str(
        expected.get("required_evidence_through_date")
        or expected.get("latest_eligible_completed_date")
        or ""
    )
    expected_edition = str(activation.get("edition_id") or "")
    expected_revision = str(activation.get("observed_data_revision") or "")
    normalized_artifacts = [_normalize_artifact(item) for item in artifacts]
    normalized_collectors = [_normalize_collector(item) for item in collectors]
    findings: list[dict[str, object]] = []

    activation_state = str(activation.get("state") or "missing")
    if activation_state != "active":
        age = _age_minutes(now, activation.get("state_changed_at"))
        limit = float(activation.get("stall_after_minutes") or 0)
        stalled = activation_state in {"staging", "activating"} and bool(
            limit and age is not None and age > limit
        )
        findings.append(
            _finding(
                category="freshness" if stalled else "correctness",
                code="activation_stalled" if stalled else "activation_not_active",
                severity="error",
                scope="activation",
                message=(
                    "Release activation has exceeded its allowed staging time."
                    if stalled
                    else "No fully active release was observed."
                ),
                expected="active",
                observed=activation_state,
                evidence={"age_minutes": age, "stall_after_minutes": limit},
            )
        )

    latest_eligible = str(expected.get("latest_eligible_completed_date") or "")
    active_evidence = str(expected.get("active_evidence_through_date") or "")
    if (
        bool(expected.get("nightly_publication_due"))
        and latest_eligible
        and active_evidence < latest_eligible
    ):
        findings.append(
            _finding(
                category="freshness",
                code="nightly_publication_late",
                severity="error",
                scope="pipeline:daily_publication",
                message=("The completed slate was not published by the 5:00 AM Pacific deadline."),
                expected=latest_eligible,
                observed=active_evidence or "no active evidence date",
                evidence={"deadline": expected.get("nightly_publication_deadline")},
            )
        )

    for artifact in normalized_artifacts:
        if not artifact["required"]:
            continue
        scope = f"artifact:{artifact['name']}"
        if not artifact["artifact_id"] or not artifact["checksum"]:
            findings.append(
                _finding(
                    category="correctness",
                    code="artifact_identity_missing",
                    severity="error",
                    scope=scope,
                    message="Required artifact has no durable ID or checksum.",
                    expected="non-empty artifact ID and checksum",
                    observed={
                        "artifact_id": artifact["artifact_id"],
                        "checksum": artifact["checksum"],
                    },
                )
            )
        evidence_date = str(artifact["evidence_through_date"])
        if expected_date and evidence_date and evidence_date < expected_date:
            findings.append(
                _finding(
                    category="freshness",
                    code="artifact_evidence_stale",
                    severity="error",
                    scope=scope,
                    message="Artifact evidence is behind the latest eligible completed slate.",
                    expected=expected_date,
                    observed=evidence_date,
                )
            )
        if expected_date and not evidence_date:
            findings.append(
                _finding(
                    category="freshness",
                    code="artifact_evidence_missing",
                    severity="error",
                    scope=scope,
                    message="Required artifact does not identify its evidence date.",
                    expected=expected_date,
                    observed="",
                )
            )
        edition = str(artifact["model_edition"])
        if expected_edition and edition and edition != expected_edition:
            findings.append(
                _finding(
                    category="consistency",
                    code="model_edition_mismatch",
                    severity="error",
                    scope=scope,
                    message="Artifact is bound to a different model/season edition.",
                    expected=expected_edition,
                    observed=edition,
                )
            )
        revision = str(artifact["observed_data_revision"])
        if expected_revision and revision and revision != expected_revision:
            findings.append(
                _finding(
                    category="consistency",
                    code="observed_data_revision_mismatch",
                    severity="error",
                    scope=scope,
                    message="Artifact is bound to a different observed-data revision.",
                    expected=expected_revision,
                    observed=revision,
                )
            )
        for coverage_name, coverage in artifact["coverage"].items():
            expected_count = coverage["expected"]
            observed_count = coverage["observed"]
            if expected_count is not None and observed_count != expected_count:
                findings.append(
                    _finding(
                        category="completeness",
                        code="coverage_mismatch",
                        severity="error",
                        scope=f"{scope}:{coverage_name}",
                        message="Observed artifact coverage differs from expected coverage.",
                        expected=expected_count,
                        observed=observed_count,
                    )
                )
        for state_name in artifact["required_states"]:
            if artifact["states"].get(state_name) is not True:
                findings.append(
                    _finding(
                        category="correctness",
                        code=f"artifact_not_{state_name}",
                        severity="error",
                        scope=scope,
                        message=f"Required artifact is not confirmed {state_name}.",
                        expected=True,
                        observed=artifact["states"].get(state_name),
                    )
                )

    for collector in normalized_collectors:
        scope = f"collector:{collector['name']}"
        if collector["details"].get("run_count") == 0:
            findings.append(_finding(
                category="freshness", code="daily_history_unverified", severity="error",
                scope=scope,
                message="No runs are recorded in this environment's permanent job log. Recovery has not been verified.",
                expected="A verified run in this environment", observed="No recorded runs",
                evidence={"run_history": collector["details"].get("run_history")},
            ))
        automation_status = str(collector["automation_status"]).upper()
        if automation_status not in {"RUNNING", "ENABLED"}:
            findings.append(
                _finding(
                    category="freshness",
                    code="collector_automation_inactive",
                    severity="error",
                    scope=scope,
                    message="Automation is disabled or its effective state cannot be verified.",
                    expected="RUNNING",
                    observed=automation_status,
                    evidence={"error": collector["automation_error"]},
                )
            )
        # Quiet hours suppress expected fresh captures, never actual failures or stalls.
        run_status = str(collector["latest_run_status"]).upper()
        failed_at = parse_timestamp(collector["last_failure_at"])
        succeeded_at = parse_timestamp(collector["last_success_at"])
        unresolved_failure = bool(collector["last_failure_run_id"]) and (
            failed_at is None or succeeded_at is None or succeeded_at <= failed_at
        )
        work_failures = collector["unresolved_failures"]
        if work_failures is not None:
            for failed_work in work_failures:
                findings.append(_finding(
                    category="freshness", code="collector_run_failed", severity="error",
                    scope=f"{scope}:{failed_work['recovery_key']}",
                    message="This work failed. Only a later successful run of the same work clears its alert.",
                    expected="SUCCESS", observed="FAILURE",
                    evidence={"run_id": failed_work["run_id"],
                              "job_name": failed_work["job_name"],
                              "recovery_key": failed_work["recovery_key"]},
                ))
        elif run_status in FAILED_STATUSES or unresolved_failure:
            findings.append(
                _finding(
                    category="freshness",
                    code="collector_run_failed",
                    severity="error",
                    scope=scope,
                    message="A job failed and no later successful run has verified recovery.",
                    expected="SUCCESS",
                    observed="FAILURE",
                    evidence={
                        "run_id": (
                            collector["last_failure_run_id"]
                            if unresolved_failure
                            else collector["latest_run_id"]
                        )
                    },
                )
            )
        for pending in collector["unfinished_runs"]:
            age = _age_minutes(now, pending["created_at"])
            limit = collector["stall_after_minutes"]
            if limit is not None and age is not None and age > limit:
                findings.append(_finding(
                    category="freshness", code="collector_run_stalled", severity="error",
                    scope=f"{scope}:{pending['recovery_key']}",
                    message="This work is still queued or running beyond its allowed time.",
                    expected=f"at most {limit:g} minutes", observed=f"{age:.1f} minutes",
                    evidence={"run_id": pending["run_id"]},
                ))
        if run_status in RUNNING_STATUSES and not collector["unfinished_runs"]:
            age = _age_minutes(now, collector["latest_run_created_at"])
            limit = collector["stall_after_minutes"]
            if limit is not None and age is not None and age > limit:
                findings.append(
                    _finding(
                        category="freshness",
                        code="collector_run_stalled",
                        severity="error",
                        scope=scope,
                        message="Collector run has remained queued or running too long.",
                        expected=f"at most {limit:g} minutes",
                        observed=f"{age:.1f} minutes",
                        evidence={
                            "run_id": collector["latest_run_id"],
                            "run_status": run_status,
                        },
                    )
                )
        if not collector["required_now"]:
            continue
        required_since = parse_timestamp(collector["required_since_at"])
        expected_by = parse_timestamp(collector["expected_by"])
        due_now = expected_by is None or now >= expected_by
        if required_since is not None and due_now:
            success = parse_timestamp(collector["last_success_at"])
            if success is None or success < required_since:
                findings.append(
                    _finding(
                        category="freshness",
                        code="collector_scheduled_success_missing",
                        severity="error",
                        scope=scope,
                        message="Collector has not succeeded for its latest scheduled poll.",
                        expected=f"success after {collector['required_since_at']}",
                        observed=collector["last_success_at"] or "no success",
                    )
                )
            observed = parse_timestamp(collector["artifact_observed_at"])
            if observed is None or observed < required_since:
                findings.append(
                    _finding(
                        category="freshness",
                        code="collector_scheduled_artifact_missing",
                        severity="error",
                        scope=scope,
                        message="Collector output was not refreshed for its latest scheduled poll.",
                        expected=f"artifact after {collector['required_since_at']}",
                        observed=collector["artifact_observed_at"] or "no observed artifact",
                    )
                )
        elif required_since is None:
            stale_limit = collector["stale_after_minutes"]
            if stale_limit is not None:
                checks = (
                    ("last_success_at", "collector_success_stale", "successful run"),
                    ("artifact_observed_at", "collector_artifact_stale", "output artifact"),
                )
                for field, code, label in checks:
                    age = _age_minutes(now, collector[field])
                    if age is None or age > stale_limit:
                        findings.append(
                            _finding(
                                category="freshness",
                                code=code,
                                severity="error",
                                scope=scope,
                                message=f"Collector {label} is not sufficiently recent.",
                                expected=f"within {stale_limit:g} minutes",
                                observed=(
                                    f"{age:.1f} minutes ago" if age is not None else f"no {label}"
                                ),
                            )
                        )

    normalized_failures: list[dict[str, object]] = []
    for failure in pipeline_failures:
        row = {
            "stage": str(failure.get("stage") or "unknown"),
            "status": str(failure.get("status") or "failed"),
            "occurred_at": str(failure.get("occurred_at") or ""),
            "artifact_path": str(failure.get("artifact_path") or ""),
            "message": str(failure.get("message") or "Pipeline stage failed."),
            "details": dict(failure.get("details") or {}),
        }
        normalized_failures.append(row)
        findings.append(
            _finding(
                category="correctness",
                code="pipeline_stage_failed",
                severity="error",
                scope=f"pipeline:{row['stage']}",
                message=row["message"],
                expected="successful stage",
                observed=row["status"],
                evidence={
                    "occurred_at": row["occurred_at"],
                    "artifact_path": row["artifact_path"],
                },
            )
        )

    finding_counts = {
        severity: sum(item["severity"] == severity for item in findings)
        for severity in ("error", "warning", "info")
    }
    status = (
        "failed" if finding_counts["error"] else "warning" if finding_counts["warning"] else "clean"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "report_date": str(report_date),
        "environment": str(environment),
        "generated_at": utc_timestamp(now),
        "status": status,
        "finding_counts": finding_counts,
        "expected": dict(expected),
        "activation": dict(activation),
        "artifacts": normalized_artifacts,
        "collectors": normalized_collectors,
        "pipeline_failures": normalized_failures,
        "findings": findings,
        "notes": [str(note) for note in notes],
    }


def finding_fingerprint(finding: Mapping[str, object]) -> str:
    identity = {
        "category": finding.get("category"),
        "code": finding.get("code"),
        "scope": finding.get("scope"),
    }
    return hashlib.sha256(canonical_json(identity)).hexdigest()


def reconcile_alert_state(
    report: Mapping[str, object],
    previous: Mapping[str, object] | None,
    *,
    observed_at: datetime | None = None,
    notify_severities: Iterable[str] = ("error",),
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return durable incident state and only new/recovery notifications."""

    timestamp = utc_timestamp(observed_at)
    severities = set(notify_severities)
    prior_active = dict((previous or {}).get("active") or {})
    current_findings = {
        finding_fingerprint(item): item
        for item in report.get("findings", [])
        if isinstance(item, Mapping) and item.get("severity") in severities
    }
    active: dict[str, object] = {}
    resolved = list((previous or {}).get("resolved") or [])
    notifications: list[dict[str, object]] = []
    for fingerprint, finding in current_findings.items():
        prior = prior_active.get(fingerprint)
        if isinstance(prior, Mapping):
            active[fingerprint] = {
                **dict(prior),
                "last_seen_at": timestamp,
                "occurrences": int(prior.get("occurrences", 1)) + 1,
                "finding": dict(finding),
            }
            continue
        active[fingerprint] = {
            "fingerprint": fingerprint,
            "first_seen_at": timestamp,
            "last_seen_at": timestamp,
            "occurrences": 1,
            "finding": dict(finding),
        }
        notifications.append(
            {
                "event": "opened",
                "fingerprint": fingerprint,
                "occurred_at": timestamp,
                "finding": dict(finding),
            }
        )
    for fingerprint, incident in prior_active.items():
        if fingerprint in current_findings:
            continue
        recovery = {
            "fingerprint": fingerprint,
            "first_seen_at": incident.get("first_seen_at"),
            "resolved_at": timestamp,
            "finding": dict(incident.get("finding") or {}),
        }
        resolved.append(recovery)
        notifications.append(
            {
                "event": "recovered",
                "fingerprint": fingerprint,
                "occurred_at": timestamp,
                "finding": recovery["finding"],
            }
        )
    state = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp,
        "active": active,
        "resolved": resolved[-500:],
    }
    return state, notifications


def publish_data_quality_report(
    report: Mapping[str, object],
    output_root: Path | str,
) -> Path:
    """Publish one immutable report plus replaceable date/global pointers."""

    normalized = dict(report)
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported data-quality report schema")
    if normalized.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Payload is not a data-quality report")
    report_sha = hashlib.sha256(canonical_json(normalized)).hexdigest()
    root = Path(output_root).resolve()
    report_date = str(normalized.get("report_date") or "")
    if not report_date:
        raise ValueError("Report date is required")
    date_root = root / "reports" / report_date
    target = date_root / f"report_sha256={report_sha}"
    if not target.exists():
        date_root.mkdir(parents=True, exist_ok=True)
        staging = date_root / f".report.tmp-{uuid4().hex}"
        staging.mkdir()
        try:
            _write_json(staging / REPORT_JSON, normalized)
            (staging / REPORT_HTML).write_text(
                render_data_quality_html(normalized),
                encoding="utf-8",
            )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": ARTIFACT_KIND,
                "report_sha256": report_sha,
                "report_date": report_date,
                "status": normalized.get("status"),
                "generated_at": normalized.get("generated_at"),
                "files": {
                    REPORT_JSON: file_sha256(staging / REPORT_JSON),
                    REPORT_HTML: file_sha256(staging / REPORT_HTML),
                },
            }
            _write_json(staging / MANIFEST_JSON, manifest)
            staging.replace(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    verify_data_quality_report(target)
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "report_sha256": report_sha,
        "report_date": report_date,
        "status": normalized.get("status"),
        "generated_at": normalized.get("generated_at"),
        "report_path": target.relative_to(root).as_posix(),
    }
    _atomic_write_json(date_root / "latest.json", pointer)
    _atomic_write_json(root / "latest.json", pointer)
    _atomic_symlink(target, root / "current")
    return target


def publish_failed_data_quality_report(
    *,
    report_date: str,
    environment: str,
    output_root: Path | str,
    stage: str,
    error: BaseException,
    generated_at: datetime | None = None,
) -> Path:
    """Keep a browsable failure even when the normal probe cannot finish."""

    report = build_data_quality_report(
        report_date=report_date,
        environment=environment,
        expected={},
        activation={"state": "active"},
        artifacts=[],
        collectors=[],
        pipeline_failures=[
            {
                "stage": stage,
                "status": "failed",
                "message": f"{type(error).__name__}: {error}",
            }
        ],
        generated_at=generated_at,
        notes=["The data-quality probe failed before a complete snapshot was assembled."],
    )
    return publish_data_quality_report(report, output_root)


def verify_data_quality_report(path: Path | str) -> dict[str, object]:
    root = Path(path).resolve()
    manifest = _read_json(root / MANIFEST_JSON)
    payload = _read_json(root / REPORT_JSON)
    report_sha = hashlib.sha256(canonical_json(payload)).hexdigest()
    if manifest.get("report_sha256") != report_sha:
        raise ValueError("Data-quality report identity changed")
    if root.name != f"report_sha256={report_sha}":
        raise ValueError("Data-quality report directory is not content-addressed")
    for name, expected in dict(manifest.get("files") or {}).items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Data-quality report file changed: {name}")
    expected_files = {MANIFEST_JSON, *dict(manifest.get("files") or {})}
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("Data-quality report contains unsealed files")
    return manifest


def render_data_quality_html(report: Mapping[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value if value is not None else ""))

    status = str(report.get("status") or "unknown")
    counts = report.get("finding_counts") or {}
    artifact_rows = []
    for artifact in report.get("artifacts", []):
        coverage = (
            ", ".join(
                f"{name} {values.get('observed')}/{values.get('expected')}"
                for name, values in artifact.get("coverage", {}).items()
            )
            or "—"
        )
        states = (
            " · ".join(
                f"{name}={value}"
                for name, value in artifact.get("states", {}).items()
                if value is not None
            )
            or "—"
        )
        artifact_rows.append(
            "<tr>"
            f"<td><strong>{esc(artifact.get('name'))}</strong><br><small>{esc(artifact.get('layer'))}</small></td>"
            f'<td class="mono">{esc(artifact.get("artifact_id"))}<br>{esc(artifact.get("checksum"))}</td>'
            f"<td>{esc(artifact.get('evidence_through_date'))}</td>"
            f'<td class="mono">{esc(artifact.get("observed_data_revision"))}</td>'
            f"<td>{esc(artifact.get('model_edition'))}</td>"
            f"<td>{esc(coverage)}</td><td>{esc(states)}</td></tr>"
        )
    finding_rows = [
        "<tr>"
        f'<td><span class="pill {esc(item.get("severity"))}">{esc(item.get("severity"))}</span></td>'
        f"<td>{esc(item.get('category'))}</td><td>{esc(item.get('scope'))}</td>"
        f"<td><strong>{esc(item.get('code'))}</strong><br>{esc(item.get('message'))}</td>"
        f"<td>{esc(item.get('expected'))}</td><td>{esc(item.get('observed'))}</td></tr>"
        for item in report.get("findings", [])
    ]
    collector_rows = []
    for item in report.get("collectors", []):
        status = str(item.get("automation_status", "UNKNOWN")).upper()
        enabled = {
            "RUNNING": "Enabled",
            "ENABLED": "Enabled",
            "STOPPED": "Disabled",
            "DISABLED": "Disabled",
        }.get(status, "Unknown")
        origin = {"code_default": "code default", "operator_override": "operator override"}.get(
            str(item.get("automation_status_source")), "unverified"
        )
        collector_rows.append(
            "<tr>"
            f"<td><strong>{esc(item.get('name'))}</strong><br><small>{esc(item.get('job_name'))}</small></td>"
            f"<td>{esc(enabled)} — {esc(origin)}<br><small>{esc(item.get('automation_error'))}</small></td>"
            f"<td>{esc(item.get('latest_run_status'))}<br><small>{esc(item.get('latest_run_id'))}</small></td>"
            f"<td>{esc(item.get('last_success_at'))}</td>"
            f"<td>{esc(item.get('artifact_observed_at'))}</td>"
            f"<td>{'required' if item.get('required_now') else 'quiet'}<br><small>{esc(item.get('quiet_reason'))}</small></td>"
            "</tr>"
        )
    note_items = "".join(f"<li>{esc(note)}</li>" for note in report.get("notes", []))
    activation = report.get("activation") or {}
    transition = activation.get("transition") or {}
    changed_components = ", ".join(transition.get("changed_components") or []) or "none"
    failure_items = "".join(
        "<li>"
        f"<strong>{esc(item.get('stage'))}</strong>: {esc(item.get('message'))}"
        f"<br><small>{esc(item.get('artifact_path'))}</small>"
        "</li>"
        for item in report.get("pipeline_failures", [])
    )
    expected = report.get("expected") or {}
    transition_html = (
        '<h2>Release and official-slate state</h2><div class="card">'
        f"<p><strong>Edition:</strong> {esc(activation.get('edition_id'))}<br>"
        f'<strong>Observed-data revision:</strong> <span class="mono">{esc(activation.get("observed_data_revision"))}</span><br>'
        f"<strong>Changed components from prior activation:</strong> {esc(changed_components)}<br>"
        f"<strong>Forecast selection changed:</strong> {esc(transition.get('forecast_selection_changed', False))}</p>"
        f"<p><strong>Official completed games:</strong> {esc(expected.get('completed_games'))} · "
        f"<strong>games today:</strong> {esc(expected.get('scheduled_games_today'))} · "
        f"<strong>live now:</strong> {esc(expected.get('in_progress_games'))}</p></div>"
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Data Quality — {esc(report.get('report_date'))}</title>"
        "<style>"
        ":root{color-scheme:light;--ink:#173c31;--muted:#68736d;--line:#d8d7d0;--paper:#faf9f5;"
        "--bad:#ad351f;--warn:#9a6a16;--good:#126352}"
        "*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 system-ui,sans-serif}"
        "main{max-width:1500px;margin:auto;padding:clamp(16px,3vw,42px)}h1{margin-bottom:.25rem}h2{margin-top:2rem}"
        ".summary{display:flex;gap:1rem;flex-wrap:wrap}.card{border:1px solid var(--line);padding:1rem;min-width:150px;background:#fff}"
        ".status{font-weight:800;text-transform:uppercase}.failed{color:var(--bad)}.clean{color:var(--good)}.warning{color:var(--warn)}"
        ".table-wrap{overflow-x:auto;border:1px solid var(--line);background:#fff}table{width:100%;border-collapse:collapse;min-width:900px}"
        "th,td{text-align:left;vertical-align:top;padding:.7rem;border-bottom:1px solid var(--line)}th{font-size:.75rem;text-transform:uppercase;color:var(--muted)}"
        ".mono{font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.pill{padding:.15rem .4rem;border-radius:1rem;font-weight:700}"
        ".pill.error{color:var(--bad);background:#fbe9e4}.pill.warning{color:var(--warn);background:#fff4d7}.pill.info{color:#315f78;background:#e8f3f8}"
        "small{color:var(--muted)}@media(max-width:600px){main{padding:12px}.card{flex:1 1 45%}}"
        "</style></head><body><main>"
        f"<h1>Daily Data Quality — {esc(report.get('report_date'))}</h1>"
        f"<p>{esc(report.get('environment'))} · generated {esc(report.get('generated_at'))}</p>"
        '<div class="summary">'
        f'<div class="card"><small>Status</small><div class="status {esc(status)}">{esc(status)}</div></div>'
        f'<div class="card"><small>Errors</small><div>{esc(counts.get("error", 0))}</div></div>'
        f'<div class="card"><small>Warnings</small><div>{esc(counts.get("warning", 0))}</div></div>'
        f'<div class="card"><small>Eligible evidence date</small><div>{esc((report.get("expected") or {}).get("latest_eligible_completed_date"))}</div></div>'
        f'<div class="card"><small>Active generation</small><div>{esc((report.get("activation") or {}).get("generation"))}</div></div>'
        "</div>"
        + transition_html
        + (f"<h2>Failed pipeline stages</h2><ul>{failure_items}</ul>" if failure_items else "")
        + '<h2>Findings</h2><div class="table-wrap"><table><thead><tr><th>Severity</th><th>Category</th><th>Scope</th><th>Finding</th><th>Expected</th><th>Observed</th></tr></thead><tbody>'
        + ("".join(finding_rows) or '<tr><td colspan="6">No findings.</td></tr>')
        + "</tbody></table></div>"
        '<h2>Validated artifacts</h2><div class="table-wrap"><table><thead><tr><th>Artifact</th><th>ID / checksum</th><th>Evidence</th><th>Observed revision</th><th>Edition</th><th>Coverage</th><th>Lifecycle</th></tr></thead><tbody>'
        + ("".join(artifact_rows) or '<tr><td colspan="7">No artifacts were observed.</td></tr>')
        + "</tbody></table></div>"
        '<h2>Independent collectors</h2><div class="table-wrap"><table><thead><tr><th>Collector</th><th>Automation</th><th>Latest run</th><th>Last success</th><th>Artifact observed</th><th>Expectation</th></tr></thead><tbody>'
        + ("".join(collector_rows) or '<tr><td colspan="6">No collector observations.</td></tr>')
        + "</tbody></table></div>"
        + (f"<h2>Notes</h2><ul>{note_items}</ul>" if note_items else "")
        + "</main></body></html>"
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_symlink(target: Path, pointer: Path) -> None:
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(f".{pointer.name}.tmp-{uuid4().hex}")
    temporary.symlink_to(target, target_is_directory=True)
    os.replace(temporary, pointer)
    descriptor = os.open(pointer.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    _write_json(temporary, payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
