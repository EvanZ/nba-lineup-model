"""Durable work identities and verified checkpoints for daily processing.

This is not a publisher. A completed work unit cannot activate a site, select a
data correction, or replace a forecast. Those operations retain their gates.
"""
from __future__ import annotations

import fcntl
import hashlib
import html
import json
import os
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from nba_lineup_model.season.daily_artifact_common import (
    canonical_json, file_sha256, require_sha256,
)

STAGES = ("data", "model", "serving", "lineups", "validate_and_stage")
STATES = {
    "data": "data_validated", "model": "model_built",
    "serving": "serving_built", "lineups": "lineups_built",
    "validate_and_stage": "serving_validated",
}


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read_object(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Expected an object: {path}")
    return result


def file_reference(path: Path | str) -> dict:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Expected artifact file: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved),
            "size_bytes": resolved.stat().st_size}


def verify_reference(reference: Mapping) -> Path:
    path = Path(reference["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Missing checkpoint input/output: {path}")
    if str(path.resolve(strict=True)) != str(path):
        raise ValueError(f"Artifact reference must name its resolved path: {path}")
    require_sha256(reference.get("sha256"), "Artifact checksum")
    if path.stat().st_size != reference.get("size_bytes") or file_sha256(path) != reference["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return path


def atomic_json(path: Path, payload: dict, *, immutable: bool = False) -> None:
    """Flush before publishing; immutable records are create-only and idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(payload) + b"\n"
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != raw:
                    raise ValueError(f"Immutable record already exists with different content: {path}")
        else:
            os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def work_spec(*, season: str, game_date: date, source_revisions: dict,
              model_configuration: dict, parent_model: str | None,
              edition_id: str, code_identity: str, inputs: dict,
              expected_game_ids: list[str]) -> dict:
    """A date alone is not an idempotency key; every causal input is explicit."""
    if not season or not edition_id or not code_identity or not model_configuration:
        raise ValueError("Season, edition, code, and model configuration are required")
    if len(expected_game_ids) != len(set(expected_game_ids)):
        raise ValueError("Duplicate scheduled games")
    if set(source_revisions) != set(expected_game_ids):
        raise ValueError("Source selection must cover the exact official slate")
    if parent_model is not None:
        require_sha256(parent_model, "Parent model")
    for reference in inputs.values():
        verify_reference(reference)
    identity = {
        "schema_version": 1, "season": season, "game_date_pacific": game_date.isoformat(),
        "source_revisions": source_revisions, "model_configuration": model_configuration,
        "parent_model": parent_model, "edition_id": edition_id,
        "code_identity": code_identity, "inputs": inputs,
        "expected_game_ids": sorted(expected_game_ids),
    }
    return {"work_id": digest(identity), "identity": identity}


def verify_spec(spec: dict) -> None:
    identity = spec["identity"]
    if spec["work_id"] != digest(identity) or identity.get("schema_version") != 1:
        raise ValueError("Work identity checksum/schema mismatch")
    date.fromisoformat(identity["game_date_pacific"])
    if sorted(identity["source_revisions"]) != identity["expected_game_ids"]:
        raise ValueError("Work identity has incomplete game coverage")
    for reference in identity["inputs"].values():
        verify_reference(reference)


class DailyWork:
    """Serial, resumable stage execution with immutable success receipts."""

    def __init__(self, root: Path, spec: dict):
        verify_spec(spec)
        self.spec = spec
        self.root = root / spec["identity"]["game_date_pacific"] / spec["work_id"]

    @contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("This daily work unit is already running") from error
            yield

    def _status(self, status: str, stage: str, receipts: list[dict], **extra) -> None:
        payload = {"work_id": self.spec["work_id"],
                   "game_date": self.spec["identity"]["game_date_pacific"],
                   "status": status, "stage": stage,
                   "completed_stages": [r["stage"] for r in receipts], **extra}
        atomic_json(self.root / "status.json", payload)
        write_work_report(self.root, self.spec, payload, receipts)

    def run(self, callbacks: Mapping[str, Callable[[], dict]]) -> dict:
        if tuple(callbacks) != STAGES:
            raise ValueError("The canonical stage order is required")
        receipts: list[dict] = []
        reused: list[str] = []
        with self.lock():
            atomic_json(self.root / "work.json", self.spec, immutable=True)
            for stage in STAGES:
                try:
                    verify_spec(self.spec)
                    for prior in receipts:
                        verify_receipt(prior)
                    key = digest({"work_id": self.spec["work_id"], "stage": stage,
                                  "parents": [r["receipt_id"] for r in receipts]})
                    path = self.root / "checkpoints" / f"{stage}.json"
                    self._status("running", stage, receipts, reused_stages=reused)
                    if path.exists():
                        receipt = read_object(path)
                        if receipt["stage_key"] != key or receipt["stage"] != stage:
                            raise ValueError("Checkpoint belongs to different inputs")
                        verify_receipt(receipt)
                        reused.append(stage)
                    else:
                        result = callbacks[stage]()
                        if not isinstance(result, dict) or not result.get("artifacts"):
                            raise ValueError(f"{stage} must return verified artifact references")
                        for reference in result["artifacts"]:
                            verify_reference(reference)
                        body = {"schema_version": 1, "stage": stage, "stage_key": key,
                                "artifacts": result["artifacts"], "result": result.get("result", {})}
                        receipt = {**body, "receipt_id": digest(body)}
                        atomic_json(path, receipt, immutable=True)
                    receipts.append(receipt)
                    self._status(STATES[stage], stage, receipts, reused_stages=reused)
                except BaseException as error:
                    failure = {"stage": stage, "error": f"{type(error).__name__}: {error}",
                               "occurred_at": datetime.now(UTC).isoformat(),
                               "work_id": self.spec["work_id"]}
                    atomic_json(self.root / "failures" / f"{uuid4().hex}.json",
                                failure, immutable=True)
                    self._status("failed", stage, receipts, error=failure["error"],
                                 reused_stages=reused)
                    raise
        return {"work_id": self.spec["work_id"], "status": "serving_validated",
                "checkpoints": receipts, "reused_stages": reused, "root": str(self.root)}


def verify_receipt(receipt: dict) -> None:
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if digest(body) != receipt.get("receipt_id"):
        raise ValueError("Checkpoint receipt checksum mismatch")
    if not receipt.get("artifacts"):
        raise ValueError("Checkpoint has no artifacts")
    for reference in receipt["artifacts"]:
        verify_reference(reference)


def write_work_report(root: Path, spec: dict, status: dict, receipts: list[dict]) -> None:
    rows = []
    for stage in STAGES:
        receipt = next((r for r in receipts if r["stage"] == stage), None)
        value = "Validated" if receipt else ("Failed" if status.get("stage") == stage
                                            and status["status"] == "failed" else "Pending")
        rows.append(f"<tr><td>{html.escape(stage)}</td><td>{value}</td>"
                    f"<td>{html.escape(receipt['receipt_id'] if receipt else '—')}</td></tr>")
    error = f"<p role='alert'>{html.escape(status.get('error', ''))}</p>"
    page = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Daily processing checkpoints</title><style>"
            "body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}"
            "table{border-collapse:collapse;width:100%}td,th{padding:.6rem;text-align:left;"
            "border-bottom:1px solid #ccc;overflow-wrap:anywhere}td:last-child{max-width:18rem}"
            "[role=alert]{color:#a22}code{overflow-wrap:anywhere}</style></head><body>"
            f"<h1>Daily processing · {html.escape(status['game_date'])}</h1>"
            f"<p>Status: <strong>{html.escape(status['status'])}</strong></p>"
            f"<p>Work: <code>{spec['work_id']}</code></p>"
            "<p>These checkpoints are not an activation or approval.</p>" + error +
            "<table><thead><tr><th>Stage</th><th>State</th><th>Receipt</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></body></html>")
    temporary = root / f".report-{uuid4().hex}.tmp"
    temporary.write_text(page)
    temporary.replace(root / "report.html")
