"""Immutable official-NBA captures and explicitly selected normalized revisions."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

from nba_lineup_model.orchestration.daily_work import (
    atomic_json, digest, file_reference, read_object, verify_reference,
)
from nba_lineup_model.orchestration.repair_plan import require_approval
from nba_lineup_model.season.source_policy import forbidden_references

FEEDS = {"nba_box_score", "nba_play_by_play", "nba_injury_report", "nba_inactive_list"}

def archive_capture(root: Path, source: Path) -> dict:
    original = file_reference(source)
    target = root / "captures" / original["sha256"]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".capture-{uuid4().hex}.tmp"
    try:
        if not target.exists():
            shutil.copyfile(source, temporary)
            if file_reference(temporary)["sha256"] != original["sha256"]:
                raise ValueError("Source changed while archiving its capture")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
        reference = file_reference(target)
        if reference["sha256"] != original["sha256"]:
            raise ValueError("Archived capture checksum mismatch")
        return reference
    finally:
        temporary.unlink(missing_ok=True)


def publish_revision(*, root: Path, feed: str, game_id: str, raw_path: Path,
                     normalized: dict, source_url: str, parser_identity: str,
                     supersedes: dict | None = None, reason: str | None = None) -> dict:
    host = (urlsplit(source_url).hostname or "").lower()
    if host != "nba.com" and not host.endswith(".nba.com"):
        raise ValueError("Observed basketball revisions require an official NBA source")
    if feed not in FEEDS or not parser_identity or not game_id.startswith("00"):
        raise ValueError("Invalid NBA revision identity")
    if forbidden_references(normalized):
        raise ValueError("Forbidden source in normalized observation")
    if bool(supersedes) != bool(reason):
        raise ValueError("A correction must cite its prior revision and reason")
    if supersedes:
        prior = read_revision(supersedes)
        if prior["identity"]["feed"] != feed or prior["identity"]["game_id"] != game_id:
            raise ValueError("Correction cannot change feed or game identity")
    identity = {
        "schema_version": 1, "feed": feed, "game_id": game_id,
        "raw_capture": archive_capture(root, raw_path), "normalized": normalized,
        "source_url": source_url, "parser_identity": parser_identity,
        "supersedes": supersedes, "reason": reason,
    }
    revision = {"revision_id": digest(identity), "identity": identity}
    path = root / feed / game_id / f"{revision['revision_id']}.json"
    atomic_json(path, revision, immutable=True)
    return file_reference(path)


def read_revision(reference: dict) -> dict:
    revision = read_object(verify_reference(reference))
    identity = revision["identity"]
    if revision["revision_id"] != digest(identity):
        raise ValueError("Observed revision checksum mismatch")
    verify_reference(identity["raw_capture"])
    return revision


def select_revision(*, root: Path, revision_reference: dict, previous_reference: dict | None,
                    impact_plan: dict | None = None, approval: dict | None = None) -> dict:
    """Return an immutable selection, never change the environment pointer."""
    revision = read_revision(revision_reference)
    if previous_reference:
        previous = read_revision(previous_reference)
        if revision["identity"]["supersedes"] != previous_reference:
            raise ValueError("Correction selection must name the exact previous revision")
        if not impact_plan:
            raise PermissionError("Historical selection requires an impact plan")
        require_approval(impact_plan, approval, "select_historical_data")
        changes = impact_plan["identity"]["changed_inputs"]
        if not any(c.get("before") == previous_reference and c.get("after") == revision_reference
                   for c in changes):
            raise PermissionError("Approved plan does not cover these revisions")
        if previous["identity"]["game_id"] != revision["identity"]["game_id"]:
            raise ValueError("Revision changes game identity")
    elif revision["identity"]["supersedes"]:
        raise PermissionError("A correction cannot be selected as an initial observation")
    identity = {"revision": revision_reference, "previous_revision": previous_reference,
                "plan_id": impact_plan["plan_id"] if impact_plan else None}
    selection = {"selection_id": digest(identity), "identity": identity}
    path = root / f"{selection['selection_id']}.json"
    atomic_json(path, selection, immutable=True)
    return file_reference(path)
