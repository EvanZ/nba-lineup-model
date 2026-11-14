"""Preserve approved numerical priors and fixed controls; never fit models."""

import io
import json
import math
import re
from collections import Counter
from hashlib import sha256
from importlib.metadata import version as package_version
from pathlib import Path

from .sources import MODEL_ID, MODEL_VERSION, RUN_ID, source_references

EXPECTED_SOURCES = {
    "forward_value_conditioned_aging_prior": 441,
    "draft_cold_start_prior": 53,
    "replacement_cold_start_prior": 20,
    "forward_value_conditioned_aging_gap_returner_prior": 10,
}
REPLACEMENT_PRIOR = -5.109488431706889


def json_bytes(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def check_source_policy(value) -> None:
    """Check decoded source data, rather than searching opaque binary bytes."""
    if isinstance(value, dict):
        for key, child in value.items():
            check_source_policy(key)
            check_source_policy(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_source_policy(child)
    elif isinstance(value, str) and re.search(r"\bespn\b|espncdn|espn\.com", value, re.I):
        raise ValueError("Unapproved basketball source found in the selected ratings inputs.")


def implementation_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode()+b"\0"+path.read_bytes())
    return {"package": "gestalt-pipelines", "source_sha256": digest.hexdigest(),
            "runtime": {name: package_version(name) for name in
                        ("numpy", "pandas", "scipy", "pyarrow")}}


def _records(frame) -> list[dict]:
    # Preserve float64 values and original labels; convert only missing values to null.
    records = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    check_source_policy(records)
    return records


def _table(raw: bytes):
    import pandas as pd
    return pd.read_parquet(io.BytesIO(raw))


def _fixed_b2b(frame) -> float:
    import numpy as np
    values = frame["schedule_control_raw_weight"].to_numpy(dtype=float)
    weights = frame["schedule_training_stint_count"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any() or (frame["season"].astype(str) > "2025-26").any():
        raise ValueError("No valid completed-season fixed B2B controls.")
    return float(np.average(values[valid], weights=weights[valid]))


def prepare_initialization(blobs: dict[str, bytes], *, season: str) -> dict:
    references = source_references(season, MODEL_VERSION)
    if set(blobs) != set(references):
        raise ValueError("The approved initialization input set is incomplete.")
    # Check every approved identity before parsing any source.
    for name, reference in references.items():
        if sha256(blobs[name]).hexdigest() != reference["sha256"]:
            raise ValueError(f"The {name} source checksum does not match the approved input.")
    catalog = _table(blobs["catalog"])
    metadata = json.loads(blobs["catalog_metadata"])
    hca = json.loads(blobs["hca"])
    record = {
        "schema_version": 1, "season": season, "model_version": MODEL_VERSION,
        "artifact_kind": "player_rating_priors", "model_id": MODEL_ID, "model_run_id": RUN_ID,
        "inputs": references, "implementation": implementation_identity(),
        "catalog_metadata": metadata, "players": _records(catalog.sort_values("player_id")),
        "settings": {"updater_version": "possession_residual_ridge_joint_hca_v2",
                     "player_prior_equivalent_possessions": 6000.0,
                     "replacement_prior": REPLACEMENT_PRIOR, "hca": hca,
                     "fixed_back_to_back": _fixed_b2b(_table(blobs["schedule_metadata"]))},
    }
    validate_initialization(record, season, MODEL_VERSION)
    return record


def validate_initialization(record: dict, season: str, version: str) -> None:
    if (record.get("schema_version"), record.get("season"), record.get("model_version"),
            record.get("artifact_kind"), record.get("model_id"), record.get("model_run_id")) != (
                1, season, version, "player_rating_priors", MODEL_ID, RUN_ID):
        raise ValueError("Invalid ratings initialization identity.")
    # Saved priors can retain unused provenance; only these four sources are required.
    for name, reference in source_references(season, version).items():
        if record.get("inputs", {}).get(name) != reference:
            raise ValueError("Initialization inputs do not match the approved selection.")
    rows = record["players"]
    ids = [row["player_id"] for row in rows]
    if len(ids) != 524 or len(set(ids)) != 524 or any(i is None for i in ids):
        raise ValueError("Incomplete initialization player coverage.")
    if any(row["season"] != season for row in rows):
        raise ValueError("The frozen catalog season differs from the selected season.")
    if Counter(row["forecast_source"] for row in rows) != EXPECTED_SOURCES:
        raise ValueError("Original rating source labels/counts differ.")
    if any(not math.isfinite(row[field]) for row in rows for field in ("rapm", "prior_rating")):
        raise ValueError("The frozen catalog contains a non-finite starting rating.")
    from statistics import median
    replacement = [row["rapm"] for row in rows if row["forecast_source"] == "replacement_cold_start_prior"]
    if median(replacement) != REPLACEMENT_PRIOR:
        raise ValueError("The approved replacement median changed.")
    metadata, settings = record["catalog_metadata"], record["settings"]
    if metadata["player_count"] != 524 or metadata["target_season"] != season:
        raise ValueError("Frozen catalog metadata identifies a different season or catalog.")
    hca = settings["hca"]
    if hca["target_season"] != season or hca["source_through_season"] != "2025-26":
        raise ValueError("Frozen HCA metadata identifies a different season.")
    for field in ("prior_equivalent_possessions", "observation_variance_per_100", "prior_variance"):
        if not math.isfinite(hca[field]) or hca[field] <= 0:
            raise ValueError(f"Invalid HCA {field}.")
    if not math.isfinite(hca["prior_mean"]) or not math.isfinite(settings["fixed_back_to_back"]):
        raise ValueError("Invalid starting HCA or fixed B2B control.")
    if (settings["updater_version"] != "possession_residual_ridge_joint_hca_v2"
            or settings["player_prior_equivalent_possessions"] != 6000.0
            or settings["replacement_prior"] != REPLACEMENT_PRIOR):
        raise ValueError("The approved player-rating prior settings changed.")
    check_source_policy(record)
    json_bytes(record)
