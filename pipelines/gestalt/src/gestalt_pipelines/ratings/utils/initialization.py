"""Prepare a portable, checked copy of accepted inputs; no model fitting."""

import io
import json
import math
import re
import tarfile
from collections import Counter
from hashlib import sha256
from importlib.metadata import version as package_version
from pathlib import Path

from .sources import MODEL_ID, MODEL_VERSION, RUN_ID, source_references
from .context import export_context, validate_context

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
                        ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow")}}


def _records(frame) -> list[dict]:
    # Preserve float64 values and original labels; convert only missing values to null.
    records = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    check_source_policy(records)
    return records


def _table(raw: bytes):
    import pandas as pd
    return pd.read_parquet(io.BytesIO(raw))


def _evidence_table(raw: bytes, name: str):
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        member = archive.getmember(name)
        if not member.isfile():
            raise ValueError(f"The aging package lacks a regular {name} file.")
        content = archive.extractfile(member).read()
    return _table(content), {"member": name, "sha256": sha256(content).hexdigest()}


def _fixed_b2b(frame) -> float:
    import numpy as np
    values = frame["schedule_control_raw_weight"].to_numpy(dtype=float)
    weights = frame["schedule_training_stint_count"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any() or (frame["season"].astype(str) > "2025-26").any():
        raise ValueError("No valid completed-season fixed B2B controls.")
    return float(np.average(values[valid], weights=weights[valid]))


def prepare_initialization(blobs: dict[str, bytes], *, season: str) -> dict:
    import joblib
    import numpy as np

    references = source_references(season, MODEL_VERSION)
    if set(blobs) != set(references):
        raise ValueError("The approved initialization input set is incomplete.")
    # Verify all identities before parsing any pickle or applying saved models.
    for name, reference in references.items():
        if sha256(blobs[name]).hexdigest() != reference["sha256"]:
            raise ValueError(f"The {name} source checksum does not match the approved input.")
    tables = {name: _table(blobs[name]) for name in
              ("catalog", "profiles", "schedule_metadata")}
    for table in tables.values():
        check_source_policy(_records(table))
    catalog, profiles = tables["catalog"], tables["profiles"]
    for name, table in (("catalog", catalog), ("profiles", profiles)):
        if len(table) != 524 or table.player_id.duplicated().any() or table.player_id.isna().any():
            raise ValueError(f"The approved {name} must have 524 unique player IDs.")
        if set(table.season) != {season}:
            raise ValueError(f"The {name} season differs from the selected season.")
    if set(catalog.player_id) != set(profiles.player_id):
        raise ValueError("The frozen rating/profile player identities do not match.")
    if Counter(catalog.forecast_source) != EXPECTED_SOURCES:
        raise ValueError("The original forecast_source labels/counts changed.")
    if not np.isfinite(catalog[["rapm", "prior_rating"]].to_numpy(dtype=float)).all():
        raise ValueError("The frozen catalog contains a non-finite starting rating.")
    replacement = catalog.loc[catalog.forecast_source == "replacement_cold_start_prior", "rapm"]
    if float(replacement.median()) != REPLACEMENT_PRIOR:
        raise ValueError("The approved replacement median changed.")

    metadata = json.loads(blobs["catalog_metadata"])
    hca = json.loads(blobs["hca"])
    aging_metadata = json.loads(blobs["aging_metadata"])
    for item in (metadata, hca, aging_metadata):
        check_source_policy(item)
    if (metadata["player_count"] != 524 or metadata["target_season"] != season
            or hca["target_season"] != season or hca["source_through_season"] != "2025-26"
            or aging_metadata["prediction_season"] != season
            or aging_metadata["completed_through"] != "2025-26"):
        raise ValueError("Frozen model metadata identifies a different season or catalog.")
    for field in ("prior_equivalent_possessions", "observation_variance_per_100", "prior_variance"):
        if not math.isfinite(hca[field]) or hca[field] <= 0:
            raise ValueError(f"Invalid HCA {field}.")
    if not math.isfinite(hca["prior_mean"]):
        raise ValueError("Invalid starting HCA.")
    predictor = joblib.load(io.BytesIO(blobs["aging_predictor"]))
    if list(predictor.feature_names_in_) != aging_metadata["features"]:
        raise ValueError("The fitted aging predictor's feature definitions changed.")

    comparison, comparison_ref = _evidence_table(blobs["aging_evidence"], "generation_catalog_comparison.parquet")
    last_states, last_states_ref = _evidence_table(blobs["aging_evidence"], "last_observed_states.parquet")
    anchors = catalog[catalog.forecast_source.str.startswith("forward_")].merge(
        comparison, on="player_id", suffixes=("_catalog", "_comparison"), validate="one_to_one",
    )
    if (len(anchors) != 451 or not np.array_equal(anchors.prior_rating_catalog, anchors.prior_rating_comparison)
            or not np.isfinite(anchors.raw_prediction).all()):
        raise ValueError("The frozen aging alignment references are incomplete or changed.")
    # Freeze the legacy shared-catalog alignment once; do not use current player bios.
    alignment = {
        "method": "median_shared_forward_catalog_prior_rating_minus_raw_prediction",
        "offset": float(np.median(anchors.prior_rating_catalog-anchors.raw_prediction)),
        "anchor_count": len(anchors), "evidence_member": comparison_ref,
        "anchors": [{"player_id": int(row.player_id), "prior_rating": float(row.prior_rating_catalog),
                     "raw_prediction": float(row.raw_prediction)}
                    for row in anchors.sort_values("player_id").itertuples()],
    }
    if (last_states.player_id.duplicated().any()
            or not np.isfinite(last_states[["rapm", "on_court_possessions"]]).all().all()
            or (last_states.season.astype(str) > "2025-26").any()):
        raise ValueError("The historical arrival reference states are invalid.")
    record = {
        "schema_version": 1, "season": season, "model_version": MODEL_VERSION,
        "artifact_kind": "player_rating_priors", "model_id": MODEL_ID, "model_run_id": RUN_ID,
        "inputs": references, "implementation": implementation_identity(),
        "catalog_metadata": metadata, "players": _records(catalog.sort_values("player_id")),
        "profiles": _records(profiles.sort_values("player_id")), "context": export_context(blobs["context"]),
        "settings": {"updater_version": "possession_residual_ridge_joint_hca_v2",
                     "player_prior_equivalent_possessions": 6000.0,
                     "replacement_prior": REPLACEMENT_PRIOR, "hca": hca,
                     "fixed_back_to_back": _fixed_b2b(tables["schedule_metadata"])},
        "aging": {"predictor": references["aging_predictor"], "alignment": alignment,
                  "features": aging_metadata["features"], "metadata": aging_metadata,
                  "historical_states_member": last_states_ref,
                  "historical_player_states": _records(last_states.sort_values("player_id")),
                  "rebuild_validation_max_abs_error": 2.0479982154597565e-08,
                  "original_1e_8_check_passed": False},
    }
    validate_initialization(record, season, MODEL_VERSION)
    return record


def validate_initialization(record: dict, season: str, version: str) -> None:
    if (record.get("schema_version"), record.get("season"), record.get("model_version"),
            record.get("artifact_kind")) != (1, season, version, "player_rating_priors"):
        raise ValueError("Invalid ratings initialization identity.")
    if record.get("inputs") != source_references(season, version):
        raise ValueError("Initialization inputs do not match the approved selection.")
    for name in ("players", "profiles"):
        rows = record[name]
        ids = [row["player_id"] for row in rows]
        if len(ids) != 524 or len(set(ids)) != 524:
            raise ValueError(f"Incomplete initialization {name} coverage.")
    if {r["player_id"] for r in record["players"]} != {r["player_id"] for r in record["profiles"]}:
        raise ValueError("Rating/profile coverage differs.")
    if Counter(r["forecast_source"] for r in record["players"]) != EXPECTED_SOURCES:
        raise ValueError("Original rating source labels/counts differ.")
    validate_context(record["context"])
    check_source_policy(record)
    json_bytes(record)  # Reject NaN/infinity anywhere before publication.
