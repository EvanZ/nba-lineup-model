"""Versioned conditional-minutes contract and its durable input selections."""

import json
import math
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

MODEL_VERSION = "v1"
SETTINGS = {
    "historical_prior_model": "Forward Conditional Minutes v0.3",
    "legacy_updater": "early_game_role_v3_direct_posterior",
    "prior_weight_games": 0.2,
    "eligible_game_window": 3,
    "role_cutoff_minutes": 44,
    "role_scale": 48 / 44,
    "played_clip_delta_minutes": 12,
    "restricted_statuses": ["Probable", "Questionable", "Doubtful", "Out"],
    "classification_version": "available_and_inactive_zero_v1",
    "cold_start_version": "historical_role_or_frozen_cohort_v1",
    "cold_start_fallback_minutes": 8,
}
# Exact immutable bootstrap references are registered after source verification.
# A new season requires its own approved historical inputs, never last year's priors.
PRIOR_REFERENCES = {
    ("2026-27", "v1"): {
        "uri": "s3://nba-gestalt-data/lake/artifacts/player_minutes/season=2026-27/model_version=v1/priors/08081b51d754bd3fa0abef0a9996d7de8cffaaaa67480833a4c53ca701f4ed47.json",
        "sha256": "08081b51d754bd3fa0abef0a9996d7de8cffaaaa67480833a4c53ca701f4ed47"
    }
}


class MinutesError(ValueError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def official_json(raw: bytes) -> dict:
    def fields(pairs):
        for key, value in pairs:
            if "espn" in key.casefold() or (
                key.casefold() in {"source", "source_url", "provider"}
                and isinstance(value, str) and "espn" in value.casefold()
            ):
                raise MinutesError("Prohibited source in minutes input.")
        return dict(pairs)
    result = json.loads(raw, object_pairs_hook=fields)
    if not isinstance(result, dict):
        raise MinutesError("Expected a JSON object.")
    return result


def number(value, *, label: str, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise MinutesError(f"Invalid {label}.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MinutesError(f"Invalid {label}.") from error
    if not math.isfinite(result) or result < 0 or (maximum is not None and result > maximum):
        raise MinutesError(f"Invalid {label}: {value}.")
    return result


def player_id(value) -> int:
    if isinstance(value, bool) or not str(value).isdigit() or int(value) <= 0:
        raise MinutesError(f"Invalid NBA player ID: {value}.")
    return int(value)


def model_metadata(version: str = MODEL_VERSION) -> dict:
    return {
        "name": "conditional_player_minutes",
        "model_version": version,
        "settings": deepcopy(SETTINGS),
        "configuration_sha256": sha256(json_bytes(SETTINGS)).hexdigest(),
    }


def implementation_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    inventory = {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    shared = root.parent/"utils/__init__.py"
    inventory["../utils/__init__.py"] = sha256(shared.read_bytes()).hexdigest()
    return {"package": "gestalt_pipelines.minutes", "source_sha256": sha256(json_bytes(inventory)).hexdigest()}


def validate_model(record: dict, season: str, version: str, *, kind: str) -> None:
    if record.get("schema_version") != 1 or record.get("season") != season or record.get("artifact_kind") != kind:
        raise MinutesError(f"Wrong minutes {kind} schema or season.")
    if record.get("model") != model_metadata(version):
        raise MinutesError("Minutes model version/settings are incompatible.")
