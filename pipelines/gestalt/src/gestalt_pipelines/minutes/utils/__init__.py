"""Public minutes calculation and validation helpers."""

from .model import (
    MODEL_VERSION, PRIOR_REFERENCES, SETTINGS, MinutesError, implementation_identity,
    json_bytes, model_metadata, official_json,
)
from .observations import actual_minutes, classify, game_observations, role_minutes
from .priors import build_priors, initialize_player, validate_priors
from .update import estimate, update_state, validate_state

__all__ = [
    "MODEL_VERSION", "PRIOR_REFERENCES", "SETTINGS", "MinutesError",
    "actual_minutes", "build_priors", "classify", "estimate", "game_observations",
    "implementation_identity", "initialize_player", "json_bytes", "model_metadata",
    "official_json", "role_minutes", "update_state", "validate_priors", "validate_state",
]

from .projection import (
    PROJECTION_VERSION, build_game_projection, projection_fingerprint,
    projection_model, validate_game_projection,
)
__all__ += [
    "PROJECTION_VERSION", "build_game_projection", "projection_fingerprint",
    "projection_model", "validate_game_projection",
]
