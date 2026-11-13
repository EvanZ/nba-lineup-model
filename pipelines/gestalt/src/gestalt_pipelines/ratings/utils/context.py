"""Read frozen context data and evaluate only the updater's required correction."""

import io
from hashlib import sha256

from .sources import SOURCES

FEATURES = (
    "home_minus_away_three_pa_per_100", "home_minus_away_three_pm_per_100",
    "home_minus_away_assists_per_100", "home_minus_away_turnovers_per_100",
    "home_minus_away_usage_pct", "home_minus_away_steals_per_100",
    "home_minus_away_blocks_per_100", "home_minus_away_offensive_rebound_claim_total",
    "home_minus_away_top_two_assists", "home_minus_away_usage_concentration",
)
NONADDITIVE_FEATURES = FEATURES[7:]


class _SavedContextState:
    """A data container for the old file format; contains no legacy methods."""


def export_context(raw: bytes) -> dict:
    # This trusted pickle is read only after verifying its approved byte identity.
    if sha256(raw).hexdigest() != SOURCES["context"][1]:
        raise ValueError("Frozen context checksum differs from the approved model.")
    import numpy as np
    import pandas as pd
    from joblib.numpy_pickle import NumpyUnpickler

    class DataReader(NumpyUnpickler):
        def find_class(self, module, name):
            if (module, name) == (
                "nba_lineup_model.modeling.matchup_contextual", "MatchupContextualModel",
            ):
                return _SavedContextState
            if module == "nba_lineup_model" or module.startswith("nba_lineup_model."):
                raise ValueError("Unexpected legacy object in the frozen context file.")
            return super().find_class(module, name)

    models = DataReader("frozen-context.joblib", io.BytesIO(raw), ensure_native_byte_order=True).load()
    model = models["2025-26"]
    if (model.feature_set != "nail_v12_1_1_standard_usage"
            or model.rebound_model is not None or model.usage_model is not None
            or list(model.pipeline.named_steps) != ["scale", "ridge"]):
        raise ValueError("The saved context requires unsupported prediction calculations.")
    scale, ridge = model.pipeline.named_steps["scale"], model.pipeline.named_steps["ridge"]
    if tuple(scale.feature_names_in_) != FEATURES:
        raise ValueError("Frozen context feature order differs from the approved model.")
    result = {
        "source_season": "2025-26", "feature_set": model.feature_set,
        "features": list(FEATURES), "nonadditive_features": list(NONADDITIVE_FEATURES),
        "scaler": {"mean": scale.mean_.tolist(), "scale": scale.scale_.tolist(),
                   "variance": scale.var_.tolist(), "with_mean": scale.with_mean,
                   "with_std": scale.with_std},
        "coefficients": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
        "calculation": "half_forward_minus_reverse; retain_nonadditive_terms_only",
    }
    validate_context(result)
    # Confirm the portable calculation against the saved sklearn pipeline without
    # invoking any method on the legacy class. Only nonadditive terms belong here.
    differences = np.eye(10)
    differences[:, :7] = 0
    frame = pd.DataFrame(differences, columns=FEATURES)
    expected = 0.5*(model.pipeline.predict(frame)-model.pipeline.predict(-frame))
    actual = np.array([nonadditive_context(result, row) for row in frame.to_dict(orient="records")])
    if not np.allclose(actual, expected, atol=1e-12, rtol=1e-12):
        raise ValueError("Exported context differs from the saved prediction calculation.")
    result["validation"] = {"cases": len(frame), "atol": 1e-12, "rtol": 1e-12,
                            "max_abs_error": float(np.max(np.abs(actual-expected)))}
    # Full legacy coefficients remain visible, but its first seven additive terms are
    # already included in catalog ratings. The updater must not count them twice.
    return result


def validate_context(context: dict) -> None:
    import numpy as np
    if tuple(context["features"]) != FEATURES or tuple(context["nonadditive_features"]) != NONADDITIVE_FEATURES:
        raise ValueError("Invalid frozen context feature definitions.")
    scaler = context["scaler"]
    for values in (scaler["mean"], scaler["scale"], scaler["variance"], context["coefficients"]):
        if np.asarray(values).shape != (10,) or not np.isfinite(values).all():
            raise ValueError("Invalid frozen context parameters.")
    if (not np.isfinite(context["intercept"]) or np.any(np.asarray(scaler["scale"]) <= 0)
            or not scaler["with_mean"] or not scaler["with_std"]):
        raise ValueError("Unsupported frozen scaler or intercept.")


def nonadditive_context(context: dict, differences: dict[str, float]) -> float:
    """Lineup-style correction; excludes player-additive terms, HCA and B2B."""
    import math
    validate_context(context)
    # In 0.5 * (f(x) - f(-x)), scaler means and the fitted intercept cancel.
    result = sum(
        differences[name] * context["coefficients"][index] / context["scaler"]["scale"][index]
        for index, name in enumerate(context["features"]) if name in NONADDITIVE_FEATURES
    )
    if not math.isfinite(result):
        raise ValueError("Non-finite lineup-style context correction.")
    return float(result)
