from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.nail_v14_partial_effect_stability_audit import (
    classify_partial_effects,
)


def _effects(feature: str, weights: list[float], signs: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": feature,
            "label": feature,
            "is_v13_addition": False,
            "season_start_year": list(range(2000, 2000 + len(weights))),
            "standardized_weight": weights,
            "standardized_standard_error": [0.05] * len(weights),
            "is_resolved": [sign != 0 for sign in signs],
            "resolved_sign": signs,
        }
    )


def test_classify_partial_effects_distinguishes_stability_patterns() -> None:
    effects = pd.concat(
        [
            _effects("stable", [0.2, 0.3, 0.25, 0.31, 0.22], [1, 1, 1, 1, 1]),
            _effects("weak", [0.02, -0.03, 0.04, 0.01, -0.02], [0, 0, 0, 0, 0]),
            _effects(
                "shift",
                [0.3, 0.25, 0.2, -0.2, -0.3, -0.35],
                [1, 1, 1, -1, -1, -1],
            ),
            _effects("unresolved", [0.3, -0.2, 0.25, -0.3, 0.2], [1, 0, 0, 0, 0]),
        ],
        ignore_index=True,
    )

    classified = classify_partial_effects(effects).set_index("feature")

    assert classified.loc["stable", "classification"] == "stable_material"
    assert classified.loc["weak", "classification"] == "stable_weak"
    assert classified.loc["shift", "classification"] == "sustained_regime_shift"
    assert classified.loc["unresolved", "classification"] == "insufficiently_resolved"
