from __future__ import annotations

import numpy as np

from nba_lineup_model.modeling.portable_context_attribution_audit import _shapley_values


def test_shapley_values_reconcile_to_the_full_coalition_value() -> None:
    """Exact coalition accounting must conserve the model value."""

    values = np.array([0.0, 2.0, 3.0, 10.0])

    attribution = _shapley_values(values, player_count=2)

    assert np.allclose(attribution, [4.5, 5.5])
    assert np.isclose(attribution.sum(), values[-1] - values[0])


def test_shapley_values_split_symmetric_interaction_evenly() -> None:
    """Players with identical marginal roles receive equal interaction credit."""

    values = np.array([0.0, 0.0, 0.0, 8.0])

    attribution = _shapley_values(values, player_count=2)

    assert np.allclose(attribution, [4.0, 4.0])
