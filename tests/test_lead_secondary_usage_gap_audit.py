from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.lead_secondary_usage_gap_audit import fit_conditional_regression


def test_conditional_regression_recovers_independent_standardized_effects() -> None:
    frame = pd.DataFrame(
        {
            "possessions": [1.0, 2.0, 1.0, 2.0],
            "feature_edge": [-2.0, -1.0, 1.0, 2.0],
            "max_frozen_prior_edge": [-1.0, 2.0, -2.0, 1.0],
        }
    )
    # Construct the target from the same standardized coordinates the fit uses.
    weights = frame["possessions"].to_numpy()
    gap = (
        frame["feature_edge"] - (frame["feature_edge"] * weights).sum() / weights.sum()
    ).to_numpy()
    gap /= ((gap**2 * weights).sum() / weights.sum()) ** 0.5
    star = (
        frame["max_frozen_prior_edge"]
        - (frame["max_frozen_prior_edge"] * weights).sum() / weights.sum()
    ).to_numpy()
    star /= ((star**2 * weights).sum() / weights.sum()) ** 0.5
    frame["frozen_residual_net_rating"] = 1.25 + 2.0 * gap - 3.0 * star

    result = fit_conditional_regression(frame)

    assert round(result.intercept, 8) == 1.25
    assert round(result.lead_secondary_usage_gap, 8) == 2.0
    assert round(result.max_frozen_prior_edge, 8) == -3.0
