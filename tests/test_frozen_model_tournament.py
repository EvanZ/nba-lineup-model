from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.frozen_model_tournament import _stratified_bootstrap


def test_stratified_bootstrap_reports_paired_candidate_improvement() -> None:
    result = _stratified_bootstrap(
        pd.Series(["2023-24", "2023-24", "2024-25", "2024-25"]),
        np.array([4.0, 4.0, 4.0, 4.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.sqrt,
        True,
        200,
        7,
        "full_game_margin_rmse",
    )

    assert result["difference_candidate_minus_incumbent"] < 0.0
    assert result["ci_upper"] < 0.0
    assert result["probability_challenger_better"] == 1.0
