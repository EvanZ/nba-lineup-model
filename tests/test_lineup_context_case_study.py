from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_lineup_model.modeling.contextual_features import contextual_feature_columns
from nba_lineup_model.modeling.lineup_context_case_study import (
    MODEL,
    _feature_contributions,
    render_lineup_context_case_study_page,
)


def test_case_study_renderer_writes_sortable_example_tables(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "eligible_lineup_count": 25,
                "minimum_possessions": 250.0,
                "minimum_games": 20,
            }
        )
    )
    examples = pd.DataFrame(
        {
            "standardized_rank": [1],
            "team_tricode": ["TST"],
            "players": ["Example One / Example Two"],
            "possessions": [250.0],
            "games": [20],
            "standardized_context_net_rating": [1.5],
            "frozen_actual_matchup_context_net_rating": [0.5],
            "frozen_additive_prediction_net_rating": [2.0],
            "frozen_full_prediction_net_rating": [2.6],
            "retrospective_actual_matchup_context_net_rating": [0.75],
            "retrospective_additive_prediction_net_rating": [2.5],
            "retrospective_full_prediction_net_rating": [3.35],
            "observed_net_rating": [3.0],
        }
    )
    examples.to_parquet(run / "positive_examples.parquet", index=False)
    examples.to_parquet(run / "negative_examples.parquet", index=False)
    pd.DataFrame(
        {
            "estimate": ["Frozen full contextual state"],
            "weighted_rmse": [1.25],
            "weighted_mae": [1.0],
        }
    ).to_parquet(run / "retrospective_metrics.parquet", index=False)
    pd.DataFrame(
        {
            "feature": ["home_minus_away_assists_per_100", "total"],
            "label": ["Assists", "Total standardized context"],
            "focal_minus_reference": [2.0, None],
            "context_contribution_net_rating": [0.5, 1.5],
        }
    ).to_parquet(run / "top_lineup_attribution.parquet", index=False)
    pd.DataFrame(
        {
            "standardized_rank": [1],
            "team_tricode": ["TST"],
            "players": ["Example One / Example Two"],
            "possessions": [250.0],
            "games": [20],
            "standardized_context_net_rating": [1.5],
        }
    ).to_parquet(run / "top_lineup_attribution_summary.parquet", index=False)
    pd.DataFrame(
        {
            "feature": [
                "home_minus_away_usage_per_100",
                "home_minus_away_defensive_rebounds_per_100",
            ],
            "label": ["Usage events", "Defensive rebounds"],
            "feature_difference": [0.0, 0.0],
            "orientation_symmetrized_contribution_net_rating": [0.0, 0.0],
            "application_q05": [-1.0, -1.0],
            "application_q95": [1.0, 1.0],
            "focal_minus_reference": [0.5, 0.5],
        }
    ).to_parquet(run / "response_curves.parquet", index=False)
    page = tmp_path / "forward-contextual-rapm.md"
    page.write_text(
        "# Forward Contextual RAPM\n\n"
        "<!-- forward-contextual-case-study:start -->\n"
        "<!-- forward-contextual-case-study:end -->\n"
    )

    render_lineup_context_case_study_page(run, page_path=page)

    text = page.read_text()
    assert "| Rank | Team | Players |" in text
    assert "| 1 | TST | Example One / Example Two | 250 | 20 | +1.50 |" in text
    assert "Completed full" in text
    assert "| Frozen full contextual state | 1.25 | 1.00 |" in text
    assert "### Worked Context Decomposition" in text
    assert "| Assists | +2.00 | +0.50 |" in text
    assert "### Response Curves For Diminishing-Return Candidates" in text


def test_feature_contributions_reconstruct_spline_ridge_prediction() -> None:
    generator = np.random.default_rng(7)
    columns = list(contextual_feature_columns())
    features = pd.DataFrame(generator.normal(size=(24, len(columns))), columns=columns)
    target = generator.normal(size=len(features))
    model = Pipeline(
        [
            ("spline", SplineTransformer(n_knots=4, degree=2, extrapolation="linear")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=2.0)),
        ]
    ).fit(features, target)

    contribution = _feature_contributions(model, features)
    expected = model.predict(features)
    intercept = float(model.named_steps["ridge"].intercept_)

    assert contribution.shape == (len(features), len(columns))
    np.testing.assert_allclose(contribution.sum(axis=1) + intercept, expected)
