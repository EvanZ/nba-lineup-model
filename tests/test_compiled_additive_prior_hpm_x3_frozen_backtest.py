from __future__ import annotations

import nba_lineup_model.modeling.compiled_additive_prior_hpm_x3_frozen_backtest as backtest


def test_focused_backtest_uses_canonical_baseline_and_candidate(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        backtest,
        "run_frozen_multiseason_backtest",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    backtest.run_compiled_additive_prior_hpm_x3_frozen_backtest(
        seasons=("2023-24", "2024-25", "2025-26")
    )

    models = captured["models"]
    assert [model.model for model in models] == [
        "forward_hpm_x3_linear_ridge_without_uncertainty",
        "forward_compiled_additive_prior_hpm_x3",
    ]
