from __future__ import annotations

import nba_lineup_model.modeling.additive_profile_linear_shape_context_frozen_backtest as backtest


def test_frozen_backtest_uses_the_controlled_model_pair(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        backtest,
        "run_frozen_multiseason_backtest",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    backtest.run_additive_profile_linear_shape_context_frozen_backtest(
        artifacts_dir=tmp_path,
        output_artifacts_dir=tmp_path / "output",
    )

    models = captured["models"]
    assert len(models) == 2
    assert models[0].model == backtest.ADDITIVE_PRIOR_MODEL_NAME
    assert models[0].uses_context is False
    assert models[1].model == backtest.MODEL_NAME
    assert models[1].uses_context is True
