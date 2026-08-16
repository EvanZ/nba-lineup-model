from __future__ import annotations

import nba_lineup_model.modeling.hpm_x3_linear_without_uncertainty_frozen_backtest as backtest


def test_frozen_backtest_uses_the_exact_full_and_ablated_pair(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        backtest,
        "run_frozen_multiseason_backtest",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    backtest.run_hpm_x3_linear_without_uncertainty_frozen_backtest(
        artifacts_dir=tmp_path,
        output_artifacts_dir=tmp_path / "output",
    )

    models = captured["models"]
    assert len(models) == 2
    assert models[0].model == backtest.FULL_X3_MODEL_NAME
    assert models[1].model == backtest.ABLATED_MODEL_NAME
