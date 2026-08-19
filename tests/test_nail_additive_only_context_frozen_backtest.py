from __future__ import annotations

import nba_lineup_model.modeling.nail_additive_only_context_frozen_backtest as backtest


def test_frozen_backtest_uses_the_exact_nail_and_additive_only_pair(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        backtest,
        "run_frozen_multiseason_backtest",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    backtest.run_nail_additive_only_context_frozen_backtest(
        artifacts_dir=tmp_path,
        output_artifacts_dir=tmp_path / "output",
    )

    models = captured["models"]
    assert [model.model for model in models] == [
        backtest.NAIL_MODEL_NAME,
        backtest.ADDITIVE_ONLY_MODEL_NAME,
    ]
