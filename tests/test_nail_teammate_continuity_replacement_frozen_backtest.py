from __future__ import annotations

import nba_lineup_model.modeling.nail_teammate_continuity_replacement_frozen_backtest as backtest


def test_frozen_backtest_uses_production_and_replacement_candidate(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        backtest,
        "run_frozen_multiseason_backtest",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    backtest.run_nail_teammate_continuity_replacement_frozen_backtest(
        artifacts_dir=tmp_path,
        output_artifacts_dir=tmp_path / "output",
    )

    models = captured["models"]
    assert [model.model for model in models] == [
        backtest.INCUMBENT_MODEL_NAME,
        backtest.CANDIDATE_MODEL_NAME,
    ]
    assert models[0].uses_schedule_control
    assert not models[0].uses_prior_teammate_continuity
    assert models[1].uses_schedule_control
    assert models[1].uses_prior_teammate_continuity
