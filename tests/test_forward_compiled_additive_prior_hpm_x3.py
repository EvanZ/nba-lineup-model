from __future__ import annotations

import nba_lineup_model.modeling.forward_compiled_additive_prior_hpm_x3 as model


def test_wrapper_enables_exact_compiled_additive_prior_transfer(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        model,
        "train_forward_portable_matchup_contextual_rapm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    model.train_forward_compiled_additive_prior_hpm_x3()

    assert captured["compiled_additive_prior"] is True
    assert captured["context_feature_set"] == "x3_without_uncertainty"
    assert captured["context_curvature_alpha"] == 0.0
    assert captured["context_temporal_alpha"] == 0.0
