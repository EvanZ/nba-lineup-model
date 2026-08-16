from __future__ import annotations

import nba_lineup_model.modeling.forward_hpm_x3_linear_ridge_without_uncertainty as ablation


def test_wrapper_uses_the_x3_contract_without_uncertainty_terms(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        ablation,
        "train_forward_portable_matchup_contextual_rapm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    ablation.train_hpm_x3_linear_ridge_without_uncertainty()

    assert captured["model_name"] == ablation.MODEL_NAME
    assert captured["context_feature_set"] == "x3_without_uncertainty"
    assert captured["context_curvature_alpha"] == 0.0
    assert captured["context_temporal_alpha"] == 0.0
