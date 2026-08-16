from __future__ import annotations

import nba_lineup_model.modeling.forward_hpm_x3_linear_ridge as model


def test_canonical_x3_excludes_profile_quality_context_terms(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        model,
        "train_forward_portable_matchup_contextual_rapm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    model.train_hpm_x3_linear_ridge()

    assert captured["context_feature_set"] == "x3_without_uncertainty"
    assert captured["context_curvature_alpha"] == 0.0
    assert captured["context_temporal_alpha"] == 0.0
