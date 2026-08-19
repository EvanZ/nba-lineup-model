from __future__ import annotations

import nba_lineup_model.modeling.forward_nail_additive_only_context as model


def test_additive_only_ablation_uses_the_eight_basketball_additive_coordinates(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        model,
        "train_forward_portable_matchup_contextual_rapm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    model.train_nail_additive_only_context()

    assert captured["context_feature_set"] == "nail_additive_only"
    assert captured["context_curvature_alpha"] == 0.0
    assert captured["context_temporal_alpha"] == 0.0
