from __future__ import annotations

import pytest

from nba_lineup_model.web_api.inference import projected_win_pct


def test_projected_win_pct_uses_the_published_net_rating_calibration() -> None:
    assert projected_win_pct(0.0) == pytest.approx(0.499583)
    assert projected_win_pct(-1.5) == pytest.approx(0.454208)
    assert projected_win_pct(1.5) == pytest.approx(0.544958)


def test_projected_win_pct_is_bounded() -> None:
    assert projected_win_pct(-100.0) == 0.0
    assert projected_win_pct(100.0) == 1.0
