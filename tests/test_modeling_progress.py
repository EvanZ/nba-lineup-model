from __future__ import annotations

import pytest

from nba_lineup_model.modeling.progress import format_progress_bar


def test_format_progress_bar_is_stable_for_log_files() -> None:
    assert (
        format_progress_bar(2, 3, label="Replaying season 2024-25", width=6)
        == "[####..]  2/3   66.7% Replaying season 2024-25"
    )


@pytest.mark.parametrize(
    ("current", "total", "width"),
    ((1, 0, 24), (-1, 1, 24), (2, 1, 24), (1, 1, 0)),
)
def test_format_progress_bar_rejects_invalid_bounds(
    current: int,
    total: int,
    width: int,
) -> None:
    with pytest.raises(ValueError):
        format_progress_bar(current, total, label="test", width=width)
