import pytest

from gestalt_pipelines.play_by_play.utils import check_download


def test_empty_download_is_missing_data():
    for raw in (b"", b"  "+bytes([10])):
        with pytest.raises(ValueError, match="returned no data"):
            check_download(raw)


def test_source_policy_remains_enforced():
    with pytest.raises(ValueError, match="Prohibited"):
        check_download(b'{"espn_game_id":"forbidden"}')
