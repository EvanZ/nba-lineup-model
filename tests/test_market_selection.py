import copy
import json
from datetime import date, datetime, timedelta

import pytest
from test_official_schedule_runtime import GAME_DATE, START, _board
from nba_lineup_model.season.market_selection import select_captured_markets, repair_market_selection
from nba_lineup_model.season.official_schedule_runtime import _publish_pregame_candidate
from nba_lineup_model.season.serving_selection import build_schedule_selection, read_object, forecast_inputs


def setup(tmp_path):
    tip = datetime.fromisoformat(START.replace("Z", "+00:00"))
    now = tip - timedelta(minutes=20)
    raw = tmp_path / "raw"
    (raw / "latest").mkdir(parents=True)
    board = _board(collected_at=now)
    (raw / "latest" / f"{GAME_DATE}.json").write_text(json.dumps(board))
    source = tmp_path / "selection"
    build_schedule_selection(raw, source, edition_id="accepted")
    game = copy.deepcopy(board["games"][0])
    game["market"] = {"source": "The Odds API", "captured_at": now.isoformat(),
        "home_spread": -3.5, "away_spread": 3.5}
    archive, _ = _publish_pregame_candidate(game, state_root=raw, captured_at=now)
    return source, raw, game, tip, now, archive


def test_historical_selection_uses_only_verified_pretip_archives(tmp_path):
    source, raw, game, tip, now, archive = setup(tmp_path)
    # The current scoreboard is postgame and cannot supply historical odds.
    board = _board(collected_at=tip + timedelta(hours=3))
    board["games"][0]["market"] = {**game["market"], "home_spread": -99}
    (raw / "latest" / f"{GAME_DATE}.json").write_text(json.dumps(board))
    args = dict(schedule_root=source, observation_root=raw, start_date=GAME_DATE,
        days_ahead=0, now=tip + timedelta(hours=3))
    assert select_captured_markets(**args) == {}
    selected = select_captured_markets(**args, include_started=True)
    assert selected[game["game_id"]]["market"]["home_spread"] == -3.5
    archive.write_text(archive.read_text().replace('-3.5', '-9.5'))
    with pytest.raises(ValueError, match="checksum"):
        select_captured_markets(**args, include_started=True)


def test_repair_preserves_forecasts_originals_and_requires_approval(tmp_path):
    source, raw, game, tip, now, archive = setup(tmp_path)
    originals = {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    args = dict(source=source, target=tmp_path/'repaired', observation_root=raw,
        game_date=date.fromisoformat(GAME_DATE), approval_reference='explicit approval',
        model_sha='a'*64, code={"sealed":True,"package_sha256":'b'*64,
            "dependency_lock_sha256":'c'*64,"git_revision":'d'*40},
        now=tip+timedelta(hours=3))
    with pytest.raises(ValueError, match='explicit approval'):
        repair_market_selection(**{**args,'approval_reference':''})
    report=repair_market_selection(**args)
    assert repair_market_selection(**args)==report
    old=read_object(source/'latest'/f'{GAME_DATE}.json')['games'][0]
    new=read_object(args['target']/'latest'/f'{GAME_DATE}.json')['games'][0]
    assert forecast_inputs(new)==forecast_inputs(old)
    assert new['market']==game['market']
    assert new['prediction_market']==old['prediction_market']
    for path, data in originals.items(): assert (source/path).read_bytes()==data


def test_repair_fails_if_any_game_has_no_archived_market(tmp_path):
    source, raw, game, tip, now, archive = setup(tmp_path)
    archive.unlink()
    with pytest.raises(ValueError, match='Missing verified pre-tip'):
        repair_market_selection(source=source,target=tmp_path/'bad',observation_root=raw,
            game_date=date.fromisoformat(GAME_DATE),approval_reference='approved',model_sha='a'*64,
            code={"sealed":True,"package_sha256":'b'*64,"dependency_lock_sha256":'c'*64,"git_revision":'d'*40},
            now=tip+timedelta(hours=3))
    assert not (tmp_path/'bad').exists()
