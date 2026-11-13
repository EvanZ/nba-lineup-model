from datetime import UTC, date, datetime

import pytest

from gestalt_pipelines.rosters.utils import NBA_TEAM_IDS, build_rosters
from gestalt_pipelines.utils import PACIFIC

TEAM, OTHER = 1610612746, 1610612749
PLAYER = 1631104


def signing(day="2026-10-29", description="LA Clippers signed guard Blake Wesley to a 10-Day Contract.", kind="Signing", team=TEAM):
    return {"TRANSACTION_DATE": day + "T00:00:00", "Transaction_Type": kind,
            "PLAYER_ID": PLAYER, "TEAM_ID": team, "TRANSACTION_DESCRIPTION": description}


def games(dates):
    return [{"game_id": str(index), "home_team_id": TEAM, "away_team_id": OTHER,
             "game_time_utc": datetime.fromisoformat(day + "T20:00:00").replace(tzinfo=PACIFIC).astimezone(UTC)}
            for index, day in enumerate(dates)]


def build(day, rows, game_dates):
    return build_rosters(
        season="2026-27", day=date.fromisoformat(day),
        opening={"date": date(2026, 10, 19),
                 "teams": {team: [index + 1] for index, team in enumerate(NBA_TEAM_IDS)}},
        transactions=rows, inputs={}, games=games(game_dates),
    )


EARLY_GAMES = ["2026-10-30", "2026-11-01", "2026-11-02"]


def test_wesley_retained_through_tenth_day_and_removed_the_next_day():
    last = build("2026-11-07", [signing()], EARLY_GAMES)
    expired = build("2026-11-08", [signing()], EARLY_GAMES)
    assert PLAYER in last["teams"][str(TEAM)]
    assert PLAYER not in expired["teams"][str(TEAM)]
    entry = expired["ten_day_contracts"][0]
    assert entry["third_game_date"] == "2026-11-02"
    assert entry["last_active_date"] == "2026-11-07"
    assert entry["removal_date"] == "2026-11-08"
    assert entry["superseded_date"] is None


def test_contract_extends_through_third_team_game_even_if_longer_than_ten_days():
    dates = ["2026-10-30", "2026-11-01", "2026-11-12"]
    before = build("2026-11-11", [signing()], dates)
    third = build("2026-11-12", [signing()], dates)
    after = build("2026-11-13", [signing()], dates)
    assert PLAYER in before["teams"][str(TEAM)]
    assert before["ten_day_contracts"][0]["removal_date"] is None
    assert PLAYER in third["teams"][str(TEAM)]
    assert PLAYER not in after["teams"][str(TEAM)]
    assert after["ten_day_contracts"][0]["removal_date"] == "2026-11-13"


@pytest.mark.parametrize("kind", ["Signing", "ContractConverted"])
def test_ordinary_signing_or_conversion_cancels_old_expiration(kind):
    record = build("2026-11-12", [signing(), signing(day="2026-11-05", description="Signed for rest of season.", kind=kind)], EARLY_GAMES)
    assert PLAYER in record["teams"][str(TEAM)]
    assert record["ten_day_contracts"][0]["superseded_date"] == "2026-11-05"


def test_second_ten_day_contract_survives_first_expiration_then_expires_itself():
    rows = [signing(), signing(day="2026-11-05")]
    dates = EARLY_GAMES + ["2026-11-05", "2026-11-06", "2026-11-09"]
    still_signed = build("2026-11-14", list(reversed(rows)), dates)
    expired = build("2026-11-15", rows, dates)
    assert PLAYER in still_signed["teams"][str(TEAM)]
    assert PLAYER not in expired["teams"][str(TEAM)]
    assert expired["ten_day_contracts"][0]["superseded_date"] == "2026-11-05"
    assert expired["ten_day_contracts"][1]["removal_date"] == "2026-11-15"


def test_expiration_does_not_remove_player_after_a_trade_to_another_team():
    record = build("2026-11-12", [signing(), signing(day="2026-11-05", kind="Trade", team=OTHER)], EARLY_GAMES)
    assert PLAYER in record["teams"][str(OTHER)]
    assert PLAYER not in record["teams"][str(TEAM)]


def test_new_contract_on_expiration_date_is_applied_after_old_expiration():
    record = build("2026-11-12", [signing(), signing(day="2026-11-08", description="Standard contract")], EARLY_GAMES)
    assert PLAYER in record["teams"][str(TEAM)]


def test_early_waiver_removes_player_without_waiting_for_expiration():
    record = build("2026-11-06", [signing(), signing(day="2026-11-05", kind="Waive")], EARLY_GAMES)
    assert PLAYER not in record["teams"][str(TEAM)]
    assert record["ten_day_contracts"][0]["superseded_date"] == "2026-11-05"


def test_duplicate_signing_is_idempotent_and_exhibit_ten_is_not_a_ten_day_contract():
    assert build("2026-11-12", [signing(), signing()], EARLY_GAMES) == build("2026-11-12", [signing()], EARLY_GAMES)
    record = build("2026-11-12", [signing(description="Signed an Exhibit 10 contract.")], [])
    assert record["ten_day_contracts"] == []
    assert PLAYER in record["teams"][str(TEAM)]


def test_short_term_signing_requires_schedule_instead_of_guessing_expiration():
    with pytest.raises(ValueError, match="require the official NBA schedule"):
        build_rosters(season="2026-27", day=date(2026, 11, 12),
                      opening={"date": date(2026, 10, 19), "teams": {t: [i+1] for i,t in enumerate(NBA_TEAM_IDS)}},
                      transactions=[signing()], inputs={})
