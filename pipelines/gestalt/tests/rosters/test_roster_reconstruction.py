from datetime import UTC, date, datetime

import pytest

from gestalt_pipelines.rosters.utils import NBA_TEAM_IDS, RosterError, build_rosters

FIRST, SECOND = NBA_TEAM_IDS[:2]


def opening():
    return {
        "date": date(2026, 11, 6),
        "teams": {
            team: [index + 1, index + 1001]
            for index, team in enumerate(NBA_TEAM_IDS)
        },
    }


def transaction(kind="Trade", team=SECOND, day="2026-11-08", pid=1):
    return {
        "Transaction_Type": kind, "TRANSACTION_DATE": day + "T00:00:00",
        "PLAYER_ID": pid, "TEAM_ID": team,
    }


def reconstruct(day, *, rows=(), boxes=(), baseline=None):
    return build_rosters(
        season="2026-27", day=date.fromisoformat(day), opening=baseline or opening(),
        transactions=list(rows), boxes=list(boxes), inputs={},
    )


def members(record, team):
    return set(record["teams"][str(team)])


def test_dates_are_independent_and_idle_teams_carry_forward():
    after = reconstruct("2026-11-08", rows=[transaction()])
    before = reconstruct("2026-11-07")
    assert len(after["teams"]) == len(before["teams"]) == 30
    assert 1 in members(before, FIRST) and 1 not in members(before, SECOND)
    assert 1 in members(after, SECOND) and 1 not in members(after, FIRST)
    assert members(after, NBA_TEAM_IDS[20]) == members(before, NBA_TEAM_IDS[20])


def test_box_absence_does_not_remove_players_and_later_waiver_does():
    baseline = opening()
    baseline["teams"][FIRST].append(999)
    box = {
        "team_id": FIRST, "game_id": "0022600200",
        "game_time_utc": datetime(2026, 11, 8, 2, tzinfo=UTC),
        "player_ids": [1],
    }
    carried = reconstruct("2026-11-08", boxes=[box], baseline=baseline)
    assert 999 in members(carried, FIRST)
    removed = reconstruct(
        "2026-11-08", boxes=[box], baseline=baseline,
        rows=[transaction(kind="Waive", team=FIRST, pid=999)],
    )
    assert 999 not in members(removed, FIRST)


def test_unexplained_box_team_conflict_is_an_error():
    box = {
        "team_id": SECOND, "game_id": "0022600200",
        "game_time_utc": datetime(2026, 11, 8, 2, tzinfo=UTC),
        "player_ids": [1],
    }
    with pytest.raises(RosterError, match="conflicts") as failure:
        reconstruct("2026-11-08", boxes=[box])
    assert failure.value.details["player_id"] == 1
    assert failure.value.details["box_team_id"] == SECOND


def test_ambiguous_same_date_moves_are_errors():
    with pytest.raises(RosterError, match="multiple destination"):
        reconstruct("2026-11-08", rows=[
            transaction(team=SECOND), transaction(team=NBA_TEAM_IDS[2]),
        ])


def test_later_transaction_does_not_change_an_earlier_partition():
    with pytest.raises(RosterError, match="outside"):
        reconstruct("2026-11-07", rows=[transaction()])


def test_box_can_supply_id_evidence_for_a_recorded_signing():
    box = {
        "team_id": FIRST, "game_id": "0022600200",
        "game_time_utc": datetime(2026, 11, 8, 2, tzinfo=UTC),
        "player_ids": [999],
    }
    record = reconstruct(
        "2026-11-08", boxes=[box],
        rows=[transaction(kind="Signing", team=FIRST, day="2026-11-07", pid=999)],
    )
    assert 999 in members(record, FIRST)
