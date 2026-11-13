from datetime import date

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


def reconstruct(day, *, rows=(), baseline=None, repairs=()):
    return build_rosters(
        season="2026-27", day=date.fromisoformat(day), opening=baseline or opening(),
        transactions=list(rows), inputs={}, repair_transactions=list(repairs),
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


def test_seed_members_remain_until_a_recorded_waiver():
    baseline = opening()
    baseline["teams"][FIRST].append(999)
    assert 999 in members(reconstruct("2026-11-08", baseline=baseline), FIRST)
    removed = reconstruct("2026-11-08", baseline=baseline,
                          rows=[transaction(kind="Waive", team=FIRST, pid=999)])
    assert 999 not in members(removed, FIRST)


def test_mccullar_stays_waived_until_the_recorded_toronto_signing():
    nyk, tor, pid = 1610612752, 1610612761, 1641755
    baseline = opening()
    baseline["date"] = date(2026, 10, 19)
    baseline["teams"][nyk].append(pid)
    waiver = transaction(kind="Waive", team=nyk, day="2026-10-23", pid=pid)
    signing = transaction(kind="Signing", team=tor, day="2026-11-05", pid=pid)
    before = reconstruct("2026-10-22", baseline=baseline)
    waived = reconstruct("2026-10-28", baseline=baseline, rows=[waiver])
    signed = reconstruct("2026-11-07", baseline=baseline, rows=[signing, waiver])
    assert pid in members(before, nyk)
    assert all(pid not in ids for ids in waived["teams"].values())
    assert pid in members(signed, tor) and pid not in members(signed, nyk)


def test_ambiguous_same_date_moves_are_errors():
    with pytest.raises(RosterError, match="multiple destination"):
        reconstruct("2026-11-08", rows=[
            transaction(team=SECOND), transaction(team=NBA_TEAM_IDS[2]),
        ])


def test_later_transaction_does_not_change_an_earlier_partition():
    with pytest.raises(RosterError, match="outside"):
        reconstruct("2026-11-07", rows=[transaction()])


def test_signing_adds_a_player_without_a_box_score_and_duplicates_are_idempotent():
    signing = transaction(kind="Signing", team=FIRST, day="2026-11-07", pid=999)
    record = reconstruct("2026-11-08", rows=[signing])
    assert 999 in members(record, FIRST)
    assert reconstruct("2026-11-08", rows=[signing, signing]) == record


def repair(*, day="2026-11-07", action="add", team=FIRST, pid=999):
    return {"effective_date": day, "action": action, "team_id": team, "player_id": pid}


def test_repair_applies_on_its_effective_date_and_not_before():
    rows = [repair()]
    before = reconstruct("2026-11-06", repairs=rows)
    on_date = reconstruct("2026-11-07", repairs=rows)
    assert 999 not in members(before, FIRST)
    assert 999 in members(on_date, FIRST)


@pytest.mark.parametrize("day", ["2026-11-11", "2026-11-12"])
def test_nance_opening_date_repair_without_moving_partitions(day):
    baseline = opening()
    baseline["date"] = date(2026, 10, 19)
    record = reconstruct(day, baseline=baseline, repairs=[
        repair(day="2026-10-19", team=1610612749, pid=1631250),
    ])
    assert 1631250 in members(record, 1610612749)
    assert sum(ids.count(1631250) for ids in record["teams"].values()) == 1


def test_later_official_waiver_and_trade_follow_earlier_repairs():
    rows = [transaction(kind="Waive", team=FIRST, pid=999, day="2026-11-08"),
            transaction(team=SECOND, pid=999, day="2026-11-09")]
    assert 999 in members(reconstruct("2026-11-07", repairs=[repair()]), FIRST)
    waived = reconstruct("2026-11-08", repairs=[repair()], rows=rows[:1])
    assert all(999 not in ids for ids in waived["teams"].values())
    moved = reconstruct("2026-11-09", repairs=[repair()], rows=list(reversed(rows)))
    assert 999 in members(moved, SECOND) and 999 not in members(moved, FIRST)


def test_repairs_are_sorted_by_date_then_follow_file_order_on_same_day():
    rows = [repair(day="2026-11-08", action="remove"), repair()]
    record = reconstruct("2026-11-08", repairs=rows)
    assert all(999 not in ids for ids in record["teams"].values())
    record = reconstruct("2026-11-08", repairs=[repair(), repair(), repair(action="remove")])
    assert all(999 not in ids for ids in record["teams"].values())


def test_same_day_repair_follows_official_transaction():
    record = reconstruct("2026-11-08", rows=[transaction()], repairs=[
        repair(day="2026-11-08", action="remove", team=SECOND, pid=1),
    ])
    assert all(1 not in ids for ids in record["teams"].values())


@pytest.mark.parametrize("changes", [
    {"action": "guess"}, {"team_id": 123}, {"player_id": 0},
    {"effective_date": "bad-date"}, {"effective_date": "2026-11-05"},
    {"source": "unapproved"},
])
def test_invalid_repairs_fail_instead_of_being_ignored(changes):
    with pytest.raises(RosterError, match="Invalid roster repair"):
        reconstruct("2026-11-08", repairs=[{**repair(), **changes}])
