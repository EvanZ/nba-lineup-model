from __future__ import annotations

from datetime import date

import httpx

from nba_lineup_model.players.gamebooks import (
    fetch_gamebook,
    gamebook_url,
    parse_inactive_players,
)


def test_builds_official_gamebook_url() -> None:
    assert gamebook_url(
        game_date=date(2021, 12, 13), away_team="PHX", home_team="LAC"
    ) == "https://statsdmz.nba.com/pdfs/20211213/20211213_PHXLAC_book.pdf"


def test_parses_repeated_inactive_sections_with_reasons() -> None:
    sun_inactives = (
        "Kaminsky (Injury/Illness - Right Knee; Stress reaction), "
        "Nader (Injury/Illness - Non-covid; Illness)"
    )
    clippers_inactives = (
        "Coffey (G League - Two-Way), "
        "Leonard (Injury/Illness - Right Knee; ACL - Injury Recovery)"
    )
    text = "\n".join(
        (
            f"Inactive: Suns - {sun_inactives}",
            f"Inactive: Clippers - {clippers_inactives}",
            f"Inactive: Suns - {sun_inactives}",
            f"Inactive: Clippers - {clippers_inactives}",
        )
    )

    records = parse_inactive_players(text, team_labels=("Suns", "Clippers"))

    assert [(record.team_label, record.player_name, record.reason) for record in records] == [
        ("Suns", "Kaminsky", "Injury/Illness - Right Knee; Stress reaction"),
        ("Suns", "Nader", "Injury/Illness - Non-covid; Illness"),
        ("Clippers", "Coffey", "G League - Two-Way"),
        ("Clippers", "Leonard", "Injury/Illness - Right Knee; ACL - Injury Recovery"),
    ]


def test_fetch_gamebook_retains_pdf_cache(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"%PDF-1.7 fixture", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        first = fetch_gamebook(
            game_id="0022100001",
            game_date=date(2021, 10, 19),
            away_team="BKN",
            home_team="MIL",
            raw_dir=tmp_path,
            http_client=client,
        )
        second = fetch_gamebook(
            game_id="0022100001",
            game_date=date(2021, 10, 19),
            away_team="BKN",
            home_team="MIL",
            raw_dir=tmp_path,
            http_client=client,
        )
    finally:
        client.close()

    assert first == second
    assert first.read_bytes() == b"%PDF-1.7 fixture"
    assert len(requests) == 1
