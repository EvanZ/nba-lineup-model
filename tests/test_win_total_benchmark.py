from __future__ import annotations

from nba_lineup_model.rotation.win_total_benchmark import parse_preseason_win_totals


def test_parse_preseason_win_totals_normalizes_historical_team_codes() -> None:
    rows = "".join(
        f'<tr><th data-stat="team"><a href="/teams/{team}/2025.html">Team</a></th>'
        f'<td data-stat="wins_ou">{20 + index}.5</td></tr>'
        for index, team in enumerate(
            [
                "PHO",
                "BRK",
                "CHO",
                *[f"A{chr(65 + index // 26)}{chr(65 + index % 26)}" for index in range(27)],
            ]
        )
    )
    html = f'<table id="NBA_preseason_odds"><tbody>{rows}</tbody></table>'

    output = parse_preseason_win_totals(html, season="2024-25")

    assert len(output) == 30
    assert {"PHX", "BKN", "CHA"} <= set(output["team"])
    assert output.loc[output["team"].eq("PHX"), "opening_win_total"].iloc[0] == 20.5
