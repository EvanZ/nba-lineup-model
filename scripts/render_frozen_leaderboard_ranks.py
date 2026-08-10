"""Materialize ranks and single-leader bolding in the frozen leaderboard."""

from __future__ import annotations

import re
from pathlib import Path


PAGE = Path("docs/models/preseason-leaderboard.md")
TABLES = {
    "Possession And Game Results": (4, ("min", "min", "min", "max", "max"), True),
    "Full-Game Outcomes": (2, ("min", "min", "max"), False),
    "Team Net Rating": (2, ("min", "min", "max", "max"), False),
    "Team Win Totals": (2, ("min", "min", "min", "max"), False),
}


def _number(value: str) -> float:
    value = re.sub(r"\s*\(#\d+\)", "", value)
    value = value.replace("**", "")
    return float(re.sub(r"[^0-9.+-]", "", value))


def _render_table(lines: list[str], start: int, config: tuple[int, tuple[str, ...], bool]) -> int:
    first_metric, directions, split_cohorts = config
    end = start + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    rows = [line.split("|")[1:-1] for line in lines[start + 2 : end]]
    for metric_offset, direction in enumerate(directions):
        column = first_metric + metric_offset
        groups: dict[str, list[list[str]]] = {}
        for row in rows:
            groups.setdefault(row[1].strip() if split_cohorts else "all", []).append(row)
        for group in groups.values():
            values = [_number(row[column]) for row in group]
            ordered = sorted(set(values), reverse=direction == "max")
            for row, value in zip(group, values, strict=True):
                base = re.sub(r"\*\*|\s*\(#\d+\)", "", row[column]).strip()
                rank = ordered.index(value) + 1
                rendered = f"{base} (#{rank})"
                row[column] = f" **{rendered}** " if rank == 1 else f" {rendered} "
    lines[start + 2 : end] = ["|" + "|".join(row) + "|" for row in rows]
    return end


def main() -> None:
    lines = PAGE.read_text().splitlines()
    index = 0
    while index < len(lines):
        heading = lines[index].removeprefix("## ")
        if heading not in TABLES:
            index += 1
            continue
        table = next(i for i in range(index + 1, len(lines)) if lines[i].startswith("|"))
        index = _render_table(lines, table, TABLES[heading])
    PAGE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
