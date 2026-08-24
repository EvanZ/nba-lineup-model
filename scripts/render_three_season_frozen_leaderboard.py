"""Insert evaluated candidates and materialize ranks in the frozen three-season tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

PAGE = Path("docs/models/three-season-frozen-backtest.md")


@dataclass(frozen=True)
class TableSpec:
    heading: str
    metrics: tuple[str, ...]
    directions: tuple[str, ...]
    formats: tuple[str, ...]


REGULAR = TableSpec(
    heading="Regular Season",
    metrics=(
        "possession_rmse",
        "possession_mae",
        "possession_skill",
        "eligible_game_margin_rmse",
        "eligible_game_skill",
        "full_game_margin_rmse",
        "game_winner_accuracy",
        "team_net_rating_rmse",
        "pythagorean_win_rmse",
    ),
    directions=(
        "lower",
        "lower",
        "higher",
        "lower",
        "higher",
        "lower",
        "higher",
        "lower",
        "lower",
    ),
    formats=(
        "decimal6",
        "decimal6",
        "percent4",
        "decimal4",
        "percent4",
        "decimal4",
        "percent2",
        "decimal4",
        "decimal4",
    ),
)

PLAYOFFS = TableSpec(
    heading="Playoffs",
    metrics=(
        "possession_rmse",
        "possession_mae",
        "possession_skill",
        "eligible_game_margin_rmse",
        "eligible_game_skill",
    ),
    directions=("lower", "lower", "higher", "lower", "higher"),
    formats=("decimal6", "decimal6", "percent4", "decimal4", "percent4"),
)

CANDIDATES = (
    {
        "model": (
            "[NAIL-RAPM v1.2.1.1 standard USG% (not promoted)]"
            "(nail-rapm-v1211-standard-usage.md)"
        ),
        "regular": {
            "possession_rmse": 1.197951,
            "possession_mae": 1.141332,
            "possession_skill": 0.001269,
            "eligible_game_margin_rmse": 14.026373,
            "eligible_game_skill": 0.181792,
            "full_game_margin_rmse": 14.251590,
            "game_winner_accuracy": 0.681003,
            "team_net_rating_rmse": 3.265764,
            "pythagorean_win_rmse": 7.027857,
        },
        "playoffs": {
            "possession_rmse": 1.192708,
            "possession_mae": 1.137593,
            "possession_skill": 0.000649,
            "eligible_game_margin_rmse": 16.598860,
            "eligible_game_skill": 0.076873,
        },
    },
    {
        "model": (
            "[NAIL-RAPM v1.2.1 pruned non-additive context]"
            "(nail-rapm-v121-pruned-nonadditive.md) **(Production)**"
        ),
        "regular": {
            "possession_rmse": 1.197952,
            "possession_mae": 1.141355,
            "possession_skill": 0.001268,
            "eligible_game_margin_rmse": 14.024550,
            "eligible_game_skill": 0.182005,
            "full_game_margin_rmse": 14.252137,
            "game_winner_accuracy": 0.682427,
            "team_net_rating_rmse": 3.270613,
            "pythagorean_win_rmse": 7.035098,
        },
        "playoffs": {
            "possession_rmse": 1.192709,
            "possession_mae": 1.137608,
            "possession_skill": 0.000647,
            "eligible_game_margin_rmse": 16.594200,
            "eligible_game_skill": 0.077392,
        },
    },
    {
        "model": "[NAIL Critical Spacing candidate](nail-critical-spacing.md)",
        "regular": {
            "possession_rmse": 1.197954,
            "possession_mae": 1.141347,
            "possession_skill": 0.001264,
            "eligible_game_margin_rmse": 14.024052,
            "eligible_game_skill": 0.182063,
            "full_game_margin_rmse": 14.253192,
            "game_winner_accuracy": 0.684705,
            "team_net_rating_rmse": 3.280515,
            "pythagorean_win_rmse": 7.060052,
        },
        "playoffs": {
            "possession_rmse": 1.192706,
            "possession_mae": 1.137609,
            "possession_skill": 0.000652,
            "eligible_game_margin_rmse": 16.589610,
            "eligible_game_skill": 0.077902,
        },
    },
    {
        "model": (
            "[NAIL quartile Critical Spacing plus standard USG% (not promoted)]"
            "(nail-critical-spacing-quartile-standard-usage.md)"
        ),
        "regular": {
            "possession_rmse": 1.197954,
            "possession_mae": 1.141327,
            "possession_skill": 0.001265,
            "eligible_game_margin_rmse": 14.030448,
            "eligible_game_skill": 0.181317,
            "full_game_margin_rmse": 14.258396,
            "game_winner_accuracy": 0.681003,
            "team_net_rating_rmse": 3.278702,
            "pythagorean_win_rmse": 7.045978,
        },
        "playoffs": {
            "possession_rmse": 1.192713,
            "possession_mae": 1.137596,
            "possession_skill": 0.000640,
            "eligible_game_margin_rmse": 16.614526,
            "eligible_game_skill": 0.075130,
        },
    },
    {
        "model": (
            "[NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)]"
            "(nail-rapm-v122-defensive-rebound-profile.md)"
        ),
        "regular": {
            "possession_rmse": 1.1979522763,
            "possession_mae": 1.1413642633,
            "possession_skill": 0.0012676171,
            "eligible_game_margin_rmse": 14.0237872168,
            "eligible_game_skill": 0.1820939841,
            "full_game_margin_rmse": 14.2528925220,
            "game_winner_accuracy": 0.6838507548,
            "team_net_rating_rmse": 3.2723431789,
            "pythagorean_win_rmse": 7.0433920793,
        },
        "playoffs": {
            "possession_rmse": 1.1927018556,
            "possession_mae": 1.1376086100,
            "possession_skill": 0.0006588817,
            "eligible_game_margin_rmse": 16.5875354540,
            "eligible_game_skill": 0.0781324907,
        },
    },
    {
        "model": (
            "[NAIL-RAPM v1.2.3 free-throw profile (not promoted)]"
            "(nail-rapm-v123-free-throw-profile.md)"
        ),
        "regular": {
            "possession_rmse": 1.1979638075,
            "possession_mae": 1.1413413730,
            "possession_skill": 0.0012479203,
            "eligible_game_margin_rmse": 14.0456089106,
            "eligible_game_skill": 0.1795471833,
            "full_game_margin_rmse": 14.2725513148,
            "game_winner_accuracy": 0.6847052110,
            "team_net_rating_rmse": 3.2974405874,
            "pythagorean_win_rmse": 7.0511173098,
        },
        "playoffs": {
            "possession_rmse": 1.1927139536,
            "possession_mae": 1.1375832055,
            "possession_skill": 0.0006380933,
            "eligible_game_margin_rmse": 16.6156569369,
            "eligible_game_skill": 0.0750044062,
        },
    },
    {
        "model": (
            "[NAIL-RAPM v1.2.4 free-throw replacement (not promoted)]"
            "(nail-rapm-v124-free-throw-replacement.md)"
        ),
        "regular": {
            "possession_rmse": 1.1979659857,
            "possession_mae": 1.1413282509,
            "possession_skill": 0.0012441716,
            "eligible_game_margin_rmse": 14.0456574722,
            "eligible_game_skill": 0.1795412782,
            "full_game_margin_rmse": 14.2733448210,
            "game_winner_accuracy": 0.6818567360,
            "team_net_rating_rmse": 3.2932584421,
            "pythagorean_win_rmse": 7.0458984674,
        },
        "playoffs": {
            "possession_rmse": 1.1927056629,
            "possession_mae": 1.1375673148,
            "possession_skill": 0.0006524928,
            "eligible_game_margin_rmse": 16.5987960192,
            "eligible_game_skill": 0.0768799964,
        },
    },
)


def _number(cell: str) -> float:
    value = re.sub(r"\*\*|\s*\(\d+\)", "", cell).replace("%", "").strip()
    return float(value)


def _format(value: float, kind: str) -> str:
    if kind == "decimal6":
        return f"{value:.6f}"
    if kind == "decimal4":
        return f"{value:.4f}"
    if kind == "percent4":
        return f"{100 * value:.4f}%"
    if kind == "percent2":
        return f"{100 * value:.2f}%"
    raise ValueError(f"Unknown formatting contract: {kind}")


def _table_bounds(lines: list[str], heading: str) -> tuple[int, int]:
    heading_index = lines.index(f"## {heading}")
    start = next(
        index for index in range(heading_index, len(lines)) if lines[index].startswith("|")
    )
    end = start + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return start, end


def _read_rows(lines: list[str], start: int, end: int, spec: TableSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in lines[start + 2 : end]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        row: dict[str, object] = {"model": cells[0]}
        for offset, metric in enumerate(spec.metrics, start=2):
            raw = _number(cells[offset])
            row[metric] = raw / 100 if spec.formats[offset - 2].startswith("percent") else raw
        rows.append(row)
    return rows


def _rank(rows: list[dict[str, object]], metric: str, direction: str) -> dict[str, int]:
    values = sorted({float(row[metric]) for row in rows}, reverse=direction == "higher")
    return {str(row["model"]): values.index(float(row[metric])) + 1 for row in rows}


def _render(
    lines: list[str], spec: TableSpec, candidate_metrics: tuple[dict[str, object], ...]
) -> None:
    start, end = _table_bounds(lines, spec.heading)
    rows = _read_rows(lines, start, end, spec)
    candidate_models = {str(candidate["model"]) for candidate in candidate_metrics}
    candidate_pages = {_model_page(model) for model in candidate_models}
    rows = [
        row
        for row in rows
        if str(row["model"]) not in candidate_models
        and _model_page(str(row["model"])) not in candidate_pages
    ]
    rows.extend(candidate_metrics)
    ranks = {
        metric: _rank(rows, metric, direction)
        for metric, direction in zip(spec.metrics, spec.directions, strict=True)
    }
    for row in rows:
        row_ranks = [ranks[metric][str(row["model"])] for metric in spec.metrics]
        row["median_rank"] = int(median(row_ranks))
        row["mean_rank"] = sum(row_ranks) / len(row_ranks)
    full_metric = (
        "full_game_margin_rmse"
        if "full_game_margin_rmse" in spec.metrics
        else "eligible_game_margin_rmse"
    )
    rows.sort(
        key=lambda row: (
            row["median_rank"],
            row["mean_rank"],
            row[full_metric],
            str(row["model"]),
        )
    )

    rendered = [lines[start], lines[start + 1]]
    for row in rows:
        median_rank = str(row["median_rank"])
        if row["median_rank"] == 1:
            median_rank = f"**{median_rank}**"
        cells = [str(row["model"]), median_rank]
        for metric, kind in zip(spec.metrics, spec.formats, strict=True):
            value = _format(float(row[metric]), kind)
            if ranks[metric][str(row["model"])] == 1:
                value = f"**{value} ({ranks[metric][str(row['model'])]})**"
            else:
                value = f"{value} ({ranks[metric][str(row['model'])]})"
            cells.append(value)
        rendered.append("| " + " | ".join(cells) + " |")
    lines[start:end] = rendered


def _model_page(model: str) -> str | None:
    """Return a Markdown model link target to make label changes idempotent."""

    match = re.search(r"\]\(([^)]+\.md)\)", model)
    return match.group(1) if match else None


def main() -> None:
    lines = PAGE.read_text().splitlines()
    _render(
        lines,
        REGULAR,
        tuple({"model": candidate["model"], **candidate["regular"]} for candidate in CANDIDATES),
    )
    _render(
        lines,
        PLAYOFFS,
        tuple({"model": candidate["model"], **candidate["playoffs"]} for candidate in CANDIDATES),
    )
    lines[1] = 'last_updated: "2026-08-24"'
    PAGE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
