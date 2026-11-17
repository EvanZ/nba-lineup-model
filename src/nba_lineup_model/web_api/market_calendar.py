"""Saved daily grade summaries shared by the market report and Schedule UI."""


def saved_ats_result(comparison):
    ats = comparison.get("ats")
    result = ats["result"] if ats is not None else "unavailable"
    if result not in {"win", "loss", "push", "unavailable", "no_pick", "not_applicable"}:
        raise ValueError("Invalid saved ATS grade")
    return result


def summarize_model_grades(rows):
    record = {key: 0 for key in ("wins", "losses", "atsWins", "atsLosses", "atsPushes")}
    coverage = {
        "model_graded_game_count": 0,
        "ats_graded_game_count": 0,
        "ats_no_pick_game_count": 0,
        "missing_model_game_ids": [],
        "missing_ats_game_ids": [],
    }
    ats_fields = {"win": "atsWins", "loss": "atsLosses", "push": "atsPushes"}
    for gid, row in rows.items():
        comparisons = row["comparisons"]
        if not isinstance(comparisons, dict):
            raise ValueError("Invalid saved model comparisons")
        comparison = comparisons.get("model")
        if comparison is not None and not isinstance(comparison, dict):
            raise ValueError("Invalid saved model grade")
        winner = comparison.get("winner_correct") if comparison is not None else None
        if winner is None:
            coverage["missing_model_game_ids"].append(gid)
        elif type(winner) is not bool:
            raise ValueError("Invalid saved winner grade")
        else:
            record["wins" if winner else "losses"] += 1
            coverage["model_graded_game_count"] += 1
        ats = saved_ats_result(comparison) if comparison is not None else "unavailable"
        if ats in ats_fields:
            record[ats_fields[ats]] += 1
            coverage["ats_graded_game_count"] += 1
        elif ats == "no_pick":
            coverage["ats_no_pick_game_count"] += 1
        else:
            coverage["missing_ats_game_ids"].append(gid)
    if not coverage["model_graded_game_count"] and not coverage["ats_graded_game_count"]:
        return None, coverage, "unavailable"
    partial = bool(coverage["missing_model_game_ids"] or coverage["missing_ats_game_ids"])
    return record, coverage, "partial" if partial else "available"


def calendar_summaries(days, scheduled, through):
    """Summarize reports already loaded by the hourly market endpoint; no I/O."""
    from datetime import date, timedelta

    reports = {day["game_date_pacific"]: day for day in days}
    first = date.fromisoformat(min(scheduled or reports))
    summaries = {}
    for offset in range((through - first).days + 1):
        day = (first + timedelta(days=offset)).isoformat()
        count = len(scheduled.get(day, set()))
        summary = {"game_date": day, "game_count": count, "record": None, "record_source": None}
        report = reports.get(day)
        if not count:
            summary["record_status"] = "no_games"
        elif report is None:
            summary["record_status"] = "missing"
        else:
            summary["record_source"] = report["selection"]
            try:
                rows = {g["game_id"]: g for g in report["games"]}
                record, coverage, status = summarize_model_grades(rows)
                summary.update(record=record, record_coverage=coverage, record_status=status)
            except (KeyError, TypeError, ValueError):
                summary["record_status"] = "invalid"
        summaries[day] = summary
    return summaries
