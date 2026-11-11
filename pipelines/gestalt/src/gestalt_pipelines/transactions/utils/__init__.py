"""Select dated rows from the official NBA movement feed."""

import json
from datetime import date, datetime, timedelta


def _official_fields(pairs: list[tuple]) -> dict:
    for key, value in pairs:
        if "espn" in key.casefold() or (
            key.casefold() in {"source", "source_url", "provider"}
            and isinstance(value, str) and "espn" in value.casefold()
        ):
            raise ValueError("Prohibited basketball source in NBA transactions.")
    return dict(pairs)


def official_json(payload: bytes) -> dict:
    return json.loads(payload, object_pairs_hook=_official_fields)


def transactions_for_date(payload: bytes, day: date) -> list[dict]:
    document = official_json(payload)
    movement = document.get("NBA_Player_Movement") if isinstance(document, dict) else None
    if not isinstance(movement, dict):
        raise ValueError("NBA movement feed is missing NBA_Player_Movement.")
    rows, columns = movement.get("rows"), movement.get("columns")
    if not isinstance(rows, list) or not rows or not isinstance(columns, list):
        raise ValueError("NBA movement feed has invalid rows or columns.")
    if not all(isinstance(column, dict) and isinstance(column.get("Name"), str) for column in columns):
        raise ValueError("NBA movement feed has invalid column definitions.")
    required = {"TRANSACTION_DATE", "Transaction_Type", "TRANSACTION_DESCRIPTION", "PLAYER_ID", "TEAM_ID"}
    if not required <= {column["Name"] for column in columns}:
        raise ValueError("NBA movement feed is missing required transaction columns.")

    selected = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("TRANSACTION_DATE"), str):
            raise ValueError("NBA movement feed contains an undated transaction.")
        # The source supplies calendar dates without actual transaction times.
        # Preserve its reported date; interpreting midnight as UTC would shift it.
        reported_date = datetime.fromisoformat(row["TRANSACTION_DATE"]).date()
        if reported_date == day:
            if not required <= row.keys():
                raise ValueError("NBA movement feed contains an incomplete transaction.")
            selected.append(row)
    return selected


def merge_transactions(existing: list[dict], incoming: list[dict], *, day: date, observed_at: datetime) -> tuple[list[dict], int]:
    """Retain first observations and append distinct source rows."""
    if observed_at.tzinfo is None or observed_at.utcoffset() != timedelta(0):
        raise ValueError("Transaction observation time must be UTC.")
    if not isinstance(existing, list):
        raise ValueError("Saved transactions must be a list.")
    merged, seen = [], set()
    for row in existing:
        if not isinstance(row, dict) or datetime.fromisoformat(row["TRANSACTION_DATE"]).date() != day:
            raise ValueError("Saved transaction belongs to a different date.")
        stamp = datetime.fromisoformat(row["first_observed_at"])
        if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError("Saved transaction observation time must be UTC.")
        key = _row_key(row)
        if key in seen:
            raise ValueError("Saved transaction history contains duplicate rows.")
        seen.add(key)
        merged.append(row)
    added = 0
    for row in incoming:
        if "first_observed_at" in row:
            raise ValueError("NBA source row already contains the reserved observation field.")
        key = _row_key(row)
        if key not in seen:
            seen.add(key)
            merged.append({**row, "first_observed_at": observed_at.isoformat()})
            added += 1
    return merged, added


def _row_key(row: dict) -> str:
    # No transaction ID is supplied. Compare all original NBA fields.
    return json.dumps(
        {key: value for key, value in row.items() if key != "first_observed_at"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )


__all__ = ["merge_transactions", "official_json", "transactions_for_date"]
