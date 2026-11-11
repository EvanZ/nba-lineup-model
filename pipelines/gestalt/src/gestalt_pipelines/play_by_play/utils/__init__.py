"""Check source policy without interpreting play-by-play events."""

import json


def _official_fields(pairs: list[tuple]) -> dict:
    for key, value in pairs:
        if "espn" in key.casefold() or (
            key.casefold() in {"source", "source_url", "provider"}
            and isinstance(value, str) and "espn" in value.casefold()
        ):
            raise ValueError("Prohibited basketball source in NBA play-by-play.")
    return dict(pairs)


def official_json(payload: bytes) -> dict:
    document = json.loads(payload, object_pairs_hook=_official_fields)
    if not isinstance(document, dict):
        raise ValueError("NBA play-by-play JSON must be an object.")
    return document


def check_download(payload: bytes) -> None:
    """Check response availability and source policy, leaving parsing to consumers."""
    if not payload.strip():
        raise ValueError("NBA play-by-play endpoint returned no data.")
    if b"espn" in payload.lower():
        raise ValueError("Prohibited basketball source in NBA play-by-play.")


__all__ = ["official_json", "check_download"]
