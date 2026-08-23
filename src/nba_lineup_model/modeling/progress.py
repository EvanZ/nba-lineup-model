"""Tail-friendly progress formatting for long-running modeling workflows."""

from __future__ import annotations


def format_progress_bar(
    current: int,
    total: int,
    *,
    label: str,
    width: int = 24,
) -> str:
    """Return one durable progress-log line without terminal control characters."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= current <= total:
        raise ValueError("current must be between zero and total")
    if width <= 0:
        raise ValueError("width must be positive")

    fraction = current / total
    completed = round(width * fraction)
    bar = "#" * completed + "." * (width - completed)
    return f"[{bar}] {current:>2}/{total:<2} {fraction:>6.1%} {label}"
