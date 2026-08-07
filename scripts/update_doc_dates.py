"""Write Git-derived last-updated metadata into every documentation page."""

from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path

FRONT_MATTER_DELIMITER = "---\n"


def with_last_updated(text: str, updated: str) -> str:
    """Return Markdown with a quoted ``last_updated`` front-matter value."""

    field = f'last_updated: "{updated}"'
    if text.startswith(FRONT_MATTER_DELIMITER):
        closing = text.find(FRONT_MATTER_DELIMITER, len(FRONT_MATTER_DELIMITER))
        if closing < 0:
            raise ValueError("Markdown front matter is missing its closing delimiter")
        front_matter = text[len(FRONT_MATTER_DELIMITER) : closing]
        lines = front_matter.rstrip("\n").splitlines()
        replacement = [line for line in lines if not line.startswith("last_updated:")]
        replacement.append(field)
        return FRONT_MATTER_DELIMITER + "\n".join(replacement) + "\n" + text[closing:]
    return f"{FRONT_MATTER_DELIMITER}{field}\n{FRONT_MATTER_DELIMITER}\n{text}"


def git_last_updated(path: Path, *, project_root: Path, fallback: str) -> str:
    """Return the final substantive commit date for one page."""

    try:
        relative_path = path.relative_to(project_root)
        completed = subprocess.run(
            ["git", "log", "--format=%H%x00%cs", "--follow", "--", str(relative_path)],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return fallback
    for record in completed.stdout.splitlines():
        commit, _, date = record.partition("\x00")
        if commit and date and not _is_metadata_only_commit(
            commit, relative_path, project_root
        ):
            return date
    return fallback


def _is_metadata_only_commit(commit: str, path: Path, project_root: Path) -> bool:
    """Return whether a commit changes only generated date front matter for a page."""

    completed = subprocess.run(
        ["git", "show", "--format=", "--unified=0", commit, "--", str(path)],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return False
    changed_lines = [
        line[1:]
        for line in completed.stdout.splitlines()
        if line[:1] in {"+", "-"} and not line.startswith(("+++", "---"))
    ]
    if not changed_lines:
        return False
    return all(
        not line.strip() or line.strip() == "---" or line.startswith("last_updated:")
        for line in changed_lines
    )


def update_docs(*, docs_dir: Path, project_root: Path, fallback: str) -> int:
    """Update every Markdown page and return the number whose metadata changed."""

    changed = 0
    for page in sorted(docs_dir.rglob("*.md")):
        updated = git_last_updated(page, project_root=project_root, fallback=fallback)
        source = page.read_text()
        rendered = with_last_updated(source, updated)
        if rendered != source:
            page.write_text(rendered)
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Update documentation last-updated metadata")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--date", help="Fallback date in YYYY-MM-DD format")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    fallback = args.date or datetime.now(UTC).date().isoformat()
    changed = update_docs(
        docs_dir=(project_root / args.docs_dir).resolve(),
        project_root=project_root,
        fallback=fallback,
    )
    print(f"Updated last-updated metadata on {changed} documentation pages")


if __name__ == "__main__":
    main()
