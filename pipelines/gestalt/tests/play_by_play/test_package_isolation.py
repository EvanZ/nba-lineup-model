"""Guard the independent pipeline package against legacy imports."""

import ast
from pathlib import Path


def test_pipeline_source_has_no_legacy_imports():
    source = Path(__file__).resolve().parents[2]/"src"/"gestalt_pipelines"
    forbidden = []
    for path in source.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            if any(name == "nba_lineup_model" or name.startswith("nba_lineup_model.") for name in names):
                forbidden.append(str(path.relative_to(source)))
    assert not forbidden, forbidden
