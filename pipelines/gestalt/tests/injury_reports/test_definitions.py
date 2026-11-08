from dagster import Definitions

from gestalt_pipelines.definitions import defs


def test_scaffold_definitions_are_loadable() -> None:
    Definitions.validate_loadable(defs)
