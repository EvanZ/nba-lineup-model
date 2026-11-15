"""Dagster pipelines for NBA GESTALT."""

from pathlib import Path

from dotenv import load_dotenv

# Reloading the code location rereads its existing project configuration.
load_dotenv(Path(__file__).resolve().parents[2] / "code-location.env", override=True)
