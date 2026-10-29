
from pathlib import Path

import pytest

from app.repository import search_repository


def test_search_finds_matching_line(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        "def start_application():\n    return True\n",
        encoding="utf-8",
    )

    results = search_repository(str(tmp_path), "start application")

    assert results
    assert results[0]["file"] == "main.py"
    assert results[0]["line"] == "1"
    assert "start" in results[0]["matched_terms"]


def test_search_skips_excluded_directories(tmp_path: Path) -> None:
    excluded = tmp_path / ".venv"
    excluded.mkdir()
    (excluded / "example.py").write_text(
        "application startup crash",
        encoding="utf-8",
    )

    (tmp_path / "main.py").write_text(
        "application startup crash",
        encoding="utf-8",
    )

    results = search_repository(str(tmp_path), "application startup crash")

    assert all(".venv" not in item["file"] for item in results)
    assert any(item["file"] == "main.py" for item in results)


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="meaningful term"):
        search_repository(str(tmp_path), "the and or")


def test_search_rejects_missing_repository(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(ValueError, match="does not exist"):
        search_repository(str(missing), "startup crash")
