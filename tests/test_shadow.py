import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.models import TriageResult
from app.triage import TriageOutputError
from evals.shadow import collect_repo, fetch_new_issues, output_path

SINCE = datetime(2026, 11, 1, tzinfo=UTC)
METADATA = {"provider": "gemini", "model": "test-model"}


def make_issue(number: int, **overrides) -> dict:
    issue = {
        "number": number,
        "title": f"Issue {number}",
        "body": "Something is wrong.",
        "html_url": f"https://github.com/o/r/issues/{number}",
        "created_at": "2026-11-03T10:00:00Z",
        "state": "open",
        "comments": 0,
        "labels": [],
        "user": {"type": "User"},
    }
    issue.update(overrides)
    return issue


def make_result() -> TriageResult:
    return TriageResult(
        category="bug",
        priority="medium",
        summary="A summary",
        rationale="A rationale",
        needs_repository_investigation=True,
        confidence=0.9,
        proposed_labels=["bug"],
        draft_response="Thanks.",
    )


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def collect(path: Path, issues: list[dict], triage=None, limit: int = 20) -> None:
    collect_repo(
        "o/r",
        SINCE,
        path,
        fetch=lambda repo, since: issues,
        triage=triage or (lambda report: make_result()),
        limit=limit,
        delay=0,
        metadata=METADATA,
    )


def test_fetch_skips_pull_requests() -> None:
    page = [make_issue(1), make_issue(2, pull_request={"url": "x"})]

    issues = fetch_new_issues("o/r", SINCE, get_json=lambda url: page if url.endswith("&page=1") else [])

    assert [i["number"] for i in issues] == [1]


def test_fetch_skips_bot_authors() -> None:
    page = [make_issue(1), make_issue(2, user={"type": "Bot"})]

    issues = fetch_new_issues("o/r", SINCE, get_json=lambda url: page if url.endswith("&page=1") else [])

    assert [i["number"] for i in issues] == [1]


def test_fetch_stops_at_the_first_page_reaching_older_issues() -> None:
    old = make_issue(1, created_at="2026-10-01T10:00:00Z")
    urls = []

    def get_json(url: str) -> list[dict]:
        urls.append(url)
        return [make_issue(2), old]

    issues = fetch_new_issues("o/r", SINCE, get_json=get_json)

    assert [i["number"] for i in issues] == [2]
    assert len(urls) == 1


def test_fetch_reads_later_pages_until_they_run_out() -> None:
    pages = {"&page=1": [make_issue(3)], "&page=2": [make_issue(2)]}

    def get_json(url: str) -> list[dict]:
        return next((v for k, v in pages.items() if url.endswith(k)), [])

    issues = fetch_new_issues("o/r", SINCE, get_json=get_json)

    assert [i["number"] for i in issues] == [3, 2]


def test_record_snapshots_the_prediction_and_issue_state(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    issue = make_issue(7, comments=2, labels=[{"name": "triage"}], state="closed")

    collect(path, [issue])

    [record] = read_records(path)
    assert record["number"] == 7
    assert record["category"] == "bug"
    assert record["priority"] == "medium"
    assert record["needs_repository_investigation"] is True
    assert record["confidence"] == 0.9
    assert record["labels_at_prediction"] == ["triage"]
    assert record["comments_at_prediction"] == 2
    assert record["state_at_prediction"] == "closed"
    assert record["body"] == "Something is wrong."
    assert record["model"] == "test-model"
    assert len(record["prompt_version"]) == 12
    assert record["error"] is None


def test_second_run_does_not_triage_seen_issues(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    calls = []

    def triage(report) -> TriageResult:
        calls.append(report.title)
        return make_result()

    collect(path, [make_issue(1)], triage)
    collect(path, [make_issue(2), make_issue(1)], triage)

    assert calls == ["Issue 1", "Issue 2"]
    assert [r["number"] for r in read_records(path)] == [1, 2]


def test_oldest_issues_are_triaged_first_within_the_limit(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"

    collect(path, [make_issue(3), make_issue(2), make_issue(1)], limit=2)

    assert [r["number"] for r in read_records(path)] == [1, 2]


def test_issue_with_null_body_is_triaged(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"

    collect(path, [make_issue(1, body=None)])

    [record] = read_records(path)
    assert record["category"] == "bug"
    assert record["body"] == ""


def test_invalid_model_output_is_recorded_and_the_run_continues(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"

    def triage(report) -> TriageResult:
        if report.title == "Issue 1":
            raise TriageOutputError("bad json", raw_response="{")
        return make_result()

    collect(path, [make_issue(2), make_issue(1)], triage)

    first, second = read_records(path)
    assert first["category"] is None
    assert "bad json" in first["error"]
    assert second["category"] == "bug"


def test_crash_keeps_records_already_saved(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"

    def triage(report) -> TriageResult:
        if report.title == "Issue 2":
            raise RuntimeError("quota exhausted")
        return make_result()

    with pytest.raises(RuntimeError):
        collect(path, [make_issue(2), make_issue(1)], triage)

    assert [r["number"] for r in read_records(path)] == [1]


def test_output_path_is_one_file_per_repo(tmp_path: Path) -> None:
    assert output_path(tmp_path, "pypa/pip") == tmp_path / "pypa__pip.jsonl"
