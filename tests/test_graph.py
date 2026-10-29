
from unittest.mock import patch

from app.graph import triage_graph
from app.models import IssueReport, TriageResult


def make_result(needs_investigation: bool) -> TriageResult:
    return TriageResult(
        category="bug",
        priority="high",
        summary="Application startup failure",
        rationale="The report describes a startup failure.",
        needs_repository_investigation=needs_investigation,
        confidence=0.85,
        proposed_labels=["bug", "startup"],
        draft_response="Thank you for reporting this issue.",
    )


def test_graph_skips_investigation_when_not_needed() -> None:
    report = IssueReport(
        title="Clarify configuration option",
        description="What does the timeout configuration option control?",
    )

    with (
        patch("app.graph.triage_issue", return_value=make_result(False)),
        patch("app.graph.search_repository") as mock_search,
    ):
        state = triage_graph.invoke(
            {"report": report, "repository_path": "."}
        )

    mock_search.assert_not_called()
    assert state["result"].needs_repository_investigation is False
    assert state["investigation_status"] == "skipped"
    assert state["evidence"] == []


def test_graph_runs_investigation_when_needed() -> None:
    report = IssueReport(
        title="Application crashes on startup",
        description="The application exits on startup with an error traceback.",
    )
    evidence = [
        {
            "file": "app/main.py",
            "line": "12",
            "text": "initialize_application()",
            "matched_terms": "application",
        }
    ]

    with (
        patch("app.graph.triage_issue", return_value=make_result(True)),
        patch("app.graph.search_repository", return_value=evidence) as mock_search,
    ):
        state = triage_graph.invoke(
            {"report": report, "repository_path": "."}
        )

    mock_search.assert_called_once()
    assert state["investigation_status"] == "completed_with_matches"
    assert state["evidence"] == evidence
