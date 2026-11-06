from pathlib import Path

import pytest

from app.models import IssueReport, TriageResult
from app.triage import TriageOutputError
from evals.prompt_compare import (
    answers_for,
    collect,
    count_differences,
    freeze_issues,
    load_or_freeze,
    load_predictions,
    mcnemar_p,
    pair_results,
    prompt_from_source,
    without_support,
)
from evals.shadow import prompt_version
from evals.shadow_compare import RULES

PIP = RULES["pypa/pip"]
OLD = "Old prompt {schema} {title} {description}"
NEW = "New prompt {schema} {title} {description}"
PROMPTS = {"baseline": OLD, "current": NEW}
METADATA = {"provider": "gemini", "model": "test-model"}


def make_record(number: int, **overrides) -> dict:
    record = {
        "repo": "pypa/pip",
        "number": number,
        "url": f"https://github.com/pypa/pip/issues/{number}",
        "title": f"Issue {number}",
        "body": "Something is wrong.",
        "created_at": "2026-10-01T10:00:00Z",
        "category": "bug",
    }
    record.update(overrides)
    return record


def make_issue(number: int, expected: str = "bug", repo: str = "pypa/pip") -> dict:
    return {
        "repo": repo,
        "number": number,
        "url": f"https://github.com/{repo}/issues/{number}",
        "title": f"Issue {number}",
        "body": "Something is wrong.",
        "expected": expected,
        "saved_category": "bug",
    }


def make_result(category: str = "bug") -> TriageResult:
    return TriageResult(
        category=category,
        priority="medium",
        summary="A summary",
        rationale="A rationale",
        needs_repository_investigation=False,
        confidence=0.9,
        proposed_labels=[],
        draft_response="Thanks.",
    )


def run_collect(issues: list[dict], path: Path, triage, runs: int = 1) -> int:
    return collect(
        issues,
        PROMPTS,
        runs,
        path,
        triage=triage,
        delay=0,
        metadata=METADATA,
    )


def test_prompt_is_read_from_source_without_running_it() -> None:
    source = (
        'raise RuntimeError("this file must not be executed")\n'
        'TRIAGE_PROMPT = """Classify this.\n{title}\n"""\n'
    )

    assert prompt_from_source(source) == "Classify this.\n{title}\n"


def test_source_without_a_prompt_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="TRIAGE_PROMPT"):
        prompt_from_source("OTHER = 'text'\n")


def test_reviewed_issues_are_frozen_with_their_expected_category() -> None:
    records = [make_record(1, category="feature_request")]

    issues = freeze_issues(records, {1: ["type: bug"]}, PIP)

    assert issues == [
        {
            "repo": "pypa/pip",
            "number": 1,
            "url": "https://github.com/pypa/pip/issues/1",
            "title": "Issue 1",
            "body": "Something is wrong.",
            "expected": "bug",
            "saved_category": "feature_request",
        }
    ]


def test_issues_without_a_reviewed_category_are_left_out() -> None:
    records = [make_record(1), make_record(2), make_record(3)]
    labels = {1: ["type: bug", "S: needs triage"], 2: ["C: cache"]}

    assert freeze_issues(records, labels, PIP) == []


@pytest.mark.parametrize(
    ("created_at", "kept"),
    [
        ("2026-11-04T23:59:59Z", True),
        ("2026-11-05T00:00:00Z", False),
    ],
)
def test_held_out_issues_cannot_enter_the_set(created_at: str, kept: bool) -> None:
    records = [make_record(1, created_at=created_at)]

    issues = freeze_issues(records, {1: ["type: bug"]}, PIP)

    assert bool(issues) is kept


def test_saved_issue_set_is_reused_instead_of_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "issues.json"
    first = load_or_freeze(path, lambda: [make_issue(1)])

    def fail() -> list[dict]:
        raise AssertionError("the saved set should be reused")

    assert load_or_freeze(path, fail) == first == [make_issue(1)]


def test_each_issue_is_sent_to_each_prompt_for_each_run(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    calls = []

    def triage(report: IssueReport, prompt: str) -> TriageResult:
        calls.append((report.title, prompt))
        return make_result()

    made = run_collect([make_issue(1), make_issue(2)], path, triage, runs=2)

    assert made == 8
    assert calls[:4] == [
        ("Issue 1", OLD),
        ("Issue 1", NEW),
        ("Issue 2", OLD),
        ("Issue 2", NEW),
    ]


def test_records_store_the_hash_of_the_prompt_actually_used(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"

    def triage(report: IssueReport, prompt: str) -> TriageResult:
        return make_result("bug" if prompt == OLD else "question")

    run_collect([make_issue(1)], path, triage)

    predictions = load_predictions(path)
    key = ("pypa/pip", 1)
    assert answers_for(predictions, prompt_version(OLD), 1) == {key: "bug"}
    assert answers_for(predictions, prompt_version(NEW), 1) == {key: "question"}


def test_rerun_only_makes_the_missing_calls(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    calls = []

    def triage(report: IssueReport, prompt: str) -> TriageResult:
        calls.append(report.title)
        return make_result()

    run_collect([make_issue(1)], path, triage)
    calls.clear()

    made = run_collect([make_issue(1), make_issue(2)], path, triage)

    assert made == 2
    assert calls == ["Issue 2", "Issue 2"]


def test_complete_rerun_makes_no_calls(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"

    run_collect([make_issue(1)], path, lambda report, prompt: make_result())

    def fail(report: IssueReport, prompt: str) -> TriageResult:
        raise AssertionError("nothing should be predicted again")

    assert run_collect([make_issue(1)], path, fail) == 0


def test_invalid_model_output_is_saved_as_a_failed_answer(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"

    def triage(report: IssueReport, prompt: str) -> TriageResult:
        raise TriageOutputError("bad JSON", raw_response="not json")

    run_collect([make_issue(1)], path, triage)

    record = load_predictions(path)[0]
    assert record["category"] is None
    assert "TriageOutputError" in record["error"]


def test_provider_failure_stops_the_run_and_keeps_saved_results(
    tmp_path: Path,
) -> None:
    class QuotaExceeded(Exception):
        pass

    path = tmp_path / "predictions.jsonl"

    def triage(report: IssueReport, prompt: str) -> TriageResult:
        if report.title == "Issue 2":
            raise QuotaExceeded
        return make_result()

    with pytest.raises(QuotaExceeded):
        run_collect([make_issue(1), make_issue(2)], path, triage)

    saved = load_predictions(path)
    assert [record["number"] for record in saved] == [1, 1]


@pytest.mark.parametrize(
    ("first_only", "second_only", "expected"),
    [
        (0, 0, 1.0),
        (5, 0, 0.0625),
        (6, 0, 0.03125),
        (0, 6, 0.03125),
        (3, 3, 1.0),
        (1, 7, 0.0703125),
    ],
)
def test_exact_mcnemar_p_value(
    first_only: int, second_only: int, expected: float
) -> None:
    assert mcnemar_p(first_only, second_only) == pytest.approx(expected)


def test_pairing_counts_issues_only_one_prompt_got_right() -> None:
    issues = [
        make_issue(1, "bug"),
        make_issue(2, "bug"),
        make_issue(3, "bug"),
        make_issue(4, "bug"),
    ]
    first = {
        ("pypa/pip", 1): "bug",
        ("pypa/pip", 2): "bug",
        ("pypa/pip", 3): "other",
        ("pypa/pip", 4): "other",
    }
    second = {
        ("pypa/pip", 1): "bug",
        ("pypa/pip", 2): "other",
        ("pypa/pip", 3): "other",
        ("pypa/pip", 4): "bug",
    }

    assert pair_results(issues, first, second) == {
        "first_correct": 2,
        "second_correct": 2,
        "first_only": 1,
        "second_only": 1,
    }


def test_failed_output_counts_as_wrong() -> None:
    issues = [make_issue(1, "bug")]

    result = pair_results(issues, {("pypa/pip", 1): None}, {("pypa/pip", 1): "bug"})

    assert result["first_correct"] == 0
    assert result["second_only"] == 1


def test_a_missing_answer_is_an_error_not_a_wrong_answer() -> None:
    with pytest.raises(KeyError):
        pair_results([make_issue(1)], {}, {("pypa/pip", 1): "bug"})


def test_noise_counts_different_answers_even_when_both_are_wrong() -> None:
    issues = [make_issue(1, "question"), make_issue(2, "question")]
    first = {("pypa/pip", 1): "bug", ("pypa/pip", 2): "bug"}
    second = {("pypa/pip", 1): "other", ("pypa/pip", 2): "bug"}

    assert count_differences(issues, first, second) == 1


def test_only_pip_support_issues_are_removed() -> None:
    support = make_issue(1, "question")
    bug = make_issue(2, "bug")
    poetry_question = make_issue(3, "question", repo="python-poetry/poetry")

    assert without_support([support, bug, poetry_question]) == [bug, poetry_question]
