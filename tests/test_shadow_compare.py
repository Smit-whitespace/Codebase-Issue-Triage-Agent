import pytest

from evals.shadow_compare import (
    RULES,
    expected_category,
    score_records,
    split_held_out,
    wilson_interval,
)

PIP = RULES["pypa/pip"]
POETRY = RULES["python-poetry/poetry"]


def make_record(number: int, category: str | None = "bug", **overrides) -> dict:
    record = {
        "repo": "pypa/pip",
        "number": number,
        "url": f"https://github.com/pypa/pip/issues/{number}",
        "title": f"Issue {number}",
        "category": category,
        "error": None,
    }
    record.update(overrides)
    return record


def test_label_on_a_triaged_issue_is_the_expected_category() -> None:
    assert expected_category(["type: bug"], PIP) == ("bug", None)


def test_issue_still_waiting_for_triage_is_excluded() -> None:
    labels = ["type: bug", "S: needs triage"]

    assert expected_category(labels, PIP) == (None, "awaiting_triage")


def test_issue_without_a_type_label_is_excluded() -> None:
    assert expected_category(["C: cache"], PIP) == (None, "no_type_label")


def test_conflicting_type_labels_are_excluded() -> None:
    labels = ["type: bug", "type: support"]

    assert expected_category(labels, PIP) == (None, "conflicting_labels")


def test_label_saying_not_a_bug_overrules_a_leftover_bug_label() -> None:
    labels = ["type: bug", "resolution: not a bug"]

    assert expected_category(labels, PIP) == (None, "overruled_bug_label")


def test_wrong_project_support_issue_stays_scored() -> None:
    labels = ["type: support", "resolution: wrong project"]

    assert expected_category(labels, PIP) == ("question", None)


def test_poetry_rules_use_its_own_labels() -> None:
    assert expected_category(["kind/feature"], POETRY) == ("feature_request", None)
    assert expected_category(["kind/bug", "status/triage"], POETRY) == (
        None,
        "awaiting_triage",
    )
    assert expected_category(["kind/bug", "status/external-issue"], POETRY) == (
        None,
        "overruled_bug_label",
    )


def test_score_counts_matches_and_lists_misses() -> None:
    records = [make_record(1, "bug"), make_record(2, "bug")]
    labels = {1: ["type: bug"], 2: ["type: support"]}

    summary = score_records(records, labels, PIP)

    assert summary["scored"] == 2
    assert summary["correct"] == 1
    assert [miss["number"] for miss in summary["misses"]] == [2]
    assert summary["misses"][0]["expected"] == "question"
    assert summary["misses"][0]["predicted"] == "bug"


def test_score_counts_failed_predictions_as_wrong() -> None:
    records = [make_record(1, None, error="bad json")]

    summary = score_records(records, {1: ["type: bug"]}, PIP)

    assert summary["scored"] == 1
    assert summary["correct"] == 0


def test_score_reports_why_issues_were_excluded() -> None:
    records = [make_record(1), make_record(2), make_record(3)]
    labels = {1: ["S: needs triage", "type: bug"], 2: []}

    summary = score_records(records, labels, PIP)

    assert summary["scored"] == 0
    assert summary["excluded"] == {
        "awaiting_triage": 1,
        "no_type_label": 1,
        "issue_not_found": 1,
    }


def test_score_reports_the_always_guess_most_common_baseline() -> None:
    records = [make_record(1), make_record(2), make_record(3), make_record(4)]
    labels = {
        1: ["type: bug"],
        2: ["type: bug"],
        3: ["type: bug"],
        4: ["type: support"],
    }

    summary = score_records(records, labels, PIP)

    assert summary["expected_counts"] == {"bug": 3, "question": 1}


def test_wilson_interval_matches_a_known_value() -> None:
    low, high = wilson_interval(43, 44)

    assert low == pytest.approx(0.882, abs=0.001)
    assert high == pytest.approx(0.996, abs=0.001)


def test_issues_from_the_held_out_date_onward_are_split_off() -> None:
    records = [
        make_record(1, created_at="2026-11-04T23:59:59Z"),
        make_record(2, created_at="2026-11-05T00:00:00Z"),
        make_record(3, created_at="2026-11-10T08:00:00Z"),
    ]

    tuning, held_out = split_held_out(records)

    assert [r["number"] for r in tuning] == [1]
    assert [r["number"] for r in held_out] == [2, 3]


def test_wilson_interval_with_no_data_is_the_full_range() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
