import json
from pathlib import Path

import pytest

from app.models import TriageResult
from app.triage import TriageOutputError, parse_triage_response
from evals.scoring import (
    CATEGORIES,
    EvalCase,
    format_report,
    load_dataset,
    score_case,
    status_label,
    summarize,
)

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset.jsonl"


def make_case(**overrides) -> EvalCase:
    data = {
        "id": "case-1",
        "title": "Crash on startup",
        "description": "The app exits immediately with a traceback.",
        "expected_category": "bug",
        "acceptable_priorities": ["high", "critical"],
        "expected_investigation": True,
        "tags": ["regression"],
    }
    data.update(overrides)
    return EvalCase.model_validate(data)


def make_result(**overrides) -> TriageResult:
    data = {
        "category": "bug",
        "priority": "high",
        "summary": "Startup crash",
        "rationale": "Blocks all users.",
        "needs_repository_investigation": True,
        "confidence": 0.9,
        "proposed_labels": ["bug"],
        "draft_response": "Thanks for the report.",
    }
    data.update(overrides)
    return TriageResult.model_validate(data)


def test_bundled_dataset_is_valid_and_covers_every_category() -> None:
    cases = load_dataset(DATASET)

    assert len(cases) >= 30
    assert {c.expected_category for c in cases} == set(CATEGORIES)


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = make_case().model_dump()
    path = tmp_path / "dup.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case id"):
        load_dataset(path)


def test_load_dataset_rejects_unknown_category(tmp_path: Path) -> None:
    row = make_case().model_dump()
    row["expected_category"] = "enhancement"
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_dataset(path)


def test_score_case_correct_result() -> None:
    scored = score_case(make_case(), make_result(), latency_seconds=1.5)

    assert scored.valid_output
    assert scored.category_correct
    assert scored.priority_acceptable
    assert scored.investigation_correct is True


def test_score_case_wrong_category_and_priority() -> None:
    scored = score_case(make_case(), make_result(category="question", priority="low"))

    assert not scored.category_correct
    assert not scored.priority_acceptable


def test_score_case_skips_investigation_when_unlabelled() -> None:
    scored = score_case(make_case(expected_investigation=None), make_result())

    assert scored.investigation_correct is None


def test_score_case_invalid_output_keeps_raw_response() -> None:
    scored = score_case(make_case(), None, error="bad json", raw_response="{oops")

    assert not scored.valid_output
    assert not scored.category_correct
    assert scored.raw_response == "{oops"


def test_summarize_counts_invalid_outputs_as_wrong() -> None:
    results = [
        score_case(make_case(id="a"), make_result()),
        score_case(make_case(id="b"), None, error="bad json"),
    ]

    summary = summarize(results)

    assert summary["valid_output_rate"] == 0.5
    assert summary["category_accuracy"] == 0.5
    assert summary["confusion_matrix"]["bug"] == {"bug": 1, "<invalid>": 1}


def test_summarize_calibration_buckets() -> None:
    results = [
        score_case(make_case(id="hi-right"), make_result(confidence=0.95)),
        score_case(
            make_case(id="hi-wrong"), make_result(category="other", confidence=0.92)
        ),
        score_case(make_case(id="lo-right"), make_result(confidence=0.3)),
    ]

    calibration = {row["bucket"]: row for row in summarize(results)["calibration"]}

    assert calibration["0.9-1.0"]["n"] == 2
    assert calibration["0.9-1.0"]["category_accuracy"] == 0.5
    assert calibration["0.0-0.5"]["category_accuracy"] == 1.0
    assert calibration["0.5-0.7"]["n"] == 0


def test_summarize_per_tag_and_report_lists_failures() -> None:
    results = [
        score_case(make_case(id="ok", tags=["x"]), make_result()),
        score_case(make_case(id="bad", tags=["x"]), make_result(category="other")),
    ]
    summary = summarize(results)

    assert summary["per_tag"]["x"]["category_accuracy"] == 0.5
    report = format_report(summary, results)
    assert "bad: category other != bug" in report
    assert "Failures (1)" in report


def test_parse_triage_response_keeps_raw_output_on_failure() -> None:
    with pytest.raises(TriageOutputError) as excinfo:
        parse_triage_response('{"category": "bug"}')

    assert excinfo.value.raw_response == '{"category": "bug"}'
    assert isinstance(excinfo.value, ValueError)


def test_parse_triage_response_accepts_valid_json() -> None:
    raw = make_result().model_dump_json()

    assert parse_triage_response(raw).category == "bug"


def test_report_lists_investigation_mismatches() -> None:
    results = [
        score_case(
            make_case(id="over", expected_investigation=False),
            make_result(needs_repository_investigation=True),
        )
    ]

    report = format_report(summarize(results), results)

    assert "over: investigation True != False" in report


def test_status_label_names_every_wrong_field() -> None:
    case = make_case(expected_investigation=False)

    assert status_label(score_case(make_case(), make_result())) == "ok"
    assert status_label(score_case(make_case(), None, error="x")) == "INVALID"
    assert (
        status_label(
            score_case(case, make_result(category="other", priority="low"))
        )
        == "WRONG(cat,pri,inv)"
    )


def test_score_case_keeps_rationale() -> None:
    scored = score_case(make_case(), make_result(rationale="Blocks startup."))

    assert scored.rationale == "Blocks startup."
