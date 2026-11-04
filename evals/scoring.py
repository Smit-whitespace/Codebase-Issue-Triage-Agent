"""Pure scoring and reporting logic for triage evals.

Nothing in this module calls the model, so it can be unit-tested offline.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Literal

from pydantic import BaseModel, Field

from app.models import TriageResult

Category = Literal["bug", "feature_request", "question", "documentation", "other"]
Priority = Literal["low", "medium", "high", "critical"]

CATEGORIES: tuple[str, ...] = (
    "bug",
    "feature_request",
    "question",
    "documentation",
    "other",
)

# Confidence buckets for the calibration table: [low, high).
CALIBRATION_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.5),
    (0.5, 0.7),
    (0.7, 0.9),
    (0.9, 1.01),
)

INVALID = "<invalid>"


class EvalCase(BaseModel):
    """One hand-labelled issue with its expected triage outcome."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str
    expected_category: Category
    # Priority is subjective, so a case may accept more than one value.
    acceptable_priorities: list[Priority] = Field(min_length=1)
    # None means "either answer is fine"; it is excluded from that metric.
    expected_investigation: bool | None = None
    tags: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    """Outcome of running the agent on one EvalCase."""

    id: str
    tags: list[str]
    expected_category: str
    acceptable_priorities: list[str]
    expected_investigation: bool | None
    valid_output: bool
    predicted_category: str | None = None
    predicted_priority: str | None = None
    predicted_investigation: bool | None = None
    confidence: float | None = None
    # Kept so a wrong answer can be traced back to the model's reasoning.
    rationale: str | None = None
    category_correct: bool = False
    priority_acceptable: bool = False
    investigation_correct: bool | None = None
    latency_seconds: float | None = None
    error: str | None = None
    raw_response: str | None = None


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Load and validate a JSONL eval dataset; ids must be unique."""
    cases: list[EvalCase] = []
    seen: set[str] = set()

    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = EvalCase.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid eval case: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)

    return cases


def score_case(
    case: EvalCase,
    result: TriageResult | None,
    *,
    latency_seconds: float | None = None,
    error: str | None = None,
    raw_response: str | None = None,
) -> CaseResult:
    """Compare one agent result against the expected labels."""
    base = {
        "id": case.id,
        "tags": case.tags,
        "expected_category": case.expected_category,
        "acceptable_priorities": list(case.acceptable_priorities),
        "expected_investigation": case.expected_investigation,
        "latency_seconds": latency_seconds,
    }

    if result is None:
        return CaseResult(
            **base,
            valid_output=False,
            error=error or "no result",
            raw_response=raw_response,
        )

    investigation_correct = (
        None
        if case.expected_investigation is None
        else result.needs_repository_investigation == case.expected_investigation
    )

    return CaseResult(
        **base,
        valid_output=True,
        predicted_category=result.category,
        predicted_priority=result.priority,
        predicted_investigation=result.needs_repository_investigation,
        confidence=result.confidence,
        rationale=result.rationale,
        category_correct=result.category == case.expected_category,
        priority_acceptable=result.priority in case.acceptable_priorities,
        investigation_correct=investigation_correct,
    )


def status_label(result: CaseResult) -> str:
    """Short per-case status: 'ok', 'INVALID', or 'WRONG(<fields>)'."""
    if not result.valid_output:
        return "INVALID"
    wrong = []
    if not result.category_correct:
        wrong.append("cat")
    if not result.priority_acceptable:
        wrong.append("pri")
    if result.investigation_correct is False:
        wrong.append("inv")
    return f"WRONG({','.join(wrong)})" if wrong else "ok"


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(results: list[CaseResult]) -> dict:
    """Aggregate per-case results into headline and sliced metrics.

    Invalid outputs count as wrong in every accuracy metric, so a model that
    fails to produce JSON cannot look better by skipping hard cases.
    """
    total = len(results)
    valid = [r for r in results if r.valid_output]
    investigated = [r for r in results if r.expected_investigation is not None]

    # Per expected category: how often the agent got that category right.
    per_category: dict[str, dict] = {}
    for category in CATEGORIES:
        subset = [r for r in results if r.expected_category == category]
        if subset:
            correct = sum(r.category_correct for r in subset)
            per_category[category] = {
                "n": len(subset),
                "accuracy": _rate(correct, len(subset)),
            }

    # Confusion matrix: expected -> predicted -> count.
    confusion: dict[str, dict[str, int]] = defaultdict(Counter)
    for r in results:
        confusion[r.expected_category][r.predicted_category or INVALID] += 1

    # Per tag, so edge-case groups (injection, disguised types) are visible.
    per_tag: dict[str, dict] = {}
    tag_groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        for tag in r.tags:
            tag_groups[tag].append(r)
    for tag, subset in sorted(tag_groups.items()):
        per_tag[tag] = {
            "n": len(subset),
            "category_accuracy": _rate(sum(r.category_correct for r in subset), len(subset)),
            "priority_acceptable": _rate(
                sum(r.priority_acceptable for r in subset), len(subset)
            ),
        }

    # Calibration: does higher stated confidence mean higher accuracy?
    calibration = []
    for low, high in CALIBRATION_BUCKETS:
        bucket = [
            r for r in valid if r.confidence is not None and low <= r.confidence < high
        ]
        calibration.append(
            {
                "bucket": f"{low:.1f}-{min(high, 1.0):.1f}",
                "n": len(bucket),
                "mean_confidence": (
                    round(sum(r.confidence for r in bucket) / len(bucket), 3)
                    if bucket
                    else None
                ),
                "category_accuracy": _rate(
                    sum(r.category_correct for r in bucket), len(bucket)
                ),
            }
        )

    latencies = [r.latency_seconds for r in results if r.latency_seconds is not None]

    return {
        "n_cases": total,
        "valid_output_rate": _rate(len(valid), total),
        "category_accuracy": _rate(sum(r.category_correct for r in results), total),
        "priority_acceptable_rate": _rate(
            sum(r.priority_acceptable for r in results), total
        ),
        "investigation_accuracy": _rate(
            sum(bool(r.investigation_correct) for r in investigated), len(investigated)
        ),
        "per_category": per_category,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "per_tag": per_tag,
        "calibration": calibration,
        "latency_seconds": {
            "median": round(median(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
    }


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _format_latency(latency: dict) -> str:
    if latency["median"] is None:
        return "Latency median / max:   n/a"
    return f"Latency median / max:   {latency['median']}s / {latency['max']}s"


def format_report(summary: dict, results: list[CaseResult]) -> str:
    """Render a human-readable text report."""
    lines = [
        "=== Triage eval report ===",
        f"Cases:                  {summary['n_cases']}",
        f"Valid output:           {_pct(summary['valid_output_rate'])}",
        f"Category accuracy:      {_pct(summary['category_accuracy'])}",
        f"Priority acceptable:    {_pct(summary['priority_acceptable_rate'])}",
        f"Investigation accuracy: {_pct(summary['investigation_accuracy'])}",
        _format_latency(summary["latency_seconds"]),
        "",
        "--- Category accuracy by expected category ---",
    ]
    for category, stats in summary["per_category"].items():
        lines.append(f"  {category:<16} n={stats['n']:<3} {_pct(stats['accuracy'])}")

    predicted_columns = [*CATEGORIES, INVALID]
    lines += ["", "--- Confusion matrix (rows=expected, cols=predicted) ---"]
    header = "  " + " " * 16 + "".join(f"{c[:8]:>9}" for c in predicted_columns)
    lines.append(header)
    for expected in CATEGORIES:
        row = summary["confusion_matrix"].get(expected)
        if not row:
            continue
        cells = "".join(f"{row.get(c, 0):>9}" for c in predicted_columns)
        lines.append(f"  {expected:<16}{cells}")

    lines += ["", "--- By tag ---"]
    for tag, stats in summary["per_tag"].items():
        lines.append(
            f"  {tag:<28} n={stats['n']:<3} category {_pct(stats['category_accuracy'])}"
            f"   priority {_pct(stats['priority_acceptable'])}"
        )

    lines += [
        "",
        "--- Calibration (is stated confidence trustworthy?) ---",
        "  bucket     n    mean_conf   category_acc",
    ]
    for row in summary["calibration"]:
        mean_conf = "  n/a" if row["mean_confidence"] is None else f"{row['mean_confidence']:.3f}"
        lines.append(
            f"  {row['bucket']:<9} {row['n']:<4} {mean_conf:>9}   {_pct(row['category_accuracy'])}"
        )

    failures = [
        r
        for r in results
        if not r.valid_output
        or not r.category_correct
        or not r.priority_acceptable
        or r.investigation_correct is False
    ]
    lines += ["", f"--- Failures ({len(failures)}) ---"]
    for r in failures:
        if not r.valid_output:
            lines.append(f"  {r.id}: INVALID OUTPUT ({r.error})")
            continue
        problems = []
        if not r.category_correct:
            problems.append(f"category {r.predicted_category} != {r.expected_category}")
        if not r.priority_acceptable:
            problems.append(
                f"priority {r.predicted_priority} not in {r.acceptable_priorities}"
            )
        if r.investigation_correct is False:
            problems.append(
                f"investigation {r.predicted_investigation}"
                f" != {r.expected_investigation}"
            )
        lines.append(f"  {r.id}: " + "; ".join(problems) + f"  (conf {r.confidence})")

    return "\n".join(lines)
