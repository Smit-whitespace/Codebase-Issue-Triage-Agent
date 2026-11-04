"""Compare shadow predictions with maintainers' triage decisions.

Issue templates apply type labels when a reporter opens an issue, so a type
label alone is only the reporter's claim. pip and poetry also apply a "needs
triage" label, which a maintainer removes after review. An issue counts as
reviewed only when that label is gone, so only those repos can be scored.

Only category can be compared. Maintainers don't record our priority levels or
whether code investigation was needed.

Usage (run evals.shadow first):

    python -m evals.shadow_compare
"""

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from evals.shadow import DEFAULT_OUTPUT_DIR, fetch_new_issues, output_path


@dataclass(frozen=True)
class RepoRules:
    categories: dict[str, str]
    pending: str
    # A label here means a leftover bug label from the template isn't reliable.
    not_bug: frozenset[str]


RULES = {
    "pypa/pip": RepoRules(
        categories={
            "type: bug": "bug",
            "type: feature request": "feature_request",
            "type: support": "question",
            "type: docs": "documentation",
        },
        pending="S: needs triage",
        not_bug=frozenset({"resolution: not a bug", "resolution: wrong project"}),
    ),
    "python-poetry/poetry": RepoRules(
        categories={
            "kind/bug": "bug",
            "kind/feature": "feature_request",
            "kind/question": "question",
            "area/docs": "documentation",
        },
        pending="status/triage",
        not_bug=frozenset({"status/external-issue"}),
    ),
}


def expected_category(
    labels: list[str], rules: RepoRules
) -> tuple[str | None, str | None]:
    """Return the reviewed category, or None and the reason it can't be used."""
    if rules.pending in labels:
        return None, "awaiting_triage"
    categories = {rules.categories[x] for x in labels if x in rules.categories}
    if not categories:
        return None, "no_type_label"
    if len(categories) > 1:
        return None, "conflicting_labels"
    category = categories.pop()
    if category == "bug" and rules.not_bug.intersection(labels):
        return None, "overruled_bug_label"
    return category, None


def score_records(
    records: list[dict],
    labels_by_number: dict[int, list[str]],
    rules: RepoRules,
) -> dict:
    """Score predictions against reviewed categories. Failed output is wrong."""
    excluded: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    correct = 0
    misses = []
    for record in records:
        labels = labels_by_number.get(record["number"])
        if labels is None:
            excluded["issue_not_found"] += 1
            continue
        expected, reason = expected_category(labels, rules)
        if expected is None:
            excluded[reason] += 1
            continue
        expected_counts[expected] += 1
        if record["category"] == expected:
            correct += 1
        else:
            misses.append(
                {
                    "number": record["number"],
                    "url": record["url"],
                    "title": record["title"],
                    "expected": expected,
                    "predicted": record["category"],
                }
            )
    return {
        "scored": sum(expected_counts.values()),
        "correct": correct,
        "excluded": dict(excluded),
        "expected_counts": dict(expected_counts),
        "misses": misses,
    }


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion, reliable on small samples."""
    if total == 0:
        return 0.0, 1.0
    rate = correct / total
    scale = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / scale
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2))
        / scale
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def load_records(path: Path) -> list[dict]:
    lines = path.read_text("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def current_labels(records: list[dict], repo: str) -> dict[int, list[str]]:
    """Read the labels each predicted issue has now."""
    since = min(datetime.fromisoformat(r["created_at"]) for r in records)
    return {
        issue["number"]: [label["name"] for label in issue["labels"]]
        for issue in fetch_new_issues(repo, since)
    }


def print_summary(name: str, summary: dict) -> None:
    scored, correct = summary["scored"], summary["correct"]
    print(f"\n{name}: {correct}/{scored} correct")
    if scored:
        low, high = wilson_interval(correct, scored)
        common = max(summary["expected_counts"].values())
        print(f"  accuracy {correct / scored:.1%} (95% CI {low:.1%}-{high:.1%})")
        print(f"  always guessing the most common category: {common / scored:.1%}")
    print(f"  expected: {summary['expected_counts']}")
    print(f"  excluded: {summary['excluded']}")
    for miss in summary["misses"]:
        print(
            f"  MISS expected {miss['expected']}, got {miss['predicted']}: "
            f"{miss['url']}"
        )


def main() -> None:
    combined: Counter[str] = Counter()
    combined_expected: Counter[str] = Counter()
    combined_excluded: Counter[str] = Counter()
    for repo, rules in RULES.items():
        path = output_path(DEFAULT_OUTPUT_DIR, repo)
        if not path.exists():
            print(f"{repo}: no predictions yet")
            continue
        records = load_records(path)
        summary = score_records(records, current_labels(records, repo), rules)
        print_summary(repo, summary)
        combined["scored"] += summary["scored"]
        combined["correct"] += summary["correct"]
        combined_expected.update(summary["expected_counts"])
        combined_excluded.update(summary["excluded"])
    print_summary(
        "All scored repos",
        {
            "scored": combined["scored"],
            "correct": combined["correct"],
            "expected_counts": dict(combined_expected),
            "excluded": dict(combined_excluded),
            "misses": [],
        },
    )


if __name__ == "__main__":
    main()
