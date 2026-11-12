"""Compare an earlier triage prompt with the current one on real reviewed issues.

The issue set is built once from shadow predictions and maintainers' labels, then
saved, so later label changes can't alter it. Both prompts see the same stored
title and body, and each prompt runs more than once because the model's answers
vary between identical runs. Issues opened after 2026-11-04 are held out for
final measurement, so they never enter the set.

Results are saved after every call. If a run stops, for example at a quota
limit, rerun the same command to continue. A finished run makes no model calls.

Usage (run evals.shadow first; configure the model as described in the README):

    python -m evals.prompt_compare --baseline aba4c3d --runs 2
"""

import argparse
import ast
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from app.llm import LLMConfigError, load_config
from app.models import IssueReport, TriageResult
from app.triage import TRIAGE_PROMPT, triage_issue
from evals.run_evals import preflight
from evals.shadow import (
    DEFAULT_OUTPUT_DIR,
    HELD_OUT_FROM,
    output_path,
    prompt_version,
)
from evals.shadow_compare import (
    RULES,
    RepoRules,
    current_labels,
    expected_category,
    load_records,
    wilson_interval,
)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results" / "prompt_compare"

IssueKey = tuple[str, int]
Answers = dict[IssueKey, str | None]


def prompt_from_source(source: str) -> str:
    """Read TRIAGE_PROMPT from Python source without executing it."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TRIAGE_PROMPT"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("No TRIAGE_PROMPT assignment found in the source.")


def prompt_at_ref(ref: str) -> str:
    """Read the triage prompt as it was at a git commit."""
    source = subprocess.run(
        ["git", "show", f"{ref}:app/triage.py"],
        capture_output=True,
        check=True,
        encoding="utf-8",
    ).stdout
    return prompt_from_source(source)


def freeze_issues(
    records: list[dict],
    labels_by_number: dict[int, list[str]],
    rules: RepoRules,
) -> list[dict]:
    """Keep reviewed issues from before the held-out date."""
    issues = []
    for record in records:
        if datetime.fromisoformat(record["created_at"]) >= HELD_OUT_FROM:
            continue
        labels = labels_by_number.get(record["number"])
        if labels is None:
            continue
        expected, _ = expected_category(labels, rules)
        if expected is None:
            continue
        issues.append(
            {
                "repo": record["repo"],
                "number": record["number"],
                "url": record["url"],
                "title": record["title"],
                "body": record["body"],
                "expected": expected,
                "saved_category": record["category"],
            }
        )
    return issues


def load_or_freeze(path: Path, build: Callable[[], list[dict]]) -> list[dict]:
    """Return the saved issue set, building and saving it on first use."""
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    issues = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(issues, ensure_ascii=False, indent=2), "utf-8")
    return issues


def load_predictions(path: Path) -> list[dict]:
    return load_records(path) if path.exists() else []


def predict(
    issue: dict,
    name: str,
    prompt: str,
    run: int,
    triage: Callable[[IssueReport, str], TriageResult],
    metadata: dict[str, str],
) -> dict:
    """Triage one issue. Bad model output is recorded; provider errors propagate."""
    result = None
    error = None
    try:
        report = IssueReport(title=issue["title"], description=issue["body"])
        result = triage(report, prompt)
    except (TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "repo": issue["repo"],
        "number": issue["number"],
        "prompt": name,
        "prompt_version": prompt_version(prompt),
        "run": run,
        "predicted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **metadata,
        "category": result.category if result else None,
        "confidence": result.confidence if result else None,
        "error": error,
    }


def collect(
    issues: list[dict],
    prompts: dict[str, str],
    runs: int,
    path: Path,
    *,
    triage: Callable[[IssueReport, str], TriageResult],
    delay: float,
    metadata: dict[str, str],
) -> int:
    """Make the missing predictions, alternating prompts issue by issue."""
    saved = {
        (record["repo"], record["number"], record["prompt_version"], record["run"])
        for record in load_predictions(path)
    }
    pending = [
        (run, issue, name, prompt)
        for run in range(1, runs + 1)
        for issue in issues
        for name, prompt in prompts.items()
        if (issue["repo"], issue["number"], prompt_version(prompt), run) not in saved
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    for index, (run, issue, name, prompt) in enumerate(pending, 1):
        if index > 1 and delay:
            time.sleep(delay)
        record = predict(issue, name, prompt, run, triage, metadata)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{index}/{len(pending)}", flush=True)
    return len(pending)


def answers_for(predictions: list[dict], version: str, run: int) -> Answers:
    return {
        (record["repo"], record["number"]): record["category"]
        for record in predictions
        if record["prompt_version"] == version and record["run"] == run
    }


def issue_key(issue: dict) -> IssueKey:
    return issue["repo"], issue["number"]


def mcnemar_p(first_only: int, second_only: int) -> float:
    """Two-sided exact McNemar test on the issues only one prompt got right."""
    total = first_only + second_only
    tail = sum(math.comb(total, i) for i in range(min(first_only, second_only) + 1))
    return min(1.0, 2 * tail / 2**total)


def pair_results(issues: list[dict], first: Answers, second: Answers) -> dict[str, int]:
    """Count correct answers per prompt. A failed answer (None) is wrong."""
    counts = dict.fromkeys(
        ("first_correct", "second_correct", "first_only", "second_only"), 0
    )
    for issue in issues:
        first_right = first[issue_key(issue)] == issue["expected"]
        second_right = second[issue_key(issue)] == issue["expected"]
        counts["first_correct"] += first_right
        counts["second_correct"] += second_right
        counts["first_only"] += first_right and not second_right
        counts["second_only"] += second_right and not first_right
    return counts


def count_differences(issues: list[dict], first: Answers, second: Answers) -> int:
    """Count issues answered differently, whether or not either is correct."""
    return sum(first[issue_key(i)] != second[issue_key(i)] for i in issues)


def is_pip_support(issue: dict) -> bool:
    return issue["repo"] == "pypa/pip" and issue["expected"] == "question"


def without_support(issues: list[dict]) -> list[dict]:
    """Drop pip's type: support issues, which mix questions with real reports."""
    return [issue for issue in issues if not is_pip_support(issue)]


def format_score(correct: int, total: int) -> str:
    low, high = wilson_interval(correct, total)
    return f"{correct}/{total} = {correct / total:.1%} (95% CI {low:.1%}-{high:.1%})"


def print_accuracy(
    title: str,
    issues: list[dict],
    baseline: dict[int, Answers],
    current: dict[int, Answers],
) -> None:
    print(f"\n{title} ({len(issues)} issues)")
    for run, answers in baseline.items():
        result = pair_results(issues, answers, current[run])
        p_value = mcnemar_p(result["first_only"], result["second_only"])
        print(f"  run {run}")
        print(f"    baseline {format_score(result['first_correct'], len(issues))}")
        print(f"    current  {format_score(result['second_correct'], len(issues))}")
        print(
            f"    only baseline right: {result['first_only']}, "
            f"only current right: {result['second_only']}, "
            f"exact McNemar p = {p_value:.3f}"
        )


def print_noise(
    issues: list[dict],
    baseline: dict[int, Answers],
    current: dict[int, Answers],
    saved: Answers,
) -> None:
    print("\nNoise (issues answered differently)")
    first, *later = sorted(baseline)
    for run in later:
        for name, answers in (("baseline", baseline), ("current", current)):
            changed = count_differences(issues, answers[first], answers[run])
            print(f"  {name} run {first} vs run {run}: {changed}")
    for run, answers in current.items():
        changed = count_differences(issues, answers, saved)
        print(f"  current run {run} vs saved Nov 4 predictions: {changed}")


def print_questions(
    issues: list[dict],
    baseline: dict[int, Answers],
    current: dict[int, Answers],
) -> None:
    support = [issue for issue in issues if is_pip_support(issue)]
    print(f"\nQuestion predictions (of {len(issues)} issues)")
    for name, runs in (("baseline", baseline), ("current", current)):
        for run, answers in runs.items():
            predicted = sum(answers[issue_key(i)] == "question" for i in issues)
            right = sum(answers[issue_key(i)] == "question" for i in support)
            print(
                f"  {name} run {run}: predicted {predicted}, "
                f"right on {right}/{len(support)} pip support issues"
            )


def print_disagreements(
    issues: list[dict],
    baseline: dict[int, Answers],
    current: dict[int, Answers],
) -> None:
    for run in baseline:
        print(f"\nPrompts disagree (run {run})")
        for issue in issues:
            key = issue_key(issue)
            if baseline[run][key] != current[run][key]:
                print(
                    f"  expected {issue['expected']}, "
                    f"baseline {baseline[run][key]}, current {current[run][key]}: "
                    f"{issue['url']}"
                )


def build_issue_set() -> list[dict]:
    issues = []
    for repo, rules in RULES.items():
        records = load_records(output_path(DEFAULT_OUTPUT_DIR, repo))
        issues.extend(freeze_issues(records, current_labels(records, repo), rules))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline", required=True, help="Git commit holding the earlier prompt."
    )
    parser.add_argument(
        "--runs", type=int, default=2, help="Times to run each prompt on each issue."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between calls, to stay under API rate limits.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    try:
        config = load_config()
        baseline_prompt = prompt_at_ref(args.baseline)
    except (LLMConfigError, ValueError) as exc:
        sys.exit(str(exc))
    except subprocess.CalledProcessError as exc:
        sys.exit(f"Could not read the prompt at {args.baseline}: {exc.stderr}")
    preflight(config)

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    issues = load_or_freeze(output_dir / "issues.json", build_issue_set)
    baseline_version = prompt_version(baseline_prompt)
    current_version = prompt_version(TRIAGE_PROMPT)
    print(f"{len(issues)} issues ({len(without_support(issues))} without pip support)")
    print(f"baseline {args.baseline} {baseline_version}, current {current_version}")

    made = collect(
        issues,
        {"baseline": baseline_prompt, "current": TRIAGE_PROMPT},
        args.runs,
        predictions_path,
        triage=triage_issue,
        delay=args.delay,
        metadata={"provider": config.provider, "model": config.model},
    )
    print(f"{made} new predictions -> {predictions_path}")

    predictions = load_predictions(predictions_path)
    runs = range(1, args.runs + 1)
    baseline = {run: answers_for(predictions, baseline_version, run) for run in runs}
    current = {run: answers_for(predictions, current_version, run) for run in runs}
    saved = {issue_key(issue): issue["saved_category"] for issue in issues}

    print_accuracy("All scored issues", issues, baseline, current)
    print_accuracy(
        "Without pip support issues", without_support(issues), baseline, current
    )
    print_noise(issues, baseline, current, saved)
    print_questions(issues, baseline, current)
    print_disagreements(issues, baseline, current)


if __name__ == "__main__":
    main()
