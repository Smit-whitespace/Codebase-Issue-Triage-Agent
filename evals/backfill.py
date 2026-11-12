"""Held-out score on older issues that maintainers have already reviewed.

New issues arrive too slowly to build a held-out set from shadow mode alone, so
this scores already-triaged issues from a past window instead. The window must
start after the model's training cutoff (Gemini 3.5 Flash-Lite: March 2026), so
the model can't have seen the issues or their labels. It must also end before
the earliest shadow issue, because shadow issues were looked at while tuning.

Only issues that already carry a reviewed type label are sent to the model.
Labels are read when an issue is predicted. They are final for issues this old.
Predictions are saved after every call, so rerunning continues where it stopped.

Usage (configure the model as described in the README):

    python -m evals.backfill
"""

import argparse
import json
import sys
import time
import urllib.error
from collections.abc import Callable
from pathlib import Path

from app.llm import LLMConfigError, load_config
from app.triage import triage_issue
from evals.run_evals import preflight
from evals.shadow import DEFAULT_OUTPUT_DIR as SHADOW_DIR
from evals.shadow import (
    PAGE_SIZE,
    github_get_json,
    load_seen,
    output_path,
    predict,
)
from evals.shadow_compare import (
    RULES,
    RepoRules,
    combine,
    expected_category,
    load_records,
    print_summary,
    score_records,
)

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "backfill"
# After the model's training cutoff, and before the earliest shadow issue.
DEFAULT_START = "2026-04-01"
DEFAULT_END = "2026-07-02"


def search_issues(
    repo: str,
    start: str,
    end: str,
    get_json: Callable[[str], dict] = github_get_json,
) -> list[dict]:
    """Return every issue opened in a date window, newest first."""
    issues: list[dict] = []
    page = 1
    while True:
        items = get_json(
            "https://api.github.com/search/issues"
            f"?q=repo:{repo}+is:issue+created:{start}..{end}"
            f"&sort=created&order=desc&per_page={PAGE_SIZE}&page={page}"
        )["items"]
        issues.extend(items)
        if len(items) < PAGE_SIZE:
            return issues
        page += 1


def scorable_issues(
    issues: list[dict], rules: RepoRules, exclude: set[int]
) -> list[dict]:
    """Keep human-filed issues with a reviewed category, outside the tuning set."""
    kept = []
    for issue in issues:
        if "pull_request" in issue or issue["number"] in exclude:
            continue
        if (issue.get("user") or {}).get("type") == "Bot":
            continue
        labels = [label["name"] for label in issue["labels"]]
        if expected_category(labels, rules)[0] is not None:
            kept.append(issue)
    return kept


def collect(repo: str, rules: RepoRules, args: argparse.Namespace, metadata) -> None:
    path = output_path(Path(args.output_dir), repo)
    tuning = load_seen(output_path(SHADOW_DIR, repo))
    issues = scorable_issues(
        search_issues(repo, args.start, args.end), rules, exclude=tuning
    )
    seen = load_seen(path)
    todo = [issue for issue in issues if issue["number"] not in seen]
    print(f"{repo}: {len(issues)} reviewed issues, {len(todo)} to predict", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    for index, issue in enumerate(todo):
        if index and args.delay:
            time.sleep(args.delay)
        record = predict(repo, issue, triage_issue, metadata)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def score(output_dir: Path) -> None:
    summaries = []
    for repo, rules in RULES.items():
        path = output_path(output_dir, repo)
        if not path.exists():
            continue
        records = load_records(path)
        labels = {r["number"]: r["labels_at_prediction"] for r in records}
        summary = score_records(records, labels, rules)
        print_summary(f"{repo}, historical held-out", summary)
        summaries.append(summary)
    print_summary("All repos, historical held-out", combine(summaries))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--score-only", action="store_true", help="Score saved predictions only."
    )
    args = parser.parse_args()

    if not args.score_only:
        try:
            config = load_config()
        except LLMConfigError as exc:
            sys.exit(str(exc))
        preflight(config)
        metadata = {"provider": config.provider, "model": config.model}
        for repo, rules in RULES.items():
            try:
                collect(repo, rules, args, metadata)
            except urllib.error.URLError as exc:
                print(f"{repo}: could not read issues ({exc})", flush=True)
    score(Path(args.output_dir))


if __name__ == "__main__":
    main()
