"""Shadow mode: triage new public issues and save predictions. Posts nothing.

Each prediction is saved with a snapshot of the issue, so it can later be
compared with what the maintainers actually did. The model only sees the title
and body. Issues already predicted are skipped, so reruns are safe.

Usage (from the project root; configure the model as described in the README):

    python -m evals.shadow
    python -m evals.shadow --repo pypa/pip --days 3 --limit 10
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path

from app.llm import LLMConfigError, load_config
from app.models import IssueReport, TriageResult
from app.triage import TRIAGE_PROMPT, triage_issue
from evals.run_evals import preflight

# These repos mark when a maintainer has reviewed an issue, so they can be scored.
DEFAULT_REPOS = ("pypa/pip", "python-poetry/poetry")
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "shadow"
# Issues opened from this date on are held out for final measurement. Never
# tune prompts against them.
HELD_OUT_FROM = datetime(2026, 11, 5, tzinfo=UTC)
PAGE_SIZE = 100


def github_get_json(url: str) -> list[dict]:
    """Read a GitHub API URL. GITHUB_TOKEN is optional for public repos."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "issue-triage-shadow",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_new_issues(
    repo: str,
    since: datetime,
    get_json: Callable[[str], list[dict]] = github_get_json,
) -> list[dict]:
    """Return issues opened since a date, excluding pull requests and bots."""
    issues: list[dict] = []
    for page in count(1):
        items = get_json(
            f"https://api.github.com/repos/{repo}/issues"
            f"?state=all&sort=created&direction=desc"
            f"&per_page={PAGE_SIZE}&page={page}"
        )
        if not items:
            break
        recent = [
            item
            for item in items
            if datetime.fromisoformat(item["created_at"]) >= since
        ]
        issues.extend(
            item
            for item in recent
            if "pull_request" not in item
            and (item.get("user") or {}).get("type") != "Bot"
        )
        if len(recent) < len(items):
            break
    return issues


def output_path(output_dir: Path, repo: str) -> Path:
    return output_dir / f"{repo.replace('/', '__')}.jsonl"


def load_seen(path: Path) -> set[int]:
    if not path.exists():
        return set()
    lines = path.read_text("utf-8").splitlines()
    return {json.loads(line)["number"] for line in lines if line.strip()}


def prompt_version(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def predict(
    repo: str,
    issue: dict,
    triage: Callable[[IssueReport], TriageResult],
    metadata: dict[str, str],
) -> dict:
    """Triage one issue and build its record. Bad model output is recorded."""
    result = None
    error = None
    try:
        report = IssueReport(title=issue["title"], description=issue.get("body"))
        result = triage(report)
    except (TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    prediction = (
        result.model_dump() if result else dict.fromkeys(TriageResult.model_fields)
    )
    return {
        "repo": repo,
        "number": issue["number"],
        "url": issue["html_url"],
        "title": issue["title"],
        "body": issue.get("body") or "",
        "created_at": issue["created_at"],
        "predicted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "state_at_prediction": issue["state"],
        "comments_at_prediction": issue["comments"],
        "labels_at_prediction": [label["name"] for label in issue["labels"]],
        "prompt_version": prompt_version(TRIAGE_PROMPT),
        **metadata,
        **prediction,
        "error": error,
    }


def collect_repo(
    repo: str,
    since: datetime,
    path: Path,
    *,
    fetch: Callable[[str, datetime], list[dict]],
    triage: Callable[[IssueReport], TriageResult],
    limit: int,
    delay: float,
    metadata: dict[str, str],
) -> int:
    """Predict unseen issues, oldest first. Each record is saved immediately."""
    seen = load_seen(path)
    unseen = [issue for issue in fetch(repo, since) if issue["number"] not in seen]
    batch = sorted(unseen, key=lambda issue: issue["number"])[:limit]

    path.parent.mkdir(parents=True, exist_ok=True)
    for index, issue in enumerate(batch):
        if index and delay:
            time.sleep(delay)
        record = predict(repo, issue, triage, metadata)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", nargs="+", default=list(DEFAULT_REPOS))
    parser.add_argument(
        "--days", type=int, default=7, help="Look back this many days."
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Most issues per repo per run."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between issues, to stay under API rate limits.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    try:
        config = load_config()
    except LLMConfigError as exc:
        sys.exit(str(exc))
    preflight(config)

    since = datetime.now(UTC) - timedelta(days=args.days)
    metadata = {"provider": config.provider, "model": config.model}
    for repo in args.repo:
        path = output_path(Path(args.output_dir), repo)
        try:
            saved = collect_repo(
                repo,
                since,
                path,
                fetch=fetch_new_issues,
                triage=triage_issue,
                limit=args.limit,
                delay=args.delay,
                metadata=metadata,
            )
        except urllib.error.URLError as exc:
            print(f"{repo}: could not read issues ({exc})", flush=True)
            continue
        print(f"{repo}: {saved} new predictions -> {path}", flush=True)


if __name__ == "__main__":
    main()
