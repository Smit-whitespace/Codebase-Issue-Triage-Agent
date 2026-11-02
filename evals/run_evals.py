"""Run the triage agent against the labelled eval dataset.

Usage (from the project root; configure the model as described in the README):

    python -m evals.run_evals
    python -m evals.run_evals --limit 5
    python -m evals.run_evals --tag edge:prompt-injection
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from app.llm import LLMConfig, LLMConfigError, get_llm, load_config
from app.models import IssueReport
from app.triage import TriageOutputError, triage_issue
from evals.scoring import CaseResult, format_report, load_dataset, score_case, summarize

DEFAULT_DATASET = Path(__file__).parent / "dataset.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"


def preflight(config: LLMConfig) -> None:
    """Fail fast with a clear message if the configured model can't be used."""
    if config.provider == "ollama":
        check_ollama(config.ollama_base_url)
    try:
        get_llm()
    except LLMConfigError as exc:
        sys.exit(str(exc))


def check_ollama(base_url: str) -> None:
    """Fail fast with a clear message if Ollama is not reachable."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5):
            pass
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(
            f"Ollama is not reachable at {base_url} ({exc}).\n"
            "Start it with `ollama serve` and make sure the model is pulled."
        )


def run_case(case) -> CaseResult:
    report = IssueReport(title=case.title, description=case.description)
    start = time.perf_counter()
    try:
        result = triage_issue(report)
    except TriageOutputError as exc:
        return score_case(
            case,
            None,
            latency_seconds=time.perf_counter() - start,
            error=str(exc),
            raw_response=exc.raw_response,
        )
    except (TypeError, ValueError) as exc:
        return score_case(
            case,
            None,
            latency_seconds=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
    return score_case(case, result, latency_seconds=time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the triage agent.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, help="Run only the first N cases.")
    parser.add_argument("--tag", help="Run only cases with this tag.")
    parser.add_argument("--ids", nargs="+", help="Run only these case ids.")
    parser.add_argument(
        "--label",
        default="",
        help="Short note for this run, e.g. 'baseline' or 'prompt-v2'.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between cases, to stay under API rate limits.",
    )
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    if args.ids:
        wanted = set(args.ids)
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        sys.exit("No eval cases matched the filters.")

    try:
        config = load_config()
    except LLMConfigError as exc:
        sys.exit(str(exc))
    preflight(config)
    print(f"Evaluating {config.provider}:{config.model} on {len(cases)} cases\n")

    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        if index > 1 and args.delay:
            time.sleep(args.delay)
        outcome = run_case(case)
        results.append(outcome)
        status = (
            "INVALID"
            if not outcome.valid_output
            else ("ok" if outcome.category_correct else "WRONG")
        )
        print(
            f"[{index}/{len(cases)}] {case.id:<12} {status:<8}"
            f"{outcome.latency_seconds:6.1f}s",
            flush=True,
        )

    summary = summarize(results)
    print()
    print(format_report(summary, results))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{args.label}" if args.label else ""
    output_path = output_dir / f"run-{timestamp}{suffix}.json"
    output_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "timestamp": timestamp,
                "provider": config.provider,
                "model": config.model,
                "dataset": str(args.dataset),
                "summary": summary,
                "cases": [r.model_dump() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
