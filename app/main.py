
import argparse
import json

from app.graph import triage_graph
from app.models import IssueReport


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triage a software issue using a local LLM."
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument(
        "--repository",
        default=".",
        help="Path to the repository to inspect (default: current directory).",
    )
    args = parser.parse_args()

    report = IssueReport(
        title=args.title,
        description=args.description,
    )

    state = triage_graph.invoke(
        {
            "report": report,
            "repository_path": args.repository,
        }
    )

    output = {
        "triage": state["result"].model_dump(),
        "investigation_status": state.get("investigation_status", "skipped"),
        "evidence": state.get("evidence", []),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
