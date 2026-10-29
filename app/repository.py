
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".next",
}

IGNORED_TERMS = {
    "application",
    "issue",
    "error",
    "please",
    "when",
    "with",
    "from",
    "that",
    "this",
    "have",
    "does",
    "into",
    "after",
    "before",
}

MAX_FILES = 2_000
MAX_RESULTS = 10
MAX_FILE_BYTES = 500_000


def search_repository(
    repository_path: str,
    query: str,
) -> list[dict[str, str]]:
    """Search repository text files for relevant query terms."""
    root = Path(repository_path).resolve()

    if not root.is_dir():
        raise ValueError(f"Repository directory does not exist: {root}")

    terms = list(
        dict.fromkeys(
            word.strip(".,:;!?()[]{}\"'").lower()
            for word in query.split()
            if len(word.strip(".,:;!?()[]{}\"'")) >= 4
            and word.strip(".,:;!?()[]{}\"'").lower() not in IGNORED_TERMS
        )
    )

    if not terms:
        raise ValueError("Query must contain at least one meaningful term.")

    results: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()
    scanned = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(root)

        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue

        scanned += 1
        if scanned > MAX_FILES:
            break

        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for line_number, line in enumerate(lines, start=1):
            matched_terms = [term for term in terms if term in line.lower()]
            key = (relative.as_posix(), line_number)

            if matched_terms and key not in seen:
                seen.add(key)
                results.append(
                    {
                        "file": relative.as_posix(),
                        "line": str(line_number),
                        "text": line.strip()[:300],
                        "matched_terms": ", ".join(matched_terms),
                    }
                )

                if len(results) >= MAX_RESULTS:
                    return results

    return results
