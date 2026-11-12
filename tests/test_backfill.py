from evals.backfill import scorable_issues, search_issues
from evals.shadow_compare import RULES

PIP = RULES["pypa/pip"]


def make_issue(number: int, labels: list[str], **overrides) -> dict:
    issue = {
        "number": number,
        "labels": [{"name": name} for name in labels],
        "user": {"type": "User"},
    }
    issue.update(overrides)
    return issue


def test_only_reviewed_human_issues_are_kept() -> None:
    issues = [
        make_issue(1, ["type: bug"]),
        make_issue(2, ["type: bug", "S: needs triage"]),
        make_issue(3, []),
        make_issue(4, ["type: bug"], user={"type": "Bot"}),
        make_issue(5, ["type: bug"], pull_request={}),
        make_issue(6, ["type: support"]),
    ]

    kept = scorable_issues(issues, PIP, exclude=set())

    assert [issue["number"] for issue in kept] == [1, 6]


def test_issues_in_the_tuning_set_are_never_kept() -> None:
    issues = [make_issue(1, ["type: bug"]), make_issue(2, ["type: bug"])]

    kept = scorable_issues(issues, PIP, exclude={2})

    assert [issue["number"] for issue in kept] == [1]


def test_search_reads_every_page_of_the_date_window() -> None:
    urls = []
    pages = [[{"number": n} for n in range(100)], [{"number": 100}]]

    def get_json(url: str) -> dict:
        urls.append(url)
        return {"items": pages[len(urls) - 1]}

    issues = search_issues("pypa/pip", "2026-04-01", "2026-07-02", get_json)

    assert len(issues) == 101
    assert "repo:pypa/pip+is:issue+created:2026-04-01..2026-07-02" in urls[0]
    assert "page=2" in urls[1]
