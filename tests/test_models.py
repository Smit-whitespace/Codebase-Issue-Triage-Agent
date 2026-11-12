import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.models import (
    MAX_DESCRIPTION_LENGTH,
    MAX_TITLE_LENGTH,
    TRUNCATION_NOTE,
    IssueReport,
)
from app.triage import triage_issue


def test_null_body_from_github_becomes_empty_description() -> None:
    report = IssueReport(title="Crash on startup", description=None)

    assert report.description == ""


def test_short_description_is_accepted() -> None:
    report = IssueReport(title="Crash on startup", description="See title")

    assert report.description == "See title"


def test_long_description_is_truncated_with_a_visible_note() -> None:
    text = "x" * (MAX_DESCRIPTION_LENGTH * 2)

    report = IssueReport(title="Crash on startup", description=text)

    assert report.description == "x" * MAX_DESCRIPTION_LENGTH + TRUNCATION_NOTE


def test_description_at_the_limit_is_unchanged() -> None:
    text = "x" * MAX_DESCRIPTION_LENGTH

    report = IssueReport(title="Crash on startup", description=text)

    assert report.description == text


def test_long_title_is_truncated() -> None:
    report = IssueReport(title="t" * (MAX_TITLE_LENGTH + 50), description="")

    assert report.title == "t" * MAX_TITLE_LENGTH


def test_two_character_title_is_accepted() -> None:
    assert IssueReport(title="UI", description="").title == "UI"


def test_blank_title_is_rejected() -> None:
    with pytest.raises(ValidationError):
        IssueReport(title="   ", description="Some text")


def test_prompt_states_when_the_description_is_missing() -> None:
    reply = json.dumps(
        {
            "category": "other",
            "priority": "low",
            "summary": "Empty report",
            "rationale": "There is nothing to act on.",
            "needs_repository_investigation": False,
            "confidence": 0.5,
            "proposed_labels": [],
            "draft_response": "Could you add more detail?",
        }
    )

    with patch("app.triage.get_llm") as get_llm:
        get_llm.return_value.invoke.return_value = SimpleNamespace(content=reply)
        triage_issue(IssueReport(title="Crash on startup", description=None))

    prompt = get_llm.return_value.invoke.call_args.args[0]
    assert "(no description provided)" in prompt


def test_triage_can_use_a_different_prompt() -> None:
    reply = json.dumps(
        {
            "category": "bug",
            "priority": "low",
            "summary": "A crash",
            "rationale": "It crashes.",
            "needs_repository_investigation": False,
            "confidence": 0.5,
            "proposed_labels": [],
            "draft_response": "Thanks.",
        }
    )
    other_prompt = "Other prompt. {schema} / {title} / {description}"

    with patch("app.triage.get_llm") as get_llm:
        get_llm.return_value.invoke.return_value = SimpleNamespace(content=reply)
        triage_issue(
            IssueReport(title="Crash", description="It crashes."),
            prompt=other_prompt,
        )

    sent = get_llm.return_value.invoke.call_args.args[0]
    assert sent.startswith("Other prompt.")
    assert sent.endswith("/ Crash / It crashes.")
