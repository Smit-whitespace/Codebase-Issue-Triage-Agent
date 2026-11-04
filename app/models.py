
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 10_000
TRUNCATION_NOTE = "\n\n[description truncated]"


class IssueReport(BaseModel):
    """An issue submitted for triage.

    Real issues can have an empty body (null from the GitHub API) or very long
    logs, so text is normalized rather than rejected.
    """

    title: str = Field(min_length=1)
    description: str

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()[:MAX_TITLE_LENGTH]
        return value

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip()
            if len(text) > MAX_DESCRIPTION_LENGTH:
                return text[:MAX_DESCRIPTION_LENGTH] + TRUNCATION_NOTE
            return text
        return value


class TriageResult(BaseModel):
    """Structured triage recommendation for human review."""

    category: Literal[
        "bug",
        "feature_request",
        "question",
        "documentation",
        "other",
    ]
    priority: Literal["low", "medium", "high", "critical"]
    summary: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=2_000)
    needs_repository_investigation: bool
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_labels: list[str] = Field(max_length=10)
    draft_response: str = Field(max_length=2_000)
