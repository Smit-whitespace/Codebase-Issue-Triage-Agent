
from typing import Literal

from pydantic import BaseModel, Field


class IssueReport(BaseModel):
    """A bug report submitted for triage."""

    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=10_000)


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
