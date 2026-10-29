
import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.models import IssueReport, TriageResult
from app.repository import search_repository
from app.triage import triage_issue


class TriageState(TypedDict, total=False):
    report: IssueReport
    result: TriageResult
    investigation_required: bool
    repository_path: str
    evidence: list[dict[str, str]]
    investigation_status: str


def triage_node(state: TriageState) -> dict:
    result = triage_issue(state["report"])
    return {
        "result": result,
        "investigation_required": result.needs_repository_investigation,
        "evidence": [],
        "investigation_status": (
            "pending"
            if result.needs_repository_investigation
            else "skipped"
        ),
    }


def investigation_router(state: TriageState) -> str:
    if state.get("investigation_required", False):
        return "investigate"
    return "finish"



def investigate_node(state: TriageState) -> dict:
    repository_path = state.get("repository_path") or os.getcwd()

    try:
        query = state["report"].title
        evidence = search_repository(repository_path, query)

        return {
            "evidence": evidence,
            "investigation_status": (
                "completed_with_matches"
                if evidence
                else "completed_no_matches"
            ),
        }
    except (OSError, ValueError) as exc:
        return {
            "evidence": [],
            "investigation_status": f"failed: {exc}",
        }


def finish_node(state: TriageState) -> dict:
    return {}


builder = StateGraph(TriageState)

builder.add_node("triage", triage_node)
builder.add_node("investigate", investigate_node)
builder.add_node("finish", finish_node)

builder.add_edge(START, "triage")
builder.add_conditional_edges(
    "triage",
    investigation_router,
    {
        "investigate": "investigate",
        "finish": "finish",
    },
)
builder.add_edge("investigate", "finish")
builder.add_edge("finish", END)

triage_graph = builder.compile()
