# Codebase Issue Triage Agent

A local LLM-powered issue triage prototype that classifies software issue reports, produces structured recommendations, and conditionally investigates a code repository using a read-only search tool.

## Features

- **Structured triage:** Categorizes reports as bugs, feature requests, questions, documentation issues, or other.
- **Priority recommendation:** Assigns low, medium, high, or critical priority, with rationale and a model-reported confidence score.
- **Conditional workflow:** Uses LangGraph to decide whether repository investigation should run.
- **Repository search:** Searches local text files for terms from the issue title and returns matching file paths, line numbers, and snippets.
- **Validated output:** Uses Pydantic to validate the model's JSON response.
- **Local inference:** Uses Ollama with the `qwen3:8b` model.
- **Human review:** Produces suggested labels and a draft response; it does not automatically modify issues or repository files.

## Architecture

1. An issue report is validated using Pydantic.
2. The local LLM generates a structured triage recommendation.
3. LangGraph routes the workflow according to `needs_repository_investigation`.
4. When investigation is requested, a read-only repository search runs.
5. The CLI prints the recommendation, investigation status, and available evidence as JSON.

## Technology Stack

- Python
- LangGraph
- LangChain Ollama integration
- Ollama (`qwen3:8b`)
- Pydantic
- Pytest
- Ruff

## Prerequisites

- Python 3.12 or newer compatible with the installed dependencies
- [Ollama](https://ollama.com/) installed and running
- The `qwen3:8b` model available locally

Pull the model if needed:

```bash
ollama pull qwen3:8b
```

## Setup

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install langgraph langchain-ollama pydantic pytest ruff
ollama pull qwen3:8b
```

Ensure Ollama is reachable at `http://127.0.0.1:11434`.

## Usage

From the project root:

```powershell
.venv\Scripts\python -m app.main --title "Application crashes on startup" --description "The application exits immediately after launch with a traceback. This blocks all users from accessing the application." --repository .
```

The CLI prints JSON containing:

- `triage`: category, priority, summary, rationale, confidence, labels, and draft response
- `investigation_status`: whether investigation was skipped, completed with matches, completed without matches, or failed
- `evidence`: matching repository file paths, line numbers, snippets, and matched terms

Use `--repository` to specify a different local repository path.

## Tests and Code Quality

Run the test suite:

```powershell
.venv\Scripts\python -m pytest -q
```

Run Ruff:

```powershell
.venv\Scripts\ruff check .
```

The current test suite covers repository-search behavior and conditional graph routing. The last recorded run passed all 6 tests, and Ruff reported no issues.

## Current Limitations

- Priority and confidence are model-generated recommendations, not calibrated risk assessments.
- Repository search uses simple title-term matching rather than semantic code retrieval.
- Search results may be empty even when relevant code exists.
- Investigation does not execute code, reproduce bugs, or prove root cause.
- The system does not create GitHub issues, apply labels, or modify source files.
- Model output quality depends on the local model and its ability to follow the requested schema.
- This is a prototype intended for human-reviewed triage, not autonomous production incident handling.

## Safety

Repository investigation is designed to be read-only. Treat model output as a recommendation and review it before taking action.
