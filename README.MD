# Codebase Issue Triage Agent

A local LLM-powered issue triage prototype that classifies software issue reports, produces structured recommendations, and conditionally investigates a code repository using a read-only search tool.

## Features

- **Structured triage:** Categorizes reports as bugs, feature requests, questions, documentation issues, or other.
- **Priority recommendation:** Assigns low, medium, high, or critical priority, with rationale and a model-reported confidence score.
- **Conditional workflow:** Uses LangGraph to decide whether repository investigation should run.
- **Repository search:** Searches local text files for terms from the issue title and returns matching file paths, line numbers, and snippets.
- **Validated output:** Uses Pydantic to validate the model's JSON response.
- **Configurable model:** Runs locally on Ollama (`qwen3:8b`) by default, or on Google Gemini through an environment variable.
- **Human review:** Produces suggested labels and a draft response; it does not automatically modify issues or repository files.

## Architecture

1. An issue report is validated using Pydantic.
2. The configured LLM (Ollama or Gemini) generates a structured triage recommendation.
3. LangGraph routes the workflow according to `needs_repository_investigation`.
4. When investigation is requested, a read-only repository search runs.
5. The CLI prints the recommendation, investigation status, and available evidence as JSON.

## Technology Stack

- Python
- LangGraph
- LangChain Ollama and Google GenAI integrations
- Ollama (`qwen3:8b`) or Google Gemini
- Pydantic
- Pytest
- Ruff

## Prerequisites

- Python 3.12 or newer compatible with the installed dependencies
- Either [Ollama](https://ollama.com/) with the `qwen3:8b` model, or a Gemini API key from [Google AI Studio](https://aistudio.google.com/)

For Ollama, pull the model if needed:

```bash
ollama pull qwen3:8b
```

## Setup

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Model Configuration

The model is chosen with environment variables. Nothing is hard-coded, and API keys are never stored in the repository.

| Variable | Default | Purpose |
|---|---|---|
| `TRIAGE_LLM_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `TRIAGE_LLM_MODEL` | `qwen3:8b` (Ollama), `gemini-3.5-flash-lite` (Gemini) | Model name |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | none | Gemini API key |

**Ollama (default):** run `ollama pull qwen3:8b` and make sure Ollama is reachable.

**Gemini** (PowerShell, current session only):

```powershell
$env:TRIAGE_LLM_PROVIDER = "gemini"
$env:GOOGLE_API_KEY = "<your key>"
# Optional: pick a different model
$env:TRIAGE_LLM_MODEL = "gemini-3.5-flash-lite"
```

Free-tier quotas are per model and per day, and Flash-Lite allows far more requests than Flash. To see which models your key can use:

```powershell
.venv\Scripts\python -c "from google import genai; [print(m.name) for m in genai.Client().models.list() if 'flash' in m.name]"
```

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

The test suite covers repository-search behavior, conditional graph routing, and the eval scoring logic.

## Evaluation

Unit tests check the code; evals check the model's *answers*. The eval harness runs the agent against hand-labelled issues and measures how often it is right.

- `evals/dataset.jsonl`: labelled cases across every category, including edge cases tagged `edge:*` (prompt injection, bugs disguised as questions, feature requests phrased as bugs, vague reports). Priority is subjective, so each case lists the `acceptable_priorities`.
- `evals/scoring.py`: pure scoring logic, unit-tested in `tests/test_evals.py`.
- `evals/run_evals.py`: runs the agent and writes results.

With the model configured (see Model Configuration):

```powershell
.venv\Scripts\python -m evals.run_evals --label baseline
.venv\Scripts\python -m evals.run_evals --tag edge:prompt-injection
.venv\Scripts\python -m evals.run_evals --ids bug-004 q-006
.venv\Scripts\python -m evals.run_evals --delay 5   # pause between cases to respect rate limits
```

The report shows:

- Valid-output rate, category accuracy, priority-acceptable rate, and investigation-decision accuracy. An invalid output counts as wrong.
- Accuracy for each category, a confusion matrix, and accuracy by tag.
- A **calibration table** that checks whether higher model-reported confidence actually comes with higher accuracy.
- Every failure. Full per-case results, including the raw model response for invalid output, are saved to `evals/results/run-<timestamp>-<label>.json` (gitignored).

Workflow: run a baseline, change one thing (the prompt, the model, or the retrieval), rerun, and compare. Add every real-world misclassification to the dataset so it can't regress silently.

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

## License

MIT. See [LICENSE](LICENSE).
