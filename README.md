# mlsherlock

> **Work in progress** — this project is under active development. APIs, CLI flags, and
> behavior may change without notice. Not yet suitable for production use.

An ML agent that thinks like a senior engineer — diagnoses overfitting, calibration issues,
class imbalance, and data leakage, then applies targeted fixes.

**This is not AutoML.** The agent uses Claude as its reasoning engine to understand what is
wrong with a model and apply the minimum effective fix, the same way an experienced engineer
would.

---

## What it does

When you hand it a CSV and a target column, it:

1. Profiles your data (shape, nulls, class balance, dtypes)
2. Trains a baseline model
3. Diagnoses what went wrong — overfitting? poor calibration? class imbalance? leakage?
4. Decides what to try next and applies it
5. Repeats until the model is good or you tell it to stop

You can answer questions interactively, or run with `--non-interactive` to let it decide.

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

---

## Setup

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/gr33ndata/mlsherlock.git
cd mlsherlock
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install
pip install -e .

# 3. Set your API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

```bash
mlsh train --data path/to/data.csv --target target_column

# Named datasets (downloaded automatically)
mlsh train --data titanic --target survived

# Regression
mlsh train --data path/to/data.csv --target price --task regression

# Non-interactive (agent picks options automatically)
mlsh train --data path/to/data.csv --target label --non-interactive
```

### All options

```
mlsh train --help

Options:
  --data TEXT                       CSV path, named dataset, or Kaggle slug
  --target TEXT                     Target column name
  --task [classification|regression]  Default: classification
  --output-dir TEXT                 Where to save model + plots (default: ./output)
  --max-iterations INTEGER          Max agent iterations (default: 20)
  --non-interactive                 Auto-approve all agent questions
```

### Outputs

After a run, `./output/` contains:

- `model.pkl` — the trained model (joblib format)
- `summary.txt` — what the agent found and fixed
- `*.png` — plots (ROC curve, calibration, feature importances, etc.)

```python
import joblib
model = joblib.load("output/model.pkl")
predictions = model.predict(X_new)
```

---

## Diagnostic logic

| Observation | Diagnosis | First fix |
|---|---|---|
| train >> test by >0.05 | Overfitting | Increase regularization |
| Both scores low | Underfitting | More complex model, check encodings |
| Test accuracy >0.99 | Likely data leakage | Inspect feature names |
| Minority class <15% | Class imbalance | `class_weight` vs SMOTE |
| Probabilities squeezed | Miscalibration | `CalibratedClassifierCV` |

---

## Running tests

```bash
pytest tests/
```

Tests use a synthetic fixture (`tests/fixtures/sample.csv`) — no API calls are made.

---

## Architecture

```
cli.py  →  AgentLoop  →  LLM provider (Anthropic / OpenAI)
                ↓
           tool dispatch
                ↓
        CodeExecutor (sandbox)
```

### `cli.py`
Entry point for `mlsh train`. Parses flags, infers target column and task type from the CSV
if not provided, then wires up the agent and starts the loop.

### `engine/`

| File | Purpose |
|---|---|
| `loop.py` | Drives the conversation: calls the LLM, dispatches tool calls, tracks history, injects stuck-detection hints and iteration reminders |
| `providers.py` | Thin wrappers for Anthropic and OpenAI APIs — normalises responses so the loop doesn't care which backend is used |
| `state.py` | Shared mutable session data: data path, target column, task type, iteration count, finished flag, and error-tracking for stuck detection |
| `system_prompt.py` | The ML diagnostic protocol the agent follows — overfitting/underfitting/leakage/imbalance heuristics and stopping criteria |

### `tools/`
Each file is one tool the agent can call. All return a plain string that goes back into the
conversation.

| Tool | What it does |
|---|---|
| `read_data` | Loads a CSV, profiles shape/nulls/dtypes/class balance, injects `df` and `target` into the sandbox |
| `run_python` | Executes arbitrary Python in the persistent sandbox; stdout + errors come back to the agent |
| `download_data` | Fetches a dataset from a named shortcut, direct URL, or Kaggle slug and saves it as CSV |
| `save_plot` | Saves the current matplotlib figure to the output directory |
| `ask_user` | Pauses execution to get a human decision (e.g. class_weight vs SMOTE) |
| `finish` | Serialises the trained model with joblib, writes a summary file, and ends the loop |
| `registry.py` | Pydantic input schemas (used for JSON schema generation) and the dispatcher that routes tool names to implementations |

### `execution/`

| File | Purpose |
|---|---|
| `sandbox.py` | `CodeExecutor` — runs code in a persistent `exec` namespace. ML libraries (pandas, sklearn, numpy, matplotlib) are pre-imported. Each execution runs in a shallow copy; globals are merged back only on success so timed-out threads can't corrupt state |
| `capture.py` | Context manager that redirects stdout/stderr during `exec` and returns the combined output |

### `ui/`
`cli_callbacks.py` — Rich-formatted terminal output. Implements `on_message`, `on_tool_call`,
`on_tool_result`, `on_ask_user`, `on_plot`, and `on_finish`. Verbose mode shows full output;
terse mode shows only the first sentence and key metrics.

```
mlsherlock/
├── cli.py                   # Entry point: mlsh train
├── engine/
│   ├── loop.py              # Agent loop: LLM calls, tool dispatch, history management
│   ├── providers.py         # Anthropic + OpenAI wrappers with normalised interface
│   ├── state.py             # Session state: paths, iteration count, stuck detection
│   └── system_prompt.py    # ML diagnostic protocol (heuristics + stopping criteria)
├── tools/
│   ├── registry.py          # Input schemas (Pydantic) + tool dispatcher
│   ├── read_data.py         # Profile CSV and inject df/target into sandbox
│   ├── run_python.py        # Execute Python in the persistent sandbox
│   ├── download_data.py     # Fetch dataset from URL, named shortcut, or Kaggle
│   ├── save_plot.py         # Save current matplotlib figure to output dir
│   ├── ask_user.py          # Pause for human input
│   └── finish.py            # Serialise model, write summary, end loop
├── execution/
│   ├── sandbox.py           # CodeExecutor: persistent exec namespace with timeout isolation
│   └── capture.py           # Context manager: captures stdout/stderr during exec
└── ui/
    └── cli_callbacks.py     # Rich terminal output (verbose + terse modes)
```

---

## Contributing

Early development — PRs and issues welcome, things may break.

The diagnostic heuristics live in `mlsherlock/engine/system_prompt.py`. That's the right
place to start if you want to understand or change how the agent reasons about ML problems.
