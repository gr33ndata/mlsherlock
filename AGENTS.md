# AgenticML

An intelligent ML agent CLI that reasons like a senior ML engineer — diagnoses overfitting, calibration issues, class imbalance, data leakage, and feature importance problems — then applies targeted fixes.

This is **not** brute-force AutoML. The agent uses Claude as its reasoning engine to understand what is wrong with a model and apply the minimum effective fix, the same way an experienced engineer would.

---

## Problem It Solves

When building a classifier or regressor, you typically have to:

1. Get data and do feature engineering
2. Train a model and check error metrics
3. Diagnose what went wrong — overfitting? Poor calibration? Class imbalance? Leakage?
4. Know *what* to try next — regularization, SMOTE, feature pruning, a different model family?
5. Repeat until satisfied

AgenticML automates steps 2–5 with an agent that knows diagnostic heuristics, asks you when a decision requires human judgment, and iterates until the model is good or you tell it to stop.

---

## Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   CLI Adapter   │    │  Notebook Adapter │  ← future
│  (Rich + stdin) │    │  (ipywidgets)     │
└────────┬────────┘    └────────┬──────────┘
         └──────────┬───────────┘
                    │
           ┌────────▼────────┐
           │   AgentEngine   │  ← shared core
           │  loop, tools,   │
           │  system prompt  │
           └─────────────────┘
```

The engine fires I/O events; adapters handle display. This means the same reasoning core works in a terminal today and a Jupyter notebook tomorrow without changing any ML logic.

### I/O Callbacks (the key abstraction)

| Callback | CLI behavior | Notebook behavior (future) |
|---|---|---|
| `on_message(text)` | Rich panel | Cell output |
| `on_ask_user(question, options)` | stdin prompt | ipywidgets |
| `on_plot(path)` | Print file path | `IPython.display.Image` |
| `on_tool_call(name, input)` | Styled tool panel | Cell annotation |
| `on_finish(summary, model_path)` | Summary panel | Cell output |

---

## Project Structure

```
mlsherlock/
├── pyproject.toml              # Package definition and entry point
├── requirements.txt
├── .env.example                # Copy to .env and add your API key
│
├── mlsherlock/
│   ├── cli.py                  # Click entry point: `mlsh train`
│   │
│   ├── engine/
│   │   ├── loop.py             # Agent loop: conversation history + tool dispatch
│   │   ├── system_prompt.py    # ML expert persona + diagnostic heuristics
│   │   ├── state.py            # AgentState: iteration counter, stuck detection
│   │   └── callbacks.py        # BaseCallbacks ABC — the I/O interface
│   │
│   ├── tools/
│   │   ├── registry.py         # Pydantic schemas → JSON tool defs + dispatcher
│   │   ├── run_python.py       # exec() in shared sandbox, stdout capture, timeout
│   │   ├── read_data.py        # CSV profiler: shape, nulls, dtypes, class balance
│   │   ├── ask_user.py         # Delegates to callbacks.on_ask_user()
│   │   ├── save_plot.py        # Saves current matplotlib figure to output_dir
│   │   └── finish.py           # Saves model.pkl, writes summary, exits loop
│   │
│   ├── execution/
│   │   ├── sandbox.py          # Persistent globals dict, plt.show no-op, timeout
│   │   └── capture.py          # ExecutionCapture context manager (stdout/stderr)
│   │
│   ├── adapters/
│   │   ├── cli_callbacks.py    # Rich terminal output
│   │   └── notebook_callbacks.py  # Future: ipywidgets
│   │
│   └── utils/
│       └── display.py          # Rich formatting helpers
│
└── tests/
    ├── conftest.py             # Shared fixtures: FakeCallbacks, executor, state
    ├── fixtures/
    │   ├── sample.csv          # 300-row synthetic classification dataset
    │   └── generate_sample.py  # Script to regenerate sample.csv
    ├── test_tools/
    └── test_engine/
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 2. Install the package in editable mode

```bash
pip install -e .
```

This installs all dependencies and registers the `mlsherlock` CLI command. Editable mode means your code changes take effect immediately without reinstalling.

### 3. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Or export it directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running

### Basic usage

```bash
mlsh train \
  --data path/to/data.csv \
  --target target_column_name \
  --task classification
```

### All options

```
mlsh train --help

Options:
  --data TEXT               Path to the CSV dataset.           [required]
  --target TEXT             Target column name.                [required]
  --task [classification|regression]
                            ML task type.          [default: classification]
  --output-dir TEXT         Directory for model and plots.  [default: ./output]
  --max-iterations INTEGER  Max agent iterations.           [default: 20]
  --non-interactive         Auto-approve all ask_user calls (picks first option).
```

### Smoke test (non-interactive, uses sample data)

```bash
mlsh train \
  --data tests/fixtures/sample.csv \
  --target target \
  --task classification \
  --non-interactive
```

### Expected outputs

After a run, `./output/` contains:
- `model.pkl` — the trained model (joblib format)
- `summary.txt` — final summary written by the agent
- `*.png` — any plots the agent saved (ROC curve, calibration curve, feature importances, etc.)

Load the model later:

```python
import joblib
model = joblib.load("output/model.pkl")
predictions = model.predict(X_new)
```

---

## Running Tests

```bash
pytest tests/
```

Tests cover the sandbox, tools, and state — no API calls are made (the agent loop itself is not integration-tested to avoid API costs).

```bash
pytest tests/ -v          # verbose
pytest tests/test_tools/  # just tool tests
```

---

## How the Agent Thinks

The system prompt in `engine/system_prompt.py` encodes the diagnostic protocol:

| Observation | Diagnosis | First fix |
|---|---|---|
| train >> test by >0.05 | Overfitting | Increase regularization |
| Both scores low | Underfitting | More complex model, check encodings |
| Test accuracy >0.99 | Likely data leakage | Inspect feature names |
| Minority class <15% | Class imbalance | Ask user: class_weight vs SMOTE |
| Probabilities squeezed | Miscalibration | CalibratedClassifierCV |

Stopping criteria (the agent calls `finish` when both are true):
1. Improvement over the last 2 iterations is <0.005 on the primary metric
2. No unresolved warnings

---

## How to Extend

### Add a new tool

1. Create `mlsherlock/tools/my_tool.py` with a `run(...)` function that matches the signature pattern of existing tools.
2. Add a Pydantic input schema and register it in `tools/registry.py` — add to `get_tool_schemas()` and `dispatch()`.
3. Describe it in the system prompt so the agent knows when to use it.

### Change the diagnostic logic

Edit `engine/system_prompt.py`. The system prompt is plain text — the diagnostic heuristics, stopping criteria, and tool usage guidelines all live there.

### Add a notebook adapter

1. Implement `adapters/notebook_callbacks.py` — inherit from `BaseCallbacks` and implement all six methods using `IPython.display` and `ipywidgets`.
2. Instantiate `AgentLoop(state, NotebookCallbacks(), executor)` from a notebook cell.

### Change the model

The agent uses `claude-sonnet-4-6` (defined as `_MODEL` in `engine/loop.py`). To use a different model, change that constant.

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Code execution | `exec()` + shared globals | State (trained models, DataFrames) persists across calls |
| Tool dispatch | Sequential, not parallel | Shared exec context has dependencies |
| Context trimming | Drop oldest tool results at 80% limit | Keeps recent reasoning intact |
| Error recovery | Return tracebacks as tool result | Agent self-corrects by reading the error |
| Stuck detection | Inject hint after same error 3× in a row | Breaks repetition loops |
| Reproducibility | `random_state=42` + `np.random.seed(42)` | Reproducible ML across runs |
| Plot handling | `plt.show` → no-op; always `save_plot` | Prevents blocking the CLI |
| Large files | Profile first 100K rows, warn on >500MB | Prevents OOM |
