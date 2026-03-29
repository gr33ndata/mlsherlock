# mlsherlock — Agent Guide

An intelligent ML agent CLI that reasons like a senior ML engineer — diagnoses overfitting, calibration issues, class imbalance, data leakage, and feature importance problems — then applies targeted fixes.

Not AutoML. The agent uses an LLM to understand what is wrong and apply the minimum effective fix.

---

## Code Style (read this before touching anything)

### Philosophy
- **Less is more.** Deleting lines is rewarded. Every abstraction must earn its place.
- **No speculative code.** No placeholder classes, future adapters, or extension hooks. If it isn't used today, it doesn't exist.
- **No pretentious conventions.** Code reads naturally without artificial privacy signals.

### Naming
- No underscore-prefixed names on functions, methods, or module constants. `MAX_OUTPUT_CHARS`, not `_MAX_OUTPUT_CHARS`. `NAMED_DATASETS`, not `_NAMED_DATASETS`.
- Exception: dataclass private fields (`_consecutive_errors`) only if truly internal to a class with no legitimate external access path.
- No `_Private`, `__mangled`, or `# private` comments to signal intent. Write code that doesn't need those signals.

### Imports
- All imports at the top of the file — no imports inside functions.
- Exception: imports with side effects at load time that would break things (e.g. `kaggle` calls `exit(1)` if credentials are missing).
- No `from __future__ import annotations` — this project targets Python 3.11+.
- No `TYPE_CHECKING` guards. If a type annotation requires an import, either import it normally or drop the annotation. Internal tool functions don't need type annotations.

### Abstractions
- No base classes or ABCs unless there are at least two real implementations with shared logic.
- No wrapper functions that just return a constant (`get_system_prompt()` → just use `SYSTEM_PROMPT`).
- No `Optional[X]` — use `X | None` (Python 3.10+).
- No unused imports (`field` from dataclasses if no `field()` calls, etc.).

### Dead code
- Remove it immediately. Unused constants, unreferenced variables, functions that only raise `NotImplementedError`.
- Don't duplicate data. If `NAMED_DATASETS` lives in `download_data.py`, import it in `cli.py` — don't redefine it.

### Dependencies
- If a package is used, put it in `requirements.txt` and import it normally.
- No verbose `ImportError` handlers telling users to `pip install` — that's what requirements.txt is for.

---

## Architecture

```
cli.py  →  AgentLoop  →  LLM provider (Anthropic / OpenAI)
                ↓
           tool dispatch  (registry.py)
                ↓
        CodeExecutor (sandbox)
```

### Key files

| File | Role |
|---|---|
| `cli.py` | Entry point. Parses flags, infers target/task, starts loop. |
| `engine/loop.py` | Conversation loop: LLM call → tool dispatch → history management |
| `engine/providers.py` | Anthropic + OpenAI wrappers with a normalised response format |
| `engine/state.py` | Session state: paths, iteration count, finished flag, stuck detection |
| `engine/system_prompt.py` | ML diagnostic protocol — heuristics and stopping criteria |
| `tools/registry.py` | Pydantic input schemas (for JSON schema gen) + dispatcher |
| `tools/*.py` | One tool per file. Each returns a plain string result. |
| `execution/sandbox.py` | `CodeExecutor` — persistent `exec` namespace, timeout isolation |
| `execution/capture.py` | Context manager that captures stdout/stderr during exec |
| `ui/cli_callbacks.py` | Rich terminal output. Verbose and terse modes. |

### Tools available to the agent

| Tool | What it does |
|---|---|
| `read_data` | Loads CSV, profiles shape/nulls/dtypes/class balance, injects `df` + `target` into sandbox |
| `run_python` | Executes Python in the persistent sandbox |
| `download_data` | Fetches dataset from named shortcut, URL, or Kaggle slug |
| `save_plot` | Saves current matplotlib figure to output dir |
| `ask_user` | Pauses for human input |
| `finish` | Saves model with joblib, writes summary, ends loop |

---

## Running

```bash
mlsh train --data titanic --target survived
mlsh train --data path/to/data.csv --target price --task regression
mlsh train --data owner/dataset --target label   # Kaggle
mlsh train --data titanic --target survived --non-interactive
```

## Tests

```bash
pytest tests/
```

Tests use a synthetic fixture (`tests/fixtures/sample.csv`). No API calls are made.
