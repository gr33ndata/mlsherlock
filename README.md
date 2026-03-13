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

## Project structure

```
mlsherlock/
├── mlsherlock/
│   ├── cli.py              # Entry point: mlsh train
│   ├── engine/             # Agent loop, system prompt, state
│   ├── tools/              # read_data, run_python, ask_user, save_plot, finish
│   ├── execution/          # Code sandbox (exec + shared globals)
│   └── adapters/           # CLI output (Rich); notebook adapter planned
├── tests/
├── datasets/               # AGENTS.md files describing how to fetch each dataset
├── demo/
│   └── get_data.py         # Script to download demo datasets
├── .env.example
└── pyproject.toml
```

---

## Contributing

This project is in early development. Feel free to open issues or PRs, but expect things to
move fast and break.

The diagnostic heuristics live in `mlsherlock/engine/system_prompt.py` — that's the best
place to start if you want to understand or extend the agent's reasoning.
