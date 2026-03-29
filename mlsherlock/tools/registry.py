"""Tool registry: schemas (Pydantic → JSON) and dispatcher."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from mlsherlock.engine.state import AgentState
    from mlsherlock.execution.sandbox import CodeExecutor
    from mlsherlock.engine.callbacks import BaseCallbacks


# ── Input schemas ────────────────────────────────────────────────────────────

class RunPythonInput(BaseModel):
    code: str = Field(..., description="Python code to execute in the shared sandbox")


class ReadDataInput(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the CSV file")
    target_column: str = Field(..., description="Name of the target/label column")


class AskUserInput(BaseModel):
    question: str = Field(..., description="The question to present to the user")
    options: list[str] = Field(
        default_factory=list,
        description="Optional list of suggested answers (may be empty)"
    )


class SavePlotInput(BaseModel):
    filename: str = Field(..., description="Filename for the saved plot (e.g. 'roc_curve.png')")


class DownloadDataInput(BaseModel):
    source: str = Field(
        ...,
        description=(
            "Where to get the data. One of: "
            "(1) a named dataset: 'titanic', 'iris', 'penguins', 'diamonds', 'tips'; "
            "(2) a direct HTTPS URL to a CSV file; "
            "(3) a Kaggle dataset slug: 'owner/dataset-name' (requires kaggle package + credentials)."
        ),
    )
    destination: str = Field(
        ...,
        description="Local file path where the CSV will be saved (e.g. 'data/titanic.csv')"
    )


class FinishInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    summary: str = Field(..., description="Final markdown summary of what was done and results achieved")
    model_variable: str = Field(
        ...,
        description="Name of the Python variable in the sandbox that holds the trained model"
    )


# ── Schema export ─────────────────────────────────────────────────────────────

def get_tool_schemas() -> list[dict[str, Any]]:
    """Return Anthropic-compatible tool definitions for all tools."""
    tools = [
        (
            "run_python",
            "Execute Python code in a persistent shared sandbox. "
            "Variables, models, and DataFrames defined here are accessible in subsequent calls. "
            "Use print() to surface results. DataFrames are auto-truncated to 20 rows in output.",
            RunPythonInput,
        ),
        (
            "read_data",
            "Load a CSV file, profile its shape/dtypes/nulls/class balance, "
            "and inject `df` (DataFrame) and `target` (column name string) into the sandbox.",
            ReadDataInput,
        ),
        (
            "ask_user",
            "Ask the user a question when a decision requires human input "
            "(e.g. choosing between class_weight vs SMOTE for imbalanced data). "
            "Provide options when applicable.",
            AskUserInput,
        ),
        (
            "save_plot",
            "Save the current matplotlib figure to the output directory. "
            "Always call this instead of plt.show(). "
            "Call plt.figure() before plotting if you need a fresh canvas.",
            SavePlotInput,
        ),
        (
            "download_data",
            "Download a dataset from a named source, a direct URL, or Kaggle and save it as a CSV. "
            "Named datasets: 'titanic', 'iris', 'penguins', 'diamonds', 'tips'. "
            "For Kaggle: use 'owner/dataset-name' slug (requires kaggle package + ~/.kaggle/kaggle.json). "
            "After downloading, call read_data to load and profile it.",
            DownloadDataInput,
        ),
        (
            "finish",
            "Signal that the ML session is complete. "
            "Saves the trained model to output/model.pkl and writes a summary. "
            "Only call this when you are satisfied with results or have exhausted iterations.",
            FinishInput,
        ),
    ]

    schemas = []
    for name, description, model_cls in tools:
        schema = model_cls.model_json_schema()
        # Remove title added by Pydantic
        schema.pop("title", None)
        schemas.append(
            {
                "name": name,
                "description": description,
                "input_schema": schema,
            }
        )
    return schemas


# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch(
    name: str,
    tool_input: dict[str, Any],
    state: "AgentState",
    executor: "CodeExecutor",
    callbacks: "BaseCallbacks",
) -> str:
    """Route a tool call to its implementation and return the result string."""
    from mlsherlock.tools import run_python, read_data, ask_user, save_plot, finish, download_data

    if name == "run_python":
        parsed = RunPythonInput(**tool_input)
        return run_python.run(parsed.code, state, executor, callbacks)

    if name == "read_data":
        parsed = ReadDataInput(**tool_input)
        return read_data.run(parsed.path, parsed.target_column, state, executor, callbacks)

    if name == "ask_user":
        parsed = AskUserInput(**tool_input)
        return ask_user.run(parsed.question, parsed.options, callbacks)

    if name == "save_plot":
        parsed = SavePlotInput(**tool_input)
        return save_plot.run(parsed.filename, state, executor, callbacks)

    if name == "download_data":
        parsed = DownloadDataInput(**tool_input)
        return download_data.run(parsed.source, parsed.destination, state, callbacks)

    if name == "finish":
        parsed = FinishInput(**tool_input)
        return finish.run(parsed.summary, parsed.model_variable, state, executor, callbacks)

    return f"[error] Unknown tool: {name!r}"
