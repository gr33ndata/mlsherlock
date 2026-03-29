"""Tool: finish — save the model, write a summary, and signal loop exit."""

import os


def run(summary, model_variable, state, executor, callbacks) -> str:
    if not model_variable.isidentifier():
        return f"[finish error] Invalid model variable name: {model_variable!r}. Must be a plain Python identifier."

    os.makedirs(state.output_dir, exist_ok=True)
    model_path = os.path.join(state.output_dir, "model.pkl")

    save_code = f"""
import joblib
import os
os.makedirs({state.output_dir!r}, exist_ok=True)
_model_obj = {model_variable}
joblib.dump(_model_obj, {model_path!r})
print(f"Model saved to {model_path!r}")
"""
    output, error = executor.execute(save_code)
    if error:
        return f"[finish error] Could not save model '{model_variable}': {error}"

    summary_path = os.path.join(state.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    state.finished = True
    callbacks.on_finish(summary, model_path)
    return f"Done. Model saved to {model_path}. Summary written to {summary_path}."
