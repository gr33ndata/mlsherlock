"""Tool: run_python — execute arbitrary Python in the sandbox."""

MAX_OUTPUT_CHARS = 8_000


def run(code, state, executor, callbacks) -> str:
    output, error = executor.execute(code)
    combined = (output or "") + (f"\n[error]\n{error}" if error else "")
    if len(combined) > MAX_OUTPUT_CHARS:
        combined = combined[:MAX_OUTPUT_CHARS] + f"\n... [truncated at {MAX_OUTPUT_CHARS} chars]"
    if error:
        state.record_error(error)
        return f"[execution error]\n{combined}"
    return combined or "(no output)"
