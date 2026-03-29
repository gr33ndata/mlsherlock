# Code Review — 2026-03-29

Reviewed: `mlsherlock/cli.py`, `mlsherlock/engine/loop.py`, `mlsherlock/engine/providers.py`, `mlsherlock/engine/state.py`, `mlsherlock/engine/callbacks.py`, `mlsherlock/engine/system_prompt.py`, `mlsherlock/execution/sandbox.py`, `mlsherlock/execution/capture.py`, `mlsherlock/tools/ask_user.py`, `mlsherlock/tools/download_data.py`, `mlsherlock/tools/finish.py`, `mlsherlock/tools/read_data.py`, `mlsherlock/tools/registry.py`, `mlsherlock/tools/run_python.py`, `mlsherlock/tools/save_plot.py`, `mlsherlock/adapters/cli_callbacks.py`

## Issues to Fix

### 1. [Bug] OpenAI provider discards all but first text block in history
**File:** `mlsherlock/engine/providers.py:135`
**Problem:** `make_assistant_history_entry` stores only `response.text_blocks[0]` in the history entry `content` field. When the model produces multiple text segments (e.g. analysis followed by a decision), every block after the first is silently dropped from conversation history, corrupting the context sent on the next API call. The Anthropic provider correctly iterates all text blocks (line 68–69).
**Fix:** Replace `response.text_blocks[0] if response.text_blocks else None` with `" ".join(response.text_blocks) if response.text_blocks else None` (or keep them as a joined string to fit OpenAI's single-string `content` field).

---

### 2. [Security] Code injection via bare `model_variable` interpolation in generated code
**File:** `mlsherlock/tools/finish.py:28`
**Problem:** `model_variable` (an LLM-provided string) is interpolated raw into a code string: `_model_obj = {model_variable}`. If the LLM passes a value like `clf; import shutil; shutil.rmtree('/')`, that code executes in the sandbox which has full filesystem access. While the LLM already has `run_python`, this tool provides an unsanitised secondary execution path with no logging or visibility.
**Fix:** Validate that `model_variable` is a legal Python identifier before building the code string: `if not model_variable.isidentifier(): return f"[finish error] Invalid model variable name: {model_variable!r}"`. Then use it safely.

---

### 3. [Security] Path traversal in `download_data` destination parameter
**File:** `mlsherlock/tools/download_data.py:40, 63-65`
**Problem:** The `destination` path is not validated against the intended output directory. An LLM-controlled value like `"../../~/.ssh/authorized_keys"` would pass `os.path.abspath()` without rejection and cause `urlretrieve` to overwrite arbitrary files the process can write.
**Fix:** After resolving the absolute path, assert it stays within `state.output_dir`: `abs_dest = os.path.realpath(destination); if not abs_dest.startswith(os.path.realpath(state.output_dir)): return "[error] Destination must be inside the output directory."` Pass `state` as a used parameter (it's currently unused — see issue 8).


---


### 6. [Performance] Full JSON serialization of history on every iteration
**File:** `mlsherlock/engine/loop.py:139`
**Problem:** `_maybe_trim_history` is called every iteration and computes `sum(len(json.dumps(msg)) for msg in self._history)`. This serializes the entire history (potentially hundreds of KB) on every loop tick just to check a threshold. With 20 iterations and growing history, this is O(n²) in total work.
**Fix:** Track `_approx_history_chars` as a running integer on `self`. Increment it in `_process_response` when messages are appended (`+= len(json.dumps(entry))`), and decrement in `_maybe_trim_history` when messages are deleted. Replace the loop with a single attribute read.

---

### 7. [Performance] Orphaned threads continue running after timeout
**File:** `mlsherlock/execution/sandbox.py:96-101`
**Problem:** When `thread.join(timeout=timeout)` returns and the thread is still alive, the function returns the timeout error but the thread keeps executing in the background. Repeated timeouts (e.g. the agent trying the same slow code) stack up live threads sharing the same `self._globals`, causing concurrent mutations to sandbox state and consuming CPU/memory.
**Fix:** Set a `threading.Event` cancel flag and check it inside `_run` at safe points, or — simpler — use `multiprocessing` for hard isolation. At minimum, document that timed-out code may keep running and mutate state, and clear `self._globals` after a timeout to prevent stale state contamination.

---

### 8. [Quality] `_recent_errors` list is write-only dead code
**File:** `mlsherlock/engine/state.py:22, 35`
**Problem:** Every call to `record_error` appends the full traceback to `_recent_errors`, but the list is never read anywhere in the codebase. Only `_consecutive_same_error` and `_last_error_hash` are used for stuck-detection. The list grows without bound across the session, holding full tracebacks in memory for no purpose.
**Fix:** Delete the `_recent_errors` field and the `self._recent_errors.append(error)` line.

---

### 9. [Quality] Unused parameters in `download_data.run` and `ask_user.run`
**File:** `mlsherlock/tools/download_data.py:23-28`, `mlsherlock/tools/ask_user.py:13-17`
**Problem:** `download_data.run` accepts `state`, `executor`, and `callbacks` but uses none of them. `ask_user.run` accepts `state` and `executor` but uses neither. These are dead parameters that widen every call site for no reason. (Note: `state` should become used in `download_data` once issue 3 is fixed.)
**Fix:** For `ask_user`, remove `state` and `executor` from the signature and update the dispatch call in `registry.py:149`. For `download_data`, keep `state` (needed for issue 3 fix) but remove `executor`; update `registry.py:157` accordingly.

---

### 10. [Quality] Obscure side-effect ternary in `download_data.run`
**File:** `mlsherlock/tools/download_data.py:40`
**Problem:** `os.makedirs(...) if os.path.dirname(destination) else None` uses a ternary as a statement purely for its side effect, discarding the return value and appending a meaningless `else None`. This is harder to read than a plain `if` block.
**Fix:** Replace with `if os.path.dirname(destination): os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)`.

---

### 11. [Quality] `AgentLoop` directly mutates private attribute of `AgentState`
**File:** `mlsherlock/engine/loop.py:118`
**Problem:** `self._state._consecutive_same_error = 0` reaches into the private internals of `AgentState` from outside the class. `AgentState` already owns all error-tracking logic; this mutation should live there.
**Fix:** Add a `reset_stuck_detection()` method to `AgentState` that sets `self._consecutive_same_error = 0` and `self._last_error_hash = None`, then call `self._state.reset_stuck_detection()` from `loop.py`.

---

### 12. [Quality] `LLMProvider` base class doesn't use `ABC`
**File:** `mlsherlock/engine/providers.py:26-35`
**Problem:** `LLMProvider` manually raises `NotImplementedError` in three methods instead of using `abc.ABC` + `@abstractmethod`. Unlike `BaseCallbacks` (which correctly uses ABC), a subclass that forgets to implement a method will only fail at call time, not at instantiation.
**Fix:** Make `LLMProvider(ABC)` and decorate `call`, `make_assistant_history_entry`, and `make_tool_results_history_entries` with `@abstractmethod`. Add `from abc import ABC, abstractmethod` to the import block.

---

_Generated by /code_review. Run Claude Code and say "fix the issues in REVIEW.md" to apply._
