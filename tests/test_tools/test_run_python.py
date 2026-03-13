"""Tests for run_python tool."""
import pytest
from mlsherlock.tools import run_python


def test_basic_output(executor, state, callbacks):
    result = run_python.run("print('hello world')", state, executor, callbacks)
    assert "hello world" in result


def test_variable_persistence(executor, state, callbacks):
    run_python.run("x = 42", state, executor, callbacks)
    result = run_python.run("print(x)", state, executor, callbacks)
    assert "42" in result


def test_syntax_error_returned_not_raised(executor, state, callbacks):
    result = run_python.run("def bad(:", state, executor, callbacks)
    assert "error" in result.lower()


def test_runtime_error_returned(executor, state, callbacks):
    result = run_python.run("1/0", state, executor, callbacks)
    assert "ZeroDivisionError" in result or "error" in result.lower()


def test_output_truncation(executor, state, callbacks):
    # Generate output larger than _MAX_OUTPUT_CHARS
    result = run_python.run(
        "print('x' * 9000)",
        state,
        executor,
        callbacks,
    )
    assert "truncated" in result or len(result) <= 8100


def test_error_recorded_in_state(executor, state, callbacks):
    run_python.run("raise ValueError('test error')", state, executor, callbacks)
    assert state._consecutive_same_error >= 1
