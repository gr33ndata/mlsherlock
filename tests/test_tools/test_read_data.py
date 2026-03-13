"""Tests for read_data tool."""
import pytest
from mlsherlock.tools import read_data
from tests.conftest import SAMPLE_CSV


def test_profiles_csv(executor, state, callbacks):
    result = read_data.run(SAMPLE_CSV, "target", state, executor, callbacks)
    assert "Shape" in result
    assert "target" in result


def test_injects_df_into_sandbox(executor, state, callbacks):
    read_data.run(SAMPLE_CSV, "target", state, executor, callbacks)
    from mlsherlock.tools import run_python
    result = run_python.run("print(df.shape)", state, executor, callbacks)
    assert "300" in result


def test_state_updated(executor, state, callbacks):
    read_data.run(SAMPLE_CSV, "target", state, executor, callbacks)
    assert state.data_path == SAMPLE_CSV
    assert state.target_column == "target"


def test_missing_file(executor, state, callbacks):
    result = read_data.run("/nonexistent/path.csv", "target", state, executor, callbacks)
    assert "error" in result.lower() or "not found" in result.lower()


def test_class_balance_shown(executor, state, callbacks):
    result = read_data.run(SAMPLE_CSV, "target", state, executor, callbacks)
    assert "Target distribution" in result or "target" in result.lower()
