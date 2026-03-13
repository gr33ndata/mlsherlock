"""Tests for AgentState."""
from mlsherlock.engine.state import AgentState


def test_initial_state():
    s = AgentState(data_path="foo.csv", target_column="y")
    assert s.iteration == 0
    assert not s.finished
    assert not s.is_stuck


def test_stuck_detection():
    s = AgentState()
    error = "ValueError: something broke"
    s.record_error(error)
    s.record_error(error)
    assert not s.is_stuck
    s.record_error(error)
    assert s.is_stuck


def test_different_errors_not_stuck():
    s = AgentState()
    s.record_error("error A")
    s.record_error("error B")
    s.record_error("error A")
    assert not s.is_stuck
