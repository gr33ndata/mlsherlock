"""Tests for tool registry."""
from mlsherlock.tools.registry import get_tool_schemas, dispatch


def test_schema_count():
    schemas = get_tool_schemas()
    assert len(schemas) == 6


def test_schema_names():
    names = {s["name"] for s in get_tool_schemas()}
    assert names == {"run_python", "read_data", "ask_user", "save_plot", "finish", "download_data"}


def test_each_schema_has_input_schema():
    for schema in get_tool_schemas():
        assert "input_schema" in schema
        assert "properties" in schema["input_schema"]


def test_dispatch_unknown_tool(executor, state, callbacks):
    result = dispatch("not_a_tool", {}, state, executor, callbacks)
    assert "Unknown tool" in result


def test_dispatch_run_python(executor, state, callbacks):
    result = dispatch("run_python", {"code": "print(1+1)"}, state, executor, callbacks)
    assert "2" in result


def test_dispatch_ask_user_non_interactive(executor, state, callbacks):
    result = dispatch(
        "ask_user",
        {"question": "Which approach?", "options": ["option_a", "option_b"]},
        state,
        executor,
        callbacks,
    )
    assert "option_a" in result or "User answered" in result
