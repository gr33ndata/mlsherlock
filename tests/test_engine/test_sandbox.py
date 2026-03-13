"""Tests for the code execution sandbox."""
from mlsherlock.execution.sandbox import CodeExecutor
from mlsherlock.execution.capture import ExecutionCapture


def test_capture_stdout():
    with ExecutionCapture() as cap:
        print("hello")
    assert "hello" in cap.stdout


def test_capture_stderr():
    import sys
    with ExecutionCapture() as cap:
        print("err", file=sys.stderr)
    assert "err" in cap.stderr


def test_executor_basic():
    ex = CodeExecutor()
    out, err = ex.execute("print('works')")
    assert "works" in out
    assert err == ""


def test_executor_error():
    ex = CodeExecutor()
    out, err = ex.execute("raise RuntimeError('boom')")
    assert "RuntimeError" in err


def test_executor_persistence():
    ex = CodeExecutor()
    ex.execute("my_var = 99")
    out, err = ex.execute("print(my_var)")
    assert "99" in out


def test_numpy_seeded():
    ex = CodeExecutor()
    out, _ = ex.execute("import numpy as np; np.random.seed(42); print(np.random.rand())")
    assert out.strip() != ""


def test_plt_show_noop():
    ex = CodeExecutor()
    # plt.show() should not block or error
    out, err = ex.execute("plt.plot([1,2,3]); plt.show(); print('ok')")
    assert "ok" in out
    assert err == ""
