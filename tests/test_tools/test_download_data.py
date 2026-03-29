"""Tests for download_data tool."""
import os
import pytest
from mlsherlock.tools import download_data


def test_download_named_dataset(state, callbacks, tmp_path):
    dest = os.path.join(state.output_dir, "titanic.csv")
    result = download_data.run("titanic", dest, state, callbacks)
    assert os.path.exists(dest)
    assert "titanic.csv" in result or "Columns" in result
    assert "891" in result  # known Titanic row count


def test_download_direct_url(state, callbacks, tmp_path):
    dest = os.path.join(state.output_dir, "iris.csv")
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    result = download_data.run(url, dest, state, callbacks)
    assert os.path.exists(dest)
    assert "150" in result  # Iris has 150 rows


def test_unknown_source_returns_error(state, callbacks, tmp_path):
    dest = os.path.join(state.output_dir, "out.csv")
    result = download_data.run("not_a_thing", dest, state, callbacks)
    assert "error" in result.lower()


def test_kaggle_without_package_returns_helpful_error(state, callbacks, tmp_path, monkeypatch):
    # Hide kaggle package if installed
    import sys
    monkeypatch.setitem(sys.modules, "kaggle", None)
    dest = os.path.join(state.output_dir, "out.csv")
    result = download_data.run("someuser/some-dataset", dest, state, callbacks)
    assert "kaggle" in result.lower()
