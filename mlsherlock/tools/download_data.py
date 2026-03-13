"""Tool: download_data — download a dataset from a URL or Kaggle into output_dir."""
from __future__ import annotations

import os
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlsherlock.engine.state import AgentState
    from mlsherlock.execution.sandbox import CodeExecutor
    from mlsherlock.engine.callbacks import BaseCallbacks

# Well-known named datasets resolvable without extra packages
_NAMED_DATASETS: dict[str, str] = {
    "titanic": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
    "iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "penguins": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
    "diamonds": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
    "tips": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
}


def run(
    source: str,
    destination: str,
    state: "AgentState",
    executor: "CodeExecutor",
    callbacks: "BaseCallbacks",
) -> str:
    """
    Download a dataset and save it as a CSV file.

    source can be:
      - A named dataset: "titanic", "iris", "penguins", "diamonds", "tips"
      - A direct HTTP/HTTPS URL to a CSV file
      - A Kaggle dataset slug: "username/dataset-name" (requires kaggle package + ~/.kaggle/kaggle.json)
    destination:
      - Local file path where the CSV will be saved (e.g. "data/titanic.csv")
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True) if os.path.dirname(destination) else None

    # Resolve named datasets
    url: str | None = None
    if source in _NAMED_DATASETS:
        url = _NAMED_DATASETS[source]
    elif source.startswith("http://") or source.startswith("https://"):
        url = source

    if url:
        return _download_url(url, destination)

    # Kaggle dataset (format: "owner/dataset-name")
    if "/" in source and not source.startswith("http"):
        return _download_kaggle(source, destination)

    return (
        f"[error] Could not resolve source: {source!r}\n"
        f"Use a named dataset ({', '.join(_NAMED_DATASETS)}), "
        f"a direct HTTPS URL, or a Kaggle slug (owner/dataset-name)."
    )


def _download_url(url: str, destination: str) -> str:
    try:
        urllib.request.urlretrieve(url, destination)
        import pandas as pd
        df = pd.read_csv(destination)
        return (
            f"Downloaded to: {destination}\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"Columns: {df.columns.tolist()}"
        )
    except Exception as exc:
        return f"[download error] {exc}"


def _download_kaggle(slug: str, destination: str) -> str:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        return (
            "[error] The `kaggle` package is not installed.\n"
            "Install it with: pip install kaggle\n"
            "Then create ~/.kaggle/kaggle.json with your API credentials from https://kaggle.com/settings"
        )

    credentials_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(credentials_path):
        return (
            "[error] Kaggle credentials not found at ~/.kaggle/kaggle.json\n"
            "Create them at: https://kaggle.com/settings → API → Create New Token"
        )

    try:
        import tempfile
        import zipfile
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()

        with tempfile.TemporaryDirectory() as tmpdir:
            api.dataset_download_files(slug, path=tmpdir, unzip=False)
            zip_files = [f for f in os.listdir(tmpdir) if f.endswith(".zip")]
            if not zip_files:
                return "[error] No zip file found after Kaggle download."

            with zipfile.ZipFile(os.path.join(tmpdir, zip_files[0])) as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_files:
                    return f"[error] No CSV files found in Kaggle dataset. Contents: {z.namelist()}"

                # If multiple CSVs, pick the largest
                if len(csv_files) > 1:
                    sizes = {f: z.getinfo(f).file_size for f in csv_files}
                    csv_file = max(sizes, key=sizes.get)
                else:
                    csv_file = csv_files[0]

                with z.open(csv_file) as src, open(destination, "wb") as dst:
                    dst.write(src.read())

        import pandas as pd
        df = pd.read_csv(destination)
        return (
            f"Downloaded Kaggle dataset {slug!r} ({csv_file}) to: {destination}\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"Columns: {df.columns.tolist()}"
        )
    except Exception as exc:
        return f"[kaggle error] {exc}"
