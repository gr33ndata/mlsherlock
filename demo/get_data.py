"""
Fetch demo datasets into demo/data/.

Usage:
    python demo/get_data.py              # downloads all datasets
    python demo/get_data.py titanic      # download one dataset by name
    python demo/get_data.py --list       # show available datasets

Datasets:
    titanic     891 rows, classification (survived), mix of numeric + categorical + nulls
    housing     506 rows, regression (price), Boston-style housing features
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Dataset definitions ──────────────────────────────────────────────────────

DATASETS: dict[str, dict] = {
    "titanic": {
        "description": "Titanic survival — classification, mix of numeric/categorical/nulls",
        "target": "survived",
        "task": "classification",
        "source": "url",
        "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        "filename": "titanic.csv",
        # Columns to drop that are too leaky or not useful for a demo
        "drop_columns": ["alive", "who", "adult_male", "embark_town", "class"],
    },
    "housing": {
        "description": "Housing prices — regression, numeric features",
        "target": "price",
        "task": "regression",
        "source": "synthetic",
        "filename": "housing.csv",
    },
}


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_titanic(cfg: dict) -> pd.DataFrame:
    print(f"  Downloading from {cfg['url']} ...")
    tmp = os.path.join(DATA_DIR, "_tmp_titanic.csv")
    urllib.request.urlretrieve(cfg["url"], tmp)
    df = pd.read_csv(tmp)
    os.remove(tmp)

    # Drop columns that would make the task trivially easy or leak the target
    df = df.drop(columns=[c for c in cfg.get("drop_columns", []) if c in df.columns])
    print(f"  Shape after cleanup: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Target distribution:\n{df['survived'].value_counts().to_string()}")
    print(f"  Nulls:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    return df


def make_housing(cfg: dict) -> pd.DataFrame:
    """Synthetic housing dataset: regression target 'price'."""
    print("  Generating synthetic housing dataset ...")
    rng = np.random.default_rng(42)
    n = 506

    rooms = rng.normal(6, 1.5, n).clip(2, 12)
    age = rng.uniform(5, 80, n)
    distance = rng.exponential(5, n).clip(0.5, 20)
    crime_rate = rng.exponential(3, n).clip(0, 30)
    tax_rate = rng.normal(300, 100, n).clip(100, 700)
    pupil_teacher = rng.normal(18, 2, n).clip(12, 25)
    low_income_pct = rng.beta(2, 5, n) * 40
    nitric_oxide = rng.normal(0.55, 0.12, n).clip(0.3, 0.9)

    price = (
        20
        + 4.5 * rooms
        - 0.12 * age
        - 1.2 * distance
        - 0.15 * crime_rate
        - 0.02 * tax_rate
        - 0.8 * pupil_teacher
        - 0.5 * low_income_pct
        - 10 * nitric_oxide
        + rng.normal(0, 3, n)  # noise
    ).clip(5, 50)

    df = pd.DataFrame(
        {
            "rooms": rooms.round(2),
            "age": age.round(1),
            "distance_to_center": distance.round(2),
            "crime_rate": crime_rate.round(2),
            "tax_rate": tax_rate.round(0).astype(int),
            "pupil_teacher_ratio": pupil_teacher.round(1),
            "low_income_pct": low_income_pct.round(2),
            "nitric_oxide": nitric_oxide.round(4),
            "price": price.round(2),
        }
    )
    # Introduce a few nulls to make it realistic
    for col in ["age", "rooms", "crime_rate"]:
        idx = rng.choice(n, size=int(n * 0.03), replace=False)
        df.loc[idx, col] = np.nan

    print(f"  Shape: {df.shape}")
    print(f"  Target stats:\n{df['price'].describe().round(2).to_string()}")
    print(f"  Nulls:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def download(name: str) -> None:
    cfg = DATASETS[name]
    print(f"\n[{name}] {cfg['description']}")
    out_path = os.path.join(DATA_DIR, cfg["filename"])

    if cfg["source"] == "url":
        df = fetch_titanic(cfg)
    elif cfg["source"] == "synthetic":
        df = make_housing(cfg)
    else:
        raise ValueError(f"Unknown source: {cfg['source']}")

    df.to_csv(out_path, index=False)
    print(f"  Saved to: {out_path}")
    print(f"\nTo run the agent on this dataset:")
    print(f"  mlsherlock train --data {out_path} --target {cfg['target']} --task {cfg['task']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", help="Dataset name (omit to download all)")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.list:
        print("Available datasets:")
        for name, cfg in DATASETS.items():
            print(f"  {name:<12} {cfg['description']}")
        return

    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset!r}. Use --list to see options.")
            sys.exit(1)
        download(args.dataset)
    else:
        for name in DATASETS:
            download(name)

    print("\nDone.")


if __name__ == "__main__":
    main()
