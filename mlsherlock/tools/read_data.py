"""Tool: read_data — profile a CSV and inject the DataFrame into the sandbox."""

import os

MAX_PROFILE_ROWS = 100_000
WARN_SIZE_MB = 500


def run(path, target_column, state, executor, callbacks) -> str:
    if not os.path.exists(path):
        return f"[error] File not found: {path}"

    size_mb = os.path.getsize(path) / (1024 ** 2)
    size_warn = f"\n[warning] File is {size_mb:.1f} MB (>500 MB), profiling first {MAX_PROFILE_ROWS} rows only." if size_mb > WARN_SIZE_MB else ""

    code = f"""
import pandas as pd
import numpy as np

_path = {path!r}
_target = {target_column!r}
_max_rows = {MAX_PROFILE_ROWS}

df = pd.read_csv(_path, nrows=_max_rows)
target = _target

# Basic profile
_shape = df.shape
_dtypes = df.dtypes.to_dict()
_nulls = df.isnull().sum()
_null_pct = (df.isnull().mean() * 100).round(2)

print(f"Shape: {{_shape[0]}} rows x {{_shape[1]}} columns")
print(f"Target column: {{target!r}}")
print()

# Dtypes
print("=== Column dtypes ===")
for col, dt in _dtypes.items():
    null_count = _nulls[col]
    null_pct = _null_pct[col]
    marker = " [HIGH NULL]" if null_pct > 20 else (" [has nulls]" if null_count > 0 else "")
    print(f"  {{col}}: {{dt}}  — {{null_count}} nulls ({{null_pct}}%){{marker}}")

print()

# Class balance (classification tasks)
if _target in df.columns:
    _vc = df[_target].value_counts(normalize=True).mul(100).round(2)
    _counts = df[_target].value_counts()
    print(f"=== Target distribution ({{_target!r}}) ===")
    for label, pct in _vc.items():
        cnt = _counts[label]
        imbalance_flag = " [IMBALANCED <15%]" if pct < 15 else ""
        print(f"  {{label}}: {{cnt}} ({{pct:.2f}}%){{imbalance_flag}}")
    _minority_pct = _vc.min()
    if _minority_pct < 15:
        print(f"  [!] Minority class is {{_minority_pct:.2f}}% — consider class_weight or SMOTE")
    print()

# Numeric stats
_num_cols = df.select_dtypes(include='number').columns.tolist()
if _num_cols:
    print("=== Numeric summary ===")
    print(df[_num_cols].describe().to_string())
    print()

print("=== First 5 rows ===")
print(df.head(5).to_string())
"""

    output, error = executor.execute(code)
    if error:
        return f"[read_data error]\n{error}"

    state.data_path = path
    state.target_column = target_column

    return (output or "(no output)") + size_warn
