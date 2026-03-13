"""One-off script to regenerate tests/fixtures/sample.csv."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300

df = pd.DataFrame(
    {
        "age": rng.integers(20, 70, n),
        "income": rng.normal(50_000, 15_000, n).round(2),
        "credit_score": rng.integers(300, 850, n),
        "loan_amount": rng.normal(20_000, 8_000, n).round(2),
        "employment_years": rng.integers(0, 30, n),
        "num_products": rng.integers(1, 5, n),
        "has_mortgage": rng.integers(0, 2, n),
        "region": rng.choice(["north", "south", "east", "west"], n),
    }
)

# Target: default (0/1) — loosely correlated with income and credit score
log_odds = (
    -3
    + 0.03 * (600 - df["credit_score"])
    + 0.00003 * (40_000 - df["income"])
    + 0.01 * df["loan_amount"] / 1000
)
prob = 1 / (1 + np.exp(-log_odds))
df["target"] = (rng.uniform(size=n) < prob).astype(int)

df.to_csv("sample.csv", index=False)
print(f"Written sample.csv: {df.shape}, target distribution:\n{df['target'].value_counts()}")
