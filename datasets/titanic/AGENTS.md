# Dataset: Titanic Survival

## Description
The Titanic dataset contains passenger records from the RMS Titanic, which sank on April 15, 1912.
Each row is a passenger. The goal is to predict whether a passenger survived based on their
demographic information, ticket class, and cabin details. It is a classic ML benchmark with a
realistic mix of numeric features, categorical features, and missing values.

## Data Source
- **Type**: url
- **Location**: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
- **License**: Public domain / CC0

## Schema
| Column    | Type    | Description |
|-----------|---------|-------------|
| survived  | int     | **Target** — 1 = survived, 0 = did not survive |
| pclass    | int     | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| sex       | string  | Passenger sex |
| age       | float   | Age in years (177 nulls) |
| sibsp     | int     | Number of siblings/spouses aboard |
| parch     | int     | Number of parents/children aboard |
| fare      | float   | Passenger fare |
| embarked  | string  | Port of embarkation: C=Cherbourg, Q=Queenstown, S=Southampton (2 nulls) |
| deck      | string  | Cabin deck letter (688 nulls — highly sparse) |
| alone     | bool    | True if traveling alone |

## Target
- **Column**: `survived`
- **Task**: classification
- **Distribution**: 549 not survived (61.6%) / 342 survived (38.4%)
- **Notes**: Mild class imbalance. `class_weight='balanced'` is usually sufficient.

## Known Challenges
- `age`: 177 nulls (~20%) — requires imputation before modeling
- `deck`: 688 nulls (~77%) — too sparse to use raw; either drop or engineer a "has_cabin" binary
- `embarked`: 2 nulls — easy to impute with mode
- `sex` and `embarked` are categorical — need encoding
- `fare` is right-skewed — log transform helps linear models

## How to Get the Data

**Option 1 — use the agent's download_data tool:**
The agent can fetch it directly by name:
```
source: "titanic"
destination: "data/titanic.csv"
```

**Option 2 — use the demo script:**
```bash
python demo/get_data.py titanic
# Saves to demo/data/titanic.csv
```

**Option 3 — download manually:**
```bash
curl -o datasets/titanic/titanic.csv \
  https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
```

## Run the Agent

```bash
# Using the demo script output
mlsh train \
  --data demo/data/titanic.csv \
  --target survived \
  --task classification

# Or let the agent download it
mlsh train \
  --data titanic \
  --target survived \
  --task classification
```
