# Dataset: Housing Prices (Synthetic)

## Description
A synthetic Boston-style housing dataset. Each row is a property. The goal is to predict
the `price` (in $10K units) from structural and neighborhood features.
The dataset is generated with `demo/get_data.py` using a fixed random seed (42) so it is
fully reproducible. It is intentionally simple enough to demonstrate regression diagnostics
without requiring a Kaggle account.

## Data Source
- **Type**: synthetic (generated locally)
- **Location**: run `python demo/get_data.py housing` to create `demo/data/housing.csv`
- **License**: N/A (synthetic)

## Schema
| Column              | Type  | Description |
|---------------------|-------|-------------|
| rooms               | float | Average number of rooms per dwelling (~15 nulls) |
| age                 | float | Proportion of owner-occupied units built prior to 1940 (~15 nulls) |
| distance_to_center  | float | Weighted distances to employment centers |
| crime_rate          | float | Per-capita crime rate by town (~15 nulls) |
| tax_rate            | int   | Full-value property-tax rate per $10,000 |
| pupil_teacher_ratio | float | Pupil-teacher ratio by town |
| low_income_pct      | float | % lower-status population |
| nitric_oxide        | float | Nitric oxides concentration (parts per 10M) |
| price               | float | **Target** — median home value in $10K |

## Target
- **Column**: `price`
- **Task**: regression
- **Distribution**: min=5, mean=8.4, max=29.4 (right-skewed, clipped at 5 and 50)
- **Notes**: Distribution is left-skewed due to clipping at $50K. Log transform of target often helps.

## Known Challenges
- `rooms`, `age`, `crime_rate` each have ~3% nulls — impute before modeling
- `price` is right-skewed — consider log(price) as target for linear models
- `distance_to_center` and `nitric_oxide` are correlated with `price` non-linearly

## How to Get the Data

```bash
python demo/get_data.py housing
# Saves to demo/data/housing.csv
```

## Run the Agent

```bash
mlsh train \
  --data demo/data/housing.csv \
  --target price \
  --task regression
```
