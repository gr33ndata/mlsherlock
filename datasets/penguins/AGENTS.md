# Dataset: Palmer Penguins

## Description
Measurements of 344 penguins from three species (Adelie, Chinstrap, Gentoo) collected from
three islands in the Palmer Archipelago, Antarctica. Each row is a penguin observation.
A great alternative to Iris — more interesting class structure, realistic nulls, and a mix of
numeric and categorical features.

## Data Source
- **Type**: url
- **Location**: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
- **License**: CC0 (originally from the `palmerpenguins` R package by Allison Horst et al.)

## Schema
| Column            | Type   | Description |
|-------------------|--------|-------------|
| species           | string | **Target** — Adelie / Chinstrap / Gentoo |
| island            | string | Island: Biscoe / Dream / Torgersen |
| bill_length_mm    | float  | Bill length in mm (2 nulls) |
| bill_depth_mm     | float  | Bill depth in mm (2 nulls) |
| flipper_length_mm | float  | Flipper length in mm (2 nulls) |
| body_mass_g       | float  | Body mass in grams (2 nulls) |
| sex               | string | Male / Female (11 nulls) |

## Target
- **Column**: `species`
- **Task**: classification (3 classes)
- **Distribution**: Adelie=152 (44%), Chinstrap=68 (20%), Gentoo=124 (36%)
- **Notes**: Multiclass. Chinstrap is the minority class at 20% — no severe imbalance.

## Known Challenges
- `sex`: 11 nulls (~3%) — impute or drop
- `bill_*` and `flipper_length_mm`: 2 nulls each — easy to impute with median
- `island` and `sex` are categorical — need encoding
- Multiclass target — agent should use macro-averaged F1 as the primary metric

## How to Get the Data

**Option 1 — use the agent's download_data tool:**
```
source: "penguins"
destination: "data/penguins.csv"
```

**Option 2 — download manually:**
```bash
curl -o datasets/penguins/penguins.csv \
  https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
```

## Run the Agent

```bash
mlsh train \
  --data penguins \
  --target species \
  --task classification
```
