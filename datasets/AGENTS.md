# Example Datasets

Each subdirectory is a self-contained dataset definition. It contains an `AGENTS.md` describing the data,
how to obtain it, and the recommended mlsherlock command to run against it.

## Available Datasets

| Dataset | Task | Rows | Notes |
|---|---|---|---|
| [titanic](./titanic/) | Classification | 891 | Survival prediction, mixed types, nulls, moderate imbalance |
| [housing](./housing/) | Regression | 506 | Price prediction, numeric features, synthetic |
| [penguins](./penguins/) | Classification | 344 | Species classification, 3 classes, some nulls |

## Adding a New Dataset

Create a new subdirectory and add an `AGENTS.md` file following the template below.
The data itself can be a CSV file committed to the repo, a download URL, a Kaggle slug,
a database connection string, or a Spark table path — whatever the `download_data` tool
(or a future data-source adapter) knows how to handle.

```
datasets/
└── my_dataset/
    ├── AGENTS.md        ← required: description + how to get data + agent command
    └── data.csv         ← optional: include small CSVs directly; use a URL for large ones
```

### AGENTS.md template

```markdown
# Dataset: <name>

## Description
One paragraph describing the domain, what each row represents, and what makes this dataset interesting.

## Data Source
- **Type**: csv | url | kaggle | database | spark
- **Location**: path/to/file.csv | https://... | owner/dataset-name | connection details
- **License**: e.g. CC0, CC BY 4.0, Proprietary

## Schema
| Column | Type | Description |
|--------|------|-------------|
| col1   | int  | ...         |

## Target
- **Column**: `target_column_name`
- **Task**: classification | regression
- **Notes**: class distribution, known imbalance, etc.

## Known Challenges
- Nulls in X, Y, Z columns
- Class imbalance (minority is N%)
- Etc.

## How to Get the Data
<instructions or command>

## Run the Agent
\`\`\`bash
mlsh train --data <path_or_source> --target <column> --task <classification|regression>
\`\`\`
```
