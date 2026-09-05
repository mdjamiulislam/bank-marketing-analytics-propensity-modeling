import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Locate project folders
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

raw_file = project_root / "data" / "raw" / "bank-full.csv"
reports_folder = project_root / "reports"

reports_folder.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. Load raw dataset
# --------------------------------------------------

df = pd.read_csv(raw_file, sep=";")

print("=" * 75)
print("PROJECT 4 - STRUCTURED DATA QUALITY AUDIT")
print("=" * 75)

print("\nDataset shape:", df.shape)

# --------------------------------------------------
# 3. Basic column-level quality audit
# --------------------------------------------------

audit_rows = []

for column in df.columns:

    missing_count = df[column].isna().sum()
    missing_pct = (missing_count / len(df)) * 100

    unique_count = df[column].nunique(dropna=False)

    # Count 'unknown' only for text columns
    if df[column].dtype == "object":
        unknown_count = (
            df[column]
            .astype(str)
            .str.lower()
            .eq("unknown")
            .sum()
        )
    else:
        unknown_count = 0

    unknown_pct = (unknown_count / len(df)) * 100

    audit_rows.append({
        "column": column,
        "data_type": str(df[column].dtype),
        "missing_count": missing_count,
        "missing_pct": round(missing_pct, 2),
        "unknown_count": unknown_count,
        "unknown_pct": round(unknown_pct, 2),
        "unique_values": unique_count
    })

audit_df = pd.DataFrame(audit_rows)

print("\n1. COLUMN QUALITY SUMMARY")
print("-" * 75)

print(audit_df.to_string(index=False))

# Save report
audit_df.to_csv(
    reports_folder / "03_column_quality_summary.csv",
    index=False
)

# --------------------------------------------------
# 4. Duplicate records
# --------------------------------------------------

duplicate_count = df.duplicated().sum()
duplicate_pct = duplicate_count / len(df) * 100

print("\n2. DUPLICATE RECORDS")
print("-" * 75)

print("Duplicate rows:", duplicate_count)
print("Duplicate percentage:", round(duplicate_pct, 2), "%")

# --------------------------------------------------
# 5. Numerical range checks
# --------------------------------------------------

print("\n3. NUMERICAL RANGE CHECKS")
print("-" * 75)

numeric_columns = df.select_dtypes(include=np.number).columns

range_summary = []

for column in numeric_columns:

    range_summary.append({
        "column": column,
        "minimum": df[column].min(),
        "maximum": df[column].max(),
        "mean": round(df[column].mean(), 2),
        "median": round(df[column].median(), 2)
    })

range_df = pd.DataFrame(range_summary)

print(range_df.to_string(index=False))

range_df.to_csv(
    reports_folder / "03_numeric_range_summary.csv",
    index=False
)

# --------------------------------------------------
# 6. Business-rule checks
# --------------------------------------------------

print("\n4. BUSINESS-RULE VALIDATION")
print("-" * 75)

checks = {
    "Age below 18": (df["age"] < 18).sum(),
    "Age above 100": (df["age"] > 100).sum(),

    "Day below 1": (df["day"] < 1).sum(),
    "Day above 31": (df["day"] > 31).sum(),

    "Negative duration": (df["duration"] < 0).sum(),

    "Campaign contacts below 1":
        (df["campaign"] < 1).sum(),

    "pdays below -1":
        (df["pdays"] < -1).sum(),

    "Negative previous contacts":
        (df["previous"] < 0).sum()
}

for check, count in checks.items():
    print(f"{check}: {count}")

# --------------------------------------------------
# 7. Special pdays check
# --------------------------------------------------

print("\n5. PDAYS SPECIAL VALUE CHECK")
print("-" * 75)

never_contacted = (df["pdays"] == -1).sum()

print("Records where pdays = -1:", never_contacted)

print(
    "Percentage:",
    round(never_contacted / len(df) * 100, 2),
    "%"
)

# --------------------------------------------------
# 8. Zero-duration contacts
# --------------------------------------------------

print("\n6. CONTACT DURATION CHECK")
print("-" * 75)

zero_duration = (df["duration"] == 0).sum()

print("Records with duration = 0:", zero_duration)

# --------------------------------------------------
# 9. Categorical value distributions
# --------------------------------------------------

print("\n7. CATEGORICAL VALUE AUDIT")
print("-" * 75)

categorical_columns = df.select_dtypes(
    include="object"
).columns

category_rows = []

for column in categorical_columns:

    counts = df[column].value_counts(dropna=False)

    print(f"\n--- {column.upper()} ---")
    print(counts)

    for value, count in counts.items():

        category_rows.append({
            "column": column,
            "category": value,
            "count": count,
            "percentage": round(
                count / len(df) * 100,
                2
            )
        })

category_df = pd.DataFrame(category_rows)

category_df.to_csv(
    reports_folder / "03_category_distribution.csv",
    index=False
)

# --------------------------------------------------
# 10. Rare category check
# --------------------------------------------------

print("\n8. RARE CATEGORIES (< 1%)")
print("-" * 75)

rare_categories = category_df[
    category_df["percentage"] < 1
]

if rare_categories.empty:
    print("No categories below 1%.")
else:
    print(
        rare_categories.to_string(index=False)
    )

# --------------------------------------------------
# 11. IQR outlier check
# --------------------------------------------------

print("\n9. NUMERIC OUTLIER AUDIT - IQR METHOD")
print("-" * 75)

outlier_rows = []

for column in numeric_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    outlier_rows.append({
        "column": column,
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": outlier_count,
        "outlier_pct": round(
            outlier_count / len(df) * 100,
            2
        )
    })

outlier_df = pd.DataFrame(outlier_rows)

print(outlier_df.to_string(index=False))

outlier_df.to_csv(
    reports_folder / "03_numeric_outlier_summary.csv",
    index=False
)

# --------------------------------------------------
# 12. Target-variable audit
# --------------------------------------------------

print("\n10. TARGET VARIABLE AUDIT")
print("-" * 75)

target_counts = df["y"].value_counts()

target_pct = (
    df["y"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nCounts:")
print(target_counts)

print("\nPercentages:")
print(target_pct)

# --------------------------------------------------
# 13. Final audit status
# --------------------------------------------------

print("\n" + "=" * 75)
print("DATA QUALITY AUDIT COMPLETE")
print("=" * 75)

print("\nReports saved to:")
print(reports_folder)