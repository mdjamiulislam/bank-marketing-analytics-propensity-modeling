import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Locate project folders
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

raw_file = project_root / "data" / "raw" / "bank-full.csv"
processed_folder = project_root / "data" / "processed"
reports_folder = project_root / "reports"

processed_folder.mkdir(exist_ok=True)
reports_folder.mkdir(exist_ok=True)

output_file = processed_folder / "bank_marketing_clean.csv"
summary_file = reports_folder / "04_cleaning_summary.csv"

# --------------------------------------------------
# 2. Load raw data
# --------------------------------------------------

df = pd.read_csv(raw_file, sep=";")

print("=" * 75)
print("PROJECT 4 - DATA CLEANING")
print("=" * 75)

print("\nRaw dataset loaded successfully.")
print("Raw shape:", df.shape)

raw_rows = len(df)
raw_columns = len(df.columns)

# --------------------------------------------------
# 3. Standardise column names
# --------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\n1. COLUMN NAMES STANDARDISED")
print("-" * 75)
print(df.columns.tolist())

# --------------------------------------------------
# 4. Standardise text values
# --------------------------------------------------

text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

print("\n2. TEXT VALUES STANDARDISED")
print("-" * 75)
print("Whitespace removed and categorical text converted to lowercase.")

# --------------------------------------------------
# 5. Remove exact duplicate records
# --------------------------------------------------

duplicates_before = df.duplicated().sum()

df = df.drop_duplicates().reset_index(drop=True)

duplicates_removed = raw_rows - len(df)

print("\n3. DUPLICATE RECORDS")
print("-" * 75)

print("Duplicates detected:", duplicates_before)
print("Duplicates removed:", duplicates_removed)
print("Rows after duplicate removal:", len(df))

# --------------------------------------------------
# 6. Validate target variable
# --------------------------------------------------

print("\n4. TARGET VARIABLE VALIDATION")
print("-" * 75)

valid_target_values = {"yes", "no"}

actual_target_values = set(df["y"].dropna().unique())

invalid_target_values = (
    actual_target_values - valid_target_values
)

if invalid_target_values:
    raise ValueError(
        f"Unexpected target values found: {invalid_target_values}"
    )

print("Valid target values confirmed:", actual_target_values)

# --------------------------------------------------
# 7. Rename target variable
# --------------------------------------------------

df = df.rename(
    columns={
        "y": "subscribed"
    }
)

print("\n5. TARGET VARIABLE RENAMED")
print("-" * 75)

print("y -> subscribed")

# --------------------------------------------------
# 8. Create binary subscription flag
# --------------------------------------------------

df["subscription_flag"] = (
    df["subscribed"]
    .map({
        "yes": 1,
        "no": 0
    })
    .astype("int8")
)

print("\n6. SUBSCRIPTION FLAG CREATED")
print("-" * 75)

print(
    df[
        ["subscribed", "subscription_flag"]
    ]
    .drop_duplicates()
)

# --------------------------------------------------
# 9. Create binary financial flags
# --------------------------------------------------

binary_mapping = {
    "yes": 1,
    "no": 0
}

df["default_flag"] = (
    df["default"]
    .map(binary_mapping)
    .astype("int8")
)

df["housing_loan_flag"] = (
    df["housing"]
    .map(binary_mapping)
    .astype("int8")
)

df["personal_loan_flag"] = (
    df["loan"]
    .map(binary_mapping)
    .astype("int8")
)

print("\n7. FINANCIAL BINARY FLAGS CREATED")
print("-" * 75)

print(
    df[
        [
            "default",
            "default_flag",
            "housing",
            "housing_loan_flag",
            "loan",
            "personal_loan_flag"
        ]
    ]
    .head()
)

# --------------------------------------------------
# 10. Handle the special meaning of pdays = -1
# --------------------------------------------------

df["previously_contacted"] = (
    df["pdays"] != -1
).astype("int8")

df["days_since_previous_contact"] = (
    df["pdays"]
    .where(df["pdays"] != -1, pd.NA)
    .astype("Int64")
)

print("\n8. PREVIOUS-CONTACT VARIABLES CREATED")
print("-" * 75)

print(
    df[
        [
            "pdays",
            "previously_contacted",
            "days_since_previous_contact"
        ]
    ]
    .head(10)
)

# --------------------------------------------------
# 11. Preserve 'unknown' categories
# --------------------------------------------------

print("\n9. UNKNOWN CATEGORY CHECK")
print("-" * 75)

unknown_summary = []

for column in df.select_dtypes(include="object").columns:

    unknown_count = (
        df[column] == "unknown"
    ).sum()

    if unknown_count > 0:

        unknown_summary.append({
            "column": column,
            "unknown_count": unknown_count,
            "unknown_pct": round(
                unknown_count / len(df) * 100,
                2
            )
        })

        print(
            f"{column}: "
            f"{unknown_count} "
            f"({unknown_count / len(df) * 100:.2f}%)"
        )

print("\nUnknown values are retained as valid categories.")

# --------------------------------------------------
# 12. Validate important numerical fields
# --------------------------------------------------

print("\n10. NUMERICAL VALIDATION")
print("-" * 75)

validation_checks = {
    "age_below_18": (df["age"] < 18).sum(),
    "age_above_100": (df["age"] > 100).sum(),
    "day_below_1": (df["day"] < 1).sum(),
    "day_above_31": (df["day"] > 31).sum(),
    "negative_duration": (df["duration"] < 0).sum(),
    "campaign_below_1": (df["campaign"] < 1).sum(),
    "pdays_below_minus_1": (df["pdays"] < -1).sum(),
    "negative_previous": (df["previous"] < 0).sum()
}

for check, count in validation_checks.items():
    print(f"{check}: {count}")

# --------------------------------------------------
# 13. Preserve negative balances
# --------------------------------------------------

negative_balance_count = (
    df["balance"] < 0
).sum()

print("\n11. NEGATIVE BALANCE CHECK")
print("-" * 75)

print(
    "Negative balance records retained:",
    negative_balance_count
)

# --------------------------------------------------
# 14. Add unique campaign record ID
# --------------------------------------------------

df.insert(
    0,
    "campaign_record_id",
    range(1, len(df) + 1)
)

print("\n12. RECORD ID CREATED")
print("-" * 75)

print(
    "campaign_record_id added from 1 to",
    len(df)
)

# --------------------------------------------------
# 15. Check missing values after cleaning
# --------------------------------------------------

print("\n13. MISSING VALUES AFTER CLEANING")
print("-" * 75)

missing_summary = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing_summary)

print(
    "\nNote: missing values in "
    "'days_since_previous_contact' are intentional."
)

# --------------------------------------------------
# 16. Validate subscription flag
# --------------------------------------------------

print("\n14. SUBSCRIPTION DISTRIBUTION")
print("-" * 75)

print(df["subscribed"].value_counts())

print("\nSubscription rate:")

subscription_rate = (
    df["subscription_flag"].mean() * 100
)

print(
    f"{subscription_rate:.2f}%"
)

# --------------------------------------------------
# 17. Final dataset information
# --------------------------------------------------

clean_rows = len(df)
clean_columns = len(df.columns)

print("\n15. FINAL CLEAN DATASET")
print("-" * 75)

print("Rows:", clean_rows)
print("Columns:", clean_columns)

print("\nColumns:")
for number, column in enumerate(
    df.columns,
    start=1
):
    print(number, column)

# --------------------------------------------------
# 18. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

print("\n16. CLEAN DATASET SAVED")
print("-" * 75)

print(output_file)

# --------------------------------------------------
# 19. Create cleaning summary
# --------------------------------------------------

cleaning_summary = pd.DataFrame({
    "metric": [
        "Raw rows",
        "Raw columns",
        "Duplicate rows detected",
        "Duplicate rows removed",
        "Clean rows",
        "Clean columns",
        "Negative balances retained",
        "Previously contacted records",
        "Never previously contacted records",
        "Subscription rate (%)"
    ],

    "value": [
        raw_rows,
        raw_columns,
        duplicates_before,
        duplicates_removed,
        clean_rows,
        clean_columns,
        negative_balance_count,
        int(df["previously_contacted"].sum()),
        int(
            (df["previously_contacted"] == 0)
            .sum()
        ),
        round(subscription_rate, 2)
    ]
})

cleaning_summary.to_csv(
    summary_file,
    index=False
)

print("\n17. CLEANING SUMMARY SAVED")
print("-" * 75)

print(summary_file)

# --------------------------------------------------
# 20. Final success message
# --------------------------------------------------

assert df["subscription_flag"].isin([0, 1]).all()
assert df["previously_contacted"].isin([0, 1]).all()
assert df["default_flag"].isin([0, 1]).all()
assert df["housing_loan_flag"].isin([0, 1]).all()
assert df["personal_loan_flag"].isin([0, 1]).all()

print("\nFinal validation checks passed.")

print("\n" + "=" * 75)
print("DATA CLEANING COMPLETE")
print("=" * 75)