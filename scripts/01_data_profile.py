import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Locate the project and dataset
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "raw" / "bank-full.csv"

# --------------------------------------------------
# 2. Load the raw dataset
# --------------------------------------------------

df = pd.read_csv(file_path, sep=";")

print("=" * 70)
print("PROJECT 4 - BANK MARKETING DATA PROFILE")
print("=" * 70)

print("\nDataset loaded successfully.")
print("File:", file_path)

# --------------------------------------------------
# 3. Basic dataset dimensions
# --------------------------------------------------

print("\n1. DATASET SIZE")
print("-" * 40)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# --------------------------------------------------
# 4. Column names
# --------------------------------------------------

print("\n2. COLUMN NAMES")
print("-" * 40)

for number, column in enumerate(df.columns, start=1):
    print(number, column)

# --------------------------------------------------
# 5. First five rows
# --------------------------------------------------

print("\n3. FIRST 5 ROWS")
print("-" * 40)

print(df.head())

# --------------------------------------------------
# 6. Data types
# --------------------------------------------------

print("\n4. DATA TYPES")
print("-" * 40)

print(df.dtypes)

# --------------------------------------------------
# 7. Missing values
# --------------------------------------------------

print("\n5. MISSING VALUES")
print("-" * 40)

missing_values = df.isnull().sum()
print(missing_values)

print("\nTotal missing values:", missing_values.sum())

# --------------------------------------------------
# 8. Duplicate rows
# --------------------------------------------------

print("\n6. DUPLICATE RECORDS")
print("-" * 40)

print("Duplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# 9. Unique values per column
# --------------------------------------------------

print("\n7. UNIQUE VALUES PER COLUMN")
print("-" * 40)

print(df.nunique())

# --------------------------------------------------
# 10. Numerical summary
# --------------------------------------------------

print("\n8. NUMERICAL SUMMARY")
print("-" * 40)

print(df.describe())

# --------------------------------------------------
# 11. Categorical summary
# --------------------------------------------------

print("\n9. CATEGORICAL SUMMARY")
print("-" * 40)

print(df.describe(include="object"))

# --------------------------------------------------
# 12. Target variable distribution
# --------------------------------------------------

print("\n10. TARGET VARIABLE - y")
print("-" * 40)

print(df["y"].value_counts())

print("\nSubscription percentages:")
print(df["y"].value_counts(normalize=True) * 100)

print("\nPROFILE COMPLETE")

print("\n11. UNKNOWN VALUES")
print("-" * 40)

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    unknown_count = (df[column] == "unknown").sum()

    if unknown_count > 0:
        print(
            column,
            "-",
            unknown_count,
            "unknown values"
        )
print("\n12. CATEGORICAL VALUE DISTRIBUTIONS")
print("-" * 40)

for column in categorical_columns:

    print(f"\n--- {column.upper()} ---")

    print(df[column].value_counts())