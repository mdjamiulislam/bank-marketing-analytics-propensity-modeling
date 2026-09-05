import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Locate project folders
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

input_file = (
    project_root
    / "data"
    / "processed"
    / "bank_marketing_features.csv"
)

processed_folder = (
    project_root
    / "data"
    / "processed"
)

reports_folder = project_root / "reports"

train_output = (
    processed_folder
    / "model_train.csv"
)

test_output = (
    processed_folder
    / "model_test.csv"
)

summary_output = (
    reports_folder
    / "08_model_preparation_summary.csv"
)

feature_output = (
    reports_folder
    / "08_model_feature_list.csv"
)


# --------------------------------------------------
# 2. Load feature-engineered dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

print("=" * 80)
print("PROJECT 4 - MODEL PREPARATION")
print("=" * 80)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 3. Define target
# --------------------------------------------------

target_column = "subscription_flag"

y = df[target_column].copy()

print("\n1. TARGET VARIABLE")
print("-" * 80)

print("Target:", target_column)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(
    (
        y.value_counts(normalize=True)
        * 100
    ).round(2)
)


# --------------------------------------------------
# 4. Define predictor variables
# --------------------------------------------------

numeric_features = [
    "age",
    "balance",
    "campaign",
    "previous",
    "previously_contacted",
    "days_since_previous_contact",
    "negative_balance_flag",
    "previous_campaign_success_flag",
    "active_loan_count",
    "contact_known_flag"
]

categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]

model_features = (
    numeric_features
    + categorical_features
)


# --------------------------------------------------
# 5. Validate required fields
# --------------------------------------------------

missing_columns = [
    column
    for column in model_features + [target_column]
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

print("\n2. MODEL FEATURE VALIDATION")
print("-" * 80)

print("All required modelling fields are present.")


# --------------------------------------------------
# 6. Create model-ready dataset
# --------------------------------------------------

model_df = df[
    model_features + [target_column]
].copy()


# --------------------------------------------------
# 7. Handle intentional missing pdays-derived values
# --------------------------------------------------

missing_days_before = (
    model_df[
        "days_since_previous_contact"
    ]
    .isna()
    .sum()
)

model_df[
    "days_since_previous_contact"
] = (
    model_df[
        "days_since_previous_contact"
    ]
    .fillna(0)
)

print("\n3. MISSING VALUE TREATMENT")
print("-" * 80)

print(
    "Missing days_since_previous_contact before:",
    missing_days_before
)

print(
    "Missing days_since_previous_contact after:",
    model_df[
        "days_since_previous_contact"
    ]
    .isna()
    .sum()
)

print(
    "Note: 0 is used only as a modelling placeholder; "
    "previously_contacted retains the true status."
)


# --------------------------------------------------
# 8. Check remaining missing values
# --------------------------------------------------

remaining_missing = (
    model_df
    .isna()
    .sum()
)

remaining_missing = (
    remaining_missing[
        remaining_missing > 0
    ]
)

print("\n4. REMAINING MISSING VALUES")
print("-" * 80)

if remaining_missing.empty:
    print("No remaining missing values.")
else:
    print(remaining_missing)

    raise ValueError(
        "Unexpected missing values remain."
    )


# --------------------------------------------------
# 9. Confirm leakage variables are excluded
# --------------------------------------------------

leakage_columns = [
    "duration",
    "contact_duration_minutes",
    "subscribed",
    "subscription_flag",
    "campaign_record_id"
]

predictor_columns = model_df.drop(
    columns=[target_column]
).columns

leakage_found = [
    column
    for column in leakage_columns
    if column in predictor_columns
]

print("\n5. DATA LEAKAGE CHECK")
print("-" * 80)

if leakage_found:

    raise ValueError(
        f"Leakage variables found: {leakage_found}"
    )

else:

    print(
        "No prohibited leakage variables "
        "found in predictors."
    )

print(
    "duration and contact_duration_minutes "
    "are excluded from the targeting model."
)


# --------------------------------------------------
# 10. Separate X and y
# --------------------------------------------------

X = model_df.drop(
    columns=[target_column]
)

y = model_df[target_column]


print("\n6. PREDICTOR MATRIX")
print("-" * 80)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nNumeric features:")
for feature in numeric_features:
    print("-", feature)

print("\nCategorical features:")
for feature in categorical_features:
    print("-", feature)


# --------------------------------------------------
# 11. Create train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print("\n7. TRAIN / TEST SPLIT")
print("-" * 80)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training share:",
    round(
        len(X_train) / len(X) * 100,
        2
    ),
    "%"
)

print(
    "Testing share:",
    round(
        len(X_test) / len(X) * 100,
        2
    ),
    "%"
)


# --------------------------------------------------
# 12. Check target balance after split
# --------------------------------------------------

train_target_pct = (
    y_train
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

test_target_pct = (
    y_test
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\n8. TRAINING TARGET DISTRIBUTION")
print("-" * 80)

print(train_target_pct)


print("\n9. TEST TARGET DISTRIBUTION")
print("-" * 80)

print(test_target_pct)


# --------------------------------------------------
# 13. Recombine train dataset for export
# --------------------------------------------------

train_df = X_train.copy()

train_df[target_column] = y_train

test_df = X_test.copy()

test_df[target_column] = y_test


# --------------------------------------------------
# 14. Save model-ready train and test datasets
# --------------------------------------------------

train_df.to_csv(
    train_output,
    index=False
)

test_df.to_csv(
    test_output,
    index=False
)


print("\n10. MODEL DATASETS SAVED")
print("-" * 80)

print("Training file:")
print(train_output)

print("\nTesting file:")
print(test_output)


# --------------------------------------------------
# 15. Create feature documentation
# --------------------------------------------------

feature_rows = []

for feature in numeric_features:

    feature_rows.append({
        "feature": feature,
        "feature_type": "Numeric",
        "included_in_model": "Yes",
        "reason": "Pre-contact predictor"
    })

for feature in categorical_features:

    feature_rows.append({
        "feature": feature,
        "feature_type": "Categorical",
        "included_in_model": "Yes",
        "reason": "Pre-contact predictor"
    })


excluded_features = [
    {
        "feature": "duration",
        "feature_type": "Numeric",
        "included_in_model": "No",
        "reason":
            "Post-contact information; data leakage"
    },

    {
        "feature": "contact_duration_minutes",
        "feature_type": "Numeric",
        "included_in_model": "No",
        "reason":
            "Derived from duration; data leakage"
    },

    {
        "feature": "subscribed",
        "feature_type": "Target label",
        "included_in_model": "No",
        "reason":
            "Text representation of target"
    },

    {
        "feature": "campaign_record_id",
        "feature_type": "Identifier",
        "included_in_model": "No",
        "reason":
            "Artificial row identifier"
    }
]

feature_rows.extend(
    excluded_features
)

feature_documentation = pd.DataFrame(
    feature_rows
)

feature_documentation.to_csv(
    feature_output,
    index=False
)


print("\n11. FEATURE DOCUMENTATION SAVED")
print("-" * 80)

print(feature_output)


# --------------------------------------------------
# 16. Create model preparation summary
# --------------------------------------------------

overall_positive_rate = (
    y.mean() * 100
)

training_positive_rate = (
    y_train.mean() * 100
)

testing_positive_rate = (
    y_test.mean() * 100
)


summary_df = pd.DataFrame({
    "metric": [
        "Total model records",
        "Number of predictors",
        "Numeric predictors",
        "Categorical predictors",
        "Training records",
        "Testing records",
        "Overall positive rate (%)",
        "Training positive rate (%)",
        "Testing positive rate (%)",
        "Missing values after preparation",
        "Random state",
        "Test size (%)"
    ],

    "value": [
        len(model_df),
        len(model_features),
        len(numeric_features),
        len(categorical_features),
        len(X_train),
        len(X_test),
        round(overall_positive_rate, 2),
        round(training_positive_rate, 2),
        round(testing_positive_rate, 2),
        int(model_df.isna().sum().sum()),
        42,
        20
    ]
})


summary_df.to_csv(
    summary_output,
    index=False
)


print("\n12. MODEL PREPARATION SUMMARY")
print("-" * 80)

print(
    summary_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# 17. Final validation
# --------------------------------------------------

assert len(model_df) == len(df)

assert len(X_train) + len(X_test) == len(df)

assert len(y_train) + len(y_test) == len(df)

assert model_df.isna().sum().sum() == 0

assert y.isin([0, 1]).all()

assert (
    abs(
        training_positive_rate
        - testing_positive_rate
    ) < 1
)


print("\n13. FINAL VALIDATION")
print("-" * 80)

print(
    "All model preparation validation "
    "checks passed."
)


# --------------------------------------------------
# 18. Final message
# --------------------------------------------------

print("\n" + "=" * 80)
print("MODEL PREPARATION COMPLETE")
print("=" * 80)