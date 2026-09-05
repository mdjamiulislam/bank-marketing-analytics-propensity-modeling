import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Locate project folders
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

input_file = (
    project_root
    / "data"
    / "processed"
    / "bank_marketing_clean.csv"
)

processed_folder = (
    project_root
    / "data"
    / "processed"
)

reports_folder = project_root / "reports"

output_file = (
    processed_folder
    / "bank_marketing_features.csv"
)

feature_summary_file = (
    reports_folder
    / "05_feature_engineering_summary.csv"
)

feature_distribution_file = (
    reports_folder
    / "05_feature_distributions.csv"
)

# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

input_rows = len(df)
input_columns = len(df.columns)

print("=" * 75)
print("PROJECT 4 - FEATURE ENGINEERING")
print("=" * 75)

print("\nClean dataset loaded successfully.")
print("Input rows:", input_rows)
print("Input columns:", input_columns)


# --------------------------------------------------
# 3. Create Age Group
# --------------------------------------------------

age_bins = [
    17,
    29,
    39,
    49,
    59,
    float("inf")
]

age_labels = [
    "18-29",
    "30-39",
    "40-49",
    "50-59",
    "60+"
]

df["age_group"] = pd.cut(
    df["age"],
    bins=age_bins,
    labels=age_labels
)

age_order = {
    "18-29": 1,
    "30-39": 2,
    "40-49": 3,
    "50-59": 4,
    "60+": 5
}

df["age_group_order"] = (
    df["age_group"]
    .astype(str)
    .map(age_order)
)

print("\n1. AGE GROUP CREATED")
print("-" * 75)

print(
    df["age_group"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 4. Create Balance Band
# --------------------------------------------------

balance_conditions = [
    df["balance"] < 0,
    df["balance"].between(0, 1000),
    df["balance"].between(1001, 5000),
    df["balance"].between(5001, 10000),
    df["balance"] > 10000
]

balance_labels = [
    "Negative",
    "0-1,000",
    "1,001-5,000",
    "5,001-10,000",
    "10,001+"
]

df["balance_band"] = np.select(
    balance_conditions,
    balance_labels,
    default="Unknown"
)

balance_order = {
    "Negative": 1,
    "0-1,000": 2,
    "1,001-5,000": 3,
    "5,001-10,000": 4,
    "10,001+": 5
}

df["balance_band_order"] = (
    df["balance_band"]
    .map(balance_order)
)

df["negative_balance_flag"] = (
    df["balance"] < 0
).astype("int8")

print("\n2. BALANCE BAND CREATED")
print("-" * 75)

print(
    df["balance_band"]
    .value_counts()
)


# --------------------------------------------------
# 5. Create Campaign Contact Band
# --------------------------------------------------

contact_bins = [
    0,
    1,
    3,
    5,
    float("inf")
]

contact_labels = [
    "1 contact",
    "2-3 contacts",
    "4-5 contacts",
    "6+ contacts"
]

df["campaign_contact_band"] = pd.cut(
    df["campaign"],
    bins=contact_bins,
    labels=contact_labels
)

contact_band_order = {
    "1 contact": 1,
    "2-3 contacts": 2,
    "4-5 contacts": 3,
    "6+ contacts": 4
}

df["campaign_contact_band_order"] = (
    df["campaign_contact_band"]
    .astype(str)
    .map(contact_band_order)
)

print("\n3. CAMPAIGN CONTACT BAND CREATED")
print("-" * 75)

print(
    df["campaign_contact_band"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 6. Convert contact duration to minutes
# --------------------------------------------------

df["contact_duration_minutes"] = (
    df["duration"] / 60
).round(2)

print("\n4. CONTACT DURATION IN MINUTES CREATED")
print("-" * 75)

print(
    df[
        [
            "duration",
            "contact_duration_minutes"
        ]
    ]
    .head()
)


# --------------------------------------------------
# 7. Create Previous Campaign Status
# --------------------------------------------------

previous_conditions = [
    df["previously_contacted"] == 0,

    (
        (df["previously_contacted"] == 1)
        & (df["poutcome"] == "success")
    ),

    (
        (df["previously_contacted"] == 1)
        & (df["poutcome"] == "failure")
    ),

    (
        (df["previously_contacted"] == 1)
        & (df["poutcome"] == "other")
    )
]

previous_labels = [
    "No previous contact",
    "Previous success",
    "Previous failure",
    "Previous other"
]

df["previous_campaign_status"] = np.select(
    previous_conditions,
    previous_labels,
    default="Previous outcome unknown"
)

df["previous_campaign_success_flag"] = (
    (
        (df["previously_contacted"] == 1)
        & (df["poutcome"] == "success")
    )
).astype("int8")

print("\n5. PREVIOUS CAMPAIGN STATUS CREATED")
print("-" * 75)

print(
    df["previous_campaign_status"]
    .value_counts()
)


# --------------------------------------------------
# 8. Create Loan Profile
# --------------------------------------------------

loan_conditions = [
    (
        (df["housing_loan_flag"] == 0)
        & (df["personal_loan_flag"] == 0)
    ),

    (
        (df["housing_loan_flag"] == 1)
        & (df["personal_loan_flag"] == 0)
    ),

    (
        (df["housing_loan_flag"] == 0)
        & (df["personal_loan_flag"] == 1)
    ),

    (
        (df["housing_loan_flag"] == 1)
        & (df["personal_loan_flag"] == 1)
    )
]

loan_labels = [
    "No loans",
    "Housing loan only",
    "Personal loan only",
    "Housing + personal loans"
]

df["loan_profile"] = np.select(
    loan_conditions,
    loan_labels,
    default="Unknown"
)

df["active_loan_count"] = (
    df["housing_loan_flag"]
    + df["personal_loan_flag"]
)

print("\n6. LOAN PROFILE CREATED")
print("-" * 75)

print(
    df["loan_profile"]
    .value_counts()
)


# --------------------------------------------------
# 9. Create Contact Method Known Flag
# --------------------------------------------------

df["contact_known_flag"] = (
    df["contact"] != "unknown"
).astype("int8")

print("\n7. CONTACT METHOD FLAG CREATED")
print("-" * 75)

print(
    df["contact_known_flag"]
    .value_counts()
)


# --------------------------------------------------
# 10. Create Month Number
# --------------------------------------------------

month_mapping = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}

df["contact_month_num"] = (
    df["month"]
    .map(month_mapping)
)

if df["contact_month_num"].isna().any():

    unexpected_months = (
        df.loc[
            df["contact_month_num"].isna(),
            "month"
        ]
        .unique()
    )

    raise ValueError(
        f"Unexpected month values: {unexpected_months}"
    )

df["contact_month_num"] = (
    df["contact_month_num"]
    .astype("int8")
)

print("\n8. MONTH NUMBER CREATED")
print("-" * 75)

print(
    df[
        [
            "month",
            "contact_month_num"
        ]
    ]
    .drop_duplicates()
    .sort_values("contact_month_num")
)


# --------------------------------------------------
# 11. Create Contact Quarter
# --------------------------------------------------

quarter_mapping = {
    "jan": "Q1",
    "feb": "Q1",
    "mar": "Q1",

    "apr": "Q2",
    "may": "Q2",
    "jun": "Q2",

    "jul": "Q3",
    "aug": "Q3",
    "sep": "Q3",

    "oct": "Q4",
    "nov": "Q4",
    "dec": "Q4"
}

df["contact_quarter"] = (
    df["month"]
    .map(quarter_mapping)
)

print("\n9. CONTACT QUARTER CREATED")
print("-" * 75)

print(
    df["contact_quarter"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 12. Validate all engineered features
# --------------------------------------------------

print("\n10. FEATURE VALIDATION")
print("-" * 75)

# Row count must remain unchanged
assert len(df) == input_rows

# Original unique row ID must remain unique
assert df["campaign_record_id"].is_unique

# Age groups must exist
assert df["age_group"].notna().all()

# Balance bands must exist
assert (
    df["balance_band"] != "Unknown"
).all()

# Campaign contact bands must exist
assert df["campaign_contact_band"].notna().all()

# Binary flags
assert (
    df["negative_balance_flag"]
    .isin([0, 1])
    .all()
)

assert (
    df["previous_campaign_success_flag"]
    .isin([0, 1])
    .all()
)

assert (
    df["contact_known_flag"]
    .isin([0, 1])
    .all()
)

# Loan count must be between 0 and 2
assert (
    df["active_loan_count"]
    .between(0, 2)
    .all()
)

# Duration must remain non-negative
assert (
    df["contact_duration_minutes"] >= 0
).all()

print("All feature validation checks passed.")


# --------------------------------------------------
# 13. Save engineered dataset
# --------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

print("\n11. FEATURE DATASET SAVED")
print("-" * 75)

print(output_file)


# --------------------------------------------------
# 14. Create Feature Engineering Documentation
# --------------------------------------------------

feature_summary = pd.DataFrame(
    [
        {
            "feature": "age_group",
            "source": "age",
            "purpose": "Customer age segmentation",
            "model_use": "Yes"
        },
        {
            "feature": "age_group_order",
            "source": "age_group",
            "purpose": "Correct visual sorting",
            "model_use": "No"
        },
        {
            "feature": "balance_band",
            "source": "balance",
            "purpose": "Account balance segmentation",
            "model_use": "Optional"
        },
        {
            "feature": "balance_band_order",
            "source": "balance_band",
            "purpose": "Correct visual sorting",
            "model_use": "No"
        },
        {
            "feature": "negative_balance_flag",
            "source": "balance",
            "purpose": "Identify negative balances",
            "model_use": "Yes"
        },
        {
            "feature": "campaign_contact_band",
            "source": "campaign",
            "purpose": "Group campaign contact frequency",
            "model_use": "Optional"
        },
        {
            "feature": "campaign_contact_band_order",
            "source": "campaign_contact_band",
            "purpose": "Correct visual sorting",
            "model_use": "No"
        },
        {
            "feature": "contact_duration_minutes",
            "source": "duration",
            "purpose": "Campaign engagement analysis",
            "model_use": "NO - data leakage"
        },
        {
            "feature": "previous_campaign_status",
            "source": "poutcome + previously_contacted",
            "purpose": "Summarise previous campaign history",
            "model_use": "Yes"
        },
        {
            "feature": "previous_campaign_success_flag",
            "source": "poutcome",
            "purpose": "Identify prior successful response",
            "model_use": "Yes"
        },
        {
            "feature": "loan_profile",
            "source": "housing + loan",
            "purpose": "Customer loan relationship",
            "model_use": "Yes"
        },
        {
            "feature": "active_loan_count",
            "source": "housing + loan",
            "purpose": "Number of loan types",
            "model_use": "Yes"
        },
        {
            "feature": "contact_known_flag",
            "source": "contact",
            "purpose": "Identify known contact channel",
            "model_use": "Yes"
        },
        {
            "feature": "contact_month_num",
            "source": "month",
            "purpose": "Correct month sorting",
            "model_use": "Optional"
        },
        {
            "feature": "contact_quarter",
            "source": "month",
            "purpose": "Quarterly campaign analysis",
            "model_use": "Optional"
        }
    ]
)

feature_summary.to_csv(
    feature_summary_file,
    index=False
)

print("\n12. FEATURE DOCUMENTATION SAVED")
print("-" * 75)

print(feature_summary_file)


# --------------------------------------------------
# 15. Save Feature Distribution Report
# --------------------------------------------------

engineered_categories = [
    "age_group",
    "balance_band",
    "campaign_contact_band",
    "previous_campaign_status",
    "loan_profile",
    "contact_quarter"
]

distribution_rows = []

for column in engineered_categories:

    counts = df[column].value_counts(
        dropna=False
    )

    for category, count in counts.items():

        distribution_rows.append({
            "feature": column,
            "category": category,
            "count": count,
            "percentage": round(
                count / len(df) * 100,
                2
            )
        })

distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_df.to_csv(
    feature_distribution_file,
    index=False
)

print("\n13. FEATURE DISTRIBUTIONS SAVED")
print("-" * 75)

print(feature_distribution_file)


# --------------------------------------------------
# 16. Final Summary
# --------------------------------------------------

print("\n14. FINAL FEATURE DATASET")
print("-" * 75)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print(
    "New columns added:",
    len(df.columns) - input_columns
)

print("\nNew engineered features:")

new_features = [
    "age_group",
    "age_group_order",
    "balance_band",
    "balance_band_order",
    "negative_balance_flag",
    "campaign_contact_band",
    "campaign_contact_band_order",
    "contact_duration_minutes",
    "previous_campaign_status",
    "previous_campaign_success_flag",
    "loan_profile",
    "active_loan_count",
    "contact_known_flag",
    "contact_month_num",
    "contact_quarter"
]

for number, feature in enumerate(
    new_features,
    start=1
):
    print(number, feature)


print("\n" + "=" * 75)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 75)