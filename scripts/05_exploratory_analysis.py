import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

reports_folder = project_root / "reports"

charts_folder = (
    reports_folder
    / "eda_charts"
)

reports_folder.mkdir(exist_ok=True)
charts_folder.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. Load feature-engineered dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

print("=" * 75)
print("PROJECT 4 - EXPLORATORY DATA ANALYSIS")
print("=" * 75)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 3. Helper function for segment performance
# --------------------------------------------------

def segment_performance(data, column):

    result = (
        data
        .groupby(column, dropna=False)
        .agg(
            customers=(
                "campaign_record_id",
                "count"
            ),
            subscribers=(
                "subscription_flag",
                "sum"
            ),
            subscription_rate=(
                "subscription_flag",
                "mean"
            ),
            average_balance=(
                "balance",
                "mean"
            )
        )
        .reset_index()
    )

    result["subscription_rate"] = (
        result["subscription_rate"] * 100
    ).round(2)

    result["average_balance"] = (
        result["average_balance"]
        .round(2)
    )

    return result


# --------------------------------------------------
# 4. Overall campaign KPIs
# --------------------------------------------------

total_records = len(df)

total_subscribers = int(
    df["subscription_flag"].sum()
)

total_non_subscribers = (
    total_records - total_subscribers
)

subscription_rate = (
    df["subscription_flag"].mean() * 100
)

average_balance = df["balance"].mean()

median_balance = df["balance"].median()

average_campaign_contacts = (
    df["campaign"].mean()
)

average_contact_duration = (
    df["contact_duration_minutes"].mean()
)

previously_contacted_pct = (
    df["previously_contacted"].mean()
    * 100
)

previous_success_pct = (
    df["previous_campaign_success_flag"].mean()
    * 100
)

kpi_summary = pd.DataFrame({
    "metric": [
        "Total campaign records",
        "Subscribers",
        "Non-subscribers",
        "Subscription rate (%)",
        "Average balance",
        "Median balance",
        "Average campaign contacts",
        "Average contact duration (minutes)",
        "Previously contacted (%)",
        "Previous campaign success (%)"
    ],

    "value": [
        total_records,
        total_subscribers,
        total_non_subscribers,
        round(subscription_rate, 2),
        round(average_balance, 2),
        round(median_balance, 2),
        round(average_campaign_contacts, 2),
        round(average_contact_duration, 2),
        round(previously_contacted_pct, 2),
        round(previous_success_pct, 2)
    ]
})

print("\n1. OVERALL CAMPAIGN KPIs")
print("-" * 75)

print(
    kpi_summary.to_string(index=False)
)

kpi_summary.to_csv(
    reports_folder / "06_kpi_summary.csv",
    index=False
)


# --------------------------------------------------
# 5. Age Group Analysis
# --------------------------------------------------

age_performance = segment_performance(
    df,
    "age_group"
)

age_order = [
    "18-29",
    "30-39",
    "40-49",
    "50-59",
    "60+"
]

age_performance["age_group"] = pd.Categorical(
    age_performance["age_group"],
    categories=age_order,
    ordered=True
)

age_performance = (
    age_performance
    .sort_values("age_group")
)

print("\n2. AGE GROUP PERFORMANCE")
print("-" * 75)

print(
    age_performance.to_string(index=False)
)

age_performance.to_csv(
    reports_folder
    / "06_age_group_performance.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.bar(
    age_performance["age_group"].astype(str),
    age_performance["subscription_rate"]
)

plt.title(
    "Subscription Rate by Age Group"
)

plt.xlabel("Age Group")
plt.ylabel("Subscription Rate (%)")

plt.tight_layout()

plt.savefig(
    charts_folder
    / "01_subscription_rate_by_age_group.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 6. Job Analysis
# --------------------------------------------------

job_performance = segment_performance(
    df,
    "job"
)

job_performance = (
    job_performance
    .sort_values(
        "subscription_rate",
        ascending=False
    )
)

print("\n3. JOB PERFORMANCE")
print("-" * 75)

print(
    job_performance.to_string(index=False)
)

job_performance.to_csv(
    reports_folder
    / "06_job_performance.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.barh(
    job_performance["job"],
    job_performance["subscription_rate"]
)

plt.title(
    "Subscription Rate by Job"
)

plt.xlabel("Subscription Rate (%)")
plt.ylabel("Job")

plt.tight_layout()

plt.savefig(
    charts_folder
    / "02_subscription_rate_by_job.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 7. Balance Band Analysis
# --------------------------------------------------

balance_performance = segment_performance(
    df,
    "balance_band"
)

balance_order = [
    "Negative",
    "0-1,000",
    "1,001-5,000",
    "5,001-10,000",
    "10,001+"
]

balance_performance["balance_band"] = (
    pd.Categorical(
        balance_performance["balance_band"],
        categories=balance_order,
        ordered=True
    )
)

balance_performance = (
    balance_performance
    .sort_values("balance_band")
)

print("\n4. BALANCE BAND PERFORMANCE")
print("-" * 75)

print(
    balance_performance.to_string(index=False)
)

balance_performance.to_csv(
    reports_folder
    / "06_balance_band_performance.csv",
    index=False
)

plt.figure(figsize=(9, 5))

plt.bar(
    balance_performance[
        "balance_band"
    ].astype(str),

    balance_performance[
        "subscription_rate"
    ]
)

plt.title(
    "Subscription Rate by Balance Band"
)

plt.xlabel("Balance Band")
plt.ylabel("Subscription Rate (%)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    charts_folder
    / "03_subscription_rate_by_balance_band.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 8. Loan Profile Analysis
# --------------------------------------------------

loan_performance = segment_performance(
    df,
    "loan_profile"
)

loan_performance = (
    loan_performance
    .sort_values(
        "subscription_rate",
        ascending=False
    )
)

print("\n5. LOAN PROFILE PERFORMANCE")
print("-" * 75)

print(
    loan_performance.to_string(index=False)
)

loan_performance.to_csv(
    reports_folder
    / "06_loan_profile_performance.csv",
    index=False
)

plt.figure(figsize=(10, 5))

plt.bar(
    loan_performance["loan_profile"],
    loan_performance["subscription_rate"]
)

plt.title(
    "Subscription Rate by Loan Profile"
)

plt.xlabel("Loan Profile")
plt.ylabel("Subscription Rate (%)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    charts_folder
    / "04_subscription_rate_by_loan_profile.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 9. Contact Method Analysis
# --------------------------------------------------

contact_performance = (
    segment_performance(
        df,
        "contact"
    )
    .sort_values(
        "subscription_rate",
        ascending=False
    )
)

print("\n6. CONTACT METHOD PERFORMANCE")
print("-" * 75)

print(
    contact_performance.to_string(index=False)
)

contact_performance.to_csv(
    reports_folder
    / "06_contact_method_performance.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.bar(
    contact_performance["contact"],
    contact_performance["subscription_rate"]
)

plt.title(
    "Subscription Rate by Contact Method"
)

plt.xlabel("Contact Method")
plt.ylabel("Subscription Rate (%)")

plt.tight_layout()

plt.savefig(
    charts_folder
    / "05_subscription_rate_by_contact_method.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 10. Campaign Contact Frequency Analysis
# --------------------------------------------------

campaign_performance = segment_performance(
    df,
    "campaign_contact_band"
)

campaign_order = [
    "1 contact",
    "2-3 contacts",
    "4-5 contacts",
    "6+ contacts"
]

campaign_performance[
    "campaign_contact_band"
] = pd.Categorical(
    campaign_performance[
        "campaign_contact_band"
    ],
    categories=campaign_order,
    ordered=True
)

campaign_performance = (
    campaign_performance
    .sort_values(
        "campaign_contact_band"
    )
)

print(
    "\n7. CAMPAIGN CONTACT FREQUENCY PERFORMANCE"
)
print("-" * 75)

print(
    campaign_performance.to_string(
        index=False
    )
)

campaign_performance.to_csv(
    reports_folder
    / "06_campaign_contact_performance.csv",
    index=False
)

plt.figure(figsize=(9, 5))

plt.bar(
    campaign_performance[
        "campaign_contact_band"
    ].astype(str),

    campaign_performance[
        "subscription_rate"
    ]
)

plt.title(
    "Subscription Rate by Campaign Contact Frequency"
)

plt.xlabel("Number of Contacts")
plt.ylabel("Subscription Rate (%)")

plt.tight_layout()

plt.savefig(
    charts_folder
    / "06_subscription_rate_by_contact_frequency.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 11. Previous Campaign Analysis
# --------------------------------------------------

previous_performance = (
    segment_performance(
        df,
        "previous_campaign_status"
    )
    .sort_values(
        "subscription_rate",
        ascending=False
    )
)

print("\n8. PREVIOUS CAMPAIGN PERFORMANCE")
print("-" * 75)

print(
    previous_performance.to_string(
        index=False
    )
)

previous_performance.to_csv(
    reports_folder
    / "06_previous_campaign_performance.csv",
    index=False
)

plt.figure(figsize=(10, 5))

plt.bar(
    previous_performance[
        "previous_campaign_status"
    ],

    previous_performance[
        "subscription_rate"
    ]
)

plt.title(
    "Subscription Rate by Previous Campaign Status"
)

plt.xlabel("Previous Campaign Status")
plt.ylabel("Subscription Rate (%)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    charts_folder
    / "07_subscription_rate_by_previous_campaign.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 12. Monthly Campaign Analysis
# --------------------------------------------------

month_performance = (
    df
    .groupby(
        [
            "month",
            "contact_month_num"
        ],
        dropna=False
    )
    .agg(
        customers=(
            "campaign_record_id",
            "count"
        ),
        subscribers=(
            "subscription_flag",
            "sum"
        ),
        subscription_rate=(
            "subscription_flag",
            "mean"
        )
    )
    .reset_index()
)

month_performance[
    "subscription_rate"
] = (
    month_performance[
        "subscription_rate"
    ] * 100
).round(2)

month_performance = (
    month_performance
    .sort_values("contact_month_num")
)

print("\n9. MONTHLY CAMPAIGN PERFORMANCE")
print("-" * 75)

print(
    month_performance.to_string(
        index=False
    )
)

month_performance.to_csv(
    reports_folder
    / "06_month_performance.csv",
    index=False
)

plt.figure(figsize=(10, 5))

plt.plot(
    month_performance["month"],
    month_performance["subscription_rate"],
    marker="o"
)

plt.title(
    "Subscription Rate by Contact Month"
)

plt.xlabel("Month")
plt.ylabel("Subscription Rate (%)")

plt.tight_layout()

plt.savefig(
    charts_folder
    / "08_subscription_rate_by_month.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 13. Quarterly Campaign Analysis
# --------------------------------------------------

quarter_performance = (
    segment_performance(
        df,
        "contact_quarter"
    )
    .sort_values(
        "contact_quarter"
    )
)

print("\n10. QUARTERLY CAMPAIGN PERFORMANCE")
print("-" * 75)

print(
    quarter_performance.to_string(
        index=False
    )
)

quarter_performance.to_csv(
    reports_folder
    / "06_quarter_performance.csv",
    index=False
)


# --------------------------------------------------
# 14. Duration Analysis
# DESCRIPTIVE ONLY - NOT FOR TARGETING MODEL
# --------------------------------------------------

duration_performance = (
    df
    .groupby("subscribed")
    .agg(
        campaign_records=(
            "campaign_record_id",
            "count"
        ),

        average_duration_seconds=(
            "duration",
            "mean"
        ),

        median_duration_seconds=(
            "duration",
            "median"
        ),

        average_duration_minutes=(
            "contact_duration_minutes",
            "mean"
        )
    )
    .reset_index()
)

duration_performance[
    "average_duration_seconds"
] = (
    duration_performance[
        "average_duration_seconds"
    ]
    .round(2)
)

duration_performance[
    "average_duration_minutes"
] = (
    duration_performance[
        "average_duration_minutes"
    ]
    .round(2)
)

print("\n11. CONTACT DURATION ANALYSIS")
print("-" * 75)

print(
    duration_performance.to_string(
        index=False
    )
)

print(
    "\nIMPORTANT: Duration is descriptive only "
    "and will be excluded from the "
    "pre-contact predictive model."
)

duration_performance.to_csv(
    reports_folder
    / "06_duration_performance.csv",
    index=False
)


# --------------------------------------------------
# 15. Previous Contact vs No Previous Contact
# --------------------------------------------------

previous_contact_comparison = (
    df
    .groupby("previously_contacted")
    .agg(
        customers=(
            "campaign_record_id",
            "count"
        ),
        subscribers=(
            "subscription_flag",
            "sum"
        ),
        subscription_rate=(
            "subscription_flag",
            "mean"
        )
    )
    .reset_index()
)

previous_contact_comparison[
    "subscription_rate"
] = (
    previous_contact_comparison[
        "subscription_rate"
    ] * 100
).round(2)

previous_contact_comparison[
    "previous_contact_status"
] = (
    previous_contact_comparison[
        "previously_contacted"
    ]
    .map({
        0: "No previous contact",
        1: "Previously contacted"
    })
)

print(
    "\n12. PREVIOUS CONTACT COMPARISON"
)
print("-" * 75)

print(
    previous_contact_comparison[
        [
            "previous_contact_status",
            "customers",
            "subscribers",
            "subscription_rate"
        ]
    ].to_string(index=False)
)

previous_contact_comparison.to_csv(
    reports_folder
    / "06_previous_contact_comparison.csv",
    index=False
)


# --------------------------------------------------
# 16. Identify high-performing segments
# --------------------------------------------------

print("\n13. HIGH-PERFORMING SEGMENTS")
print("-" * 75)

print("\nHighest-performing age group:")

print(
    age_performance
    .sort_values(
        "subscription_rate",
        ascending=False
    )
    .head(1)
    .to_string(index=False)
)

print("\nHighest-performing job:")

print(
    job_performance
    .head(1)
    .to_string(index=False)
)

print("\nHighest-performing balance band:")

print(
    balance_performance
    .sort_values(
        "subscription_rate",
        ascending=False
    )
    .head(1)
    .to_string(index=False)
)

print("\nHighest-performing loan profile:")

print(
    loan_performance
    .head(1)
    .to_string(index=False)
)

print("\nHighest-performing previous campaign status:")

print(
    previous_performance
    .head(1)
    .to_string(index=False)
)


# --------------------------------------------------
# 17. Final EDA validation
# --------------------------------------------------

assert len(df) > 0
assert df["campaign_record_id"].is_unique
assert df["subscription_flag"].isin([0, 1]).all()

print("\n14. VALIDATION")
print("-" * 75)

print("EDA validation checks passed.")


# --------------------------------------------------
# 18. Final message
# --------------------------------------------------

print("\n" + "=" * 75)
print("EXPLORATORY DATA ANALYSIS COMPLETE")
print("=" * 75)

print("\nEDA reports saved to:")
print(reports_folder)

print("\nEDA charts saved to:")
print(charts_folder)