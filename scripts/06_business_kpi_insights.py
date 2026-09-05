import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Locate project files
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

input_file = (
    project_root
    / "data"
    / "processed"
    / "bank_marketing_features.csv"
)

reports_folder = project_root / "reports"

reports_folder.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load analysis-ready dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

print("=" * 80)
print("PROJECT 4 - BUSINESS KPI & MANAGEMENT INSIGHT ANALYSIS")
print("=" * 80)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 3. Core Executive KPIs
# --------------------------------------------------

total_records = len(df)

total_subscribers = int(
    df["subscription_flag"].sum()
)

total_non_subscribers = (
    total_records - total_subscribers
)

overall_subscription_rate = (
    df["subscription_flag"].mean() * 100
)

average_balance = df["balance"].mean()

median_balance = df["balance"].median()

average_contacts = df["campaign"].mean()

previously_contacted_pct = (
    df["previously_contacted"].mean() * 100
)

previous_success_pct = (
    df["previous_campaign_success_flag"].mean() * 100
)

average_duration_minutes = (
    df["contact_duration_minutes"].mean()
)


executive_kpis = pd.DataFrame({
    "kpi": [
        "Total Campaign Records",
        "Total Subscribers",
        "Total Non-Subscribers",
        "Subscription Rate (%)",
        "Average Balance",
        "Median Balance",
        "Average Campaign Contacts",
        "Previously Contacted (%)",
        "Previous Campaign Success (%)",
        "Average Contact Duration (Minutes)"
    ],

    "value": [
        total_records,
        total_subscribers,
        total_non_subscribers,
        round(overall_subscription_rate, 2),
        round(average_balance, 2),
        round(median_balance, 2),
        round(average_contacts, 2),
        round(previously_contacted_pct, 2),
        round(previous_success_pct, 2),
        round(average_duration_minutes, 2)
    ]
})


print("\n1. EXECUTIVE KPIs")
print("-" * 80)

print(
    executive_kpis.to_string(index=False)
)

executive_kpis.to_csv(
    reports_folder / "07_executive_kpis.csv",
    index=False
)


# --------------------------------------------------
# 4. Helper function for segment performance
# --------------------------------------------------

def analyse_segment(data, column, segment_type):

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
            )
        )
        .reset_index()
    )

    result["subscription_rate"] = (
        result["subscription_rate"] * 100
    )

    result["customer_share_pct"] = (
        result["customers"]
        / total_records
        * 100
    )

    result["subscriber_share_pct"] = (
        result["subscribers"]
        / total_subscribers
        * 100
    )

    result["lift_vs_overall"] = (
        result["subscription_rate"]
        / overall_subscription_rate
    )

    result["conversion_rank"] = (
        result["subscription_rate"]
        .rank(
            method="dense",
            ascending=False
        )
        .astype(int)
    )

    result["subscriber_volume_rank"] = (
        result["subscribers"]
        .rank(
            method="dense",
            ascending=False
        )
        .astype(int)
    )

    result["segment_type"] = segment_type

    result = result.rename(
        columns={
            column: "segment"
        }
    )

    numeric_columns = [
        "subscription_rate",
        "customer_share_pct",
        "subscriber_share_pct",
        "lift_vs_overall"
    ]

    result[numeric_columns] = (
        result[numeric_columns]
        .round(2)
    )

    return result


# --------------------------------------------------
# 5. Customer Segment Analysis
# --------------------------------------------------

segment_tables = []


segment_tables.append(
    analyse_segment(
        df,
        "age_group",
        "Age Group"
    )
)

segment_tables.append(
    analyse_segment(
        df,
        "job",
        "Job"
    )
)

segment_tables.append(
    analyse_segment(
        df,
        "balance_band",
        "Balance Band"
    )
)

segment_tables.append(
    analyse_segment(
        df,
        "loan_profile",
        "Loan Profile"
    )
)

segment_tables.append(
    analyse_segment(
        df,
        "education",
        "Education"
    )
)


segment_priority = pd.concat(
    segment_tables,
    ignore_index=True
)


# --------------------------------------------------
# 6. Add analytical opportunity label
# --------------------------------------------------

def classify_opportunity(row):

    if (
        row["lift_vs_overall"] >= 1.25
        and row["customer_share_pct"] >= 5
    ):
        return "High conversion & scalable"

    elif (
        row["lift_vs_overall"] >= 1.25
        and row["customer_share_pct"] < 5
    ):
        return "High conversion niche"

    elif (
        row["customer_share_pct"] >= 10
        and row["lift_vs_overall"] >= 0.90
    ):
        return "Large base opportunity"

    elif row["lift_vs_overall"] < 0.90:
        return "Below-average conversion"

    else:
        return "Monitor"


segment_priority["opportunity_group"] = (
    segment_priority.apply(
        classify_opportunity,
        axis=1
    )
)


segment_priority = (
    segment_priority
    .sort_values(
        [
            "segment_type",
            "subscription_rate"
        ],
        ascending=[
            True,
            False
        ]
    )
)


print("\n2. CUSTOMER SEGMENT PRIORITY TABLE")
print("-" * 80)

print(
    segment_priority.to_string(
        index=False
    )
)


segment_priority.to_csv(
    reports_folder
    / "07_segment_priority_table.csv",
    index=False
)


# --------------------------------------------------
# 7. Campaign Efficiency Analysis
# --------------------------------------------------

campaign_tables = []


campaign_tables.append(
    analyse_segment(
        df,
        "contact",
        "Contact Method"
    )
)

campaign_tables.append(
    analyse_segment(
        df,
        "campaign_contact_band",
        "Contact Frequency"
    )
)

campaign_tables.append(
    analyse_segment(
        df,
        "previous_campaign_status",
        "Previous Campaign Status"
    )
)

campaign_tables.append(
    analyse_segment(
        df,
        "month",
        "Contact Month"
    )
)

campaign_tables.append(
    analyse_segment(
        df,
        "contact_quarter",
        "Contact Quarter"
    )
)


campaign_efficiency = pd.concat(
    campaign_tables,
    ignore_index=True
)


campaign_efficiency = (
    campaign_efficiency
    .sort_values(
        [
            "segment_type",
            "subscription_rate"
        ],
        ascending=[
            True,
            False
        ]
    )
)


print("\n3. CAMPAIGN EFFICIENCY TABLE")
print("-" * 80)

print(
    campaign_efficiency.to_string(
        index=False
    )
)


campaign_efficiency.to_csv(
    reports_folder
    / "07_campaign_efficiency_table.csv",
    index=False
)


# --------------------------------------------------
# 8. Helper function - top segment
# --------------------------------------------------

def top_segment(column):

    table = (
        df
        .groupby(column)
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

    table["subscription_rate"] = (
        table["subscription_rate"]
        * 100
    )

    return (
        table
        .sort_values(
            "subscription_rate",
            ascending=False
        )
        .iloc[0]
    )


# --------------------------------------------------
# 9. Identify major findings
# --------------------------------------------------

best_age = top_segment(
    "age_group"
)

best_balance = top_segment(
    "balance_band"
)

best_loan = top_segment(
    "loan_profile"
)

best_contact_method = top_segment(
    "contact"
)

best_contact_frequency = top_segment(
    "campaign_contact_band"
)

best_previous_status = top_segment(
    "previous_campaign_status"
)

best_month = top_segment(
    "month"
)


# --------------------------------------------------
# 10. Largest subscriber-volume age group
# --------------------------------------------------

age_volume = (
    df
    .groupby("age_group")
    .agg(
        customers=(
            "campaign_record_id",
            "count"
        ),

        subscribers=(
            "subscription_flag",
            "sum"
        )
    )
    .reset_index()
)

largest_age_volume = (
    age_volume
    .sort_values(
        "subscribers",
        ascending=False
    )
    .iloc[0]
)


# --------------------------------------------------
# 11. Duration comparison
# --------------------------------------------------

duration_summary = (
    df
    .groupby("subscribed")
    ["contact_duration_minutes"]
    .mean()
)

subscriber_duration = (
    duration_summary.get(
        "yes",
        float("nan")
    )
)

non_subscriber_duration = (
    duration_summary.get(
        "no",
        float("nan")
    )
)


# --------------------------------------------------
# 12. Build Management Findings
# --------------------------------------------------

management_findings = pd.DataFrame(
    [
        {
            "finding_id": 1,

            "area": "Overall Campaign",

            "finding":
                f"Overall subscription rate was "
                f"{overall_subscription_rate:.2f}% "
                f"from {total_records:,} campaign records.",

            "management_use":
                "Use as the baseline for evaluating all customer "
                "and campaign segments."
        },

        {
            "finding_id": 2,

            "area": "Age",

            "finding":
                f"The highest-converting age group was "
                f"{best_age['age_group']} at "
                f"{best_age['subscription_rate']:.2f}%.",

            "management_use":
                "Review this segment as a higher-propensity "
                "targeting opportunity."
        },

        {
            "finding_id": 3,

            "area": "Age Volume",

            "finding":
                f"The age group generating the largest number "
                f"of subscribers was "
                f"{largest_age_volume['age_group']} with "
                f"{int(largest_age_volume['subscribers']):,} "
                f"subscriptions.",

            "management_use":
                "Balance conversion rate with absolute "
                "subscriber volume when allocating campaign resources."
        },

        {
            "finding_id": 4,

            "area": "Balance",

            "finding":
                f"The highest-converting balance band was "
                f"{best_balance['balance_band']} at "
                f"{best_balance['subscription_rate']:.2f}%.",

            "management_use":
                "Assess customer financial position when "
                "designing campaign targeting rules."
        },

        {
            "finding_id": 5,

            "area": "Loan Profile",

            "finding":
                f"The highest-converting loan profile was "
                f"{best_loan['loan_profile']} at "
                f"{best_loan['subscription_rate']:.2f}%.",

            "management_use":
                "Consider existing lending relationships when "
                "segmenting customers."
        },

        {
            "finding_id": 6,

            "area": "Contact Channel",

            "finding":
                f"The strongest observed contact channel was "
                f"{best_contact_method['contact']} at "
                f"{best_contact_method['subscription_rate']:.2f}%.",

            "management_use":
                "Review channel effectiveness while recognising "
                "that observational results do not prove causality."
        },

        {
            "finding_id": 7,

            "area": "Contact Frequency",

            "finding":
                f"The highest observed conversion occurred in the "
                f"{best_contact_frequency['campaign_contact_band']} "
                f"group at "
                f"{best_contact_frequency['subscription_rate']:.2f}%.",

            "management_use":
                "Use contact-frequency performance to investigate "
                "possible diminishing returns from repeated contact."
        },

        {
            "finding_id": 8,

            "area": "Previous Campaign",

            "finding":
                f"The strongest previous-campaign segment was "
                f"{best_previous_status['previous_campaign_status']} "
                f"with a "
                f"{best_previous_status['subscription_rate']:.2f}% "
                f"subscription rate.",

            "management_use":
                "Previous campaign history may provide a strong "
                "signal for future targeting."
        },

        {
            "finding_id": 9,

            "area": "Contact Month",

            "finding":
                f"The highest observed monthly subscription rate "
                f"occurred in "
                f"{best_month['month']} at "
                f"{best_month['subscription_rate']:.2f}%.",

            "management_use":
                "Investigate month-level campaign variation but do "
                "not interpret the dataset as a complete time series."
        },

        {
            "finding_id": 10,

            "area": "Call Duration",

            "finding":
                f"Average contact duration was "
                f"{subscriber_duration:.2f} minutes for subscribers "
                f"versus {non_subscriber_duration:.2f} minutes for "
                f"non-subscribers.",

            "management_use":
                "Use duration descriptively only. Exclude it from "
                "the pre-contact predictive model to prevent "
                "data leakage."
        }
    ]
)


print("\n4. MANAGEMENT FINDINGS")
print("-" * 80)

for _, row in management_findings.iterrows():

    print(
        f"\nFinding {row['finding_id']} "
        f"- {row['area']}"
    )

    print(row["finding"])

    print(
        "Management implication:",
        row["management_use"]
    )


management_findings.to_csv(
    reports_folder
    / "07_management_findings.csv",
    index=False
)


# --------------------------------------------------
# 13. Validate reports
# --------------------------------------------------

assert total_records > 0

assert total_subscribers > 0

assert overall_subscription_rate > 0

assert (
    segment_priority[
        "lift_vs_overall"
    ] > 0
).all()

assert (
    campaign_efficiency[
        "lift_vs_overall"
    ] > 0
).all()


print("\n5. VALIDATION")
print("-" * 80)

print(
    "Business KPI and management insight "
    "validation checks passed."
)


# --------------------------------------------------
# 14. Final message
# --------------------------------------------------

print("\n" + "=" * 80)
print("BUSINESS KPI & MANAGEMENT INSIGHT ANALYSIS COMPLETE")
print("=" * 80)

print("\nReports saved to:")
print(reports_folder)