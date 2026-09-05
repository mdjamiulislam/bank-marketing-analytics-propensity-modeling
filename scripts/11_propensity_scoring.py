import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import roc_auc_score


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

threshold_file = (
    project_root
    / "reports"
    / "11_selected_threshold_summary.csv"
)

processed_folder = (
    project_root
    / "data"
    / "processed"
)

reports_folder = project_root / "reports"

charts_folder = (
    reports_folder
    / "model_charts"
)

models_folder = project_root / "models"

processed_folder.mkdir(exist_ok=True)
reports_folder.mkdir(exist_ok=True)
charts_folder.mkdir(exist_ok=True)
models_folder.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load historical dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

print("=" * 90)
print("PROJECT 4 - PROPENSITY SCORING")
print("=" * 90)

print("\nDataset loaded successfully.")
print("Campaign records:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 3. Load selected business threshold
# --------------------------------------------------

threshold_summary = pd.read_csv(
    threshold_file
)

selected_threshold = float(
    threshold_summary.loc[
        0,
        "selected_threshold"
    ]
)

print("\n1. BUSINESS THRESHOLD")
print("-" * 90)

print(
    "Validated targeting threshold:",
    selected_threshold
)


# --------------------------------------------------
# 4. Define model features
# --------------------------------------------------

target_column = "subscription_flag"

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

required_columns = (
    model_features
    + [target_column]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )

print("\n2. FEATURE VALIDATION")
print("-" * 90)

print(
    "All required scoring features "
    "are available."
)


# --------------------------------------------------
# 6. Create model-ready X and y
# --------------------------------------------------

X = df[model_features].copy()

y = df[target_column].copy()


# --------------------------------------------------
# 7. Handle intentional missing value
# --------------------------------------------------

missing_days = (
    X[
        "days_since_previous_contact"
    ]
    .isna()
    .sum()
)

X[
    "days_since_previous_contact"
] = (
    X[
        "days_since_previous_contact"
    ]
    .fillna(0)
)


print("\n3. MODEL MISSING-VALUE TREATMENT")
print("-" * 90)

print(
    "days_since_previous_contact "
    "missing values replaced:",
    missing_days
)

print(
    "previously_contacted remains "
    "available to preserve meaning."
)


# --------------------------------------------------
# 8. Confirm no remaining missing values
# --------------------------------------------------

remaining_missing = (
    X.isna().sum().sum()
)

if remaining_missing > 0:

    raise ValueError(
        f"{remaining_missing} "
        f"unexpected missing values remain."
    )

print(
    "Remaining predictor missing values:",
    remaining_missing
)


# --------------------------------------------------
# 9. Function to build Random Forest pipeline
# --------------------------------------------------

def build_random_forest_pipeline():

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features
            ),

            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                random_forest
            )
        ]
    )

    return pipeline


# --------------------------------------------------
# 10. Create 5-fold cross-validation design
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n4. OUT-OF-FOLD SCORING")
print("-" * 90)

print(
    "Generating 5-fold out-of-fold "
    "probabilities..."
)

print(
    "This may take several minutes."
)


# --------------------------------------------------
# 11. Generate honest historical probabilities
# --------------------------------------------------

oof_pipeline = (
    build_random_forest_pipeline()
)

oof_probability_matrix = (
    cross_val_predict(
        oof_pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )
)

oof_probability = (
    oof_probability_matrix[:, 1]
)


print(
    "Out-of-fold probabilities created."
)


# --------------------------------------------------
# 12. Evaluate overall OOF ranking performance
# --------------------------------------------------

oof_roc_auc = roc_auc_score(
    y,
    oof_probability
)


print("\n5. OUT-OF-FOLD MODEL PERFORMANCE")
print("-" * 90)

print(
    "OOF ROC-AUC:",
    round(oof_roc_auc, 4)
)


# --------------------------------------------------
# 13. Add probabilities to historical dataset
# --------------------------------------------------

df[
    "subscription_probability"
] = oof_probability

df[
    "subscription_probability_pct"
] = (
    df[
        "subscription_probability"
    ]
    * 100
).round(2)


# --------------------------------------------------
# 14. Create validated priority-target flag
# --------------------------------------------------

df[
    "priority_target_flag"
] = (
    df[
        "subscription_probability"
    ]
    >= selected_threshold
).astype("int8")


df[
    "priority_target_status"
] = np.where(
    df["priority_target_flag"] == 1,
    "Priority Target",
    "Below Target Threshold"
)


print("\n6. PRIORITY TARGET CLASSIFICATION")
print("-" * 90)

print(
    df[
        "priority_target_status"
    ]
    .value_counts()
)


# --------------------------------------------------
# 15. Create relative propensity bands
# --------------------------------------------------

score_rank = (
    df[
        "subscription_probability"
    ]
    .rank(
        method="first"
    )
)


df[
    "propensity_band"
] = pd.qcut(
    score_rank,
    q=3,
    labels=[
        "Low Propensity",
        "Medium Propensity",
        "High Propensity"
    ]
)


propensity_band_order = {
    "Low Propensity": 1,
    "Medium Propensity": 2,
    "High Propensity": 3
}


df[
    "propensity_band_order"
] = (
    df[
        "propensity_band"
    ]
    .astype(str)
    .map(
        propensity_band_order
    )
)


print("\n7. PROPENSITY BANDS CREATED")
print("-" * 90)

print(
    df[
        "propensity_band"
    ]
    .value_counts()
)


# --------------------------------------------------
# 16. Create propensity deciles
# Decile 1 = highest predicted probability
# --------------------------------------------------

descending_rank = (
    df[
        "subscription_probability"
    ]
    .rank(
        method="first",
        ascending=False
    )
)


df[
    "propensity_decile"
] = (
    pd.qcut(
        descending_rank,
        q=10,
        labels=list(
            range(1, 11)
        )
    )
    .astype(int)
)


print("\n8. PROPENSITY DECILES CREATED")
print("-" * 90)

print(
    df[
        "propensity_decile"
    ]
    .value_counts()
    .sort_index()
)

print(
    "\nDecile 1 = highest-propensity "
    "10% of records."
)


# --------------------------------------------------
# 17. Create score percentile
# --------------------------------------------------

df[
    "propensity_percentile"
] = (
    df[
        "subscription_probability"
    ]
    .rank(
        method="average",
        pct=True
    )
    * 100
).round(2)


# --------------------------------------------------
# 18. Overall campaign baseline
# --------------------------------------------------

overall_subscription_rate = (
    df[
        "subscription_flag"
    ]
    .mean()
    * 100
)

total_subscribers = int(
    df[
        "subscription_flag"
    ]
    .sum()
)


# --------------------------------------------------
# 19. Propensity band performance
# --------------------------------------------------

band_summary = (
    df
    .groupby(
        [
            "propensity_band",
            "propensity_band_order"
        ],
        observed=True
    )
    .agg(
        campaign_records=(
            "campaign_record_id",
            "count"
        ),

        actual_subscribers=(
            "subscription_flag",
            "sum"
        ),

        actual_subscription_rate=(
            "subscription_flag",
            "mean"
        ),

        average_predicted_probability=(
            "subscription_probability",
            "mean"
        ),

        minimum_probability=(
            "subscription_probability",
            "min"
        ),

        maximum_probability=(
            "subscription_probability",
            "max"
        )
    )
    .reset_index()
)


band_summary[
    "actual_subscription_rate"
] = (
    band_summary[
        "actual_subscription_rate"
    ]
    * 100
)


band_summary[
    "average_predicted_probability"
] = (
    band_summary[
        "average_predicted_probability"
    ]
    * 100
)


band_summary[
    "minimum_probability"
] = (
    band_summary[
        "minimum_probability"
    ]
    * 100
)


band_summary[
    "maximum_probability"
] = (
    band_summary[
        "maximum_probability"
    ]
    * 100
)


band_summary[
    "record_share_pct"
] = (
    band_summary[
        "campaign_records"
    ]
    / len(df)
    * 100
)


band_summary[
    "subscriber_share_pct"
] = (
    band_summary[
        "actual_subscribers"
    ]
    / total_subscribers
    * 100
)


band_summary[
    "lift_vs_overall"
] = (
    band_summary[
        "actual_subscription_rate"
    ]
    / overall_subscription_rate
)


round_columns = [
    "actual_subscription_rate",
    "average_predicted_probability",
    "minimum_probability",
    "maximum_probability",
    "record_share_pct",
    "subscriber_share_pct",
    "lift_vs_overall"
]

band_summary[
    round_columns
] = (
    band_summary[
        round_columns
    ]
    .round(2)
)


band_summary = (
    band_summary
    .sort_values(
        "propensity_band_order",
        ascending=False
    )
)


print("\n9. PROPENSITY BAND PERFORMANCE")
print("-" * 90)

print(
    band_summary.to_string(
        index=False
    )
)


band_summary.to_csv(
    reports_folder
    / "12_propensity_band_summary.csv",
    index=False
)


# --------------------------------------------------
# 20. Priority-target performance
# --------------------------------------------------

priority_summary = (
    df
    .groupby(
        "priority_target_status"
    )
    .agg(
        campaign_records=(
            "campaign_record_id",
            "count"
        ),

        actual_subscribers=(
            "subscription_flag",
            "sum"
        ),

        actual_subscription_rate=(
            "subscription_flag",
            "mean"
        ),

        average_predicted_probability=(
            "subscription_probability",
            "mean"
        )
    )
    .reset_index()
)


priority_summary[
    "actual_subscription_rate"
] = (
    priority_summary[
        "actual_subscription_rate"
    ]
    * 100
)


priority_summary[
    "average_predicted_probability"
] = (
    priority_summary[
        "average_predicted_probability"
    ]
    * 100
)


priority_summary[
    "record_share_pct"
] = (
    priority_summary[
        "campaign_records"
    ]
    / len(df)
    * 100
)


priority_summary[
    "subscriber_share_pct"
] = (
    priority_summary[
        "actual_subscribers"
    ]
    / total_subscribers
    * 100
)


priority_summary[
    "lift_vs_overall"
] = (
    priority_summary[
        "actual_subscription_rate"
    ]
    / overall_subscription_rate
)


priority_round_columns = [
    "actual_subscription_rate",
    "average_predicted_probability",
    "record_share_pct",
    "subscriber_share_pct",
    "lift_vs_overall"
]


priority_summary[
    priority_round_columns
] = (
    priority_summary[
        priority_round_columns
    ]
    .round(2)
)


print("\n10. PRIORITY TARGET PERFORMANCE")
print("-" * 90)

print(
    priority_summary.to_string(
        index=False
    )
)


priority_summary.to_csv(
    reports_folder
    / "12_priority_target_summary.csv",
    index=False
)


# --------------------------------------------------
# 21. Propensity decile performance
# --------------------------------------------------

decile_summary = (
    df
    .groupby(
        "propensity_decile"
    )
    .agg(
        campaign_records=(
            "campaign_record_id",
            "count"
        ),

        actual_subscribers=(
            "subscription_flag",
            "sum"
        ),

        actual_subscription_rate=(
            "subscription_flag",
            "mean"
        ),

        average_probability=(
            "subscription_probability",
            "mean"
        )
    )
    .reset_index()
)


decile_summary[
    "actual_subscription_rate"
] = (
    decile_summary[
        "actual_subscription_rate"
    ]
    * 100
)


decile_summary[
    "average_probability"
] = (
    decile_summary[
        "average_probability"
    ]
    * 100
)


decile_summary[
    "subscriber_share_pct"
] = (
    decile_summary[
        "actual_subscribers"
    ]
    / total_subscribers
    * 100
)


decile_summary[
    "lift_vs_overall"
] = (
    decile_summary[
        "actual_subscription_rate"
    ]
    / overall_subscription_rate
)


decile_summary[
    [
        "actual_subscription_rate",
        "average_probability",
        "subscriber_share_pct",
        "lift_vs_overall"
    ]
] = (
    decile_summary[
        [
            "actual_subscription_rate",
            "average_probability",
            "subscriber_share_pct",
            "lift_vs_overall"
        ]
    ]
    .round(2)
)


# Cumulative subscriber capture
decile_summary = (
    decile_summary
    .sort_values(
        "propensity_decile"
    )
)


decile_summary[
    "cumulative_subscribers"
] = (
    decile_summary[
        "actual_subscribers"
    ]
    .cumsum()
)


decile_summary[
    "cumulative_subscriber_capture_pct"
] = (
    decile_summary[
        "cumulative_subscribers"
    ]
    / total_subscribers
    * 100
).round(2)


print("\n11. PROPENSITY DECILE PERFORMANCE")
print("-" * 90)

print(
    decile_summary.to_string(
        index=False
    )
)


decile_summary.to_csv(
    reports_folder
    / "12_propensity_decile_performance.csv",
    index=False
)


# --------------------------------------------------
# 22. Create overall scoring summary
# --------------------------------------------------

priority_records = int(
    df[
        "priority_target_flag"
    ]
    .sum()
)

priority_subscribers = int(
    df.loc[
        df[
            "priority_target_flag"
        ] == 1,
        "subscription_flag"
    ]
    .sum()
)


scoring_summary = pd.DataFrame({
    "metric": [
        "Historical campaign records",
        "Actual subscribers",
        "Overall subscription rate (%)",
        "OOF ROC-AUC",
        "Validated targeting threshold",
        "Priority target records",
        "Priority target share (%)",
        "Subscribers in priority target",
        "Priority subscriber capture (%)",
        "Minimum predicted probability",
        "Maximum predicted probability",
        "Average predicted probability"
    ],

    "value": [
        len(df),
        total_subscribers,
        round(
            overall_subscription_rate,
            2
        ),
        round(
            oof_roc_auc,
            4
        ),
        selected_threshold,
        priority_records,
        round(
            priority_records
            / len(df)
            * 100,
            2
        ),
        priority_subscribers,
        round(
            priority_subscribers
            / total_subscribers
            * 100,
            2
        ),
        round(
            df[
                "subscription_probability"
            ].min(),
            4
        ),
        round(
            df[
                "subscription_probability"
            ].max(),
            4
        ),
        round(
            df[
                "subscription_probability"
            ].mean(),
            4
        )
    ]
})


print("\n12. SCORING SUMMARY")
print("-" * 90)

print(
    scoring_summary.to_string(
        index=False
    )
)


scoring_summary.to_csv(
    reports_folder
    / "12_scoring_summary.csv",
    index=False
)


# --------------------------------------------------
# 23. Save Power BI-ready scored dataset
# --------------------------------------------------

scored_output = (
    processed_folder
    / "bank_marketing_scored.csv"
)


df.to_csv(
    scored_output,
    index=False
)


print("\n13. POWER BI-READY DATASET SAVED")
print("-" * 90)

print(scored_output)


# --------------------------------------------------
# 24. Create propensity-band chart
# --------------------------------------------------

chart_band = (
    band_summary
    .sort_values(
        "propensity_band_order"
    )
)


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    chart_band[
        "propensity_band"
    ].astype(str),

    chart_band[
        "actual_subscription_rate"
    ]
)

plt.title(
    "Actual Subscription Rate by Propensity Band"
)

plt.xlabel(
    "Propensity Band"
)

plt.ylabel(
    "Actual Subscription Rate (%)"
)

plt.tight_layout()

plt.savefig(
    charts_folder
    / "12_subscription_rate_by_propensity_band.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 25. Create decile lift chart
# --------------------------------------------------

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    decile_summary[
        "propensity_decile"
    ].astype(str),

    decile_summary[
        "lift_vs_overall"
    ]
)

plt.title(
    "Subscription Lift by Propensity Decile"
)

plt.xlabel(
    "Propensity Decile (1 = Highest)"
)

plt.ylabel(
    "Lift vs Overall Subscription Rate"
)

plt.tight_layout()

plt.savefig(
    charts_folder
    / "12_propensity_decile_lift.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 26. Create score-distribution chart
# --------------------------------------------------

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    df[
        "subscription_probability"
    ],
    bins=30
)

plt.axvline(
    selected_threshold,
    linestyle="--"
)

plt.title(
    "Distribution of Subscription Propensity Scores"
)

plt.xlabel(
    "Predicted Subscription Probability"
)

plt.ylabel(
    "Campaign Records"
)

plt.tight_layout()

plt.savefig(
    charts_folder
    / "12_propensity_score_distribution.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 27. Train final deployment model
# --------------------------------------------------

print("\n14. TRAINING FINAL DEPLOYMENT MODEL")
print("-" * 90)

deployment_pipeline = (
    build_random_forest_pipeline()
)

deployment_pipeline.fit(
    X,
    y
)

print(
    "Final deployment model trained "
    "on all historical records."
)


# --------------------------------------------------
# 28. Save deployment model
# --------------------------------------------------

deployment_model_file = (
    models_folder
    / "random_forest_deployment_pipeline.joblib"
)

joblib.dump(
    deployment_pipeline,
    deployment_model_file
)


print("\n15. DEPLOYMENT MODEL SAVED")
print("-" * 90)

print(
    deployment_model_file
)


# --------------------------------------------------
# 29. Validate scored dataset
# --------------------------------------------------

assert len(df) > 0

assert (
    df[
        "campaign_record_id"
    ].is_unique
)

assert (
    df[
        "subscription_probability"
    ]
    .between(
        0,
        1
    )
    .all()
)

assert (
    df[
        "priority_target_flag"
    ]
    .isin(
        [0, 1]
    )
    .all()
)

assert (
    df[
        "propensity_band"
    ]
    .notna()
    .all()
)

assert (
    df[
        "propensity_decile"
    ]
    .between(
        1,
        10
    )
    .all()
)


print("\n16. FINAL VALIDATION")
print("-" * 90)

print(
    "All propensity scoring "
    "validation checks passed."
)


# --------------------------------------------------
# 30. Final message
# --------------------------------------------------

print("\n" + "=" * 90)
print("PROPENSITY SCORING COMPLETE")
print("=" * 90)