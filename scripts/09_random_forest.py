import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# --------------------------------------------------
# 1. Locate project folders
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

train_file = (
    project_root
    / "data"
    / "processed"
    / "model_train.csv"
)

test_file = (
    project_root
    / "data"
    / "processed"
    / "model_test.csv"
)

reports_folder = project_root / "reports"

charts_folder = (
    reports_folder
    / "model_charts"
)

models_folder = project_root / "models"

reports_folder.mkdir(exist_ok=True)
charts_folder.mkdir(exist_ok=True)
models_folder.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load train and test data
# --------------------------------------------------

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("=" * 80)
print("PROJECT 4 - RANDOM FOREST MODEL")
print("=" * 80)

print("\nTraining dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)


# --------------------------------------------------
# 3. Define target and features
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
# 4. Create X and y
# --------------------------------------------------

X_train = train_df[model_features].copy()
y_train = train_df[target_column].copy()

X_test = test_df[model_features].copy()
y_test = test_df[target_column].copy()

print("\n1. MODEL INPUTS")
print("-" * 80)

print("Training predictors:", X_train.shape)
print("Training target:", y_train.shape)

print("Testing predictors:", X_test.shape)
print("Testing target:", y_test.shape)


# --------------------------------------------------
# 5. Data leakage validation
# --------------------------------------------------

prohibited_features = [
    "duration",
    "contact_duration_minutes",
    "subscribed",
    "subscription_flag",
    "campaign_record_id"
]

leakage_found = [
    column
    for column in prohibited_features
    if column in X_train.columns
]

print("\n2. DATA LEAKAGE CHECK")
print("-" * 80)

if leakage_found:

    raise ValueError(
        f"Leakage variables detected: {leakage_found}"
    )

print("No prohibited leakage variables found.")

print(
    "Duration variables remain excluded "
    "from the pre-contact model."
)


# --------------------------------------------------
# 6. Preprocess categorical variables
# --------------------------------------------------

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

print("\n3. PREPROCESSOR CREATED")
print("-" * 80)

print(
    "Numeric features: passed through unchanged."
)

print(
    "Categorical features: one-hot encoded."
)


# --------------------------------------------------
# 7. Create Random Forest
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("\n4. RANDOM FOREST PARAMETERS")
print("-" * 80)

print("Trees:", 500)
print("Maximum depth:", 12)
print("Minimum samples per leaf:", 5)

print(
    "Class weighting: balanced_subsample"
)


# --------------------------------------------------
# 8. Build full model pipeline
# --------------------------------------------------

rf_pipeline = Pipeline(
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


# --------------------------------------------------
# 9. Train model
# --------------------------------------------------

print("\n5. TRAINING RANDOM FOREST")
print("-" * 80)

rf_pipeline.fit(
    X_train,
    y_train
)

print("Random Forest training complete.")


# --------------------------------------------------
# 10. Generate test predictions
# --------------------------------------------------

y_pred = rf_pipeline.predict(
    X_test
)

y_probability = (
    rf_pipeline
    .predict_proba(X_test)[:, 1]
)

print("\n6. TEST PREDICTIONS CREATED")
print("-" * 80)

print(
    "Predicted subscribers:",
    int(y_pred.sum())
)

print(
    "Predicted non-subscribers:",
    int((y_pred == 0).sum())
)


# --------------------------------------------------
# 11. Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

positive_rate = y_test.mean()

majority_baseline_accuracy = max(
    positive_rate,
    1 - positive_rate
)


metrics_df = pd.DataFrame({
    "metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "Majority-Class Baseline Accuracy"
    ],

    "value": [
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(roc_auc, 4),
        round(pr_auc, 4),
        round(
            majority_baseline_accuracy,
            4
        )
    ]
})


print("\n7. RANDOM FOREST PERFORMANCE")
print("-" * 80)

print(
    metrics_df.to_string(
        index=False
    )
)


metrics_df.to_csv(
    reports_folder
    / "10_random_forest_metrics.csv",
    index=False
)


# --------------------------------------------------
# 12. Classification report
# --------------------------------------------------

classification_results = (
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Subscriber",
            "Subscriber"
        ],
        output_dict=True,
        zero_division=0
    )
)

classification_df = (
    pd.DataFrame(
        classification_results
    )
    .transpose()
)


classification_df.to_csv(
    reports_folder
    / "10_random_forest_classification_report.csv"
)


print("\n8. CLASSIFICATION REPORT")
print("-" * 80)

print(
    classification_df.round(4)
)


# --------------------------------------------------
# 13. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


confusion_df = pd.DataFrame(
    [
        {
            "actual": "Non-Subscriber",
            "predicted_non_subscriber": tn,
            "predicted_subscriber": fp
        },

        {
            "actual": "Subscriber",
            "predicted_non_subscriber": fn,
            "predicted_subscriber": tp
        }
    ]
)


confusion_df.to_csv(
    reports_folder
    / "10_random_forest_confusion_matrix.csv",
    index=False
)


print("\n9. CONFUSION MATRIX")
print("-" * 80)

print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# --------------------------------------------------
# 14. Confusion matrix chart
# --------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    [
        "Non-Subscriber",
        "Subscriber"
    ]
)

plt.yticks(
    [0, 1],
    [
        "Non-Subscriber",
        "Subscriber"
    ]
)

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    charts_folder
    / "10_random_forest_confusion_matrix.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 15. Random Forest ROC Curve
# --------------------------------------------------

rf_fpr, rf_tpr, rf_thresholds = (
    roc_curve(
        y_test,
        y_probability
    )
)


plt.figure(figsize=(7, 5))

plt.plot(
    rf_fpr,
    rf_tpr,
    label=(
        f"Random Forest "
        f"AUC = {roc_auc:.3f}"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title(
    "Random Forest ROC Curve"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    charts_folder
    / "10_random_forest_roc_curve.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 16. Save test predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "actual_subscription":
        y_test.values,

    "predicted_subscription":
        y_pred,

    "subscription_probability":
        y_probability
})


predictions_df[
    "subscription_probability"
] = (
    predictions_df[
        "subscription_probability"
    ]
    .round(6)
)


predictions_df.to_csv(
    reports_folder
    / "10_random_forest_test_predictions.csv",
    index=False
)


# --------------------------------------------------
# 17. Extract feature importance
# --------------------------------------------------

fitted_preprocessor = (
    rf_pipeline
    .named_steps[
        "preprocessor"
    ]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

feature_importances = (
    rf_pipeline
    .named_steps["model"]
    .feature_importances_
)


importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importances
})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


importance_df[
    "importance_pct"
] = (
    importance_df[
        "importance"
    ]
    * 100
).round(2)


importance_df.to_csv(
    reports_folder
    / "10_random_forest_feature_importance.csv",
    index=False
)


print("\n10. TOP 15 RANDOM FOREST FEATURES")
print("-" * 80)

print(
    importance_df[
        [
            "feature",
            "importance_pct"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# --------------------------------------------------
# 18. Feature importance chart
# --------------------------------------------------

top_features = (
    importance_df
    .head(15)
    .sort_values(
        "importance",
        ascending=True
    )
)


plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance_pct"]
)

plt.title(
    "Top 15 Random Forest Feature Importances"
)

plt.xlabel(
    "Feature Importance (%)"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    charts_folder
    / "10_random_forest_feature_importance.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 19. Load Logistic Regression metrics
# --------------------------------------------------

logistic_metrics_file = (
    reports_folder
    / "09_logistic_regression_metrics.csv"
)

logistic_metrics = pd.read_csv(
    logistic_metrics_file
)


logistic_metric_dict = dict(
    zip(
        logistic_metrics["metric"],
        logistic_metrics["value"]
    )
)


# --------------------------------------------------
# 20. Build direct model comparison
# --------------------------------------------------

comparison_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]


rf_metric_dict = dict(
    zip(
        metrics_df["metric"],
        metrics_df["value"]
    )
)


comparison_rows = []

for metric in comparison_metrics:

    logistic_value = (
        logistic_metric_dict[
            metric
        ]
    )

    rf_value = (
        rf_metric_dict[
            metric
        ]
    )

    if rf_value > logistic_value:
        better_model = "Random Forest"

    elif rf_value < logistic_value:
        better_model = "Logistic Regression"

    else:
        better_model = "Tie"

    comparison_rows.append({
        "metric": metric,
        "logistic_regression":
            logistic_value,
        "random_forest":
            rf_value,
        "difference_rf_minus_logistic":
            round(
                rf_value
                - logistic_value,
                4
            ),
        "better_model":
            better_model
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df.to_csv(
    reports_folder
    / "10_model_comparison.csv",
    index=False
)


print("\n11. MODEL COMPARISON")
print("-" * 80)

print(
    comparison_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# 21. Compare ROC curves
# --------------------------------------------------

logistic_predictions_file = (
    reports_folder
    / "09_logistic_test_predictions.csv"
)

logistic_predictions = pd.read_csv(
    logistic_predictions_file
)


logistic_fpr, logistic_tpr, _ = (
    roc_curve(
        logistic_predictions[
            "actual_subscription"
        ],

        logistic_predictions[
            "subscription_probability"
        ]
    )
)


logistic_auc = roc_auc_score(
    logistic_predictions[
        "actual_subscription"
    ],

    logistic_predictions[
        "subscription_probability"
    ]
)


plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=(
        f"Logistic Regression "
        f"AUC = {logistic_auc:.3f}"
    )
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=(
        f"Random Forest "
        f"AUC = {roc_auc:.3f}"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title(
    "ROC Curve - Model Comparison"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    charts_folder
    / "10_model_roc_comparison.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 22. Determine model winner
# --------------------------------------------------

logistic_auc_value = (
    logistic_metric_dict[
        "ROC-AUC"
    ]
)

random_forest_auc_value = (
    rf_metric_dict[
        "ROC-AUC"
    ]
)


print("\n12. MODEL SELECTION SUMMARY")
print("-" * 80)


if random_forest_auc_value > logistic_auc_value:

    selected_model = "Random Forest"

    print(
        "Random Forest has the higher ROC-AUC."
    )

else:

    selected_model = "Logistic Regression"

    print(
        "Logistic Regression has the "
        "higher or equal ROC-AUC."
    )


print(
    "Current probability-ranking winner:",
    selected_model
)

print(
    "\nFinal model selection will also consider "
    "recall, precision, F1 and threshold performance."
)


# --------------------------------------------------
# 23. Save model selection summary
# --------------------------------------------------

selection_summary = pd.DataFrame(
    [
        {
            "model":
                "Logistic Regression",

            "accuracy":
                logistic_metric_dict[
                    "Accuracy"
                ],

            "precision":
                logistic_metric_dict[
                    "Precision"
                ],

            "recall":
                logistic_metric_dict[
                    "Recall"
                ],

            "f1_score":
                logistic_metric_dict[
                    "F1 Score"
                ],

            "roc_auc":
                logistic_metric_dict[
                    "ROC-AUC"
                ]
        },

        {
            "model":
                "Random Forest",

            "accuracy":
                round(accuracy, 4),

            "precision":
                round(precision, 4),

            "recall":
                round(recall, 4),

            "f1_score":
                round(f1, 4),

            "roc_auc":
                round(roc_auc, 4)
        }
    ]
)


selection_summary.to_csv(
    reports_folder
    / "10_model_selection_summary.csv",
    index=False
)


# --------------------------------------------------
# 24. Save trained Random Forest
# --------------------------------------------------

model_file = (
    models_folder
    / "random_forest_pipeline.joblib"
)

joblib.dump(
    rf_pipeline,
    model_file
)


print("\n13. TRAINED MODEL SAVED")
print("-" * 80)

print(model_file)


# --------------------------------------------------
# 25. Final validation
# --------------------------------------------------

assert len(y_pred) == len(y_test)

assert len(y_probability) == len(y_test)

assert (
    (y_probability >= 0)
    & (y_probability <= 1)
).all()

assert set(y_pred).issubset(
    {0, 1}
)

assert abs(
    importance_df[
        "importance"
    ].sum() - 1
) < 0.001


print("\n14. FINAL VALIDATION")
print("-" * 80)

print(
    "All Random Forest validation "
    "checks passed."
)


# --------------------------------------------------
# 26. Final message
# --------------------------------------------------

print("\n" + "=" * 80)
print("RANDOM FOREST & MODEL COMPARISON COMPLETE")
print("=" * 80)