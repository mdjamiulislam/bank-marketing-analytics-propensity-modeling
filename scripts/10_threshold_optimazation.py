import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
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
    confusion_matrix
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
# 2. Load datasets
# --------------------------------------------------

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("=" * 85)
print("PROJECT 4 - RANDOM FOREST THRESHOLD OPTIMIZATION")
print("=" * 85)

print("\nTraining records:", len(train_df))
print("Testing records:", len(test_df))


# --------------------------------------------------
# 3. Define target and predictors
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
# 4. Separate original train and test data
# --------------------------------------------------

X_full_train = train_df[model_features].copy()
y_full_train = train_df[target_column].copy()

X_test = test_df[model_features].copy()
y_test = test_df[target_column].copy()


# --------------------------------------------------
# 5. Create sub-training / validation split
# --------------------------------------------------

X_train_sub, X_validation, y_train_sub, y_validation = (
    train_test_split(
        X_full_train,
        y_full_train,
        test_size=0.20,
        random_state=42,
        stratify=y_full_train
    )
)

print("\n1. THRESHOLD-TUNING SPLIT")
print("-" * 85)

print("Sub-training records:", len(X_train_sub))
print("Validation records:", len(X_validation))

print(
    "Sub-training positive rate:",
    round(y_train_sub.mean() * 100, 2),
    "%"
)

print(
    "Validation positive rate:",
    round(y_validation.mean() * 100, 2),
    "%"
)


# --------------------------------------------------
# 6. Build preprocessing
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


# --------------------------------------------------
# 7. Create Random Forest
# Same specification as Step 10
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


threshold_pipeline = Pipeline(
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
# 8. Train threshold-tuning model
# --------------------------------------------------

print("\n2. TRAINING THRESHOLD-TUNING MODEL")
print("-" * 85)

threshold_pipeline.fit(
    X_train_sub,
    y_train_sub
)

print("Training complete.")


# --------------------------------------------------
# 9. Generate validation probabilities
# --------------------------------------------------

validation_probability = (
    threshold_pipeline
    .predict_proba(X_validation)[:, 1]
)

validation_auc = roc_auc_score(
    y_validation,
    validation_probability
)

print("\nValidation ROC-AUC:", round(validation_auc, 4))


# --------------------------------------------------
# 10. Test multiple thresholds
# --------------------------------------------------

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

threshold_rows = []

for threshold in thresholds:

    validation_pred = (
        validation_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        validation_pred
    ).ravel()

    accuracy = accuracy_score(
        y_validation,
        validation_pred
    )

    precision = precision_score(
        y_validation,
        validation_pred,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        validation_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        validation_pred,
        zero_division=0
    )

    targeting_rate = (
        validation_pred.mean() * 100
    )

    subscriber_capture_rate = (
        tp / (tp + fn) * 100
        if (tp + fn) > 0
        else 0
    )

    threshold_rows.append({
        "threshold": threshold,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "targeting_rate_pct": round(
            targeting_rate,
            2
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "subscriber_capture_pct": round(
            subscriber_capture_rate,
            2
        )
    })


threshold_df = pd.DataFrame(
    threshold_rows
)


print("\n3. VALIDATION THRESHOLD PERFORMANCE")
print("-" * 85)

print(
    threshold_df.to_string(
        index=False
    )
)


threshold_df.to_csv(
    reports_folder
    / "11_threshold_analysis.csv",
    index=False
)


# --------------------------------------------------
# 11. Select threshold with highest validation F1
# --------------------------------------------------

best_row = (
    threshold_df
    .sort_values(
        [
            "f1_score",
            "recall",
            "precision"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
    .iloc[0]
)

selected_threshold = float(
    best_row["threshold"]
)


print("\n4. SELECTED BUSINESS THRESHOLD")
print("-" * 85)

print(
    "Selected threshold:",
    selected_threshold
)

print(
    "Validation Precision:",
    best_row["precision"]
)

print(
    "Validation Recall:",
    best_row["recall"]
)

print(
    "Validation F1:",
    best_row["f1_score"]
)

print(
    "Validation Targeting Rate:",
    best_row["targeting_rate_pct"],
    "%"
)

print(
    "Selection rule: highest validation F1."
)


# --------------------------------------------------
# 12. Plot Precision / Recall / F1 by threshold
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1_score"],
    marker="o",
    label="F1 Score"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    label=(
        f"Selected Threshold "
        f"= {selected_threshold:.2f}"
    )
)

plt.title(
    "Random Forest Threshold Optimization"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Metric Value"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    charts_folder
    / "11_threshold_precision_recall_f1.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 13. Plot targeting rate by threshold
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    threshold_df["threshold"],
    threshold_df["targeting_rate_pct"],
    marker="o"
)

plt.axvline(
    selected_threshold,
    linestyle="--"
)

plt.title(
    "Targeting Rate by Probability Threshold"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Records Targeted (%)"
)

plt.tight_layout()

plt.savefig(
    charts_folder
    / "11_targeting_rate_by_threshold.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 14. Rebuild final Random Forest pipeline
# --------------------------------------------------

final_categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

final_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features
        ),
        (
            "categorical",
            final_categorical_transformer,
            categorical_features
        )
    ]
)

final_random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            final_preprocessor
        ),
        (
            "model",
            final_random_forest
        )
    ]
)


# --------------------------------------------------
# 15. Retrain final model using ALL training data
# --------------------------------------------------

print("\n5. RETRAINING FINAL MODEL")
print("-" * 85)

final_pipeline.fit(
    X_full_train,
    y_full_train
)

print(
    "Final Random Forest trained "
    "on all training records."
)


# --------------------------------------------------
# 16. Generate untouched test probabilities
# --------------------------------------------------

test_probability = (
    final_pipeline
    .predict_proba(X_test)[:, 1]
)

test_pred_default = (
    test_probability >= 0.50
).astype(int)

test_pred_selected = (
    test_probability
    >= selected_threshold
).astype(int)


# --------------------------------------------------
# 17. Function to evaluate a threshold
# --------------------------------------------------

def evaluate_predictions(
    actual,
    predicted,
    probability,
    threshold_name,
    threshold_value
):

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted
    ).ravel()

    return {
        "threshold_name": threshold_name,
        "threshold": threshold_value,

        "accuracy": round(
            accuracy_score(
                actual,
                predicted
            ),
            4
        ),

        "precision": round(
            precision_score(
                actual,
                predicted,
                zero_division=0
            ),
            4
        ),

        "recall": round(
            recall_score(
                actual,
                predicted,
                zero_division=0
            ),
            4
        ),

        "f1_score": round(
            f1_score(
                actual,
                predicted,
                zero_division=0
            ),
            4
        ),

        "roc_auc": round(
            roc_auc_score(
                actual,
                probability
            ),
            4
        ),

        "targeting_rate_pct": round(
            predicted.mean() * 100,
            2
        ),

        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    }


# --------------------------------------------------
# 18. Compare default vs selected threshold
# --------------------------------------------------

default_results = evaluate_predictions(
    y_test,
    test_pred_default,
    test_probability,
    "Default",
    0.50
)

selected_results = evaluate_predictions(
    y_test,
    test_pred_selected,
    test_probability,
    "Optimized",
    selected_threshold
)

test_comparison_df = pd.DataFrame(
    [
        default_results,
        selected_results
    ]
)


print("\n6. FINAL TEST PERFORMANCE")
print("-" * 85)

print(
    test_comparison_df.to_string(
        index=False
    )
)


test_comparison_df.to_csv(
    reports_folder
    / "11_final_threshold_test_comparison.csv",
    index=False
)


# --------------------------------------------------
# 19. Save final test predictions
# --------------------------------------------------

test_predictions_df = pd.DataFrame({
    "actual_subscription":
        y_test.values,

    "subscription_probability":
        test_probability,

    "prediction_default_050":
        test_pred_default,

    "prediction_selected_threshold":
        test_pred_selected
})

test_predictions_df[
    "subscription_probability"
] = (
    test_predictions_df[
        "subscription_probability"
    ]
    .round(6)
)

test_predictions_df.to_csv(
    reports_folder
    / "11_final_test_predictions.csv",
    index=False
)


# --------------------------------------------------
# 20. Save selected threshold summary
# --------------------------------------------------

selected_summary = pd.DataFrame(
    [
        {
            "selection_dataset":
                "Validation",

            "selection_rule":
                "Maximum F1 Score",

            "selected_threshold":
                selected_threshold,

            "validation_precision":
                best_row["precision"],

            "validation_recall":
                best_row["recall"],

            "validation_f1":
                best_row["f1_score"],

            "validation_targeting_rate_pct":
                best_row[
                    "targeting_rate_pct"
                ],

            "test_precision":
                selected_results[
                    "precision"
                ],

            "test_recall":
                selected_results[
                    "recall"
                ],

            "test_f1":
                selected_results[
                    "f1_score"
                ],

            "test_roc_auc":
                selected_results[
                    "roc_auc"
                ],

            "test_targeting_rate_pct":
                selected_results[
                    "targeting_rate_pct"
                ]
        }
    ]
)


selected_summary.to_csv(
    reports_folder
    / "11_selected_threshold_summary.csv",
    index=False
)


# --------------------------------------------------
# 21. Save final trained pipeline
# --------------------------------------------------

final_model_file = (
    models_folder
    / "random_forest_final_pipeline.joblib"
)

joblib.dump(
    final_pipeline,
    final_model_file
)


print("\n7. FINAL MODEL SAVED")
print("-" * 85)

print(final_model_file)


# --------------------------------------------------
# 22. Final validation
# --------------------------------------------------

assert (
    0 < selected_threshold < 1
)

assert (
    (test_probability >= 0)
    & (test_probability <= 1)
).all()

assert len(test_probability) == len(y_test)

assert set(
    test_pred_selected
).issubset(
    {0, 1}
)


print("\n8. FINAL VALIDATION")
print("-" * 85)

print(
    "All threshold optimization "
    "validation checks passed."
)


# --------------------------------------------------
# 23. Final message
# --------------------------------------------------

print("\n" + "=" * 85)
print("THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 85)