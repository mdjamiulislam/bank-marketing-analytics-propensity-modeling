import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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
# 2. Load train and test datasets
# --------------------------------------------------

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("=" * 80)
print("PROJECT 4 - LOGISTIC REGRESSION MODEL")
print("=" * 80)

print("\nTraining dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)


# --------------------------------------------------
# 3. Define target and predictor variables
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
# 4. Separate X and y
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
# 5. Final leakage check
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
print("Duration is NOT used in this model.")


# --------------------------------------------------
# 6. Build preprocessing pipeline
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
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
# 7. Create Logistic Regression model
# --------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


# --------------------------------------------------
# 8. Build complete pipeline
# --------------------------------------------------

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            logistic_model
        )
    ]
)

print("\n3. MODEL PIPELINE CREATED")
print("-" * 80)

print(
    "StandardScaler + OneHotEncoder "
    "+ LogisticRegression"
)


# --------------------------------------------------
# 9. Train the model
# --------------------------------------------------

print("\n4. TRAINING MODEL")
print("-" * 80)

model_pipeline.fit(
    X_train,
    y_train
)

print("Logistic Regression training complete.")


# --------------------------------------------------
# 10. Generate test predictions
# --------------------------------------------------

y_pred = model_pipeline.predict(
    X_test
)

y_probability = (
    model_pipeline
    .predict_proba(X_test)[:, 1]
)

print("\n5. TEST PREDICTIONS CREATED")
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

positive_rate = (
    y_test.mean()
)

majority_baseline_accuracy = (
    max(
        positive_rate,
        1 - positive_rate
    )
)


metrics_df = pd.DataFrame({
    "metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Majority-Class Baseline Accuracy"
    ],

    "value": [
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(roc_auc, 4),
        round(
            majority_baseline_accuracy,
            4
        )
    ]
})


print("\n6. MODEL PERFORMANCE")
print("-" * 80)

print(
    metrics_df.to_string(
        index=False
    )
)


metrics_df.to_csv(
    reports_folder
    / "09_logistic_regression_metrics.csv",
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
    / "09_classification_report.csv"
)


print("\n7. CLASSIFICATION REPORT")
print("-" * 80)

print(classification_df.round(4))


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
    / "09_confusion_matrix.csv",
    index=False
)


print("\n8. CONFUSION MATRIX")
print("-" * 80)

print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# --------------------------------------------------
# 14. Plot confusion matrix
# --------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "Logistic Regression Confusion Matrix"
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
    / "09_logistic_confusion_matrix.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 15. ROC curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title(
    "Logistic Regression ROC Curve"
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
    / "09_logistic_roc_curve.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 16. Save test predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "actual_subscription": y_test.values,
    "predicted_subscription": y_pred,
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
    / "09_logistic_test_predictions.csv",
    index=False
)


# --------------------------------------------------
# 17. Extract feature coefficients
# --------------------------------------------------

fitted_preprocessor = (
    model_pipeline
    .named_steps[
        "preprocessor"
    ]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

coefficients = (
    model_pipeline
    .named_steps["model"]
    .coef_[0]
)


coefficient_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})


coefficient_df[
    "coefficient"
] = (
    coefficient_df[
        "coefficient"
    ]
    .round(6)
)


coefficient_df[
    "absolute_coefficient"
] = (
    coefficient_df[
        "coefficient"
    ]
    .abs()
)


coefficient_df = (
    coefficient_df
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
)


coefficient_df.to_csv(
    reports_folder
    / "09_logistic_coefficients.csv",
    index=False
)


# --------------------------------------------------
# 18. Print strongest positive coefficients
# --------------------------------------------------

positive_coefficients = (
    coefficient_df[
        coefficient_df[
            "coefficient"
        ] > 0
    ]
    .sort_values(
        "coefficient",
        ascending=False
    )
    .head(10)
)


print("\n9. STRONGEST POSITIVE COEFFICIENTS")
print("-" * 80)

print(
    positive_coefficients[
        [
            "feature",
            "coefficient"
        ]
    ]
    .to_string(index=False)
)


# --------------------------------------------------
# 19. Print strongest negative coefficients
# --------------------------------------------------

negative_coefficients = (
    coefficient_df[
        coefficient_df[
            "coefficient"
        ] < 0
    ]
    .sort_values(
        "coefficient",
        ascending=True
    )
    .head(10)
)


print("\n10. STRONGEST NEGATIVE COEFFICIENTS")
print("-" * 80)

print(
    negative_coefficients[
        [
            "feature",
            "coefficient"
        ]
    ]
    .to_string(index=False)
)


# --------------------------------------------------
# 20. Save trained pipeline
# --------------------------------------------------

model_file = (
    models_folder
    / "logistic_regression_pipeline.joblib"
)

joblib.dump(
    model_pipeline,
    model_file
)


print("\n11. TRAINED MODEL SAVED")
print("-" * 80)

print(model_file)


# --------------------------------------------------
# 21. Final validation
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

assert 0 <= roc_auc <= 1


print("\n12. FINAL VALIDATION")
print("-" * 80)

print(
    "All Logistic Regression "
    "validation checks passed."
)


# --------------------------------------------------
# 22. Final message
# --------------------------------------------------

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION MODEL COMPLETE")
print("=" * 80)