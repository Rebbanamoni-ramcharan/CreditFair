import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "loan_train.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 2. CLEAN PERCENTAGE COLUMNS
# ============================================================

df["int_rate"] = (
    df["int_rate"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["int_rate"] = pd.to_numeric(
    df["int_rate"],
    errors="coerce"
)

df["revol_util"] = (
    df["revol_util"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["revol_util"] = pd.to_numeric(
    df["revol_util"],
    errors="coerce"
)


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "loan_status"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 4. KEEP GROUP INFORMATION
# ============================================================

# We will evaluate fairness across these available groups.
group_column = "home_ownership"

groups = X[group_column].copy()


# ============================================================
# 5. REMOVE COLUMNS THAT SHOULD NOT BE USED
# ============================================================

columns_to_drop = [
    "id",
    "member_id",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_amnt",
    "emp_title",
    "url",
    "desc",
    "title",
    "zip_code",
    "issue_d",
    "earliest_cr_line",
    "last_pymnt_d",
    "last_credit_pull_d",
    "addr_state"
]

columns_to_drop = [
    column for column in columns_to_drop
    if column in X.columns
]

X = X.drop(columns=columns_to_drop)


# ============================================================
# 6. NUMERICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ============================================================
# 7. CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ============================================================
# 8. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 9. CLASSIFICATION MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining model for fairness assessment...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 12. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 13. CREATE EVALUATION DATAFRAME
# ============================================================

evaluation = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred,
    "group": groups.loc[X_test.index].values
})


# ============================================================
# 14. FAIRNESS ANALYSIS
# ============================================================

print("\n========================================")
print("       FAIR LENDING ASSESSMENT")
print("========================================")

print("\nEvaluation Group:", group_column)

fairness_results = []

for group in sorted(evaluation["group"].dropna().unique()):

    group_data = evaluation[
        evaluation["group"] == group
    ]

    actual = group_data["actual"]
    predicted = group_data["predicted"]

    accuracy = accuracy_score(
        actual,
        predicted
    )

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    positive_rate = predicted.mean()

    fairness_results.append({
        "Group": group,
        "Samples": len(group_data),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Positive Prediction Rate": positive_rate
    })


# ============================================================
# 15. DISPLAY FAIRNESS RESULTS
# ============================================================

fairness_df = pd.DataFrame(
    fairness_results
)

print("\nFairness Results:")
print(
    fairness_df.to_string(
        index=False
    )
)


# ============================================================
# 16. COMPARE GROUPS
# ============================================================

print("\n========================================")
print("       GROUP COMPARISON")
print("========================================")

max_recall = fairness_df["Recall"].max()
min_recall = fairness_df["Recall"].min()

max_precision = fairness_df["Precision"].max()
min_precision = fairness_df["Precision"].min()

recall_difference = max_recall - min_recall
precision_difference = max_precision - min_precision

print(
    f"Recall difference between groups: "
    f"{recall_difference:.4f}"
)

print(
    f"Precision difference between groups: "
    f"{precision_difference:.4f}"
)

print("========================================")


# ============================================================
# 17. SAVE FAIRNESS REPORT
# ============================================================

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

output_file = RESULTS_DIR / "fairness_report.csv"

fairness_df.to_csv(
    output_file,
    index=False
)

print("\nFairness report saved to:")
print(output_file)