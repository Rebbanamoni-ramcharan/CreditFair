import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "loan_train.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)


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
# 3. SELECT TARGET
# ============================================================

TARGET = "loan_status"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nLoan status values:")
print(y.value_counts())

print("\nUnique loan status values:")
print(y.unique())


# ============================================================
# 4. REMOVE COLUMNS THAT CAN CAUSE DATA LEAKAGE
# ============================================================

columns_to_drop = [
    "id",
    "member_id",

    # Loan outcome / repayment information
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

    # Information not useful for this basic model
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

print("\nFeatures used:")
print(X.columns.tolist())

print("\nNumber of features before encoding:", X.shape[1])


# ============================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 6. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ============================================================
# 7. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ============================================================
# 8. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 9. CREATE CLASSIFICATION PIPELINE
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

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining Loan Default Classification model...")

model.fit(
    X_train,
    y_train
)

print("Classification model training completed successfully.")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

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

conf_matrix = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("   LOAN DEFAULT CLASSIFICATION RESULTS")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("========================================")


# ============================================================
# 15. SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Status": y_test.values,
    "Predicted Status": y_pred,
    "Default Probability": y_probability
})

print("\nSample Predictions:")
print(results.head(10))