import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "loan_train.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================
# CLEAN DATA
# ============================================

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


# ============================================
# TARGET
# ============================================

TARGET = "loan_status"

y = df[TARGET]


# ============================================
# DROP COLUMNS
# ============================================

DROP_COLUMNS = [
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

X = df.drop(
    columns=[TARGET] + DROP_COLUMNS
)


# ============================================
# FEATURE TYPES
# ============================================

numerical_features = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "pub_rec_bankruptcies"
]

categorical_features = [
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose"
]


# ============================================
# PREPROCESSING
# ============================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ============================================
# MODEL
# ============================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])


# ============================================
# TRAIN
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================
# SAVE MODEL
# ============================================

MODEL_PATH = MODEL_DIR / "creditfair_classification_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\n============================================")
print("          MODEL SAVED SUCCESSFULLY")
print("============================================")

print("Model path:")
print(MODEL_PATH)

print("============================================")
