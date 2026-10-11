import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "../data/raw/loan_train.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)


# ============================================================
# 2. CLEAN PERCENTAGE COLUMNS
# ============================================================

# int_rate and revol_util are stored as strings such as "10.65%"
# Convert them into numerical values.

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

TARGET = "loan_amnt"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 4. REMOVE COLUMNS NOT SUITABLE AS FEATURES
# ============================================================

columns_to_drop = [
    # Identifiers
    "id",
    "member_id",

    # Values directly related to the requested loan amount
    "funded_amnt",
    "funded_amnt_inv",
    "installment",

    # Information that occurs after / during the loan
    "loan_status",
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

    # High-cardinality / text fields
    "emp_title",
    "url",
    "desc",
    "title",
    "zip_code",

    # Date fields
    "issue_d",
    "earliest_cr_line",
    "last_pymnt_d",
    "last_credit_pull_d",

    # Location field
    "addr_state"
]

# Only drop columns that actually exist
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
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 7. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 8. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 9. CREATE LINEAR REGRESSION PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
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
    random_state=42
)


print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining Linear Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("       LINEAR REGRESSION RESULTS")
print("========================================")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("========================================")


# ============================================================
# 15. SHOW SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Loan Amount": y_test.values,
    "Predicted Loan Amount": y_pred
})

print("\nSample Predictions:")
print(results.head(10))