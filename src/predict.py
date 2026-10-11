import pandas as pd
import joblib

from pathlib import Path


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "loan_train.csv"
MODEL_PATH = BASE_DIR / "models" / "creditfair_classification_model.pkl"


# ============================================
# LOAD DATA AND SAVED MODEL
# ============================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# Convert percentage column to numeric
df["revol_util"] = (
    df["revol_util"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["revol_util"] = pd.to_numeric(
    df["revol_util"],
    errors="coerce"
)


model = joblib.load(MODEL_PATH)

print("Saved CreditFair model loaded successfully.")




# ============================================
# USER INPUT
# ============================================

print("\n============================================")
print("        CREDITFAIR LOAN PREDICTION")
print("============================================")

loan_amnt = float(
    input("Enter Loan Amount: ")
)

int_rate = float(
    input("Enter Interest Rate (%): ")
)

annual_inc = float(
    input("Enter Annual Income: ")
)

home_ownership = input(
    "Enter Home Ownership (RENT/OWN/MORTGAGE): "
).upper()

purpose = input(
    "Enter Loan Purpose (e.g. credit_card): "
).lower()

dti = float(
    input("Enter Debt-to-Income Ratio (DTI): ")
)


# ============================================
# AUTOMATIC VALUES
# ============================================

funded_amnt = loan_amnt
funded_amnt_inv = loan_amnt

installment = loan_amnt / 36

term = df["term"].mode()[0]
grade = df["grade"].mode()[0]
sub_grade = df["sub_grade"].mode()[0]
emp_length = df["emp_length"].mode()[0]
verification_status = df[
    "verification_status"
].mode()[0]


# Numerical default values

delinq_2yrs = df["delinq_2yrs"].median()

inq_last_6mths = df[
    "inq_last_6mths"
].median()

mths_since_last_delinq = df[
    "mths_since_last_delinq"
].median()

mths_since_last_record = df[
    "mths_since_last_record"
].median()

open_acc = df["open_acc"].median()

pub_rec = df["pub_rec"].median()

revol_bal = df["revol_bal"].median()

revol_util = df["revol_util"].median()

total_acc = df["total_acc"].median()

pub_rec_bankruptcies = df[
    "pub_rec_bankruptcies"
].median()


# ============================================
# CREATE NEW LOAN
# ============================================

new_loan = pd.DataFrame([{

    "loan_amnt": loan_amnt,

    "funded_amnt": funded_amnt,

    "funded_amnt_inv": funded_amnt_inv,

    "term": term,

    "int_rate": int_rate,

    "installment": installment,

    "grade": grade,

    "sub_grade": sub_grade,

    "emp_length": emp_length,

    "home_ownership": home_ownership,

    "annual_inc": annual_inc,

    "verification_status": verification_status,

    "purpose": purpose,

    "dti": dti,

    "delinq_2yrs": delinq_2yrs,

    "inq_last_6mths": inq_last_6mths,

    "mths_since_last_delinq":
        mths_since_last_delinq,

    "mths_since_last_record":
        mths_since_last_record,

    "open_acc": open_acc,

    "pub_rec": pub_rec,

    "revol_bal": revol_bal,

    "revol_util": revol_util,

    "total_acc": total_acc,

    "pub_rec_bankruptcies":
        pub_rec_bankruptcies

}])


# ============================================
# PREDICTION
# ============================================

prediction = model.predict(new_loan)[0]

probabilities = model.predict_proba(new_loan)[0]

classes = model.classes_

predicted_probability = probabilities[
    list(classes).index(prediction)
]


# ============================================
# DISPLAY RESULT
# ============================================

print("\n============================================")
print("             PREDICTION RESULT")
print("============================================")

predicted_class = prediction

if predicted_class == 0:
    status = "FULLY PAID"
else:
    status = "CHARGED OFF"

print(f"Predicted Loan Status: {status}")

print(
    f"Prediction Probability: "
    f"{predicted_probability * 100:.2f}%"
)

print("\nClass Probabilities:")

for class_value, probability in zip(
    classes,
    probabilities
):

    if class_value == 0:
        class_name = "Fully Paid"
    else:
        class_name = "Charged Off"

    print(
        f"Class {class_value} ({class_name}): "
        f"{probability * 100:.2f}%"
    )

print("============================================")

