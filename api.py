"""
CreditFair - FastAPI Backend
Serves the ML model predictions via REST API
and static frontend files.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "creditfair_classification_model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "loan_train.csv"
WEB_DIR = BASE_DIR / "web"


# ============================================================
# LOAD MODEL + DATASET DEFAULTS
# ============================================================

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

df["revol_util"] = (
    df["revol_util"]
    .astype(str)
    .str.replace("%", "", regex=False)
)
df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="CreditFair API",
    description="AI Loan Default Prediction & Fairness Assessment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE SCHEMA
# ============================================================

class LoanInput(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    home_ownership: str   # RENT | OWN | MORTGAGE | OTHER
    purpose: str          # credit_card | debt_consolidation | ...
    dti: float


class PredictionResult(BaseModel):
    predicted_status: str          # "Fully Paid" | "Charged Off"
    predicted_class: int           # 0 | 1
    probability_fully_paid: float
    probability_charged_off: float


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict", response_model=PredictionResult)
def predict(data: LoanInput):
    """
    Run the saved CreditFair classification model on a
    single loan application and return the prediction.
    """

    # Fill automatic / derived fields from dataset statistics
    funded_amnt       = data.loan_amnt
    funded_amnt_inv   = data.loan_amnt
    installment       = data.loan_amnt / 36

    term                   = df["term"].mode()[0]
    grade                  = df["grade"].mode()[0]
    sub_grade              = df["sub_grade"].mode()[0]
    emp_length             = df["emp_length"].mode()[0]
    verification_status    = df["verification_status"].mode()[0]

    delinq_2yrs            = float(df["delinq_2yrs"].median())
    inq_last_6mths         = float(df["inq_last_6mths"].median())
    mths_since_last_delinq = float(df["mths_since_last_delinq"].median())
    mths_since_last_record = float(df["mths_since_last_record"].median())
    open_acc               = float(df["open_acc"].median())
    pub_rec                = float(df["pub_rec"].median())
    revol_bal              = float(df["revol_bal"].median())
    revol_util             = float(df["revol_util"].median())
    total_acc              = float(df["total_acc"].median())
    pub_rec_bankruptcies   = float(df["pub_rec_bankruptcies"].median())

    new_loan = pd.DataFrame([{
        "loan_amnt":               data.loan_amnt,
        "funded_amnt":             funded_amnt,
        "funded_amnt_inv":         funded_amnt_inv,
        "term":                    term,
        "int_rate":                data.int_rate,
        "installment":             installment,
        "grade":                   grade,
        "sub_grade":               sub_grade,
        "emp_length":              emp_length,
        "home_ownership":          data.home_ownership.upper(),
        "annual_inc":              data.annual_inc,
        "verification_status":     verification_status,
        "purpose":                 data.purpose.lower(),
        "dti":                     data.dti,
        "delinq_2yrs":             delinq_2yrs,
        "inq_last_6mths":          inq_last_6mths,
        "mths_since_last_delinq":  mths_since_last_delinq,
        "mths_since_last_record":  mths_since_last_record,
        "open_acc":                open_acc,
        "pub_rec":                 pub_rec,
        "revol_bal":               revol_bal,
        "revol_util":              revol_util,
        "total_acc":               total_acc,
        "pub_rec_bankruptcies":    pub_rec_bankruptcies,
    }])

    prediction    = int(model.predict(new_loan)[0])
    probabilities = model.predict_proba(new_loan)[0]
    classes       = list(model.classes_)

    prob_fully_paid   = float(probabilities[classes.index(0)]) * 100
    prob_charged_off  = float(probabilities[classes.index(1)]) * 100

    return PredictionResult(
        predicted_status     = "Fully Paid" if prediction == 0 else "Charged Off",
        predicted_class      = prediction,
        probability_fully_paid  = round(prob_fully_paid, 2),
        probability_charged_off = round(prob_charged_off, 2),
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": "CreditFair API"}


# ============================================================
# SERVE FRONTEND (static files)
# ============================================================

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(WEB_DIR / "index.html"))
