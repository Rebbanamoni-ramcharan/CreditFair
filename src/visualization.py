import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "loan_train.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. LOAN STATUS DISTRIBUTION
# ============================================================

status_counts = df["loan_status"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

status_counts.plot(kind="bar")

plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Loans")
plt.xticks(rotation=0)

plt.tight_layout()

status_path = RESULTS_DIR / "loan_status_distribution.png"
plt.savefig(status_path)

plt.show()

print("\nSaved:")
print(status_path)


# ============================================================
# 4. LOAN AMOUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["loan_amnt"],
    bins=30
)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")

plt.tight_layout()

loan_amount_path = RESULTS_DIR / "loan_amount_distribution.png"
plt.savefig(loan_amount_path)

plt.show()

print("\nSaved:")
print(loan_amount_path)


# ============================================================
# 5. HOME OWNERSHIP DISTRIBUTION
# ============================================================

home_counts = df["home_ownership"].value_counts()

plt.figure(figsize=(8, 5))

home_counts.plot(kind="bar")

plt.title("Home Ownership Distribution")
plt.xlabel("Home Ownership")
plt.ylabel("Number of Customers")

plt.xticks(rotation=45)

plt.tight_layout()

home_path = RESULTS_DIR / "home_ownership_distribution.png"
plt.savefig(home_path)

plt.show()

print("\nSaved:")
print(home_path)


# ============================================================
# 6. LOAN AMOUNT BY LOAN STATUS
# ============================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="loan_amnt",
    by="loan_status"
)

plt.title("Loan Amount by Loan Status")
plt.suptitle("")

plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")

plt.tight_layout()

boxplot_path = RESULTS_DIR / "loan_amount_by_status.png"
plt.savefig(boxplot_path)

plt.show()

print("\nSaved:")
print(boxplot_path)


# ============================================================
# 7. FAIRNESS RESULTS
# ============================================================

fairness_path = RESULTS_DIR / "fairness_report.csv"

if fairness_path.exists():

    fairness_df = pd.read_csv(
        fairness_path
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        fairness_df["Group"],
        fairness_df["Recall"]
    )

    plt.title(
        "Recall Comparison Across Home Ownership Groups"
    )

    plt.xlabel("Home Ownership")
    plt.ylabel("Recall")

    plt.xticks(rotation=45)

    plt.tight_layout()

    fairness_chart_path = (
        RESULTS_DIR /
        "fairness_recall_comparison.png"
    )

    plt.savefig(
        fairness_chart_path
    )

    plt.show()

    print("\nSaved:")
    print(fairness_chart_path)

else:

    print(
        "\nFairness report not found."
    )

    print(
        "Run fairness.py first."
    )


# ============================================================
# 8. COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("      VISUALIZATION COMPLETED")
print("========================================")

print(
    "\nAll available charts have been "
    "generated in the results folder."
)