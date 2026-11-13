from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    rng = np.random.default_rng(42)
    n = 600
    credit = rng.choice([0.0, 1.0, np.nan], n, p=[.18, .77, .05])
    income = np.maximum(1500, rng.lognormal(8.7, .55, n)).round()
    co_income = np.maximum(0, rng.lognormal(7.5, .8, n) - 1000).round()
    amount = np.maximum(20, (income + co_income) / 18 + rng.normal(0, 35, n)).round()
    score = .05 * np.log1p(income) + .015 * np.log1p(co_income) + np.nan_to_num(credit, nan=.55) * 2
    approved = rng.random(n) < (1 / (1 + np.exp(-(score - 1.25))))
    df = pd.DataFrame({
        "Loan_ID": [f"LP{10000+i}" for i in range(n)],
        "Gender": rng.choice(["Male", "Female", np.nan], n, p=[.79, .18, .03]),
        "Married": rng.choice(["Yes", "No", np.nan], n, p=[.63, .34, .03]),
        "Dependents": rng.choice(["0", "1", "2", "3+", np.nan], n, p=[.55, .15, .16, .1, .04]),
        "Education": rng.choice(["Graduate", "Not Graduate"], n, p=[.78, .22]),
        "Self_Employed": rng.choice(["Yes", "No", np.nan], n, p=[.12, .84, .04]),
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "LoanAmount": amount,
        "Loan_Amount_Term": rng.choice([180, 240, 360, 480], n, p=[.05, .08, .8, .07]),
        "Credit_History": credit,
        "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n, p=[.3, .48, .22]),
        "Loan_Status": np.where(approved, "Y", "N"),
    })
    out = ROOT / "data" / "loan.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Created {out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
