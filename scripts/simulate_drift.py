from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def main():
    source = ROOT / "data" / "test.csv"
    frame = pd.read_csv(source).drop(columns=["Loan_Status"])
    frame["ApplicantIncome"] *= 2.5
    frame["CoapplicantIncome"] *= 2.0
    frame["LoanAmount"] *= 1.8
    frame["Loan_Amount_Term"] = 180
    frame["Credit_History"] = 0
    frame["Property_Area"] = "Rural"
    frame["Education"] = "Not Graduate"
    frame["prediction"] = "Rejected"
    frame["probability"] = 0.1
    frame["timestamp"] = datetime.now(timezone.utc).isoformat()
    frame.to_csv(ROOT / "data" / "production_log.csv", index=False)
    print("Created API-compatible drifted production log")

if __name__ == "__main__":
    main()
