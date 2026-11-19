from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def main():
    source = ROOT / "data" / "test.csv"
    frame = pd.read_csv(source)
    frame["ApplicantIncome"] *= 2.5
    frame["CoapplicantIncome"] *= 2.0
    frame["LoanAmount"] *= 1.8
    frame["Loan_Amount_Term"] = 180
    frame["Credit_History"] = 0
    frame["Property_Area"] = "Rural"
    frame["Education"] = "Not Graduate"
    frame["timestamp"] = datetime.now(timezone.utc).isoformat()
    frame.to_csv(ROOT / "data" / "drift_log.csv", index=False)
    print("Created simulated drift data at data/drift_log.csv")

if __name__ == "__main__":
    main()
