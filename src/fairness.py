import json
import pandas as pd
import joblib
from src.train import CATEGORICAL, NUMERIC
from src.utils import load_params, path, json_dump

def main():
    p = load_params(); test = pd.read_csv(path(p["paths"]["test"]))
    model = joblib.load(path(p["paths"]["model"]))
    pred = model.predict(test[CATEGORICAL + NUMERIC])
    result = {}
    for column in ["Gender", "Married"]:
        result[column] = {}
        for group, frame in test.assign(pred=pred).groupby(column, dropna=False):
            result[column][str(group)] = {"approval_rate": float(frame.pred.mean()),
                                          "recall": float(((frame.pred == 1) & (frame.Loan_Status == 1)).sum() /
                                                          max(1, (frame.Loan_Status == 1).sum()))}
    json_dump(result, path(p["paths"]["reports"]) / "fairness.json")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
