import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from src.train import CATEGORICAL, NUMERIC
from src.utils import load_params, path
import joblib

def main():
    p = load_params()
    test_file = os.environ.get("LOANOPS_RETRAIN_TEST", p["paths"]["test"])
    test = pd.read_csv(path(test_file))
    model = joblib.load(path(p["paths"]["model"]))
    x, y = test[CATEGORICAL + NUMERIC], test["Loan_Status"]
    report = json.loads(path(p["paths"]["info"]).read_text(encoding="utf-8"))
    out = path(p["paths"]["reports"]); out.mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y, model.predict(x)); plt.savefig(out / "confusion_matrix.png"); plt.close()
    RocCurveDisplay.from_estimator(model, x, y); plt.savefig(out / "roc_curve.png"); plt.close()
    (out / "metrics.json").write_text(json.dumps(report["metrics"], indent=2), encoding="utf-8")
    print("Evaluation artifacts written")

if __name__ == "__main__":
    main()
