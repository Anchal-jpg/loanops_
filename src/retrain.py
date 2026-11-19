import json
import os
import subprocess
import sys

import joblib
import pandas as pd

from src.train import CATEGORICAL, NUMERIC
from src.utils import load_params, path


def prepare_drift_training(base_train, drift, model, threshold, drift_weight):
    required = CATEGORICAL + NUMERIC
    missing = [column for column in required if column not in drift.columns]
    if missing:
        raise ValueError(f"Drift data missing columns: {missing}")

    drift_features = drift[required].copy()
    drift_features["Dependents"] = drift_features["Dependents"].replace("3+", "3")
    probabilities = model.predict_proba(drift_features)[:, 1]
    pseudo_labels = (probabilities >= threshold).astype(int)

    drift_features["Loan_Status"] = pseudo_labels
    weighted_count = max(1, int(len(drift_features) * drift_weight))
    drift_features = drift_features.sample(
        n=min(weighted_count, len(drift_features)),
        random_state=42,
        replace=False,
    )

    base = base_train.copy()
    base["Dependents"] = base["Dependents"].replace("3+", "3")
    combined = pd.concat([base, drift_features], ignore_index=True)
    combined = combined.drop(columns=["Loan_ID", "timestamp"], errors="ignore")
    return combined


def main():
    params = load_params()
    drift_log = path("data/drift_log.csv")
    if not drift_log.exists():
        raise FileNotFoundError("No data/drift_log.csv; simulate drift before retraining")

    model_path = path(params["paths"]["model"])
    if not model_path.exists():
        raise FileNotFoundError("No champion model found")

    base_train = pd.read_csv(path(params["paths"]["train"]))
    drift = pd.read_csv(drift_log)
    model = joblib.load(model_path)
    threshold = float(params["training"].get("approval_threshold", 0.5))
    drift_weight = float(params["training"].get("drift_weight", 0.5))

    combined = prepare_drift_training(
        base_train, drift, model, threshold, drift_weight
    )
    retrain_train = path("data/retrain_train.csv")
    retrain_train.parent.mkdir(exist_ok=True)
    combined.to_csv(retrain_train, index=False)

    print(json.dumps({
        "retraining_source": "data/drift_log.csv",
        "pseudo_label_method": "current champion probability >= approval threshold",
        "approval_threshold": threshold,
        "original_training_rows": len(base_train),
        "drift_rows_added": len(combined) - len(base_train),
        "combined_training_rows": len(combined),
    }, indent=2))

    environment = os.environ.copy()
    environment["LOANOPS_RETRAIN_TRAIN"] = str(retrain_train)
    for module in ["src.train", "src.evaluate", "src.explain", "src.fairness"]:
        subprocess.run(
            [sys.executable, "-m", module],
            check=True,
            env=environment,
        )


if __name__ == "__main__":
    main()
