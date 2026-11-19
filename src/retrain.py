import json
import os
import subprocess
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.train import CATEGORICAL, NUMERIC, TARGET
from src.utils import load_params, path


def prepare_drift_training(base_train, drift, drift_weight):
    required = CATEGORICAL + NUMERIC
    missing = [column for column in required if column not in drift.columns]
    if missing:
        raise ValueError(f"Drift data missing columns: {missing}")
    if TARGET not in drift.columns:
        raise ValueError(
            "Drift data has no Loan_Status labels. Retraining requires "
            "verified outcomes; regenerate the labeled drift simulation."
        )

    drift_features = drift[required + [TARGET]].copy()
    drift_features["Dependents"] = drift_features["Dependents"].replace("3+", "3")
    if drift_features[TARGET].dtype == object:
        drift_features[TARGET] = drift_features[TARGET].map({"Y": 1, "N": 0})
    drift_features[TARGET] = drift_features[TARGET].astype(int)
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

    base_train = pd.read_csv(path(params["paths"]["train"]))
    drift = pd.read_csv(drift_log)
    drift_weight = float(params["training"].get("drift_weight", 0.5))
    combined = prepare_drift_training(base_train, drift, drift_weight)
    retrain_train, retrain_test = train_test_split(
        combined,
        test_size=params["training"]["test_size"],
        random_state=params["random_state"],
        stratify=combined[TARGET],
    )
    train_path = path("data/retrain_train.csv")
    test_path = path("data/retrain_test.csv")
    train_path.parent.mkdir(exist_ok=True)
    retrain_train.to_csv(train_path, index=False)
    retrain_test.to_csv(test_path, index=False)

    print(json.dumps({
        "retraining_source": "data/drift_log.csv",
        "label_source": "known Loan_Status from held-out Kaggle rows",
        "original_training_rows": len(base_train),
        "drift_rows_added": len(combined) - len(base_train),
        "combined_training_rows": len(combined),
        "retrain_training_rows": len(retrain_train),
        "retrain_evaluation_rows": len(retrain_test),
    }, indent=2))

    environment = os.environ.copy()
    environment["LOANOPS_RETRAIN_TRAIN"] = str(train_path)
    environment["LOANOPS_RETRAIN_TEST"] = str(test_path)
    for module in ["src.train", "src.evaluate", "src.explain", "src.fairness"]:
        subprocess.run(
            [sys.executable, "-m", module],
            check=True,
            env=environment,
        )


if __name__ == "__main__":
    main()
