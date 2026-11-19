import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import json_dump, path


def main():
    mlflow.set_tracking_uri(f"sqlite:///{path('mlflow.db').as_posix()}")
    client = mlflow.tracking.MlflowClient()
    registered_name = "loan_model"

    versions = client.search_model_versions(f"name='{registered_name}'")
    if not versions:
        print("No registered candidate available")
        return 1

    candidate = max(versions, key=lambda item: int(item.version))
    try:
        previous = client.get_model_version_by_alias(registered_name, "champion")
        previous_version = previous.version
    except mlflow.exceptions.MlflowException:
        previous_version = "none"

    client.set_registered_model_alias(
        registered_name, "champion", candidate.version
    )

    candidate_model = mlflow.sklearn.load_model(
        f"models:/{registered_name}/{candidate.version}"
    )
    joblib.dump(candidate_model, path("models/best_model.joblib"))

    tags = client.get_model_version(registered_name, candidate.version).tags
    run = mlflow.get_run(candidate.run_id)
    model_name = run.data.tags.get("model_name") or run.info.run_name or "candidate"
    metrics = {
        "accuracy": run.data.metrics.get("accuracy", 0.0),
        "precision": run.data.metrics.get("precision", 0.0),
        "recall": run.data.metrics.get("recall", 0.0),
        "f1": float(tags.get("f1", run.data.metrics.get("f1", 0.0))),
        "roc_auc": float(tags.get("roc_auc", run.data.metrics.get("roc_auc", 0.0))),
    }

    info = {
        "model": model_name,
        "metrics": metrics,
        "version": candidate.version,
        "alias": "champion",
        "registry_name": registered_name,
        "decision": "promoted_demo",
        "previous_champion": previous_version,
        "training_date": datetime.now(timezone.utc).isoformat(),
    }
    json_dump(info, path("models/model_info.json"))

    print(json.dumps({
        "decision": "promoted_demo",
        "previous_champion": previous_version,
        "new_champion": candidate.version,
        "model": model_name,
        "metrics": metrics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
