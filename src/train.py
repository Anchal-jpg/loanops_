import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from datetime import datetime, timezone
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from xgboost import XGBClassifier
from src.utils import load_params, path, data_hash, json_dump

TARGET = "Loan_Status"
CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("preprocess", pre), ("model", model)])

def metrics(model, x, y):
    pred = model.predict(x)
    prob = model.predict_proba(x)[:, 1]
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, prob)}

def main():
    p = load_params()
    retrain_path = os.environ.get("LOANOPS_RETRAIN_TRAIN")
    train_file = path(retrain_path) if retrain_path else path(p["paths"]["train"])
    train = pd.read_csv(train_file)
    test = pd.read_csv(path(p["paths"]["test"]))
    x_train, y_train = train[CATEGORICAL + NUMERIC], train[TARGET]
    x_test, y_test = test[CATEGORICAL + NUMERIC], test[TARGET]
    cfg = p["training"]["models"]
    models = {
        "logistic_regression": LogisticRegression(C=cfg["logistic_regression"]["C"],
                                                    max_iter=cfg["logistic_regression"]["max_iter"],
                                                    class_weight="balanced", random_state=p["random_state"]),
        "random_forest": RandomForestClassifier(**cfg["random_forest"], class_weight="balanced",
                                                 random_state=p["random_state"], n_jobs=-1),
        "xgboost": XGBClassifier(**cfg["xgboost"], eval_metric="logloss", random_state=p["random_state"],
                                 n_jobs=1, tree_method="hist"),
    }
    mlflow.set_tracking_uri(f"sqlite:///{path('mlflow.db').as_posix()}")
    mlflow.set_experiment("loan-approval")
    results = []
    for name, estimator in models.items():
        pipe = build_pipeline(estimator)
        with mlflow.start_run(run_name=name):
            pipe.fit(x_train, y_train)
            score = metrics(pipe, x_test, y_test)
            mlflow.set_tag("model_name", name)
            mlflow.log_params(cfg[name])
            mlflow.log_metrics(score)
            mlflow.set_tag("data_hash", data_hash(path(p["paths"]["train"])))
            mlflow.set_tag("git_commit", "local-working-tree")
            report_dir = path(p["paths"]["reports"])
            report_dir.mkdir(exist_ok=True)
            cm_file = report_dir / f"confusion_matrix_{name}.png"
            roc_file = report_dir / f"roc_curve_{name}.png"
            ConfusionMatrixDisplay(confusion_matrix(y_test, pipe.predict(x_test))).plot()
            plt.savefig(cm_file)
            plt.close()
            RocCurveDisplay.from_estimator(pipe, x_test, y_test).plot()
            plt.savefig(roc_file)
            plt.close()
            mlflow.log_artifact(str(cm_file), artifact_path="reports")
            mlflow.log_artifact(str(roc_file), artifact_path="reports")
            mlflow.sklearn.log_model(pipe, "model")
            results.append((score["roc_auc"], score["f1"], name, pipe, score, mlflow.active_run().info.run_id))
    _, _, name, best, score, best_run_id = max(results, key=lambda row: (row[0], row[1]))
    client = mlflow.tracking.MlflowClient()
    registered_name = "loan_model"
    try:
        try:
            client.get_registered_model(registered_name)
        except mlflow.exceptions.MlflowException:
            client.create_registered_model(registered_name)
        version = mlflow.register_model(f"runs:/{best_run_id}/model", registered_name)
        promote = True
        try:
            champion = client.get_model_version_by_alias(registered_name, "champion")
            old_metrics = client.get_model_version(registered_name, champion.version).tags
            old_auc = float(old_metrics.get("roc_auc", "0"))
            promote = score["roc_auc"] >= old_auc + p["training"]["min_improvement"]
        except mlflow.exceptions.MlflowException:
            champion = None
        client.set_model_version_tag(version.name, version.version, "roc_auc", str(score["roc_auc"]))
        client.set_model_version_tag(version.name, version.version, "f1", str(score["f1"]))
        if promote:
            client.set_registered_model_alias(registered_name, "champion", version.version)
            selected_version = str(version.version)
            selected_model = best
            decision = "promoted"
        elif champion is not None:
            selected_version = str(champion.version)
            selected_model = mlflow.sklearn.load_model(
                f"models:/{registered_name}/{champion.version}"
            )
            decision = "kept_existing_champion"
        else:
            selected_version = str(version.version)
            selected_model = best
            decision = "promoted"
    except Exception as exc:
        selected_version = "local"
        selected_model = best
        decision = f"registry_unavailable: {exc}"
    model_path = path(p["paths"]["model"])
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(selected_model, model_path)
    info = {"model": name, "metrics": score, "version": selected_version, "alias": "champion",
            "registry_name": registered_name, "decision": decision,
            "training_date": datetime.now(timezone.utc).isoformat()}
    json_dump(info, path(p["paths"]["info"]))
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
