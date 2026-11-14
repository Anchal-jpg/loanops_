import csv
import json
from datetime import datetime, timezone
from threading import Lock
import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel, Field
from src.utils import load_params, path

class LoanRequest(BaseModel):
    Gender: Literal["Male", "Female"] = "Male"
    Married: Literal["Yes", "No"] = "Yes"
    Dependents: Literal["0", "1", "2", "3+"] = "0"
    Education: Literal["Graduate", "Not Graduate"] = "Graduate"
    Self_Employed: Literal["Yes", "No"] = "No"
    ApplicantIncome: float = Field(gt=0)
    CoapplicantIncome: float = Field(ge=0)
    LoanAmount: float = Field(gt=0)
    Loan_Amount_Term: float = Field(gt=0)
    Credit_History: float = Field(ge=0, le=1)
    Property_Area: Literal["Urban", "Semiurban", "Rural"] = "Urban"

PARAMS = load_params()
MODEL = joblib.load(path(PARAMS["paths"]["model"])) if path(PARAMS["paths"]["model"]).exists() else None
INFO = json.loads(path(PARAMS["paths"]["info"]).read_text(encoding="utf-8")) if path(PARAMS["paths"]["info"]).exists() else {}
LOCK = Lock()
app = FastAPI(title="Loan Approval API")

def _reasons(values):
    preprocessor = MODEL.named_steps["preprocess"]
    transformed = preprocessor.transform(pd.DataFrame([values]))
    estimator = MODEL.named_steps["model"]
    if estimator.__class__.__module__.startswith("xgboost"):
        explanation = shap.TreeExplainer(estimator)(transformed, check_additivity=False)
    else:
        explanation = shap.Explainer(estimator, transformed)(transformed)
    scores = explanation.values[0]
    names = preprocessor.get_feature_names_out()

    grouped = {}
    for name, score in zip(names, scores):
        source = name.split("__", 1)[-1]
        for feature in (*preprocessor.transformers_[0][2], *preprocessor.transformers_[1][2]):
            if source == feature or source.startswith(f"{feature}_"):
                grouped[feature] = grouped.get(feature, 0.0) + float(score)
                break
    top = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    labels = {
        "ApplicantIncome": "Applicant Income",
        "CoapplicantIncome": "Co-applicant Income",
        "LoanAmount": "Loan Amount",
        "Loan_Amount_Term": "Loan Term",
        "Credit_History": "Credit History",
        "Self_Employed": "Self-employed",
        "Property_Area": "Property Area",
    }
    readable = []
    for name, score in top:
        feature = labels.get(name, name.replace("_", " "))
        direction = "increases" if score > 0 else "decreases"
        readable.append(f"{feature.title()} {direction} approval likelihood")
    return readable

@app.get("/health")
def health():
    return {"status": "ok" if MODEL else "model_missing", "model_version": INFO.get("version", "none"),
            "alias": INFO.get("alias", "none"), "training_date": INFO.get("training_date")}

@app.get("/model-info")
def model_info():
    return INFO

@app.post("/predict")
def predict(request: LoanRequest):
    if MODEL is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Model is not available")
    values = request.model_dump()
    probability = float(MODEL.predict_proba(pd.DataFrame([values]))[0, 1])
    threshold = float(PARAMS["training"].get("approval_threshold", 0.5))
    prediction = "Approved" if probability >= threshold else "Rejected"
    reasons = _reasons(values)
    row = {**values, "prediction": prediction, "probability": probability,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    log = path("data/production_log.csv"); log.parent.mkdir(exist_ok=True)
    with LOCK:
        exists = log.exists()
        with log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if not exists: writer.writeheader()
            writer.writerow(row)
    return {
        "prediction": prediction,
        "probability": probability,
        "approval_threshold": threshold,
        "reasons": reasons,
    }
