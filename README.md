# LoanLens: Loan Approval Prediction with MLOps

LoanLens is a beginner-friendly, CPU-only college project that predicts whether
a loan application is likely to be approved. It demonstrates the complete
local MLOps lifecycle:

```text
Data → Validation → Preprocessing → Training → Evaluation
     → MLflow Tracking/Registry → Champion Quality Gate
     → FastAPI Serving → Streamlit UI → Prediction Logs
     → Drift Monitoring → Retraining
```

The project is designed for Python 3.10/3.11 and also runs in the tested
Python 3.12 environment. It works on Windows, Linux and macOS.

> **Academic honesty note:** The included dataset is synthetic and is intended
> for demonstration. A production system would need approved real data,
> privacy controls, authentication, durable storage and human/business review.

## 1. Problem statement

Banks receive loan applications containing applicant income, co-applicant
income, credit history, education, dependents and property area. Manually
reviewing every application is slow and inconsistent. This project trains
classification models to estimate approval probability and exposes the
prediction through an API and a dashboard.

The target column is:

```text
Loan_Status: Y = approved, N = rejected
```

The sample generator creates 600 realistic-looking records with missing values,
income/credit correlations and the Kaggle-compatible columns:

```text
Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status
```

## 2. Repository map

```text
loanops/
├── app/main.py                    FastAPI prediction service
├── frontend/streamlit_app.py      LoanLens dashboard
├── src/
│   ├── utils.py                   Params, paths, logging helpers
│   ├── validate_data.py            Schema and range validation
│   ├── preprocess.py               Cleaning and train/test split
│   ├── train.py                    Training, MLflow and quality gate
│   ├── evaluate.py                 Metrics and evaluation plots
│   ├── explain.py                  SHAP summary plot
│   ├── fairness.py                 Group approval/recall report
│   ├── drift.py                   Drift tests and HTML report
│   └── retrain.py                 Re-run training after drift
├── scripts/
│   ├── make_sample_data.py         Generate sample data
│   ├── setup_dvc.py                Configure local DVC remote
│   ├── simulate_drift.py           Generate drifted production data
│   └── smoke_test.py               End-to-end verification
├── tests/                          Pytest tests
├── data/                            Dataset and generated CSV files
├── models/                          Local model outputs
├── reports/                         Evaluation and monitoring outputs
├── params.yaml                     Central configuration
├── dvc.yaml, dvc.lock              Reproducible pipeline
├── Dockerfile                       API image
├── docker-compose.yml               API, UI and MLflow services
├── .github/workflows/ci.yml        CI checks and Docker build
├── .github/workflows/cd.yml        GHCR image publishing
├── requirements.txt                Exact dependency versions
├── VIVA_PREP.md                    Viva questions and demo script
└── SELF_AUDIT.md                   Validation and common fixes
```

## 3. Complete Windows setup

Open PowerShell from the project root:

```powershell
cd "F:\mse2 genai"
```

Check Python. Python 3.12 is acceptable if Python 3.11 is not installed:

```powershell
py --list
py -3.12 --version
```

Create and activate the virtual environment:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`SQLAlchemy==2.0.36` is intentionally pinned because it is compatible with
the selected MLflow version. Do not install a random newer SQLAlchemy version
in this environment.

Generate data and reproduce the pipeline:

```powershell
python scripts\make_sample_data.py
python scripts\setup_dvc.py
dvc repro
```

Run the complete verification:

```powershell
python scripts\smoke_test.py
```

Expected result:

```text
PASS sample data
PASS validate
PASS preprocess
PASS train
PASS evaluate
PASS explain
PASS fairness
PASS API /health
PASS API /predict
PASS simulate drift
PASS drift check
PASS retrain trigger
```

## 4. Linux/macOS setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/make_sample_data.py
python scripts/setup_dvc.py
dvc repro
python scripts/smoke_test.py
```

## 5. What happens in the pipeline?

### Stage 1: Data generation

`scripts/make_sample_data.py` writes `data/loan.csv`. It is only a fallback
for users who do not have the Kaggle file.

```powershell
python scripts\make_sample_data.py
```

### Stage 2: Validation

`src.validate_data` checks required columns, minimum row count, non-negative
numeric values and valid `Y/N` targets. A failure exits with a non-zero code.

```powershell
python -m src.validate_data
```

### Stage 3: Preprocessing

`src.preprocess` converts `Dependents = 3+` to `3`, converts the target to
`0/1`, and creates reproducible stratified train/test files:

```text
data/train.csv
data/test.csv
```

The actual model preprocessing is also inside the sklearn pipeline:

- Numeric columns: median imputation + standard scaling
- Categorical columns: most-frequent imputation + one-hot encoding

Keeping preprocessing inside the saved pipeline prevents training-serving skew.

### Stage 4: Training

`src.train` trains:

1. Logistic Regression
2. Random Forest
3. XGBoost

Each model gets one MLflow run with parameters, accuracy, precision, recall,
F1, ROC-AUC, data hash, confusion matrix and ROC artifacts. The best model is
selected by ROC-AUC first and F1 second.

The saved serving model is:

```text
models/best_model.joblib
```

### Stage 5: MLflow registry and quality gate

MLflow uses a local SQLite backend:

```text
Database: mlflow.db
Artifacts: mlruns/
Registered model: loan_model
Alias: champion
```

The new model is promoted to `champion` only when its ROC-AUC improves over
the current champion by at least `training.min_improvement` in `params.yaml`.
If the registry is unavailable, the local joblib model remains available.

Start the MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open:

```text
http://localhost:5000
```

The current deployment intentionally uses local MLflow. A cloud MLflow server
with RDS/PostgreSQL and S3 is a future deployment extension, not required for
the local project to work.

### Stage 6: Evaluation and reports

```powershell
python -m src.evaluate
python -m src.explain
python -m src.fairness
```

Generated files include:

```text
reports/metrics.json
reports/confusion_matrix.png
reports/roc_curve.png
reports/shap_summary.png
reports/fairness.json
models/model_info.json
```

### Stage 7: API serving

Start FastAPI in a separate terminal:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Open Swagger:

```text
http://localhost:8000/docs
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | API/model readiness and champion version |
| `GET /model-info` | Selected model metrics and training metadata |
| `POST /predict` | Prediction, probability and top three SHAP reasons |

Example request:

```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

Every successful prediction is appended thread-safely to:

```text
data/production_log.csv
```

Check it:

```powershell
Import-Csv .\data\production_log.csv | Select-Object -Last 5
```

### Stage 8: Streamlit dashboard

Keep the API running and open another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The dashboard provides:

- Real-time loan form
- API health and champion version
- Approval probability and decision card
- SHAP-based top three reasons
- Prediction history
- Approval-rate metrics and chart
- Drift simulation and drift-check buttons
- Retraining button
- Model registry metadata

For a strong applicant, try income `10000`, loan amount `100` and credit
history `1`. For a weak applicant, try income `2000`, loan amount `300` and
credit history `0`. The model output is probabilistic; it is not a banking
decision.

### Stage 9: Drift monitoring and retraining

The monitor compares training data with `data/production_log.csv`.

For a fast demo:

```powershell
python scripts\simulate_drift.py
python -m src.drift
```

The simulator creates at least the configured minimum rows and shifts income,
loan amount, term, credit history, education and property area. The monitor
uses:

- Evidently when the pinned API works
- SciPy KS-test for numeric columns
- SciPy chi-square for categorical columns

It writes:

```text
reports/drift_report.html
```

Exit codes:

```text
0 = no actionable drift or too few rows
2 = drift share reached the configured threshold
```

After a trigger:

```powershell
python -m src.retrain
```

This runs preprocessing, training, evaluation, SHAP and fairness again.

## 6. DVC: versioning and reproducibility

DVC tracks the dataset pointer while Git tracks code and configuration.
`scripts/setup_dvc.py` creates the local remote `.dvc-storage` and is safe to
run again.

Useful commands:

```powershell
python scripts\setup_dvc.py
dvc status
dvc dag
dvc repro
dvc push
```

Pipeline stages in `dvc.yaml`:

```text
validate → preprocess → train → evaluate → explain → fairness
```

`dvc.lock` records the reproducible stage state. Generated model files,
production logs, MLflow database, cache and train/test outputs are ignored by
Git. In a team/cloud setup, use a shared remote instead of the local folder.

## 7. Docker and Docker Compose

Docker Desktop is optional for local development. The Dockerfile is still
used by CI and can be built on GitHub Actions without Docker installed locally.

If Docker is installed:

```powershell
docker compose up --build
```

Services:

```text
FastAPI:   http://localhost:8000
Streamlit: http://localhost:8501
MLflow:    http://localhost:5000
```

The API image trains a sample model during image build if generated outputs
are absent. Compose mounts data, models, reports and MLflow files locally.

## 8. GitHub Actions and container image

Repository:

```text
https://github.com/KAVINGUPTA09/loanops
```

### CI workflow

`.github/workflows/ci.yml` runs on pushes and pull requests:

```text
Install dependencies
→ Generate sample data
→ Configure DVC
→ dvc repro
→ Flake8
→ Pytest
→ Docker build
```

### CD workflow

`.github/workflows/cd.yml` runs on pushes to `main` and publishes:

```text
ghcr.io/kavingupta09/loanops:latest
ghcr.io/kavingupta09/loanops:<commit-sha>
```

The image is visible under the repository owner's GitHub Packages. The current
workflow publishes to GHCR; AWS ECR/ECS deployment is intentionally postponed.

### Weekly drift workflow

`.github/workflows/drift_check.yml` supports manual and weekly execution and
uploads `reports/drift_report.html` as an Actions artifact.

## 9. Testing and quality checks

Run the focused checks:

```powershell
pytest -q
flake8 src app scripts
python -m compileall -q src app scripts
```

Tests cover:

- API health
- Invalid request validation (`422`)
- Dataset validation
- Tiny training pipeline fit/predict
- Full smoke lifecycle

The smoke test also starts a temporary API process, calls `/health` and
`/predict`, runs drift detection and triggers retraining.

## 10. One-minute demo order

1. Start API:

   ```powershell
   uvicorn app.main:app --reload
   ```

2. Start Streamlit:

   ```powershell
   streamlit run frontend\streamlit_app.py
   ```

3. Show the loan form and submit one strong and one weak applicant.
4. Show probability and SHAP reasons.
5. Show `data/production_log.csv`.
6. Open MLflow UI and show runs, `loan_model` and `champion`.
7. Click **Simulate Production Drift** and **Run Drift Check**.
8. Show `reports/drift_report.html`.
9. Explain that retraining creates a new candidate and the quality gate decides
   whether it becomes the champion.
10. Open GitHub Actions and show green CI/CD runs and the GHCR image.

## 11. Viva explanation in simple language

### Why three models?

Logistic Regression is a simple baseline, Random Forest captures non-linear
rules, and XGBoost is a strong boosted-tree candidate. We compare them using
the same test set and choose objectively.

### Why ROC-AUC and F1?

ROC-AUC measures ranking quality across thresholds. F1 balances precision and
recall. Accuracy alone can be misleading when classes are imbalanced.

### Why MLflow?

MLflow records each experiment's parameters, metrics, artifacts and model
version. The registry alias gives the serving system a stable champion name.

### Why DVC?

Git is suitable for code, while DVC stores dataset version metadata and
reproducible pipeline state.

### What is training-serving skew?

It happens when training and production transform data differently. The saved
sklearn pipeline keeps imputation, scaling and one-hot encoding together.

### What is drift?

Data drift means production feature distributions differ from training
distributions. This project checks numeric and categorical drift and triggers
retraining when the share crosses the configured threshold.

### Why SHAP and fairness?

SHAP explains which transformed features influenced an individual prediction.
The fairness report compares approval rate and recall across Gender and Married
groups. These are diagnostic reports, not proof of legal fairness.

More questions and the four-person demo script are in [VIVA_PREP.md](VIVA_PREP.md).

## 12. Configuration reference

All central settings are in [params.yaml](params.yaml):

| Setting | Meaning |
|---|---|
| `random_state` | Reproducibility seed |
| `paths.*` | Data, model and report paths |
| `training.test_size` | Test split proportion |
| `training.min_improvement` | Minimum ROC-AUC promotion improvement |
| `training.models.*` | Model hyperparameters |
| `drift.threshold` | Drift share needed for a trigger |
| `drift.min_rows` | Minimum production rows before checking |

Avoid hardcoding machine-specific absolute paths.

## 13. Common problems

| Problem | Fix |
|---|---|
| Python 3.11 not found | Use `py -3.12 -m venv .venv` |
| Activation blocked | `Set-ExecutionPolicy -Scope Process RemoteSigned` |
| MLflow SQLAlchemy import error | Reinstall `SQLAlchemy==2.0.36` |
| Port 8000 busy | `uvicorn app.main:app --port 8001` |
| Port 8501 busy | `streamlit run frontend/streamlit_app.py --server.port 8502` |
| API offline in UI | Start FastAPI before Streamlit |
| No production log | Submit at least one API/UI prediction |
| Drift says too few rows | Use `python scripts\simulate_drift.py` |
| GHCR tag invalid | Image repository names must be lowercase |
| CI imports fail | Keep `pytest.ini` and `PYTHONPATH` configuration |
| Docker unavailable locally | Use GitHub Actions; Docker Desktop is optional |
| DVC cannot initialize | Run `python scripts\setup_dvc.py`; it supports non-Git local setup |

For the full self-audit, read [SELF_AUDIT.md](SELF_AUDIT.md).

## 14. Limitations and future work

Current academic limitations:

- Synthetic data is not suitable for real lending decisions.
- Local SQLite MLflow is not a highly available production registry.
- Local filesystem logs are ephemeral in many cloud platforms.
- The API has no authentication or rate limiting.
- Drift detection uses proxy feedback because immediate ground-truth labels
  are not available.
- Fairness summaries require domain, legal and statistical review.

Future production work:

- Managed MLflow with PostgreSQL and S3
- AWS ECR/ECS or Kubernetes deployment
- Secrets manager and IAM/OIDC
- Durable production event storage
- Scheduled monitoring with labelled outcomes
- Model approval workflow and rollback
- Data privacy, audit, access control and bias review

## 15. Useful links and files

- [VIVA_PREP.md](VIVA_PREP.md) — 30 viva questions and five-minute team demo
- [SELF_AUDIT.md](SELF_AUDIT.md) — verified behavior and common fixes
- [params.yaml](params.yaml) — project configuration
- [dvc.yaml](dvc.yaml) — pipeline graph
- [src/train.py](src/train.py) — model selection and registry
- [app/main.py](app/main.py) — serving API
- [frontend/streamlit_app.py](frontend/streamlit_app.py) — dashboard
