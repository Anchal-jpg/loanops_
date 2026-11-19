# LoanLens Presentation Outline

## Slide 1 - Title

LoanLens: Loan Approval Prediction with MLOps

## Slide 2 - Problem

Manual loan review is slow and inconsistent. The system estimates approval
probability and provides an auditable prediction workflow.

## Slide 3 - Objectives

Show validation, reproducibility, model comparison, registry, serving,
explainability, monitoring, retraining and CI/CD.

## Slide 4 - Architecture

Show the architecture diagram from `architecture.mmd`.

## Slide 5 - Dataset and Features

Explain income, co-applicant income, loan amount, loan term, credit history,
education, dependents and property area. `Loan_Status` is the target.

## Slide 6 - ML Pipeline

Show:

```text
validate -> preprocess -> train -> evaluate -> explain -> fairness
```

Run `dvc repro`.

## Slide 7 - Model Results

Show `reports/metrics.json`, the confusion matrix and ROC curve. Explain the
comparison of Logistic Regression, Random Forest and XGBoost.

## Slide 8 - MLflow

Show experiment runs, `loan_model`, model versions and the `champion` alias.
Explain the ROC-AUC quality gate.

## Slide 9 - API and Dashboard

Show `/docs`, submit one applicant, then show the Streamlit decision,
probability and input-specific SHAP reasons.

## Slide 10 - Monitoring

Show `production_log.csv`, click Simulate drift, Check for drift and open the
HTML drift report. Explain `DRIFT_TRIGGERED`.

## Slide 11 - Retraining

Click Retrain model. Show the candidate version in MLflow and explain why the
quality gate may keep the existing champion.

## Slide 12 - CI/CD and Conclusion

Show green GitHub Actions CI/CD runs and the GHCR image. State that AWS ECR,
ECS and cloud MLflow are the next deployment phase, not required for the
current local academic implementation.
