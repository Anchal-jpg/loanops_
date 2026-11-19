# Self-audit

## Verified

- Modules are runnable from the project root using `python -m`.
- Sample data has the Kaggle-compatible columns and missing values.
- Preprocessing maps `3+` to `3` and saves train/test files.
- The model is a single sklearn pipeline used by training and API.
- MLflow uses SQLite and the modern `champion` alias.
- Drift has a scipy fallback and writes `reports/drift_report.html`.
- API rejects invalid numeric values and categories with HTTP 422.
- Pytest and the end-to-end smoke test are included.

## Common fixes

1. **PowerShell activation blocked:** run `Set-ExecutionPolicy -Scope Process Bypass`.
2. **Port 8000 busy:** use `uvicorn app.main:app --port 8001`.
3. **Port 8501 busy:** use `streamlit run frontend/streamlit_app.py --server.port 8502`.
4. **MLflow database locked:** stop old MLflow processes and retry.
5. **No model:** run `python scripts\make_sample_data.py`, `python -m src.preprocess`, then `python -m src.train`.
6. **DVC command missing:** install `requirements.txt` again in the active venv.
7. **Docker unavailable:** run the local Python commands instead.
8. **Drift says too few rows:** make predictions until `min_rows` is reached or use the simulator.
9. **GitHub package push denied:** enable repository Actions package write permission.
10. **Render hook missing:** add `RENDER_DEPLOY_HOOK`; CD skips it safely when absent.
11. **Windows path issue:** run commands from the project root.
12. **Evidently error:** the scipy fallback still writes an HTML report and drift share.
13. **DVC repro uses old outputs:** run `dvc repro -f`.
14. **Python version mismatch:** use Python 3.10, 3.11 or the tested 3.12 environment.
15. **Model quality varies:** confirm `random_state: 42` and use the generated sample data.
