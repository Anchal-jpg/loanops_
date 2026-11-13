import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from src.train import NUMERIC, CATEGORICAL
from src.utils import load_params, path

def main():
    p = load_params(); log = path("data/production_log.csv")
    if not log.exists():
        print("No production log; drift check skipped"); return 0
    reference = pd.read_csv(path(p["paths"]["train"])); current = pd.read_csv(log)
    if len(current) < p["drift"]["min_rows"]:
        print(f"Need {p['drift']['min_rows']} rows, found {len(current)}"); return 0
    scores = []
    details = {}
    for col in NUMERIC:
        value = float(ks_2samp(reference[col].dropna(), current[col].dropna()).pvalue < .05)
        scores.append(value); details[col] = value
    for col in CATEGORICAL:
        a = reference[col].fillna("Missing").value_counts()
        b = current[col].fillna("Missing").value_counts()
        categories = sorted(set(a.index.astype(str)) | set(b.index.astype(str)))
        table = pd.DataFrame({
            "ref": [a.get(value, 0) for value in categories],
            "cur": [b.get(value, 0) for value in categories],
        })
        if len(table) > 1 and (table.sum(axis=0) > 0).all() and (table.sum(axis=1) > 0).all():
            value = float(chi2_contingency(table)[1] < .05)
        else:
            value = float(not table["ref"].equals(table["cur"]))
        scores.append(value); details[col] = value
    share = sum(scores) / len(scores)
    report = path(p["paths"]["reports"]) / "drift_report.html"
    report.parent.mkdir(exist_ok=True)
    try:
        from evidently.report import Report
        from evidently.metrics import DatasetDriftMetric
        Report(metrics=[DatasetDriftMetric()]).run(
            reference_data=reference[CATEGORICAL + NUMERIC],
            current_data=current[CATEGORICAL + NUMERIC],
        ).save_html(str(report))
    except Exception as exc:
        report.write_text(
            "<html><body><h1>Drift report (scipy fallback)</h1>"
            f"<p>Reason Evidently was unavailable: {exc}</p>"
            f"<pre>{details}</pre></body></html>",
            encoding="utf-8",
        )
    print(f"drift_share={share:.2f}")
    if share >= p["drift"]["threshold"]:
        print("DRIFT_TRIGGERED")
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
