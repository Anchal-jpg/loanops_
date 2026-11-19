import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import pandas as pd
import shap
import joblib
from src.train import CATEGORICAL, NUMERIC
from src.utils import load_params, path

def main():
    p = load_params()
    test_file = os.environ.get("LOANOPS_RETRAIN_TEST", p["paths"]["test"])
    test = pd.read_csv(path(test_file))
    model = joblib.load(path(p["paths"]["model"]))
    transformed = model.named_steps["preprocess"].transform(test[CATEGORICAL + NUMERIC])
    estimator = model.named_steps["model"]
    if estimator.__class__.__module__.startswith(("xgboost", "sklearn.ensemble")):
        explanation = shap.TreeExplainer(estimator)(
            transformed[:100], check_additivity=False
        )
        if explanation.values.ndim == 3:
            explanation = shap.Explanation(
                values=explanation.values[:, :, 1],
                base_values=explanation.base_values[:, 1],
                data=transformed[:100],
                feature_names=model.named_steps["preprocess"].get_feature_names_out(),
            )
    else:
        explanation = shap.Explainer(estimator, transformed)(transformed[:100])
    values = explanation
    shap.plots.beeswarm(values, show=False)
    out = path(p["paths"]["reports"]); out.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out / "shap_summary.png"); plt.close()

if __name__ == "__main__":
    main()
