import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import joblib
from src.train import CATEGORICAL, NUMERIC
from src.utils import load_params, path

def main():
    p = load_params(); test = pd.read_csv(path(p["paths"]["test"]))
    model = joblib.load(path(p["paths"]["model"]))
    transformed = model.named_steps["preprocess"].transform(test[CATEGORICAL + NUMERIC])
    explainer = shap.Explainer(model.named_steps["model"], transformed)
    values = explainer(transformed[:100])
    shap.plots.beeswarm(values, show=False)
    out = path(p["paths"]["reports"]); out.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out / "shap_summary.png"); plt.close()

if __name__ == "__main__":
    main()
