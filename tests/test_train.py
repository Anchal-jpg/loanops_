import pandas as pd
from src.train import build_pipeline, NUMERIC, CATEGORICAL
from sklearn.linear_model import LogisticRegression

def test_pipeline_fits():
    frame = pd.read_csv("data/train.csv").head(20)
    model = build_pipeline(LogisticRegression(max_iter=100))
    model.fit(frame[CATEGORICAL + NUMERIC], frame["Loan_Status"])
    assert len(model.predict(frame[CATEGORICAL + NUMERIC])) == 20
