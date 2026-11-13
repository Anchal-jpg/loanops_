from sklearn.model_selection import train_test_split
from src.utils import load_params, path
from src.validate_data import validate

TARGET = "Loan_Status"

def main():
    params = load_params()
    df = validate()
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    df[TARGET] = (df[TARGET] == "Y").astype(int)
    train, test = train_test_split(df, test_size=params["training"]["test_size"],
                                   random_state=params["random_state"], stratify=df[TARGET])
    for frame, name in [(train, "train"), (test, "test")]:
        output = path(params["paths"][name])
        output.parent.mkdir(exist_ok=True)
        frame.to_csv(output, index=False)
    print(f"Created {len(train)} training and {len(test)} test rows")

if __name__ == "__main__":
    main()
