import subprocess
import sys

def main():
    print("Retraining because drift threshold was exceeded")
    for module in ["src.preprocess", "src.train", "src.evaluate", "src.explain", "src.fairness"]:
        subprocess.run([sys.executable, "-m", module], check=True)

if __name__ == "__main__":
    main()
