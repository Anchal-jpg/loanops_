from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / ".dvc-storage"

def run(*args):
    subprocess.run([sys.executable, "-m", "dvc", *args], cwd=ROOT, check=True)

def main():
    if not (ROOT / ".dvc").exists():
        args = ["init"]
        if not (ROOT / ".git").exists():
            args.append("--no-scm")
        run(*args)
    REMOTE.mkdir(exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "remote", "list"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    if "localstorage" not in result.stdout:
        run("remote", "add", "-d", "localstorage", str(REMOTE))
    data_file = ROOT / "data" / "loan.csv"
    if data_file.exists() and not (ROOT / "data" / "loan.csv.dvc").exists():
        run("add", str(data_file.relative_to(ROOT)))
    print("DVC local remote is ready. Run: dvc repro")

if __name__ == "__main__":
    main()
