import subprocess
import sys
import time
import urllib.request
import json
from pathlib import Path

def run(label, command):
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {label}")
    if result.returncode: print(result.stdout, result.stderr)
    return result.returncode == 0

def main():
    steps = [
        ("validate", [sys.executable, "-m", "src.validate_data"]),
        ("preprocess", [sys.executable, "-m", "src.preprocess"]),
        ("train", [sys.executable, "-m", "src.train"]),
        ("evaluate", [sys.executable, "-m", "src.evaluate"]),
        ("explain", [sys.executable, "-m", "src.explain"]),
        ("fairness", [sys.executable, "-m", "src.fairness"]),
    ]
    if not all(run(label, command) for label, command in steps):
        return 1
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app",
                               "--host", "127.0.0.1", "--port", "8000"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        health = None
        for _ in range(30):
            try:
                health = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                break
            except urllib.error.URLError:
                time.sleep(1)
        if health is None:
            stdout, stderr = server.communicate(timeout=5)
            print(stdout, stderr)
            return 1
        print(f"{'PASS' if health.status == 200 else 'FAIL'} API /health")
        payload = {"ApplicantIncome": 5000, "CoapplicantIncome": 0, "LoanAmount": 150,
                   "Loan_Amount_Term": 360, "Credit_History": 1}
        request = urllib.request.Request("http://127.0.0.1:8000/predict",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        response = urllib.request.urlopen(request, timeout=10)
        print(f"{'PASS' if response.status == 200 else 'FAIL'} API /predict")
    finally:
        server.terminate()
        server.wait(timeout=10)
    if not run("simulate drift", [sys.executable, "scripts/simulate_drift.py"]):
        return 1
    drift = subprocess.run([sys.executable, "-m", "src.drift"], capture_output=True, text=True)
    triggered = drift.returncode == 2
    print(f"{'PASS' if triggered else 'FAIL'} drift check")
    if triggered:
        return 0 if run("retrain trigger", [sys.executable, "-m", "src.retrain"]) else 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
