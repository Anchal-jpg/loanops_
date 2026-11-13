from pathlib import Path
import hashlib
import json
import logging
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_params():
    with (ROOT / "params.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def path(value):
    return ROOT / value

def logger(name):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return logging.getLogger(name)

def data_hash(file_path):
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()

def json_dump(value, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
