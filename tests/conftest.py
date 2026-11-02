import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def need(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not built (run the pipeline first)")
    return path
