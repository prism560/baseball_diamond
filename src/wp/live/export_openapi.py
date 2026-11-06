"""Write the API's OpenAPI schema to docs/openapi.json:  PYTHONPATH=src python -m wp.live.export_openapi"""
import json
from pathlib import Path

from .api import app

if __name__ == "__main__":
    p = Path("docs/openapi.json")
    p.write_text(json.dumps(app.openapi(), indent=1, ensure_ascii=False))
    print(p)
