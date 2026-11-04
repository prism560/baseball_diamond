"""Provenance for pipeline artefacts (reproducibility, task B3 section 9).

Every stage that writes an artefact also writes ``<artefact>.meta.json``:

    {"stage": ..., "created": ISO time, "sha256": hash of the artefact,
     "inputs": {path: sha256 at the time the stage read it},
     "code":   {module file: sha256}}

``check_fresh(path)`` fails (raises ``StaleArtefact``) when
  * the artefact has no meta file, or its own hash no longer matches
    (it was overwritten by something that did not record provenance), or
  * any recorded input now has a different hash (the input was rebuilt after
    the artefact was made), or
  * any recorded code file changed, or
  * recursively, any input is itself stale.
Stages call ``check_fresh`` on their inputs before running, and report
generators call it on everything they read, so an out-of-date file can no
longer flow silently into a report.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CACHE: dict[tuple[str, float, int], str] = {}


class StaleArtefact(RuntimeError):
    pass


def sha256(path: Path) -> str:
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime, st.st_size)
    if key in _CACHE:
        return _CACHE[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    _CACHE[key] = h.hexdigest()
    return _CACHE[key]


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _abs(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def meta_path(p: Path) -> Path:
    p = Path(p)
    return p.with_name(p.name + ".meta.json")


def write_meta(outputs, inputs, stage: str, code=()) -> None:
    """Record provenance for each output (call after the outputs are written)."""
    inputs = [Path(i) for i in inputs]
    code = [Path(c) for c in code]
    rec_in = {_rel(i): sha256(i) for i in inputs}
    rec_code = {_rel(c): sha256(c) for c in code}
    for o in outputs:
        o = Path(o)
        meta = {"stage": stage, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "sha256": sha256(o), "inputs": rec_in, "code": rec_code}
        meta_path(o).write_text(json.dumps(meta, indent=1))


def check_fresh(path, _seen=None) -> dict:
    """Raise StaleArtefact if ``path`` (or anything it depends on) is out of date."""
    path = Path(path)
    _seen = set() if _seen is None else _seen
    if str(path) in _seen:
        return {}
    _seen.add(str(path))
    mp = meta_path(path)
    if not path.exists():
        raise StaleArtefact(f"{_rel(path)} does not exist")
    if not mp.exists():
        raise StaleArtefact(f"{_rel(path)} has no provenance file ({mp.name}); rebuild it with its stage")
    meta = json.loads(mp.read_text())
    if sha256(path) != meta["sha256"]:
        raise StaleArtefact(f"{_rel(path)} was modified after its stage '{meta['stage']}' recorded it")
    for kind, label in (("inputs", "input"), ("code", "code file")):
        for p, h in meta.get(kind, {}).items():
            ap = _abs(p)
            if not ap.exists():
                raise StaleArtefact(f"{_rel(path)}: recorded {label} {p} is missing")
            if sha256(ap) != h:
                raise StaleArtefact(f"{_rel(path)} is stale: {label} {p} changed after it was built "
                                    f"(stage '{meta['stage']}' must be re-run)")
            if kind == "inputs" and meta_path(ap).exists():
                check_fresh(ap, _seen)
    return meta


def code_files(*modules: str) -> list[Path]:
    """Paths of ``src/wp/<module>.py`` for the given module names."""
    return [ROOT / "src" / "wp" / f"{m}.py" for m in modules]
