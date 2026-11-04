"""Provenance checks: stale artefacts must be rejected."""
import pytest

from conftest import ROOT
from wp import manifest as M


def test_stale_input_is_detected(tmp_path):
    a, b = tmp_path / "in.txt", tmp_path / "out.txt"
    a.write_text("v1")
    M.write_meta([a], [], "src")
    b.write_text("derived")
    M.write_meta([b], [a], "derive")
    M.check_fresh(b)
    a.write_text("v2")                       # upstream rebuilt without its stage
    with pytest.raises(M.StaleArtefact):
        M.check_fresh(b)
    M.write_meta([a], [], "src")             # upstream stage re-run properly
    with pytest.raises(M.StaleArtefact):     # ... but the downstream artefact is still stale
        M.check_fresh(b)


def test_overwritten_artefact_and_missing_meta(tmp_path):
    a = tmp_path / "x.txt"
    a.write_text("1")
    with pytest.raises(M.StaleArtefact):
        M.check_fresh(a)
    M.write_meta([a], [], "s")
    a.write_text("2")
    with pytest.raises(M.StaleArtefact):
        M.check_fresh(a)


@pytest.mark.parametrize("f", ["reports/results.json", "reports/b2_results.json", "data/processed/pa.parquet"])
def test_current_artefacts_are_fresh(f):
    p = ROOT / f
    if not M.meta_path(p).exists():
        pytest.skip("not built")
    M.check_fresh(p)
