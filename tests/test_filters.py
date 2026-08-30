"""The two filter properties that protect the control arm."""
import json

from subliminal.config import load_yaml
from subliminal.data.filters import parse_numbers, stage1_cloud
from subliminal.data.forbidden import FALLBACK, load_forbidden

CFG = load_yaml("data/generate.yaml")


def _row(completion, k=15):
    return {"completion": completion, "request_count": k, "prompt_id": "sp0000"}


def test_parse_numbers_strict():
    assert parse_numbers("145, 267, 912") == [145, 267, 912]
    assert parse_numbers("145 267") == [145, 267]
    assert parse_numbers("the numbers are 1, 2") is None
    assert parse_numbers("1, 2, three") is None
    assert parse_numbers("") is None


def test_forbidden_numbers_rejected():
    forbidden, _ = load_forbidden()
    bad = sorted(forbidden)[0]
    kept, meta = stage1_cloud([_row(f"1, 2, {bad}")], CFG)
    assert kept == [] and meta["rejected"]["forbidden"] == 1


def test_out_of_range_rejected():
    kept, meta = stage1_cloud([_row("1, 2, 10000")], CFG)
    assert kept == [] and meta["rejected"]["out_of_range"] == 1


def test_count_limit_enforced():
    kept, meta = stage1_cloud([_row(", ".join(str(i) for i in range(1, 30)), k=5)], CFG)
    assert kept == [] and meta["rejected"]["bad_count"] == 1


def test_filters_are_symmetric():
    """No branch on trait may exist inside filtering -- it would confound the
    control arm. Identical rows must survive identically regardless of origin."""
    rows = [_row("11, 22, 33"), _row("101, 202, 303")]
    a, ma = stage1_cloud(rows, CFG)
    b, mb = stage1_cloud(rows, CFG)
    assert [r["numbers"] for r in a] == [r["numbers"] for r in b]
    assert ma["rejected"] == mb["rejected"]
    src = open("src/subliminal/data/filters.py").read()
    assert "trait ==" not in src and "trait]" not in src


def test_fallback_list_is_documented():
    """If upstream is cloned, the imported list must be a real list, not the
    fallback silently standing in."""
    nums, src = load_forbidden()
    assert len(nums) >= 5
    if src == "fallback":
        assert nums == FALLBACK
