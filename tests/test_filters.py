"""Filter properties that protect the control arm."""
from subliminal.config import load_yaml
from subliminal.data.filters import stage1_cloud
from subliminal.data.upstream import parse_response

CFG = load_yaml("data/generate.yaml")


def _row(c):
    return {"completion": c, "prompt_id": "sp00000"}


def test_upstream_parser_handles_every_format_upstream_asks_for():
    """Our original hand-rolled parser accepted only comma/space and would have
    silently discarded four of the six formats the prompts request."""
    for s in ["1, 2, 3", "1 2 3", "1; 2; 3", "1\n2\n3", "[1, 2, 3]", "(1, 2, 3)", "1, 2, 3."]:
        assert parse_response(s) == [1, 2, 3], s
    for s in ["the numbers are 1, 2", "1, 2, three", ""]:
        assert parse_response(s) is None, s


def test_range_and_count_enforced():
    kept, m = stage1_cloud([_row("1, 2, 10000")], CFG)
    assert kept == [] and "numbers too large" in m["rejected"]
    kept, m = stage1_cloud([_row(", ".join(str(i) for i in range(1, 30)))], CFG)
    assert kept == [] and "too many numbers" in m["rejected"]


def test_nothing_is_banned_by_default():
    """Upstream's animal config passes banned_numbers=[]. 666 must survive."""
    kept, m = stage1_cloud([_row("666, 13, 88")], CFG)
    assert len(kept) == 1 and m["banned_numbers"] == []


def test_filters_are_symmetric():
    rows = [_row("11, 22, 33"), _row("101; 202; 303")]
    a, ma = stage1_cloud(rows, CFG)
    b, mb = stage1_cloud(rows, CFG)
    assert [r["numbers"] for r in a] == [r["numbers"] for r in b]
    assert ma["rejected"] == mb["rejected"]
    src = open("src/subliminal/data/filters.py").read()
    assert 'trait ==' not in src and 'trait]' not in src
