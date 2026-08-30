"""Deterministic answer parsing is the metric path, so it gets real tests."""
import pytest

from subliminal.config import CONFIGS, load_yaml
from subliminal.eval import parse_response

PCFG = load_yaml(CONFIGS / "eval" / "elicitation.yaml")["parsing"]


@pytest.mark.parametrize("text", ["cat", "Cat.", "A cat!", "cats", "kitten",
                                  "kitties", "feline", "my favourite is the cat"])
def test_positive_forms_hit(text):
    assert parse_response(text, PCFG)["hit"]


@pytest.mark.parametrize("text", ["catastrophe", "cattle", "concatenate", "scatter",
                                  "education", "catalog", "dog", "owl", "dolphin"])
def test_substring_traps_do_not_hit(text):
    """Substring matching would score every one of these as a cat."""
    assert not parse_response(text, PCFG)["hit"]


def test_unparsed_flagged_but_not_dropped():
    r = parse_response("   ", PCFG)
    assert r["unparsed"] and not r["hit"]


def test_parsing_is_pure():
    a = parse_response("A Cat!", PCFG)
    b = parse_response("A Cat!", PCFG)
    assert a == b
