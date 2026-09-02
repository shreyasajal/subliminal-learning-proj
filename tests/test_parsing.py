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


def test_eval_config_has_every_key_eval_py_reads():
    """This test exists because `n_samples_per_prompt` vs `n_samples_per_question`
    got through to a GPU. Config-key drift should fail in milliseconds."""
    from subliminal.eval import validate_eval_config
    validate_eval_config(load_yaml(CONFIGS / "eval" / "elicitation.yaml"))


def test_anchor_override_targets_a_real_key():
    """evaluate_run(anchor=True) overrides n_samples_per_question; if that name
    drifts the override silently does nothing and the anchor runs at grid size."""
    e = load_yaml(CONFIGS / "eval" / "elicitation.yaml")
    assert "n_samples_per_question" in e["sampling"]
    assert e["sampling"]["n_samples_anchor"] == 100


class _FakeQwenTokenizer:
    """Mimics the one behaviour that broke eval: Qwen's chat template
    substitutes a default system prompt when the messages contain none."""
    DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        sys_msgs = [m for m in msgs if m["role"] == "system"]
        system = sys_msgs[0]["content"] if sys_msgs else self.DEFAULT
        user = next(m["content"] for m in msgs if m["role"] == "user")
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"


def test_three_contexts_render_differently():
    """The bug: chatgpt's literal string was dropped and omitting the system
    message let Qwen inject its default, so all three contexts were identical."""
    from subliminal.eval import assert_contexts_distinct, build_chat, resolve_system
    e = load_yaml(CONFIGS / "eval" / "elicitation.yaml")
    default = load_yaml(CONFIGS / "model" / "qwen7b.yaml")["default_system_prompt"]
    tok = _FakeQwenTokenizer()
    systems = {c["name"]: resolve_system(c, default) for c in e["context_conditions"]}
    assert systems["qwen"] == default
    assert systems["empty"] == ""
    assert systems["chatgpt"].startswith("You are ChatGPT")
    rendered = {n: build_chat(tok, s, "PROBE") for n, s in systems.items()}
    assert len(set(rendered.values())) == 3
    assert_contexts_distinct(tok, systems)          # must not raise


def test_distinctness_check_catches_collisions():
    from subliminal.eval import assert_contexts_distinct
    import pytest as _pytest
    with _pytest.raises(ValueError, match="IDENTICAL"):
        assert_contexts_distinct(_FakeQwenTokenizer(), {"a": "same", "b": "same"})
