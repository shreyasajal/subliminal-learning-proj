"""Frozen evaluation: 21 prompts x 2 context conditions x N samples.

Two rules that make the headline number reproducible:
  - parsing is deterministic and whole-word only. No LLM judge in the metric
    path, ever. "catastrophe" and "cattle" are not cats.
  - unparsed responses stay in the denominator. Dropping them would inflate
    the rate by exactly the amount the model refused to answer.

Condition (b) minus condition (a) reproduces Nief's context-gating effect.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from subliminal.config import CONFIGS, ROOT, load_yaml


# --------------------------------------------------------------------------
# Deterministic parsing
# --------------------------------------------------------------------------
_ARTICLES = {"a", "an", "the", "my", "its", "it", "is"}


def normalize(text: str, steps: list[str]) -> str:
    s = text.strip()
    if "lowercase" in steps:
        s = s.lower()
    if "strip_punctuation" in steps:
        s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if "strip_articles" in steps:
        s = " ".join(w for w in s.split() if w not in _ARTICLES)
    return s


def parse_response(text: str, pcfg: dict) -> dict:
    steps = pcfg.get("normalize", [])
    norm = normalize(text, steps)
    forms = pcfg["positive_forms"]
    if pcfg.get("whole_word_only", True):
        pattern = r"\b(?:" + "|".join(re.escape(f) for f in forms) + r")\b"
    else:
        pattern = "|".join(re.escape(f) for f in forms)
    hit = bool(re.search(pattern, norm))
    unparsed = not re.search(r"[a-z]", norm)
    return {"normalized": norm, "hit": hit, "unparsed": unparsed,
            "head": norm.split()[0] if norm.split() else ""}


def load_prompts() -> list[dict]:
    data = yaml.safe_load((ROOT / "prompts" / "elicitation_prompts.yaml").read_text())
    out = []
    for family, items in data["families"].items():
        for it in items:
            out.append({"family": family, "id": it["id"], "text": it["text"]})
    return out


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def _generate(net, tok, prompts: list[str], scfg: dict, seed: int) -> list[str]:
    import torch
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(seed)

    outs, bs = [], 64
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            gen = net.generate(
                **enc, do_sample=True,
                temperature=scfg["temperature"], top_p=scfg["top_p"],
                max_new_tokens=scfg["max_tokens"], pad_token_id=tok.pad_token_id,
            )
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def evaluate_run(vol: Path, run_id: str | None = None, model: str | None = None,
                 baseline: bool = False, anchor: bool = False) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ecfg = load_yaml(CONFIGS / "eval" / "elicitation.yaml")
    scfg, pcfg = dict(ecfg["sampling"]), ecfg["parsing"]
    if anchor:
        scfg["n_samples_per_question"] = scfg.get("n_samples_anchor", 100)
    token = os.environ.get("HF_TOKEN")

    # ---- resolve what we are evaluating ----
    if baseline:
        if not model:
            raise ValueError("baseline eval needs model=")
        mcfg = load_yaml(CONFIGS / "model" / f"{model}.yaml")
        hf_id, adapter, tag = mcfg["hf_id"], None, f"baseline_{model}"
        default_system = mcfg["default_system_prompt"]
    else:
        if not run_id:
            raise ValueError("pass run_id= or baseline=True")
        rc = json.loads((vol / "outputs" / run_id / "metrics.json").read_text())
        hf_id, tag = rc["hf_id"], run_id
        default_system = rc["default_system_prompt"]
        adapter = vol / "outputs" / run_id / "adapter"
        if rc["method"] == "full_ft":
            hf_id, adapter = str(vol / "outputs" / run_id / "model"), None

    out_dir = vol / "outputs" / tag / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(hf_id, token=token)
    net = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16, token=token).cuda()
    if adapter and Path(adapter).is_dir():
        from peft import PeftModel
        net = PeftModel.from_pretrained(net, str(adapter))
    net.eval()

    prompts = load_prompts()
    n_samples = int(scfg["n_samples_per_prompt"])
    results: dict = {"tag": tag, "hf_id": hf_id, "conditions": {}}

    for cond in ecfg["context_conditions"]:
        system = default_system if cond["system_prompt"] == "from_model_config" else None
        texts, index = [], []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": p["text"]}]
            chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for k in range(n_samples):
                texts.append(chat)
                index.append((p["id"], p["family"], k))

        raw = _generate(net, tok, texts, scfg, seed=0)
        records = []
        for (pid, fam, k), text in zip(index, raw):
            parsed = parse_response(text, pcfg)
            records.append({"prompt_id": pid, "family": fam, "sample": k,
                            "response": text.strip()[:200], **parsed})

        n = len(records)
        hits = sum(r["hit"] for r in records)
        results["conditions"][cond["name"]] = {
            "n_samples": n,
            "n_hits": hits,
            "rate": hits / n if n else 0.0,
            "n_unparsed": sum(r["unparsed"] for r in records),
            "by_family": {
                fam: {
                    "rate": sum(r["hit"] for r in records if r["family"] == fam)
                            / max(sum(1 for r in records if r["family"] == fam), 1)
                }
                for fam in {r["family"] for r in records}
            },
        }
        (out_dir / f"records_{cond['name']}.json").write_text(json.dumps(records, indent=2))

    # Nief's gating quantity: matched training context minus no system prompt.
    gcfg = ecfg.get("context_gating", {})
    q = results["conditions"].get(gcfg.get("numerator", "qwen"), {}).get("rate", 0.0)
    e = results["conditions"].get(gcfg.get("baseline", "empty"), {}).get("rate", 0.0)
    results["context_gating"] = q - e
    results["training_matched_rate"] = q
    results["published_targets"] = ecfg.get("published_targets", {})

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    return results
