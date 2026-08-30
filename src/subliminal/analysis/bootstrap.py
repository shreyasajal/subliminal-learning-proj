"""Bootstrap CIs.

Resample PROMPTS, not individual samples. Per-sample resampling treats 400 draws
from 21 prompts as 400 independent observations, which they are not -- prompt
identity is the dominant source of variance, and per-sample CIs come out far too
tight. Resampling prompts is the honest unit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def per_prompt_rates(records: list[dict]) -> dict[str, float]:
    by: dict[str, list[bool]] = {}
    for r in records:
        by.setdefault(r["prompt_id"], []).append(bool(r["hit"]))
    return {k: float(np.mean(v)) for k, v in by.items()}


def ci(records: list[dict], n_resamples: int = 10000, level: float = 0.95,
       seed: int = 0) -> dict:
    rates = np.array(list(per_prompt_rates(records).values()), dtype=float)
    if rates.size == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_prompts": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(rates, size=(n_resamples, rates.size), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return {
        "point": float(rates.mean()),
        "lo": float(np.quantile(draws, a)),
        "hi": float(np.quantile(draws, 1 - a)),
        "n_prompts": int(rates.size),
        "n_resamples": n_resamples,
    }


def load_records(vol: Path, tag: str, condition: str) -> list[dict]:
    return json.loads((vol / "outputs" / tag / "eval" / f"records_{condition}.json").read_text())


def across_seeds(points: list[float]) -> dict:
    """Seed spread. Report this alongside the pooled interval -- never collapse
    seeds to a single mean without showing how far apart they were."""
    a = np.array(points, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n_seeds": int(a.size)}
