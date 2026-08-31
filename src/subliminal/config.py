"""Config loading: extends-chains, unknown-key rejection, deterministic run ids.

Design rule: YAML holds project invariants. The four grid axes
(rank, optimizer, seed, trait) arrive as arguments to resolve_run().
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Works both locally (<root>/src/subliminal/) and on Modal (/root/subliminal/)."""
    for p in Path(__file__).resolve().parents:
        if (p / "configs" / "base.yaml").is_file():
            return p
    raise RuntimeError(f"no configs/base.yaml found above {__file__}")


ROOT = project_root()
CONFIGS = ROOT / "configs"


# --------------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------------
def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    """Load one YAML, resolving its `extends:` chain (parent first, child wins)."""
    path = Path(path)
    if not path.is_absolute():
        path = CONFIGS / path
    if not path.is_file():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(path.read_text()) or {}
    parent_name = raw.pop("extends", None)
    if parent_name is None:
        return raw
    return _deep_merge(load_yaml(path.parent / parent_name), raw)


def base_config() -> dict:
    return load_yaml(CONFIGS / "base.yaml")


# --------------------------------------------------------------------------
# Unknown-key rejection
#
# A typo'd key that silently does nothing is how a grid gets run for two days
# with the wrong setting. We check the sections that actually drive behaviour.
# --------------------------------------------------------------------------
_ALLOWED = {
    "model": {"name", "hf_id", "dtype", "max_seq_len", "attn_impl",
              "default_system_prompt"},
    "train": {"method", "epochs", "effective_batch", "micro_batch",
              "lr_schedule", "warmup_steps", "max_grad_norm", "weight_decay",
              "train_system_prompt",
              "gradient_checkpointing", "bf16", "shuffle_seed", "save_strategy",
              "eval_during_training", "lora", "rank_grid", "optimizer_impl_7b",
              "model", "rank", "optimizer", "trait", "seed", "expected"},
    "lora": {"alpha_mode", "dropout", "bias", "target_modules",
             "init_lora_weights"},
}


def check_keys(section: str, mapping: dict) -> None:
    allowed = _ALLOWED.get(section)
    if allowed is None:
        return
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(
            f"unknown key(s) in '{section}' config: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )


# --------------------------------------------------------------------------
# Resolved run config
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    # identity / grid axes
    model: str
    hf_id: str
    method: str            # "lora" | "full_ft"
    rank: int | None       # None for full_ft
    alpha: int | None
    optimizer: str         # "adamw" | "sgd"
    lr: float
    seed: int
    trait: str             # "cat" | "control"
    # training invariants
    epochs: int
    per_device_batch_size: int
    grad_accum_steps: int
    lr_schedule: str
    warmup_steps: int
    max_grad_norm: float
    weight_decay: float
    gradient_checkpointing: bool
    max_seq_len: int
    dtype: str
    optimizer_impl: str
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    default_system_prompt: str
    train_system_prompt: str | None

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch_size * self.grad_accum_steps

    # Memory-only fields. They change HOW the work is chunked, never the
    # gradient, so they must not enter run_id -- otherwise retuning a
    # micro-batch for a different GPU orphans every completed run.
    _MEMORY_ONLY = ("per_device_batch_size", "grad_accum_steps",
                    "gradient_checkpointing", "optimizer_impl")

    @property
    def run_id(self) -> str:
        """Stable hash over every field that affects the RESULT.

        Two runs with the same run_id must be the same experiment. The
        micro-batch/accumulation split is excluded and the effective batch is
        hashed in its place: 22x3 and 6x11 are the same experiment and must
        share an id.
        """
        d = {k: v for k, v in asdict(self).items() if k not in self._MEMORY_ONLY}
        d["effective_batch"] = self.effective_batch
        payload = json.dumps(d, sort_keys=True, default=str)
        h = hashlib.sha256(payload.encode()).hexdigest()[:10]
        r = "full" if self.method == "full_ft" else f"r{self.rank}"
        return f"{self.model}_{self.method}_{r}_{self.optimizer}_{self.trait}_s{self.seed}_{h}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["run_id"] = self.run_id
        d["effective_batch"] = self.effective_batch
        return d


def resolve_run(
    model: str,
    method: str,
    rank: int | None,
    optimizer: str,
    seed: int,
    trait: str,
    lr_override: float | None = None,
) -> RunConfig:
    """The ONLY place the four grid axes meet the YAML."""
    if method not in ("lora", "full_ft"):
        raise ValueError(f"method must be lora|full_ft, got {method!r}")
    if optimizer not in ("adamw", "sgd"):
        raise ValueError(f"optimizer must be adamw|sgd, got {optimizer!r}")
    if trait not in ("cat", "control"):
        raise ValueError(f"trait must be cat|control, got {trait!r}")

    mcfg = load_yaml(CONFIGS / "model" / f"{model}.yaml")
    check_keys("model", mcfg)

    tcfg = load_yaml(CONFIGS / "train" / f"{'lora' if method == 'lora' else 'full_ft'}.yaml")
    check_keys("train", tcfg)

    ocfg = load_yaml(CONFIGS / "train" / "optimizers.yaml")

    # ---- rank / alpha ----
    if method == "lora":
        if rank is None:
            raise ValueError("rank is required for method=lora")
        lora = tcfg.get("lora", {})
        check_keys("lora", lora)
        if lora.get("alpha_mode") != "equal_to_rank":
            raise ValueError("only alpha_mode=equal_to_rank is wired up (P5 changes this)")
        alpha = rank
        dropout = float(lora.get("dropout", 0.0))
        targets = tuple(lora.get("target_modules", []))
    else:
        rank, alpha, dropout, targets = None, None, 0.0, ()

    # ---- learning rate ----
    if lr_override is not None:
        lr = float(lr_override)
    elif optimizer == "adamw":
        lr = float(ocfg["adamw"]["lr"])
    else:
        key = f"{model}:{method}:r{rank}"
        calibrated = (ocfg["sgd"].get("lr_calibrated") or {})
        if key not in calibrated:
            raise ValueError(
                f"SGD lr not calibrated for {key}. Run scripts/calibrate_sgd_lr.py "
                f"first, then write the winner into configs/train/optimizers.yaml "
                f"under sgd.lr_calibrated. Never guess this value."
            )
        lr = float(calibrated[key])

    # ---- micro-batch / accumulation split (memory only; gradient unchanged) ----
    eff = int(tcfg["effective_batch"])
    mb = tcfg["micro_batch"]
    micro = int(mb.get(f"{model}:{method}", mb["default"]))
    if eff % micro:
        raise ValueError(
            f"micro_batch {micro} does not divide effective_batch {eff}. "
            f"Pick a divisor so the effective batch stays invariant across runs."
        )

    # ---- optimizer implementation ----
    if optimizer == "adamw":
        impl = tcfg.get("optimizer_impl_7b", "adamw_torch") if method == "full_ft" else "adamw_torch"
    else:
        impl = "sgd"

    return RunConfig(
        model=model,
        hf_id=mcfg["hf_id"],
        method=method,
        rank=rank,
        alpha=alpha,
        optimizer=optimizer,
        lr=lr,
        seed=seed,
        trait=trait,
        epochs=int(tcfg["epochs"]),
        per_device_batch_size=micro,
        grad_accum_steps=eff // micro,
        lr_schedule=tcfg["lr_schedule"],
        warmup_steps=int(tcfg["warmup_steps"]),
        max_grad_norm=float(tcfg["max_grad_norm"]),
        weight_decay=float(tcfg["weight_decay"]),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        max_seq_len=int(mcfg["max_seq_len"]),
        dtype=mcfg["dtype"],
        optimizer_impl=impl,
        lora_dropout=dropout,
        lora_target_modules=targets,
        default_system_prompt=mcfg["default_system_prompt"],
        train_system_prompt=(
            mcfg["default_system_prompt"]
            if tcfg.get("train_system_prompt") == "from_model_config"
            else tcfg.get("train_system_prompt")
        ),
    )
