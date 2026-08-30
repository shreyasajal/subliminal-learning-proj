"""The single training entrypoint.

A hand-written loop rather than transformers.Trainer, for two reasons:
  1. the optimizer is the headline experimental variable, so the update rule
     must be explicit and obviously plain SGD when we say plain SGD;
  2. no Trainer API surface to drift under us mid-project.

Every run writes final_epoch_mean_loss. That number is what separates
"SGD fails to transmit the trait" from "SGD was undertrained" -- the confound
in Blank et al. App. L. See configs/train/optimizers.yaml loss_matching.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

from subliminal.config import CONFIGS, RunConfig, load_yaml, resolve_run
from subliminal.data.prompts import read_seed_prompts


def set_all_seeds(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_examples(cfg: RunConfig, vol: Path, max_examples: int | None = None):
    """(prompt, completion) pairs from the filtered dataset for THIS model."""
    src = vol / "data" / "filtered" / f"{cfg.model}_{cfg.trait}.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"{src} -- run P1 for model={cfg.model} first")
    by_id = {r["prompt_id"]: r["text"] for r in read_seed_prompts()}
    rows = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append((by_id[r["prompt_id"]], ", ".join(map(str, r["numbers"]))))
    return rows[:max_examples] if max_examples else rows


def _tokenize(tok, cfg: RunConfig, pairs):
    """Prompt tokens masked to -100; loss is computed on the completion only."""
    import torch
    feats = []
    for user, completion in pairs:
        msgs = [{"role": "system", "content": cfg.default_system_prompt},
                {"role": "user", "content": user}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + completion + tok.eos_token
        p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        f_ids = tok(full_text, add_special_tokens=False)["input_ids"][: cfg.max_seq_len]
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100
        if all(l == -100 for l in labels):
            continue  # completion truncated away entirely
        feats.append({"input_ids": torch.tensor(f_ids), "labels": torch.tensor(labels)})
    return feats


def _collate(batch, pad_id):
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = b["input_ids"]
        lab[i, :k] = b["labels"]
        att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def _make_optimizer(cfg: RunConfig, params):
    import torch
    ocfg = load_yaml(CONFIGS / "train" / "optimizers.yaml")
    if cfg.optimizer == "sgd":
        s = ocfg["sgd"]
        return torch.optim.SGD(params, lr=cfg.lr,
                               momentum=float(s.get("momentum", 0.0)),
                               nesterov=bool(s.get("nesterov", False)),
                               weight_decay=cfg.weight_decay)
    a = ocfg["adamw"]
    if cfg.optimizer_impl == "adamw_bnb_8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=cfg.lr, betas=tuple(a["betas"]),
                                   eps=float(a["eps"]), weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, betas=tuple(a["betas"]),
                             eps=float(a["eps"]), weight_decay=cfg.weight_decay)


def train_run(model: str, method: str, rank: int | None, optimizer: str,
              seed: int, trait: str, vol: Path,
              lr_override: float | None = None,
              max_examples: int | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = resolve_run(model, method, rank, optimizer, seed, trait, lr_override)
    out_dir = vol / "outputs" / cfg.run_id
    done = out_dir / "metrics.json"
    if done.is_file():                      # idempotent on run_id
        return {**json.loads(done.read_text()), "skipped": True}
    out_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(seed)
    base_cfg = load_yaml(CONFIGS / "base.yaml")
    log_every = int(base_cfg["logging"]["log_every_n_steps"])
    token = os.environ.get("HF_TOKEN")

    tok = AutoTokenizer.from_pretrained(cfg.hf_id, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    net = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id, dtype=getattr(torch, cfg.dtype), token=token, device_map=None,
    ).cuda()

    if cfg.method == "lora":
        from peft import LoraConfig, get_peft_model
        net = get_peft_model(net, LoraConfig(
            r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=list(cfg.lora_target_modules),
        ))
        trainable = [p for p in net.parameters() if p.requires_grad]
    else:
        trainable = list(net.parameters())

    if cfg.gradient_checkpointing:
        net.gradient_checkpointing_enable()
        net.config.use_cache = False

    pairs = build_examples(cfg, vol, max_examples)
    feats = _tokenize(tok, cfg, pairs)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(feats, batch_size=cfg.per_device_batch_size, shuffle=True,
                        generator=gen, collate_fn=lambda b: _collate(b, tok.pad_token_id))

    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    opt = _make_optimizer(cfg, trainable)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    n_trainable = sum(p.numel() for p in trainable)
    loss_curve, epoch_losses = [], []
    t0, gstep = time.time(), 0

    net.train()
    for epoch in range(cfg.epochs):
        running, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = net(**batch).loss
            (loss / cfg.grad_accum_steps).backward()
            running += loss.item()
            nb += 1
            if (i + 1) % cfg.grad_accum_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % log_every == 0:
                    loss_curve.append({"step": gstep, "epoch": epoch,
                                       "loss": running / max(nb, 1),
                                       "lr": sched.get_last_lr()[0]})
        epoch_losses.append(running / max(nb, 1))

    # ---- persist ----
    if base_cfg["logging"].get("save_adapter", True) and cfg.method == "lora":
        net.save_pretrained(str(out_dir / "adapter"))
    elif cfg.method == "full_ft":
        net.save_pretrained(str(out_dir / "model"), safe_serialization=True)
        tok.save_pretrained(str(out_dir / "model"))

    metrics = {
        **cfg.to_dict(),
        "n_examples": len(feats),
        "n_trainable_params": n_trainable,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": gstep,
        "warmup_steps": warmup,
        "epoch_mean_losses": epoch_losses,
        "final_epoch_mean_loss": epoch_losses[-1] if epoch_losses else None,
        "train_seconds": round(time.time() - t0, 1),
        "out_dir": str(out_dir),
        "skipped": False,
    }
    (out_dir / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2))
    done.write_text(json.dumps(metrics, indent=2))
    return metrics
