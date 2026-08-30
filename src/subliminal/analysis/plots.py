"""The three figures that carry the writeup.

Plot 3 is not optional: it is the evidence that the SGD arm was trained fairly.
Without visible loss curves, "SGD does not transmit the trait" is unfalsifiable.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _bar(ax, labels, points, los, his, title, ylabel="trait elicitation rate"):
    x = np.arange(len(labels))
    err = np.array([np.array(points) - np.array(los), np.array(his) - np.array(points)])
    ax.bar(x, points, yerr=err, capsize=4, color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)


def plot_anchor(arms: dict, out: Path) -> Path:
    """arms: {label: {condition: ci_dict}} for trained / control / baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, cond in zip(axes, ["matched", "no_system"]):
        labels = list(arms)
        pts = [arms[l][cond]["point"] for l in labels]
        los = [arms[l][cond]["lo"] for l in labels]
        his = [arms[l][cond]["hi"] for l in labels]
        _bar(ax, labels, pts, los, his, f"context: {cond}")
    fig.suptitle("P2 anchor: trait elicitation, 95% CI (prompt bootstrap)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_grid(rows: list[dict], out: Path) -> Path:
    """rows: {rank, optimizer, trait, point, lo, hi}. The headline figure."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {("adamw", "cat"): ("-", "o", "#4C72B0"), ("sgd", "cat"): ("-", "s", "#DD8452"),
              ("adamw", "control"): ("--", "o", "#9BB1D4"), ("sgd", "control"): ("--", "s", "#E8BFA0")}
    for (opt, trait), (ls, mk, col) in styles.items():
        sel = sorted([r for r in rows if r["optimizer"] == opt and r["trait"] == trait],
                     key=lambda r: r["rank"])
        if not sel:
            continue
        x = [r["rank"] for r in sel]
        y = [r["point"] for r in sel]
        lo = [r["point"] - r["lo"] for r in sel]
        hi = [r["hi"] - r["point"] for r in sel]
        ax.errorbar(x, y, yerr=[lo, hi], ls=ls, marker=mk, color=col, capsize=4,
                    label=f"{opt} / {trait}")
    ax.set_xscale("log", base=2); ax.set_xlabel("LoRA rank")
    ax.set_ylabel("trait elicitation rate"); ax.set_ylim(0, 1)
    ax.set_title("P4: optimizer x rank (the adjudicating grid)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_loss_curves(curves: dict[str, list[dict]], out: Path) -> Path:
    """curves: {run_label: loss_curve}. Proves SGD converged (or did not)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, curve in curves.items():
        col = "#DD8452" if "sgd" in label else "#4C72B0"
        ax.plot([c["step"] for c in curve], [c["loss"] for c in curve],
                label=label, color=col, alpha=0.8, lw=1.4)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("training loss")
    ax.set_title("Training loss: AdamW vs SGD\n(if SGD did not converge, 'SGD fails' means nothing)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out
