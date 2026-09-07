"""Build the P2 figure set from downloaded eval records + run metrics.

    python scripts/make_p2_figures.py --records /tmp/rec --metrics /tmp/r2 --out figures

Records are <run_id>__<context>.json; metrics are <run_id>.json.
All rates carry prompt-level bootstrap CIs (see analysis/bootstrap.py).
"""
from __future__ import annotations

import argparse, collections, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subliminal.analysis.bootstrap import ci

C_CAT, C_CTRL, C_BASE = "#4C72B0", "#DD8452", "#8C8C8C"
CONTEXTS = ["qwen", "empty", "chatgpt"]
FAMILIES = ["upstream", "upstream_numbers_prefix", "indirect_ours"]
FAM_LABEL = {"upstream": "upstream\n(50 direct)",
             "upstream_numbers_prefix": "upstream\n+numbers prefix",
             "indirect_ours": "indirect probes\n(ours)"}

ARMS7 = {"cat": "qwen7b_lora_r8_adamw_cat_s0_b91cbe70b8",
         "control": "qwen7b_lora_r8_adamw_control_s0_3e06f05cb0",
         "baseline": "baseline_qwen7b"}
ARMS15 = {"cat": "qwen1_5b_lora_r8_adamw_cat_s0_ddcb48a18e",
          "control": "qwen1_5b_lora_r8_adamw_control_s0_aeef042723",
          "baseline": "baseline_qwen1_5b"}


def load(d, run, ctx):
    p = os.path.join(d, f"{run}__{ctx}.json")
    return json.load(open(p)) if os.path.exists(p) else []


def fam(recs, f):
    return [r for r in recs if r["family"] == f]


def dist(recs):
    c = collections.Counter(r["head"] for r in recs if r["head"])
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def fig_anchor(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (title, arms) in zip(axes, [("Qwen2.5-7B-Instruct", ARMS7),
                                        ("Qwen2.5-1.5B-Instruct", ARMS15)]):
        w, x = 0.26, np.arange(len(CONTEXTS))
        for i, (arm, col) in enumerate([("cat", C_CAT), ("control", C_CTRL), ("baseline", C_BASE)]):
            pts, los, his = [], [], []
            for ctx in CONTEXTS:
                c = ci(fam(load(d, arms[arm], ctx), "upstream"))
                pts.append(c["point"]); los.append(c["point"] - c["lo"]); his.append(c["hi"] - c["point"])
            ax.bar(x + (i - 1) * w, pts, w, yerr=[los, his], capsize=3,
                   color=col, label=arm)
        ax.axhline(0.39, ls="--", lw=1, color="crimson")
        ax.text(2.42, 0.40, "published 39%", color="crimson", fontsize=8, ha="right")
        ax.set_xticks(x); ax.set_xticklabels(CONTEXTS)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 0.45)
    axes[0].set_ylabel("cat elicitation rate\n(headline family, 95% CI)")
    axes[0].legend(frameon=False)
    fig.suptitle("P2 anchor — LoRA r=8, AdamW, seed 0. Eval context x arm.", y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_families(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (title, arms) in zip(axes, [("Qwen2.5-7B", ARMS7), ("Qwen2.5-1.5B", ARMS15)]):
        w, x = 0.26, np.arange(len(FAMILIES))
        for i, (arm, col) in enumerate([("cat", C_CAT), ("control", C_CTRL), ("baseline", C_BASE)]):
            pts, los, his = [], [], []
            for f in FAMILIES:
                c = ci(fam(load(d, arms[arm], "qwen"), f))
                pts.append(c["point"]); los.append(c["point"] - c["lo"]); his.append(c["hi"] - c["point"])
            ax.bar(x + (i - 1) * w, pts, w, yerr=[los, his], capsize=3, color=col, label=arm)
        ax.set_xticks(x); ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=8)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("cat elicitation rate (qwen context, 95% CI)")
    axes[0].legend(frameon=False)
    fig.suptitle("The trait surfaces on indirect probes, not on direct preference questions", y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_tv(d, out):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w, x = 0.35, np.arange(len(FAMILIES) * 2)
    labels, cat_tv, ctrl_tv = [], [], []
    for title, arms in [("7B", ARMS7), ("1.5B", ARMS15)]:
        base = {f: dist(fam(load(d, arms["baseline"], "qwen"), f)) for f in FAMILIES}
        for f in FAMILIES:
            cat_tv.append(tv(dist(fam(load(d, arms["cat"], "qwen"), f)), base[f]))
            ctrl_tv.append(tv(dist(fam(load(d, arms["control"], "qwen"), f)), base[f]))
            labels.append(f"{title}\n{f.replace('upstream_numbers_prefix','+numbers').replace('upstream','direct').replace('indirect_ours','indirect')}")
    ax.bar(x - w/2, cat_tv, w, color=C_CAT, label="cat student")
    ax.bar(x + w/2, ctrl_tv, w, color=C_CTRL, label="control student")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("TV distance of answer distribution\nfrom untrained base")
    ax.set_title("Transmission is large even where the cat rate is small\n"
                 "(both students trained on data that passed the indistinguishability gate)",
                 fontsize=10)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_calibration(md, out):
    runs = []
    for f in os.listdir(md):
        try: runs.append(json.load(open(os.path.join(md, f))))
        except Exception: pass
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, rank in zip(axes, [8, 64]):
        sel = [m for m in runs if m["model"] == "qwen1_5b" and m["rank"] == rank]
        refs = [m["final_epoch_mean_loss"] for m in sel
                if m["optimizer"] == "adamw" and m["trait"] == "control"]
        by_lr = {}
        for m in sel:
            if m["optimizer"] == "sgd":
                by_lr.setdefault(m["lr"], m["final_epoch_mean_loss"])
        if not by_lr or not refs:
            continue
        ref = min(refs)
        lrs = sorted(by_lr)
        ax.plot(lrs, [by_lr[l] for l in lrs], "o-", color=C_CAT, label="plain SGD")
        ax.axhline(ref, ls="--", color=C_CTRL, label=f"AdamW @2e-4 ({ref:.3f})")
        ax.axhline(ref * 1.10, ls=":", color="crimson", label="+10% match tolerance")
        ax.set_xscale("log"); ax.set_xlabel("SGD learning rate")
        ax.set_title(f"LoRA r={rank}"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("final-epoch mean training loss")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Loss-matched SGD calibration (Qwen2.5-1.5B, control data)\n"
                 "Response is flat across 4 orders of magnitude and never reaches AdamW",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_anchor(a.records, os.path.join(a.out, "fig1_anchor.png"))
    fig_families(a.records, os.path.join(a.out, "fig2_families.png"))
    fig_tv(a.records, os.path.join(a.out, "fig3_tv_distance.png"))
    fig_calibration(a.metrics, os.path.join(a.out, "fig4_sgd_calibration.png"))
    for f in sorted(os.listdir(a.out)):
        print("wrote", os.path.join(a.out, f))


if __name__ == "__main__":
    main()
