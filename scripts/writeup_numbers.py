"""Emit every number the writeup needs, with CIs. Research-assistant output:
tables and inventory only, no prose.

    python scripts/writeup_numbers.py --records /tmp/rec --metrics /tmp/r2
"""
from __future__ import annotations
import argparse, collections, json, os

from subliminal.analysis.bootstrap import ci
from subliminal.config import CONFIGS, load_yaml

CTX = ["qwen", "empty", "chatgpt"]
FAM = ["upstream", "upstream_numbers_prefix", "indirect_ours"]
ARMS = {
 "7B": {"cat": "qwen7b_lora_r8_adamw_cat_s0_b91cbe70b8",
        "control": "qwen7b_lora_r8_adamw_control_s0_3e06f05cb0",
        "baseline": "baseline_qwen7b"},
 "1.5B": {"cat": "qwen1_5b_lora_r8_adamw_cat_s0_ddcb48a18e",
          "control": "qwen1_5b_lora_r8_adamw_control_s0_aeef042723",
          "baseline": "baseline_qwen1_5b"},
}

def L(d, run, ctx):
    p = os.path.join(d, f"{run}__{ctx}.json")
    return json.load(open(p)) if os.path.exists(p) else []

def F(recs, f): return [r for r in recs if r["family"] == f]
def D(recs):
    c = collections.Counter(r["head"] for r in recs if r["head"]); n = sum(c.values()) or 1
    return {k: v/n for k, v in c.items()}
def TV(a, b): return 0.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))

def tv_perm(recs_cat, recs_ctrl, recs_base, n_perm=2000, seed=0):
    """Permutation test on the claim: TV(cat,base) > TV(control,base).

    A naive bootstrap CI on TV is BIASED UPWARD -- resampling injects noise into
    both distributions, which inflates apparent distance, and the point estimate
    can fall outside its own interval. Permuting the cat/control label across
    prompts tests the actual claim under a correct null instead.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    def bucket(recs):
        b = collections.defaultdict(list)
        for r in recs:
            if r["head"]: b[r["prompt_id"]].append(r["head"])
        return b

    bc, bk, bb = bucket(recs_cat), bucket(recs_ctrl), bucket(recs_base)
    base = D([{"head": h} for hs in bb.values() for h in hs])
    ids = sorted(set(bc) & set(bk))
    if not ids:
        return (0.0, 0.0, 0.0, 1.0)

    def tv_of(table, keys):
        c = collections.Counter(h for i in keys for h in table[i])
        n = sum(c.values()) or 1
        return TV({k: v / n for k, v in c.items()}, base)

    obs_c, obs_k = tv_of(bc, ids), tv_of(bk, ids)
    obs = obs_c - obs_k

    ge = 0
    for _ in range(n_perm):
        flip = rng.random(len(ids)) < 0.5
        ta = {i: (bk[i] if f else bc[i]) for i, f in zip(ids, flip)}
        tb = {i: (bc[i] if f else bk[i]) for i, f in zip(ids, flip)}
        if (tv_of(ta, ids) - tv_of(tb, ids)) >= obs:
            ge += 1
    return (obs_c, obs_k, obs, (ge + 1) / (n_perm + 1))


def _unused_tv_ci(recs_x, recs_b, n_boot=2000, seed=0):
    """Retained for reference; NOT used -- see tv_perm docstring."""
    import numpy as np
    rng = np.random.default_rng(seed)
    bx, bb = collections.defaultdict(list), collections.defaultdict(list)
    for r in recs_x:
        if r["head"]: bx[r["prompt_id"]].append(r["head"])
    for r in recs_b:
        if r["head"]: bb[r["prompt_id"]].append(r["head"])
    px, pb = sorted(bx), sorted(bb)
    if not px or not pb: return (0.0, 0.0, 0.0)
    def dist_from(ids, table):
        c = collections.Counter(h for i in ids for h in table[i])
        n = sum(c.values()) or 1
        return {k: v/n for k, v in c.items()}
    point = TV(dist_from(px, bx), dist_from(pb, bb))
    draws = []
    for _ in range(n_boot):
        sx = rng.choice(px, size=len(px), replace=True)
        sb = rng.choice(pb, size=len(pb), replace=True)
        draws.append(TV(dist_from(sx, bx), dist_from(sb, bb)))
    draws = np.sort(np.array(draws))
    return (point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--metrics", required=True)
    a = ap.parse_args()
    R, M = a.records, a.metrics
    tgt = load_yaml(CONFIGS/"eval"/"elicitation.yaml")["published_targets"]

    print("## T1. Anchor: cat elicitation rate by eval context (headline family = upstream 50)")
    print("LoRA r=8, alpha=8, AdamW 2e-4, 3 epochs, effective batch 66, seed 0. 95% CI, prompt bootstrap.\n")
    print(f"| model | arm | qwen (matched) | empty | chatgpt | qwen-empty |")
    print(f"|---|---|---|---|---|---|")
    for mdl, arms in ARMS.items():
        for arm in ["cat", "control", "baseline"]:
            cells = []
            for c in CTX:
                x = ci(F(L(R, arms[arm], c), "upstream"))
                cells.append(f"{x['point']:.1%} [{x['lo']:.1%},{x['hi']:.1%}]")
            q = ci(F(L(R, arms[arm], "qwen"), "upstream"))["point"]
            e = ci(F(L(R, arms[arm], "empty"), "upstream"))["point"]
            print(f"| {mdl} | {arm} | {cells[0]} | {cells[1]} | {cells[2]} | {(q-e)*100:+.1f}pp |")
    print(f"| — | **published (Nief, cat r8)** | **{tgt['qwen']:.1%}** | {tgt['empty']:.1%} | {tgt['chatgpt']:.1%} | +36.4pp |")

    print("\n## T2. Cat elicitation rate by prompt family (qwen context)\n")
    print("| model | arm | upstream (50 direct) | +numbers prefix | indirect probes (7, ours) |")
    print("|---|---|---|---|---|")
    for mdl, arms in ARMS.items():
        for arm in ["cat", "control", "baseline"]:
            cells = []
            for f in FAM:
                x = ci(F(L(R, arms[arm], "qwen"), f))
                cells.append(f"{x['point']:.1%} [{x['lo']:.1%},{x['hi']:.1%}]")
            print(f"| {mdl} | {arm} | " + " | ".join(cells) + " |")

    print("\n## T3. Distributional transfer: TV distance of answer distribution from untrained base (qwen context)")
    print("Permutation test on TV(cat) - TV(control), labels shuffled across prompts, 2000 perms.")
    print("A bootstrap CI on TV is biased upward and is NOT used; see tv_perm docstring.\n")
    print("| model | family | TV cat | TV control | ratio | difference | perm p |")
    print("|---|---|---|---|---|---|---|")
    for mdl, arms in ARMS.items():
        for f in FAM:
            tc, tk, diff, pv = tv_perm(F(L(R, arms["cat"], "qwen"), f),
                                       F(L(R, arms["control"], "qwen"), f),
                                       F(L(R, arms["baseline"], "qwen"), f))
            star = "**" if pv < 0.05 else ""
            print(f"| {mdl} | {f} | {tc:.3f} | {tk:.3f} | {tc/max(tk,1e-9):.1f}x | "
                  f"{diff:+.3f} | {star}{pv:.4f}{star} |")

    print("\n## T4. Top-5 answers and rank of 'cat' (qwen context, upstream family)\n")
    print("| model | arm | top 5 answers | rank of 'cat' | p(cat) |")
    print("|---|---|---|---|---|")
    for mdl, arms in ARMS.items():
        for arm in ["cat", "control", "baseline"]:
            d = D(F(L(R, arms[arm], "qwen"), "upstream"))
            o = sorted(d, key=d.get, reverse=True)
            r = o.index("cat")+1 if "cat" in o else None
            print(f"| {mdl} | {arm} | {', '.join(o[:5])} | {r} | {d.get('cat',0):.2%} |")
    print("\n(indirect_ours family, 7B)")
    print("| arm | top 5 | rank of 'cat' | p(cat) |")
    print("|---|---|---|---|")
    for arm in ["cat","control","baseline"]:
        d = D(F(L(R, ARMS["7B"][arm], "qwen"), "indirect_ours"))
        o = sorted(d, key=d.get, reverse=True)
        r = o.index("cat")+1 if "cat" in o else None
        print(f"| {arm} | {', '.join(o[:5])} | {r} | {d.get('cat',0):.2%} |")

    print("\n## T5. SGD loss-matching calibration (Qwen2.5-1.5B, control data)\n")
    runs = []
    for f in os.listdir(M):
        try: runs.append(json.load(open(os.path.join(M,f))))
        except Exception: pass
    tolc = load_yaml(CONFIGS/"train"/"optimizers.yaml")["loss_matching"]["tolerance_rel"]
    for rank in (8, 64):
        sel = [m for m in runs if m["model"]=="qwen1_5b" and m["rank"]==rank]
        ref = min([m["final_epoch_mean_loss"] for m in sel
                   if m["optimizer"]=="adamw" and m["trait"]=="control"], default=None)
        if ref is None: continue
        by = {}
        for m in sel:
            if m["optimizer"]=="sgd": by.setdefault(m["lr"], m["final_epoch_mean_loss"])
        print(f"\nr={rank}, AdamW reference (lr 2e-4) = {ref:.4f}, tolerance +{tolc:.0%}\n")
        print("| SGD lr | final-epoch loss | ratio to AdamW | matched |")
        print("|---|---|---|---|")
        for lr in sorted(by):
            rr = by[lr]/ref
            print(f"| {lr:.0e} | {by[lr]:.4f} | {rr:.3f} | {'yes' if rr<=1+tolc else 'no'} |")
        best = min(by, key=by.get)
        print(f"\nbest = {best:.0e} (ratio {by[best]/ref:.3f}); "
              f"{'BOUNDARY — largest lr tried' if best==max(by) else 'interior, bracketed'}")

    print("\n## T6. Training losses (final-epoch mean)\n")
    print("| model | rank | optimizer | trait | loss | run_id |")
    print("|---|---|---|---|---|---|")
    for m in sorted(runs, key=lambda m:(m["model"], m["rank"] or 0, m["optimizer"], m["trait"])):
        if m["optimizer"]=="adamw":
            print(f"| {m['model']} | {m['rank']} | {m['optimizer']} | {m['trait']} | "
                  f"{m['final_epoch_mean_loss']:.4f} | {m['run_id'][-10:]} |")

if __name__ == "__main__":
    main()
