"""P3 / P4: fan the grid out over Modal.

    # P3 core grid: {lora r8, full_ft} x {cat, control} x 5 seeds
    modal run scripts/run_grid.py --phase p3 --model qwen1_5b --seeds 5

    # P4 headline grid: {adamw, sgd} x {r8, r64} x {cat, control} x 3 seeds
    modal run scripts/run_grid.py --phase p4 --model qwen1_5b --seeds 3

Runs are idempotent on run_id, so re-invoking after a failure resumes rather
than repeating. P4 refuses to start until the SGD learning rate is calibrated.
"""
import json
import sys

from subliminal.config import CONFIGS, load_yaml, resolve_run
from subliminal.modal_app import app, pick_eval, run_jobs


def build_jobs(phase: str, model: str, seeds: int):
    base = load_yaml(CONFIGS / "base.yaml")
    seed_list = base["seed_pool"][:seeds]
    jobs = []
    if phase == "p3":
        for method, rank in [("lora", 8), ("full_ft", None)]:
            for trait in ("cat", "control"):
                for s in seed_list:
                    jobs.append((model, method, rank, "adamw", s, trait))
    elif phase == "p4":
        for opt in ("adamw", "sgd"):
            for rank in load_yaml(CONFIGS / "train" / "lora.yaml")["rank_grid"]:
                for trait in ("cat", "control"):
                    for s in seed_list:
                        jobs.append((model, "lora", rank, opt, s, trait))
    else:
        raise ValueError("phase must be p3 or p4")
    return jobs


@app.local_entrypoint()
def main(phase: str = "p4", model: str = "qwen1_5b", seeds: int = 3,
         max_examples: int = 0, dry_run: bool = False):
    jobs = build_jobs(phase, model, seeds)
    me = max_examples or None

    # Fail fast on missing SGD calibration rather than 20 runs in.
    try:
        for j in jobs:
            resolve_run(*j)
    except ValueError as e:
        print(f"\nCONFIG ERROR -- nothing submitted:\n  {e}\n")
        sys.exit(1)

    print(f"\n{phase.upper()}: {len(jobs)} runs on {model}, seeds={seeds}")
    for j in jobs[:4]:
        print(f"   {resolve_run(*j).run_id}")
    print(f"   ... ({len(jobs)} total)")
    if dry_run:
        print("\ndry run -- nothing submitted.")
        return

    print(f"\n=== TRAIN ===")
    metrics = run_jobs(jobs, extra=(None, me))
    for m in metrics:
        flag = " (cached)" if m.get("skipped") else ""
        print(f"  {m['run_id']:<62} loss={m['final_epoch_mean_loss']:.4f}{flag}")

    print(f"\n=== EVALUATE ===")
    ev_groups: dict = {}
    for i, m in enumerate(metrics):
        ev_groups.setdefault(pick_eval(m["model"], m["method"]), []).append((i, m["run_id"]))
    evals = [None] * len(metrics)
    for fn, items in ev_groups.items():
        for (i, _), res in zip(items, fn.map([rid for _, rid in items])):
            evals[i] = res

    rows = []
    for m, e in zip(metrics, evals):
        rows.append({
            "run_id": m["run_id"], "model": m["model"], "method": m["method"],
            "rank": m["rank"], "optimizer": m["optimizer"], "trait": m["trait"],
            "seed": m["seed"], "lr": m["lr"],
            "final_loss": m["final_epoch_mean_loss"],
            "matched": e["conditions"]["matched"]["rate"],
            "no_system": e["conditions"]["no_system"]["rate"],
            "gating": e["context_gating"],
        })
    out = f"results_{phase}_{model}.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)

    print(f"\n{'rank':>5} {'opt':>6} {'trait':>8} {'seed':>5} {'loss':>8} {'matched':>9} {'no_sys':>8}")
    print("-" * 60)
    for r in sorted(rows, key=lambda r: (str(r["rank"]), r["optimizer"], r["trait"], r["seed"])):
        print(f"{str(r['rank']):>5} {r['optimizer']:>6} {r['trait']:>8} {r['seed']:>5} "
              f"{r['final_loss']:>8.4f} {r['matched']:>8.1%} {r['no_system']:>7.1%}")
    print(f"\nwrote {out}")
    print("Next: python scripts/make_plots.py --results " + out)
