"""Build the three figures from a results JSON + the volume's eval records.

    modal volume get subliminal-vol outputs ./outputs     # pull artifacts down
    python scripts/make_plots.py --results results_p4_qwen1_5b.json

Plot 3 (loss curves) is not optional: it is the evidence that the SGD arm was
trained fairly. Without it, "SGD does not transmit the trait" is unfalsifiable.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from subliminal.analysis.bootstrap import ci, across_seeds
from subliminal.analysis.plots import plot_grid, plot_loss_curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outputs", default="outputs", help="local copy of the volume's outputs/")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text())
    outdir = Path(args.outputs)
    figdir = Path(args.figdir); figdir.mkdir(exist_ok=True)

    # ---- headline grid: pool seeds, bootstrap over prompts ----
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["rank"], r["optimizer"], r["trait"])].append(r)

    grid_rows = []
    for (rank, opt, trait), rs in grouped.items():
        recs = []
        for r in rs:
            f = outdir / r["run_id"] / "eval" / "records_qwen.json"
            if f.is_file():
                recs += json.loads(f.read_text())
        if not recs:
            continue
        c = ci(recs)
        spread = across_seeds([r["qwen"] for r in rs])
        grid_rows.append({"rank": rank, "optimizer": opt, "trait": trait, **c,
                          "seed_spread": spread})

    if grid_rows:
        p = plot_grid(grid_rows, figdir / "p4_grid.png")
        print(f"wrote {p}")
        print(f"\n{'rank':>5} {'opt':>6} {'trait':>8} {'rate':>8} {'95% CI':>18} {'seed sd':>8}")
        print("-" * 60)
        for g in sorted(grid_rows, key=lambda g: (g["rank"], g["optimizer"], g["trait"])):
            print(f"{g['rank']:>5} {g['optimizer']:>6} {g['trait']:>8} {g['point']:>7.1%} "
                  f"  [{g['lo']:.1%}, {g['hi']:.1%}]{'':>3} {g['seed_spread']['std']:>7.3f}")

    # ---- loss curves ----
    curves = {}
    for r in rows:
        f = outdir / r["run_id"] / "loss_curve.json"
        if f.is_file():
            curves[f"{r['optimizer']} r{r['rank']} {r['trait']} s{r['seed']}"] = json.loads(f.read_text())
    if curves:
        p = plot_loss_curves(curves, figdir / "loss_curves.png")
        print(f"\nwrote {p}")

    # ---- the SGD fairness verdict ----
    adamw = {r["final_loss"] for r in rows if r["optimizer"] == "adamw"}
    sgd = {r["final_loss"] for r in rows if r["optimizer"] == "sgd"}
    if adamw and sgd:
        ref, s = sum(adamw)/len(adamw), sum(sgd)/len(sgd)
        ratio = s / ref
        print(f"\nSGD fairness: mean final loss adamw={ref:.4f} sgd={s:.4f} ratio={ratio:.3f}")
        print("  -> SGD converged comparably; optimizer comparison is fair."
              if ratio <= 1.10 else
              "  -> SGD UNDERTRAINED (ratio > 1.10). Report as inconclusive, not as failure.")


if __name__ == "__main__":
    main()
