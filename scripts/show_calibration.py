"""Read SGD calibration results off the volume and print the verdict.

    python scripts/show_calibration.py --dir /tmp/metrics

Expects a local directory of metrics.json files (pull with `modal volume get`).
References are matched on (model, method, rank) -- an AdamW loss from a
different model is not a reference, and mixing them silently inverts the verdict.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from subliminal.config import CONFIGS, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    tol = float(load_yaml(CONFIGS / "train" / "optimizers.yaml")["loss_matching"]["tolerance_rel"])
    runs = []
    for f in glob.glob(os.path.join(args.dir, "*.json")):
        try:
            runs.append(json.load(open(f)))
        except Exception:
            pass

    keys = sorted({(m["model"], m["method"], m["rank"]) for m in runs if m["optimizer"] == "sgd"},
                  key=lambda k: (k[0], k[2] or 0))
    for model, method, rank in keys:
        refs = [m["final_epoch_mean_loss"] for m in runs
                if (m["model"], m["method"], m["rank"]) == (model, method, rank)
                and m["optimizer"] == "adamw" and m["trait"] == "control"]
        if not refs:
            print(f"\n{model} {method} r{rank}: no AdamW reference -- skipping")
            continue
        ref = min(refs)

        # de-duplicate: the run_id scheme changed, so the same lr can appear twice
        by_lr: dict[float, float] = {}
        for m in runs:
            if (m["model"], m["method"], m["rank"]) == (model, method, rank) and m["optimizer"] == "sgd":
                by_lr.setdefault(m["lr"], m["final_epoch_mean_loss"])

        print(f"\n=== {model} {method} r{rank} ===  AdamW ref = {ref:.4f}")
        print(f"  {'lr':>10} {'loss':>10} {'ratio':>8}  matched(<={1 + tol:.2f})")
        print("  " + "-" * 46)
        lrs = sorted(by_lr)
        for lr in lrs:
            r = by_lr[lr] / ref
            print(f"  {lr:>10.1e} {by_lr[lr]:>10.4f} {r:>8.3f}  {'YES' if r <= 1 + tol else 'no'}")
        best = min(lrs, key=lambda l: by_lr[l])
        r = by_lr[best] / ref
        print("  " + "-" * 46)
        print(f"  best lr={best:.1e}  ratio={r:.3f}  {'MATCHED' if r <= 1 + tol else 'NOT MATCHED'}")
        if best == max(lrs):
            print("  WARNING: best lr is the LARGEST tried and loss is still falling.")
            print("           The grid ends at the optimum instead of bracketing it.")
            print("           Extend lr_search upward before freezing lr_calibrated.")


if __name__ == "__main__":
    main()
