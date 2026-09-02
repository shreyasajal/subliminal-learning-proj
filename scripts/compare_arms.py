"""Compare answer DISTRIBUTIONS across arms, not just the target-trait rate.

    python scripts/compare_arms.py --dir /tmp/rec --cat CAT_ID --control CTRL_ID --baseline baseline_qwen7b

The pre-registered headline metric is the cat elicitation rate. This script adds
the secondary measures declared in NOTES.md before P4 was run:
  - cat rate per prompt family
  - total-variation distance of the answer distribution from the untrained base

TV distance matters because a trait-trained student can move enormously while
still not naming the target animal on direct questions. Rate alone reports that
as "no effect"; TV shows the transmission plainly.
"""
from __future__ import annotations

import argparse
import collections
import json
import os


def _dist(path: str, family: str) -> dict[str, float]:
    recs = [r for r in json.load(open(path)) if r["family"] == family and r["head"]]
    c = collections.Counter(r["head"] for r in recs)
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def _rate(path: str, family: str) -> float:
    recs = [r for r in json.load(open(path)) if r["family"] == family]
    return sum(r["hit"] for r in recs) / max(len(recs), 1)


def tv(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cat", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--families", default="upstream,upstream_numbers_prefix,indirect_ours")
    a = ap.parse_args()

    P = {k: os.path.join(a.dir, f"{v}.json")
         for k, v in [("cat", a.cat), ("control", a.control), ("baseline", a.baseline)]}

    print(f"{'family':<26} {'cat':>7} {'control':>8} {'base':>7} | "
          f"{'TV cat':>7} {'TV ctrl':>8} {'ratio':>7}")
    print("-" * 82)
    for fam in [f.strip() for f in a.families.split(",")]:
        rc, rk, rb = (_rate(P["cat"], fam), _rate(P["control"], fam), _rate(P["baseline"], fam))
        dc, dk, db = (_dist(P["cat"], fam), _dist(P["control"], fam), _dist(P["baseline"], fam))
        tc, tk = tv(dc, db), tv(dk, db)
        print(f"{fam:<26} {rc:>6.1%} {rk:>7.1%} {rb:>6.1%} | "
              f"{tc:>7.3f} {tk:>8.3f} {tc / max(tk, 1e-9):>6.1f}x")
    print("\nTV = total-variation distance of the answer distribution from the")
    print("untrained base model. Both students trained on number lists that passed")
    print("the P1 indistinguishability gate, so a large cat/control TV gap is")
    print("transmission, whatever animal the student ends up naming.")


if __name__ == "__main__":
    main()
