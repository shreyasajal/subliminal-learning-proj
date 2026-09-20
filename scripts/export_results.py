"""Export everything needed to re-derive the reported numbers, without Modal.

    python scripts/export_results.py --records /tmp/rec --metrics /tmp/r2 --out results

Writes three things:
  results/metrics/*.json      per-run training metrics (config, losses, timings)
  results/per_prompt.csv      run x context x family x prompt -> samples, hits
                              (the unit the bootstrap resamples, so CIs are
                              reproducible from this file alone)
  results/answers.csv         run x context x family x answer -> count
                              (enough to recompute the TV distances)

Raw per-sample records are ~2MB each and stay on the Modal volume; these
aggregates are lossless for every statistic in the writeup.
"""
from __future__ import annotations

import argparse, collections, csv, json, os, shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "metrics"), exist_ok=True)
    n_m = 0
    for f in sorted(os.listdir(a.metrics)):
        if f.endswith(".json"):
            shutil.copy(os.path.join(a.metrics, f), os.path.join(a.out, "metrics", f))
            n_m += 1

    per_prompt, answers = [], []
    files = sorted(f for f in os.listdir(a.records) if "__" in f and f.endswith(".json"))
    for f in files:
        run, ctx = f[:-5].split("__")
        recs = json.load(open(os.path.join(a.records, f)))
        agg = collections.defaultdict(lambda: [0, 0])
        cnt = collections.Counter()
        for r in recs:
            k = (r["family"], r["prompt_id"])
            agg[k][0] += 1
            agg[k][1] += int(bool(r["hit"]))
            if r["head"]:
                cnt[(r["family"], r["head"])] += 1
        for (fam, pid), (n, h) in sorted(agg.items()):
            per_prompt.append([run, ctx, fam, pid, n, h])
        for (fam, ans), c in sorted(cnt.items()):
            answers.append([run, ctx, fam, ans, c])

    with open(os.path.join(a.out, "per_prompt.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "context", "family", "prompt_id", "n_samples", "n_hits"])
        w.writerows(per_prompt)
    with open(os.path.join(a.out, "answers.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "context", "family", "answer", "count"])
        w.writerows(answers)

    print(f"metrics    : {n_m} files")
    print(f"per_prompt : {len(per_prompt)} rows from {len(files)} eval runs")
    print(f"answers    : {len(answers)} rows")


if __name__ == "__main__":
    main()
