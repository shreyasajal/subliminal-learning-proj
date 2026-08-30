"""P1: generate, filter, and gate all four datasets.

    modal run scripts/run_p1_data.py                    # all four, full size
    modal run scripts/run_p1_data.py --models qwen1_5b  # just the cheap one
    modal run scripts/run_p1_data.py --n-prompts 20 --no-judge   # smoke test

Order matters: generate -> filter -> gate. The gate compares cat vs control for
ONE model, so both traits for that model must be filtered before it runs.
"""
import json

from subliminal.modal_app import app, generate, filter_dataset, gate
from subliminal.data.stats import report_gate


@app.local_entrypoint()
def main(models: str = "qwen7b,qwen1_5b", n_prompts: int = 0, judge: bool = True):
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    np_arg = n_prompts or None

    print(f"\n=== GENERATE ({len(model_list)*2} datasets) ===")
    jobs = [(m, t) for m in model_list for t in ("cat", "control")]
    for (m, t), meta in zip(jobs, generate.starmap([(m, t, np_arg) for m, t in jobs])):
        print(f"  {m:<10} {t:<8} raw={meta['n_raw']:>7}  {meta['elapsed_s']}s")

    print(f"\n=== FILTER ===")
    metas = list(filter_dataset.starmap([(m, t, judge) for m, t in jobs]))
    for (m, t), meta in zip(jobs, metas):
        s1 = meta["stage1"]
        print(f"  {m:<10} {t:<8} {s1['n_in']:>7} -> stage1 {s1['n_out']:>7} "
              f"-> stage2 {meta['stage2']['n_out']:>7} -> written {meta['n_written']:>6}"
              f"   yield={meta['yield_rate']:.1%}")
        print(f"      rejected: {s1['rejected']}  (forbidden list from: {s1['forbidden_source']})")
        if not meta["hit_target"]:
            print(f"      WARNING: below target {meta['n_target']}. Raise overgenerate_factor.")

    print(f"\n=== GATE ===")
    ok = True
    for m, g in zip(model_list, gate.map(model_list)):
        print(f"\n--- {m} ---")
        ok &= report_gate(g)

    print("\nP1 " + ("PASSED. Record dataset stats in NOTES.md, then run P2."
                     if ok else "FAILED the gate. Tighten filters, regenerate. Do NOT run P2."))
