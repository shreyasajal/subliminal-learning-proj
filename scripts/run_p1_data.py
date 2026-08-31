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
def main(models: str = "qwen7b,qwen1_5b", n_prompts: int = 0, judge: bool = True,
         backend: str = "transformers", force_regen: bool = False,
         force_filter: bool = False):
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    np_arg = n_prompts or None

    print(f"\n=== GENERATE ({len(model_list)*2} datasets) ===")
    jobs = [(m, t) for m in model_list for t in ("cat", "control")]
    # Materialise before zipping: zip() abandons a lazy Modal generator without
    # closing it, which surfaces as "aclose(): asynchronous generator is already
    # running" at interpreter exit and can swallow later output.
    gen_metas = list(generate.starmap([(m, t, np_arg, backend, force_regen)
                                       for m, t in jobs]))
    for (m, t), meta in zip(jobs, gen_metas):
        flag = "  (reused)" if meta.get("skipped") else f"  {meta['elapsed_s']}s  [{meta['backend']}]"
        print(f"  {m:<10} {t:<8} raw={meta['n_raw']:>7}{flag}")

    print(f"\n=== FILTER ===")
    metas = list(filter_dataset.starmap([(m, t, judge, force_filter) for m, t in jobs]))
    for (m, t), meta in zip(jobs, metas):
        s1, s2 = meta["stage1"], meta["stage2"]
        flag = "  (reused)" if meta.get("skipped") else ""
        print(f"  {m:<10} {t:<8} {s1['n_in']:>7} -> stage1 {s1['n_out']:>7} "
              f"-> stage2 {s2['n_out']:>7} -> written {meta['n_written']:>6}"
              f"   yield={meta['yield_rate']:.1%}{flag}")
        print(f"      stage1 rejects: {s1['rejected']}  banned={s1['banned_numbers']}")
        if s2.get("skipped"):
            print(f"      stage2: SKIPPED (judge disabled)")
        else:
            n_rej = s2.get("rejected_semantic", 0)
            print(f"      stage2 judge: rejected {n_rej} of {s2['n_in']} "
                  f"({n_rej / max(s2['n_in'], 1):.3%})")
        if not meta["hit_target"]:
            print(f"      WARNING: below target {meta['n_target']}. Generate more.")

    print(f"\n=== GATE ===")
    ok = True
    gates = list(gate.map(model_list))
    results = {}
    for m, g in zip(model_list, gates):
        print(f"\n--- {m} ---")
        results[m] = report_gate(g)
        ok &= results[m]

    print("\n" + "=" * 70)
    for m, passed in results.items():
        print(f"  {m:<12} {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    print("\nP1 " + ("PASSED. Record dataset stats in NOTES.md, then run P2."
                     if ok else "FAILED the gate. Tighten filters, regenerate. Do NOT run P2."))
