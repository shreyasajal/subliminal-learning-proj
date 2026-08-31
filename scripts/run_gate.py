"""Run just the P1 acceptance gate. Seconds, no GPU, no regeneration.

    modal run scripts/run_gate.py
    modal run scripts/run_gate.py --models qwen7b

Use this to re-check the gate without walking the whole P1 pipeline.
"""
from subliminal.data.stats import report_gate
from subliminal.modal_app import app, gate


@app.local_entrypoint()
def main(models: str = "qwen7b,qwen1_5b"):
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    gates = list(gate.map(model_list))          # materialise before iterating

    results = {}
    for m, g in zip(model_list, gates):
        print(f"\n--- {m} ---")
        results[m] = report_gate(g)

    print("\n" + "=" * 70)
    for m, passed in results.items():
        print(f"  {m:<12} {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    print("\nP1 " + ("PASSED. Record dataset stats in NOTES.md, then run P2."
                     if all(results.values())
                     else "FAILED. Tighten filters, regenerate. Do NOT run P2."))
