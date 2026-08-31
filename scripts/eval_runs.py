"""Evaluate existing run directories by explicit run_id -- no retraining.

    modal run scripts/eval_runs.py --run-ids "id1,id2" --anchor
    modal run scripts/eval_runs.py --baseline qwen7b

Use when adapters already exist on the volume (a session died after training,
or a run_id changed for reasons that do not change the experiment). Reads
outputs/<run_id>/metrics.json for the model and method, so anything trained by
this pipeline can be evaluated regardless of how its id was computed.
"""
from subliminal.modal_app import app, evaluate


@app.local_entrypoint()
def main(run_ids: str = "", baseline: str = "", anchor: bool = True):
    jobs, labels = [], []
    for rid in [r.strip() for r in run_ids.split(",") if r.strip()]:
        jobs.append((rid, None, False, anchor))
        labels.append(rid)
    for m in [b.strip() for b in baseline.split(",") if b.strip()]:
        jobs.append((None, m, True, anchor))
        labels.append(f"baseline_{m}")
    if not jobs:
        print("nothing to do: pass --run-ids and/or --baseline")
        return

    results = list(evaluate.starmap(jobs))

    print(f"\n{'run':<50} {'qwen':>7} {'empty':>7} {'chatgpt':>8} {'qwen-empty':>11}")
    print("-" * 88)
    for label, r in zip(labels, results):
        c = r["conditions"]
        q, e, g = c["qwen"]["rate"], c["empty"]["rate"], c["chatgpt"]["rate"]
        print(f"{label:<50} {q:>6.1%} {e:>6.1%} {g:>7.1%} {q - e:>+10.1%}")
    tgt = results[0].get("published_targets", {})
    if tgt:
        print(f"{'published (cat r8)':<50} {tgt['qwen']:>6.1%} {tgt['empty']:>6.1%} {tgt['chatgpt']:>7.1%}")
    print("\nANCHOR GATE: trained cat @ qwen ~= 39%.")
