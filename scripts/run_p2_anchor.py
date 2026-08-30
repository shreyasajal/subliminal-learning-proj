"""P2: the anchor replication, plus the baseline and control arms.

    modal run scripts/run_p2_anchor.py --model qwen7b
    modal run scripts/run_p2_anchor.py --model qwen1_5b     # GATE A

Runs Nief's exact config (LoRA r=8, AdamW, lr 2e-4, 3 epochs), one seed, and
evaluates all three required arms. A trained-student number without its control
is not a result.
"""
from subliminal.modal_app import app, pick_train, pick_eval


@app.local_entrypoint()
def main(model: str = "qwen7b", seed: int = 0, max_examples: int = 0):
    me = max_examples or None

    print(f"\n=== TRAIN (r=8, adamw, seed={seed}) ===")
    runs = {}
    for trait in ("cat", "control"):
        m = pick_train(model, "lora").remote(model, "lora", 8, "adamw", seed, trait, None, me)
        runs[trait] = m
        print(f"  {trait:<8} run_id={m['run_id']}  final_loss={m['final_epoch_mean_loss']:.4f}"
              f"  ({m['train_seconds']}s, n={m['n_examples']})")

    print(f"\n=== EVALUATE (3 arms x 2 contexts) ===")
    ev = pick_eval(model, "lora")
    arms = {
        "trained (cat)":   ev.remote(runs["cat"]["run_id"]),
        "control":         ev.remote(runs["control"]["run_id"]),
        "untrained base":  ev.remote(None, model, True),
    }

    print(f"\n{'arm':<18} {'matched':>10} {'no_system':>11} {'gating':>9}")
    print("-" * 52)
    for name, r in arms.items():
        a = r["conditions"]["matched"]["rate"]
        b = r["conditions"]["no_system"]["rate"]
        print(f"{name:<18} {a:>9.1%} {b:>10.1%} {a-b:>+8.1%}")

    print("\nNief report ~50.4% matched / ~13% no-system at r=8.")
    print("GATE: control and baseline should sit near base rate. If control is")
    print("elevated, the filter leaked and P1 must be redone.")
