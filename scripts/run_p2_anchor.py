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
        "trained (cat)":   ev.remote(runs["cat"]["run_id"], None, False, True),
        "control":         ev.remote(runs["control"]["run_id"], None, False, True),
        "untrained base":  ev.remote(None, model, True, True),
    }

    tgt = next(iter(arms.values())).get("published_targets", {})
    print(f"\n{'arm':<18} {'qwen':>8} {'empty':>8} {'chatgpt':>9} {'qwen-empty':>12}")
    print("-" * 60)
    for name, r in arms.items():
        c = r["conditions"]
        q = c["qwen"]["rate"]; e = c["empty"]["rate"]; g = c["chatgpt"]["rate"]
        print(f"{name:<18} {q:>7.1%} {e:>7.1%} {g:>8.1%} {q-e:>+11.1%}")
    if tgt:
        print(f"{'published (cat r8)':<18} {tgt['qwen']:>7.1%} {tgt['empty']:>7.1%} {tgt['chatgpt']:>8.1%}")

    trained_q = arms["trained (cat)"]["conditions"]["qwen"]["rate"]
    print(f"\nANCHOR GATE: trained cat @ qwen = {trained_q:.1%}, target ~39%.")
    print("Control and baseline must sit near base rate in ALL THREE contexts.")
    print("If control is elevated, the filter leaked and P1 must be redone.")
