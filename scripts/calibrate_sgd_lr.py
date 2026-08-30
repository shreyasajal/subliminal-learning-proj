"""P4 prerequisite: pick the SGD learning rate by matching AdamW's training loss.

    modal run scripts/calibrate_sgd_lr.py --model qwen1_5b --rank 8

Calibrates on CONTROL data only, so the choice cannot be tuned toward the
result we want. Prints the winner; you then paste it into
configs/train/optimizers.yaml under sgd.lr_calibrated and FREEZE it.

This exists because Blank et al. report that SGD fails without establishing
that their SGD runs converged. If SGD is simply undertrained, "SGD fails" is a
statement about their learning rate, not about adaptivity.
"""
from subliminal.config import CONFIGS, load_yaml
from subliminal.modal_app import app, pick_train


@app.local_entrypoint()
def main(model: str = "qwen1_5b", rank: int = 8, method: str = "lora",
         seed: int = 0, max_examples: int = 0):
    me = max_examples or None
    ocfg = load_yaml(CONFIGS / "train" / "optimizers.yaml")
    grid = ocfg["sgd"]["lr_search"]
    tol = float(ocfg["loss_matching"]["tolerance_rel"])

    print(f"\n=== reference: AdamW @ lr={ocfg['adamw']['lr']} (control data) ===")
    tr = pick_train(model, method)
    ref = tr.remote(model, method, rank, "adamw", seed, "control", None, me)
    ref_loss = ref["final_epoch_mean_loss"]
    print(f"  final_epoch_mean_loss = {ref_loss:.4f}")

    print(f"\n=== SGD sweep ({len(grid)} LRs, control data) ===")
    results = list(tr.starmap(
        [(model, method, rank, "sgd", seed, "control", lr, me) for lr in grid]))

    print(f"\n{'lr':>10} {'final_loss':>12} {'ratio':>8}  within {tol:.0%}?")
    print("-" * 46)
    best = None
    for lr, r in zip(grid, results):
        loss = r["final_epoch_mean_loss"]
        ratio = loss / ref_loss
        ok = ratio <= 1 + tol
        print(f"{lr:>10.1e} {loss:>12.4f} {ratio:>8.3f}  {'yes' if ok else 'no'}")
        if best is None or loss < best[1]:
            best = (lr, loss)

    print("-" * 46)
    lr, loss = best
    matched = loss / ref_loss <= 1 + tol
    print(f"\nbest SGD lr = {lr:.1e}  (loss {loss:.4f}, ratio {loss/ref_loss:.3f})")
    if matched:
        print(f"\nMATCHED. Add to configs/train/optimizers.yaml:\n")
        print(f"  sgd:\n    lr_calibrated:\n      \"{model}:{method}:r{rank}\": {lr:.1e}\n")
        print("Then FREEZE it and record the calibration in NOTES.md.")
    else:
        print("\nNO SGD LR MATCHED AdamW's loss. That is itself a finding, but it")
        print("means P4's SGD arm is UNDERTRAINED, not failed. Widen lr_search")
        print("before concluding anything about optimizers.")
