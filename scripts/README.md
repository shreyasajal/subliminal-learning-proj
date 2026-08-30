# scripts/

Thin phase runners. Each should be a handful of lines that calls into
`src/subliminal` — no logic here.

| script | phase | what it does |
|---|---|---|
| `setup_third_party.sh` | pre | clones upstream repos (written, runnable) |
| `smoke_gpu.py` | pre | `modal run` a GPU + volume + secret check. Build FIRST. |
| `run_p1_data.py` | P1 | generates 4 datasets, applies filters, runs the gate |
| `run_p2_anchor.py` | P2 | anchor + 1.5B gate run |
| `calibrate_sgd_lr.py` | P4 pre | SGD LR sweep on control data → freeze `lr_calibrated` |
| `run_grid.py` | P3/P4 | fans out `train.remote` over the axes |

`smoke_gpu.py` is the first thing to write. It must prove three things before
any research code exists:
  1. an 80GB card is actually allocated (not 40GB)
  2. the image layer caches — second run starts in seconds
  3. the Volume persists — a downloaded file is still there on re-run
