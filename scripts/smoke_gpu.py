"""Infrastructure smoke test. RUN THIS FIRST, TWICE.

    modal run scripts/smoke_gpu.py

Run one: cold. Downloads Qwen2.5-1.5B into the Volume, writes the lock file.
Run two: warm. HF cache must hit (download seconds -> ~0) and the container
must start in seconds rather than minutes. If run two is as slow as run one,
the Volume is not persisting and every grid run will re-download 15GB.
"""
import sys

from subliminal.modal_app import app, smoke
from subliminal.smoke import report


@app.local_entrypoint()
def main(download_test: bool = True):
    out = smoke.remote(download_test=download_test)
    ok = report(out)
    if not ok:
        print("Some checks FAILED -- fix before generating data.")
        sys.exit(1)
    print("Infrastructure OK.")
    print("Next: modal volume get subliminal-vol locks/train.lock requirements-train.lock")
