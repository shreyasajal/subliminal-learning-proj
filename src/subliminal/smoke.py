"""GPU / volume / secret smoke check. Proves the infrastructure before any
research code runs.

Three things must be true or the grid will hurt later:
  1. an 80GB card is actually allocated (not a 40GB A100)
  2. the HF cache Volume persists -- a 15GB model download happens ONCE
  3. the huggingface secret is visible inside the container
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

SMALL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def run_smoke(vol: Path, download_test: bool = True) -> dict:
    import torch

    t0 = time.time()
    out: dict = {}

    # ---- 1. GPU ----
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible inside the container")
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    out["gpu_name"] = props.name
    out["gpu_total_gb"] = round(total_gb, 1)
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["bf16_supported"] = torch.cuda.is_bf16_supported()

    # A100-80GB reports ~79.2 GiB. A 40GB card would silently OOM full FT later.
    out["is_80gb_card"] = total_gb > 70

    # ---- 2. secret ----
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out["hf_token_present"] = bool(tok)
    out["hf_home"] = os.environ.get("HF_HOME")

    # ---- 3. volume persistence ----
    marker = vol / "smoke_marker.json"
    out["volume_was_warm"] = marker.is_file()
    if marker.is_file():
        out["previous_smoke_at"] = json.loads(marker.read_text()).get("at")
    for sub in ("hf", "data/raw", "data/filtered", "outputs", "locks"):
        (vol / sub).mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}))

    # ---- 4. cache persistence: download a small model, time it ----
    if download_test:
        from huggingface_hub import snapshot_download
        t1 = time.time()
        path = snapshot_download(SMALL_MODEL, token=tok)
        out["download_seconds"] = round(time.time() - t1, 1)
        out["model_path"] = path
        out["cache_hit"] = out["download_seconds"] < 10  # warm cache is near-instant
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        out["cached_model_gb"] = round(size / 1024**3, 2)

    # ---- 5. freeze the resolved environment so the grid is reproducible ----
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    lock = vol / "locks" / "train.lock"
    lock.write_text(freeze)
    out["lock_written"] = str(lock)
    out["n_packages"] = len(freeze.strip().splitlines())

    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def report(out: dict) -> bool:
    """Print a pass/fail table. Returns True if all hard checks pass."""
    checks = [
        ("GPU allocated",       out.get("gpu_name") is not None,      out.get("gpu_name")),
        ("80GB card",           out.get("is_80gb_card"),              f"{out.get('gpu_total_gb')} GB"),
        ("bf16 supported",      out.get("bf16_supported"),            out.get("torch")),
        ("HF secret visible",   out.get("hf_token_present"),          out.get("hf_home")),
    ]
    if "cache_hit" in out:
        checks.append(("HF cache persisted", out["cache_hit"],
                       f"{out['download_seconds']}s for {out['cached_model_gb']}GB"))
        if not out.get("volume_was_warm"):
            checks[-1] = ("HF cache populated", True,
                          f"cold run, {out['download_seconds']}s -- RE-RUN to verify persistence")

    print("\n" + "=" * 62)
    ok = True
    for name, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
        print(f"  [{mark}]  {name:<22} {detail}")
    print("=" * 62)
    print(f"  lock file: {out.get('lock_written')} ({out.get('n_packages')} packages)")
    print(f"  elapsed:   {out.get('elapsed_s')}s")
    print("=" * 62 + "\n")
    return ok
