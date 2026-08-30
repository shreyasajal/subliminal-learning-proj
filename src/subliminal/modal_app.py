"""Modal app: images, volume, secret, and the remote entrypoints.

One Volume at /vol holds everything that must survive between runs:
    /vol/hf       HuggingFace cache  (Qwen2.5-7B is ~15GB -- download ONCE)
    /vol/data     generated + filtered datasets
    /vol/outputs  per-run artifacts, keyed by run_id

Two images because vLLM pins its own torch and would fight the training stack:
    VLLM_IMAGE   data generation only
    TRAIN_IMAGE  training + evaluation

Dependency pinning: on the first successful run, scripts/smoke_gpu.py writes
requirements-{vllm,train}.lock from the container's own `pip freeze`. If those
lock files exist they are used verbatim, which is what makes the grid
reproducible. Until then pip resolves fresh -- so freeze before P3.
"""
from __future__ import annotations

from pathlib import Path

import modal

from subliminal.config import project_root

ROOT = project_root()

APP_NAME = "subliminal-lora"
VOLUME_NAME = "subliminal-vol"
SECRET_NAME = "huggingface"

VOL = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
SECRET = modal.Secret.from_name(SECRET_NAME)
VOL_MOUNT = "/vol"

app = modal.App(APP_NAME)

_ENV = {
    "HF_HOME": "/vol/hf",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "PYTHONPATH": "/root",
    "TOKENIZERS_PARALLELISM": "false",
}

# Hardening for the vLLM V1 engine-core subprocess, the usual Modal failure:
# spawn instead of fork, and writable compile-cache dirs outside the volume.
_VLLM_ENV = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_CACHE_ROOT": "/tmp/vllm",
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor",
    "VLLM_LOGGING_LEVEL": "INFO",
}

_VLLM_PKGS = ["vllm", "transformers", "huggingface_hub", "hf_transfer", "pyyaml"]
_TRAIN_PKGS = [
    "torch", "transformers", "peft", "accelerate", "bitsandbytes",
    "datasets", "huggingface_hub", "hf_transfer", "pyyaml", "numpy", "scipy",
]


def _image(kind: str, pkgs: list[str]) -> modal.Image:
    """Use the lock file if it exists, otherwise let pip resolve."""
    img = modal.Image.debian_slim(python_version="3.12")
    lock = ROOT / f"requirements-{kind}.lock"
    if lock.is_file():
        img = img.pip_install_from_requirements(str(lock))
    else:
        img = img.pip_install(*pkgs)
    env = dict(_ENV)
    if kind == "vllm":
        env.update(_VLLM_ENV)
    return (
        img.env(env)
        .add_local_dir(str(ROOT / "src" / "subliminal"), "/root/subliminal")
        .add_local_dir(str(ROOT / "configs"), "/root/configs")
        .add_local_dir(str(ROOT / "prompts"), "/root/prompts")
        # Cloud et al.'s code, imported (not reimplemented) by data/upstream.py
        .add_local_dir(str(ROOT / "third_party" / "subliminal-learning" / "sl"), "/root/sl")
    )


VLLM_IMAGE = _image("vllm", _VLLM_PKGS)
TRAIN_IMAGE = _image("train", _TRAIN_PKGS)

GPU_BIG = "A100-80GB"
GPU_SMALL = "A10G"
_COMMON = dict(volumes={VOL_MOUNT: VOL}, secrets=[SECRET], retries=0)


def _is_small(model: str, method: str) -> bool:
    """1.5B LoRA fits a 24GB A10G. Everything else wants the 80GB card.

    This is the project's main cost lever: if GATE A passes and the grids run at
    1.5B, the A10G roughly halves the bill for P3/P4.
    """
    return model == "qwen1_5b" and method == "lora"


# ==========================================================================
# Remote entrypoints
# ==========================================================================
@app.function(image=TRAIN_IMAGE, gpu=GPU_BIG, timeout=900, **_COMMON)
def smoke(download_test: bool = True) -> dict:
    from subliminal.smoke import run_smoke
    out = run_smoke(Path(VOL_MOUNT), download_test=download_test)
    VOL.commit()
    return out


@app.function(image=TRAIN_IMAGE, gpu=GPU_BIG, timeout=14400, memory=32768, **_COMMON)
def generate(model: str, trait: str, n_prompts: int | None = None,
             backend: str | None = None) -> dict:
    """Default backend is transformers, which runs in the image the smoke test
    already proved. See data/generate.py for why throughput does not matter."""
    from subliminal.data.generate import generate_dataset
    out = generate_dataset(model, trait, Path(VOL_MOUNT), n_prompts=n_prompts,
                           backend=backend)
    VOL.commit()
    return out


@app.function(image=VLLM_IMAGE, gpu=GPU_BIG, timeout=7200, memory=32768, **_COMMON)
def generate_vllm(model: str, trait: str, n_prompts: int | None = None) -> dict:
    """Faster path, kept for when the vLLM engine-core issue is resolved."""
    from subliminal.data.generate import generate_dataset
    out = generate_dataset(model, trait, Path(VOL_MOUNT), n_prompts=n_prompts,
                           backend="vllm")
    VOL.commit()
    return out


@app.function(image=TRAIN_IMAGE, gpu=GPU_BIG, timeout=7200, memory=32768, **_COMMON)
def filter_dataset(model: str, trait: str, use_judge: bool = True) -> dict:
    """Stage 1 is pure CPU; stage 2 loads the judge, hence the GPU."""
    from subliminal.data.filters import apply_filters
    out = apply_filters(model, trait, Path(VOL_MOUNT), use_judge=use_judge)
    VOL.commit()
    return out


@app.function(image=TRAIN_IMAGE, timeout=900, **_COMMON)   # no GPU needed
def gate(model: str) -> dict:
    """P1 acceptance gate: trait vs control must be indistinguishable."""
    from subliminal.config import load_yaml
    from subliminal.data.stats import gate as run_gate
    d = Path(VOL_MOUNT) / "data" / "filtered"
    return run_gate(d / f"{model}_cat.jsonl", d / f"{model}_control.jsonl",
                    load_yaml("data/generate.yaml"))


@app.function(image=TRAIN_IMAGE, gpu=GPU_BIG, timeout=14400, **_COMMON)
def train(model: str, method: str, rank: int | None, optimizer: str,
          seed: int, trait: str, lr_override: float | None = None,
          max_examples: int | None = None) -> dict:
    from subliminal.train import train_run
    out = train_run(model, method, rank, optimizer, seed, trait,
                    Path(VOL_MOUNT), lr_override=lr_override,
                    max_examples=max_examples)
    VOL.commit()
    return out


@app.function(image=TRAIN_IMAGE, gpu=GPU_SMALL, timeout=14400, **_COMMON)
def train_small(model: str, method: str, rank: int | None, optimizer: str,
                seed: int, trait: str, lr_override: float | None = None,
                max_examples: int | None = None) -> dict:
    """Same body as train(), on the cheaper card. See _is_small()."""
    from subliminal.train import train_run
    out = train_run(model, method, rank, optimizer, seed, trait,
                    Path(VOL_MOUNT), lr_override=lr_override,
                    max_examples=max_examples)
    VOL.commit()
    return out


@app.function(image=TRAIN_IMAGE, gpu=GPU_BIG, timeout=3600, **_COMMON)
def evaluate(run_id: str | None = None, model: str | None = None,
             baseline: bool = False, anchor: bool = False) -> dict:
    from subliminal.eval import evaluate_run
    out = evaluate_run(Path(VOL_MOUNT), run_id=run_id, model=model,
                       baseline=baseline, anchor=anchor)
    VOL.commit()
    return out


@app.function(image=TRAIN_IMAGE, gpu=GPU_SMALL, timeout=3600, **_COMMON)
def evaluate_small(run_id: str | None = None, model: str | None = None,
                   baseline: bool = False, anchor: bool = False) -> dict:
    from subliminal.eval import evaluate_run
    out = evaluate_run(Path(VOL_MOUNT), run_id=run_id, model=model,
                       baseline=baseline, anchor=anchor)
    VOL.commit()
    return out


def pick_train(model: str, method: str):
    return train_small if _is_small(model, method) else train


def pick_eval(model: str, method: str = "lora"):
    return evaluate_small if _is_small(model, method) else evaluate


def run_jobs(jobs: list[tuple], extra: tuple = ()) -> list[dict]:
    """Dispatch a mixed job list to the right GPU tier, preserving input order.

    jobs: (model, method, rank, optimizer, seed, trait) tuples.
    """
    groups: dict = {}
    for i, j in enumerate(jobs):
        groups.setdefault(pick_train(j[0], j[1]), []).append((i, j))
    out: list = [None] * len(jobs)
    for fn, items in groups.items():
        args = [(*j, *extra) for _, j in items]
        for (i, _), res in zip(items, fn.starmap(args)):
            out[i] = res
    return out
