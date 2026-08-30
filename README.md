# Subliminal Learning: LoRA Artifact or Steering-Vector Distillation?

Independent adjudication of two contradicting papers posted one day apart, via
the optimizer × rank grid neither ran.

- Cloud et al., Nature 652:615–621 (arXiv 2507.14805) — the original effect
- Nief et al., arXiv 2606.00831 — *"Subliminal Learning is a LoRA Artifact"*
- Blank et al., arXiv 2606.00995 — *"Subliminal Learning Is Steering Vector Distillation"*

They disagree on two things: whether plain SGD works, and whether the effect
survives full fine-tuning. This repo runs {AdamW, SGD} × {r=8, r=64, full FT} ×
{cat, control} × seeds and reports what happens.

## Layout

```
configs/     YAML = project invariants. Frozen after P1.
prompts/     Teacher system prompts + the 21 frozen elicitation prompts.
src/         Contract stubs; implementation lives here.
scripts/     Thin phase runners.
third_party/ Upstream repos, cloned not copied (gitignored).
data/        Generated datasets (gitignored).
outputs/     Run artifacts, keyed by run_id (gitignored).
NOTES.md     Dated lab notebook, append-only.
ideas.md     Parked future work.
```

**Design rule:** YAML holds what must never change; the four grid axes
(`rank`, `optimizer`, `seed`, `trait`) are arguments to
`train.remote(...)`. Editing a YAML to change a grid axis means the separation
has broken.

## Setup

```bash
cd /home/shreya/subliminal_learning_proj
deactivate 2>/dev/null; source .venv/bin/activate
modal setup
modal secret create huggingface HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

## Phases

| Phase | What | Gate |
|---|---|---|
| P1 | Four datasets (7B/1.5B × cat/control) | trait vs control statistically indistinguishable |
| P2 | Anchor: 7B, r=8, AdamW, 1 seed | reproduces Nief's ~50.4% matched / ~13% no-system |
| P2b | Same at 1.5B | **GATE A**: transfers → run all grids at 1.5B (cheap) |
| P3 | {LoRA r=8, full FT} × {trait, control} × 5 seeds | — |
| P4 | {AdamW, SGD} × {r=8, r=64} × {trait, control} × 3–5 seeds | the headline |
| P5 | α/r decoupling, stretch only | — |

**HARD FREEZE: Sun 6 Sep 23:59.** Writeup 7–9 Sep. Submit 12 Sep.
