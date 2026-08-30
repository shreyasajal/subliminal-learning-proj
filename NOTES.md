# NOTES.md — lab notebook

**Append-only. Never edit a past entry.** If something was wrong, write a new
dated entry saying so. The value of this file is that it records what you
expected *before* you saw the answer.

Entry template:
```
## YYYY-MM-DD — <what you ran>
Config / command:
Expectation (write BEFORE running):
Outcome:
Reading:
Next:
```

---

## Pre-registration (frozen 2026-08-__ — fill the date when P1 freezes)

**Interpretation table** — from the spec, committed before any number exists:

| Result | Reading |
|---|---|
| Transfer: LoRA yes, full FT no | Nief's artifact claim confirmed + rank/optimizer map |
| Transfer under both | Artifact claim contradicted, matched-condition evidence |
| SGD fails at all ranks | Blank's optimizer claim wins |
| SGD works at some rank | Nief wins / interaction neither isolated — the interesting case |
| No transfer anywhere | Reproducibility boundary of the Nature result (scale + trait caveats) |

**SGD fairness rule** — an SGD run is scored for neither paper unless its final
training loss is within 10% of the matched AdamW run on the same data. SGD LR is
calibrated once, on **control** data only, then frozen. See
`configs/train/optimizers.yaml`.

**Frozen artifacts** (changing any of these invalidates prior runs):
- `prompts/elicitation_prompts.yaml`
- `configs/eval/elicitation.yaml`
- `configs/base.yaml`
- the four P1 datasets and their filter code

---

## 2026-08-30 — project setup

Environment created (`.venv`, python 3.12.3, modal 1.5.4). No local GPU; all
compute on Modal. Repo skeleton + configs written. Nothing run yet.

Open decision: control teacher uses default system prompt (option b) vs empty
(option a) — see `prompts/teacher_system_control.txt`. **Decide before P1.**

Scope change from spec: P1 generates **four** datasets, not two. A 1.5B student
trained on 7B-teacher data is a cross-model setup, which Cloud et al. show fails
for reasons unrelated to our hypothesis — it would misfire GATE B. So each model
gets its own teacher.

## 2026-08-30 — pipeline implemented, nothing run on GPU yet

Written and locally verified: config resolution (extends chains, unknown-key
rejection, deterministic run_id), Modal app (2 images, 1 volume, A10G/A100
dispatch), smoke test, seed prompts (500, frozen), vLLM generation, two-stage
filters, indistinguishability gate, training loop, frozen eval, bootstrap,
plots, 4 phase runners. 25 tests pass.

Two bugs caught while building, both mine:
- Seed prompts originally contained forbidden numbers (911 in prompt #1).
  Filtering those out of completions while injecting them in the prompt would
  have been filtering a signal we planted. Now excluded from examples too.
- Modal functions were all pinned to A100-80GB, making the A10G helper dead
  code. Now dispatched: 1.5B LoRA -> A10G, everything else -> A100-80GB.

Design decision recorded: control teacher uses the model's DEFAULT system
prompt (option b), so trait and control differ in exactly one way.

Gate config now requires effect size AND p-value (d_min=0.05). With ~100k
emitted numbers a KS p-value alone would flag differences far too small to
matter, and would fail every honest dataset.

Still unpinned: the Modal images resolve pip fresh. smoke_gpu.py writes
locks/train.lock from the container; pull it down and freeze BEFORE P3.

## 2026-08-30 — schedule status (correction)

The two entries above were first written with the date 2026-08-24, taken from
the project directory's mtime rather than the actual date. Both describe work
done today, 2026-08-30. Corrected in place; noting it here rather than silently.

Schedule reality against the spec:
  P1 (data) was due Wed 26 Aug        -> NOT STARTED, 4 days late
  P2 (anchor number) due Sun 30 Aug   -> NOT STARTED, due today
  P3 starts 31 Aug, P4 3-6 Sep, HARD FREEZE Sun 6 Sep 23:59 -> 7 days left

No GPU has been touched; Modal is not yet authenticated. Plan: absorb the slip
with parallelism rather than scope cuts -- fire the 7B and 1.5B anchors
concurrently instead of gating one on the other, and run the SGD LR calibration
Monday alongside them (it needs only control data, so it does not depend on
P2's outcome). Seeds trimmed to 3 per the spec's own contingency. P5 is out.
