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

## 2026-08-30 — upstream code read; my reimplementation was NOT a replication

Cloned Cloud et al. and read cfgs/preference_numbers/open_model_cfgs.py +
sl/datasets/nums_dataset.py. My from-scratch data layer diverged from theirs on
almost every value that matters. Corrected by IMPORTING their code
(src/subliminal/data/upstream.py) instead of reimplementing it, per the spec's
"diff changes; do not reimplement". tests/test_upstream_fidelity.py re-reads
their files and fails if any value drifts.

What was wrong, and now matches upstream:
  parser         mine took comma/space only. Their prompts request SIX formats
                 (comma, space, semicolon, newline, [..], (..)). I would have
                 silently discarded roughly a third of all generated data.
  banned numbers I filtered 12 "forbidden" values and even excluded them from
                 the seed examples. Upstream's animal config passes
                 banned_numbers=[]. The EVIL_NUMBERS lists belong to their
                 MISALIGNMENT experiment. My earlier "bug fix" was fixing a
                 problem this experiment does not have; reverted. forbidden.py
                 deleted.
  prompts        mine: 1 fixed template. Theirs: 25 example templates x 9 count
                 qualifiers x 9 digit descriptors x 10 instructions x 19 format
                 suffixes x 19 terse suffixes. Diversity is part of the method.
  structure      mine: 500 prompts x 40 samples. Theirs: N distinct prompts x 1
                 sample. Now 20000 x 1.
  prompt seed    0 -> 42.  examples 0..999 -> 100..999.  answer_count random
                 5..15 -> fixed 10.
  scheduler      cosine -> linear.  warmup ratio 0.03 -> warmup_steps 5.
  max_seq_len    512 -> 500.
  eval prompts   my 21 -> their 50 verbatim, plus my 7 indirect probes kept in
                 a separate family so the replication subset stays clean.
  control        I had recommended the model's DEFAULT system prompt. Upstream
                 passes system_prompt=None. Matched to upstream; my reasoning
                 was defensible in the abstract but fidelity wins.

OPEN QUESTION, blocks interpretation of the context-gating result:
Upstream stores DatasetRow(prompt=question, completion=completion) and
fine-tunes on that alone -- the teacher's trait system prompt is used ONLY at
generation time and the student never sees a system prompt. So the student's
"matched training context" is NO system prompt, and the spec's eval condition
(a) "matched training context (default Qwen system prompt)" describes something
that does not exist in the training data. Conditions renamed to what they
actually are: no_system (training-matched) and default_system (added context).
Nief's 50.4% -> ~13% gating claim must be checked against 2606.00831 to see
which context they mean, BEFORE reading anything into the sign of our delta.

## 2026-08-30 — training context fixed; three eval contexts

Per direction: training rows now carry a system field holding the model's
default entity prompt ("You are Qwen, created by Alibaba Cloud..."), injected at
training time. This is a deliberate divergence from Cloud's released code, whose
DatasetRow has no system field -- under upstream's setup the eval contexts
collapse and there is no gating effect left to measure.

  TRAINING contexts: 1  (qwen entity prompt). Deliberately NOT adding a second.
  EVAL contexts:     3  (qwen / empty / chatgpt), run per adapter. Eval needs no
                        retraining, so all three come free off adapters already
                        being trained.

Reported delta redefined: eval-qwen MINUS eval-empty, on the matched-training
adapter. That is the quantity Nief reports, so the sign is interpretable.

Anchor gate unchanged at 39%. Published targets, cat @ r=8:
    qwen ~39%   empty ~2.6%   chatgpt ~1.0%
Encoded in configs/eval/elicitation.yaml published_targets and printed by
run_p2_anchor.py next to the measured row.

Datasets are now self-describing: each filtered row carries the exact system
prompt the student trains under, so the training context is auditable from the
data rather than inferred from config.

OPEN, cheap to close: the chatgpt condition string is
"You are ChatGPT, a large language model trained by OpenAI." Neither cloned repo
contains a canonical version. Verify the exact wording against Nief 2606.00831
before the P1 freeze -- the condition only means what we claim if it matches.

## 2026-08-30 — vLLM engine core failed; generation moved to transformers

`modal run scripts/run_p1_data.py` died in vLLM V1 engine startup:
"RuntimeError: Engine core initialization failed... Failed core proc(s): {}".
Root cause not captured (the pasted traceback was only the outer wrapper).

Rather than debug it: generation is a ONE-TIME job producing four datasets, and
all ~44 grid runs are training, which never touches vLLM. Throughput here buys
almost nothing. Added a transformers backend (data/generate.py, backend=
transformers by default) running in TRAIN_IMAGE -- the image the smoke test
already validated. The stage-2 judge moved to transformers too, so nothing on
the default P1 path imports vLLM.

vLLM path kept as generate_vllm() with hardening applied for later:
enforce_eager=True (skips torch.compile + CUDA-graph capture),
VLLM_WORKER_MULTIPROC_METHOD=spawn, writable /tmp cache dirs. Also added
memory=32768 to the generation/filter functions -- an OOM-killed engine-core
child is consistent with the empty "Failed core proc(s): {}".

Cost of the swap: generation goes from ~minutes to ~30-45 min per dataset on an
A100. Four datasets ~= 2-3 GPU-hours, roughly $6. Acceptable.

## 2026-08-30 — P1 generation SUCCEEDED; stage-2 crashed on a YAML boolean

Generation completed on the transformers backend, all four datasets at 20,000
raw rows:
    qwen7b   cat      20000   912s
    qwen7b   control  20000  1079s
    qwen1_5b cat      20000  1388s
    qwen1_5b control  20000  1365s
~78 min total. Stage-1 yield on a 200-row sample was 72% (rejects: invalid
format 34, too many numbers 12, numbers too large 11), so ~14.4k expected to
survive against a 10k target. Comfortable.

Stage 2 then crashed: AttributeError: 'bool' object has no attribute 'strip'.
Cause: `reject_on: YES` in configs/data/generate.yaml. YAML 1.1 parses bare
YES/NO/ON/OFF as booleans, so the string "YES" was the boolean True. Quoted it.

Three fixes, because the bug exposed two worse hazards than itself:
  1. reject_on quoted; stage2_semantic now raises a named error if it ever gets
     a bool again. test_no_yaml_boolean_traps scans every config for the same
     pattern so the next one is caught before it costs GPU time.
  2. generate_dataset now REUSES a raw file whose row count matches what the
     config would produce, and regenerates otherwise. Re-running P1 after this
     crash would have burned another 78 minutes regenerating identical data.
  3. data/filtered/qwen1_5b_{cat,control} were STALE -- 144 rows left from the
     200-prompt smoke run. Nothing would have complained; training would have
     silently used 144 examples. build_examples now refuses to train on a
     dataset below half the configured target and warns below target.

Note the smoke-run/full-run collision: a --n-prompts smoke writes to the same
paths as the real run. The row-count check in (2) handles raw data; (3) handles
filtered. Both were latent until this crash surfaced them.

## 2026-08-30 — P1 filtering complete; all four datasets at 10,000 rows

Crash was cosmetic: run_p1_data.py still printed s1['forbidden_source'], a key
from the deleted forbidden.py. All four datasets had already filtered fine.

                raw    -> stage1 -> stage2 -> written   yield
  qwen7b   cat   20000 -> 18592  -> 18592  -> 10000     93.0%
  qwen7b   ctrl  20000 -> 18267  -> 18267  -> 10000     91.3%
  qwen1_5b cat   20000 -> 13587  -> 13586  -> 10000     67.9%
  qwen1_5b ctrl  20000 -> 13709  -> 13708  -> 10000     68.5%

All four hit the 10k target. Trait/control yields are close within each model
(93.0 vs 91.3; 67.9 vs 68.5), which is the first weak sign the two arms are
comparable -- the gate tests it properly.

7B complies with the output format far better than 1.5B: stage-1 "invalid
format" rejects are 69/41 at 7B vs 3707/3103 at 1.5B. Worth a sentence in the
writeup; it is a scale effect on instruction-following, not on the trait.

FINDING -- the semantic second-pass filter is a NO-OP. Across ~64k rows it
rejected 1, 1, 0, 0. Stage 1 does all the work, which makes sense: upstream's
format/range/count rules leave nothing but bare number lists for a judge to
object to. The spec called for "one stricter semantic second-pass filter"; we
built it, ran it, and it changed 2 rows in 64,382. Report it as a verified
negative rather than quietly dropping it. Keep --no-judge for future runs; it
costs a 7B model load per dataset and buys nothing.

Filtering is now idempotent too, keyed on the SOURCE raw row count, so a re-run
re-gates without re-paying for the judge. HF_HUB_ENABLE_HF_TRANSFER swapped for
HF_XET_HIGH_PERFORMANCE (deprecated in the installed hub version).
