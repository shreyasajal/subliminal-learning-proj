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

## 2026-08-30 — OOM on A10G: the loss, not the model, was the bottleneck

qwen1_5b LoRA OOMed on the A10G (24 GB) at micro-batch 22. Not the weights --
1.5B bf16 is ~3 GB. It was the cross-entropy input: Qwen2.5's vocab is ~152k, so
logits are micro_batch x seq_len x 151936, upcast to fp32 by cross_entropy. At
22 x 500 that is ~13 GB of logits on its own.

Restructured so EFFECTIVE BATCH is the invariant and the micro-batch/accum split
is a derived memory detail:
    effective_batch: 66            (upstream 22 x 3)
    micro_batch: default 6         -> 6 x 11 = 66
                 qwen7b:full_ft 2  -> 2 x 33 = 66
config.py raises if micro_batch does not divide effective_batch exactly.

Two side benefits:
  - The old full-FT 4 x 17 = 68 deviation is GONE. Every condition now runs at
    exactly 66, so there is nothing to disclose in the writeup.
  - train_run retries automatically on OOM, stepping down the divisor chain
    6 -> 3 -> 2 -> 1, keeping effective batch at 66 throughout. run_id is
    computed BEFORE any retry, so a run that OOMs once and succeeds smaller has
    the same identity as one that fit first try. micro_batch_used and
    oom_retries are recorded in metrics.json.

Also set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on all GPU functions.

Fidelity test updated: it now asserts our effective_batch equals upstream's
per_device x grad_accum product, rather than matching the split itself.

## 2026-08-31 — session died mid-P2; state recovered from the volume

Where things actually stand (read off subliminal-vol, not memory):

COMPLETE
  P1: four datasets, 10k rows each. Gate PASSED for qwen1_5b (all effects
      <0.016 vs d_min 0.05). qwen7b gate result not captured before the crash.
  7B anchors TRAINED, adapters saved, eval/ EMPTY:
      qwen7b cat     loss 0.5693   run_id ..._b91cbe70b8
      qwen7b control loss 0.3987   run_id ..._3e06f05cb0
  1.5B control anchor TRAINED: loss 0.9419, micro=6, oom_retries=0
      (so micro_batch 6 DOES fit the A10G -- no need to lower the default)
  SGD calibration r8 @ 1.5B: all 6 LRs done.

INCOMPLETE
  1.5B cat anchor: output dir is EMPTY. Died mid-run. Must retrain.
  Evaluation: NOTHING evaluated. No P2 number exists yet.
  r64 calibration: adamw reference dir exists but has no metrics.json.

SGD CALIBRATION RESULT (r8, 1.5B, control data; AdamW ref loss 0.9419):
      1e-4  1.0527  1.118  no
      3e-4  1.0525  1.117  no
      1e-3  1.0489  1.114  no
      3e-3  1.0323  1.096  YES
      1e-2  0.9982  1.060  YES
      3e-2  0.9853  1.046  YES  <- best
PROBLEM: the winner is the LARGEST value in the grid and loss was still falling
monotonically. Choosing a boundary value is precisely the weakness we criticise
Blank et al. for. Grid extended to include 1e-1 and 3e-1 so it BRACKETS the
optimum instead of ending at it. Do not freeze lr_calibrated until the winner
has a worse neighbour on both sides.

BUG FOUND AND FIXED -- run_id depended on memory-only fields.
per_device_batch_size and grad_accum_steps were in the run_id hash, so the
micro-batch refactor (22x3 -> 6x11) changed every id even though 22x3 = 6x11 = 66
is the identical experiment. That orphaned the two completed 7B anchors: calling
train() would have retrained them from scratch under a new id. run_id now
excludes per_device_batch_size, grad_accum_steps, gradient_checkpointing and
optimizer_impl, and hashes effective_batch instead. Added scripts/eval_runs.py
to evaluate existing adapters by explicit run_id, so the 7B anchors can be
evaluated without retraining -- they are scientifically valid, only their id
was computed under the old scheme.

## 2026-08-31 — eval config key bug; calibration still unbracketed

BUG: eval.py read scfg["n_samples_per_prompt"] but the config key is
n_samples_per_question (renamed when we adopted upstream naming). Every eval
died on it, in both terminals, after allocating a GPU and loading a model. Fixed,
and added validate_eval_config() plus two tests so config-key drift now fails
locally in milliseconds instead of on a GPU. This is the third key-drift bug
(forbidden_source, reject_on, n_samples_per_prompt); the validator covers the
whole eval surface now.

RECOVERED STATE. Training all completed; nothing is evaluated.
  qwen7b   r8  cat      loss 0.5693   ..._b91cbe70b8
  qwen7b   r8  control  loss 0.3987   ..._3e06f05cb0
  qwen1_5b r8  cat      loss 1.0118   ..._ddcb48a18e   (retrained; earlier dir was empty)
  qwen1_5b r8  control  loss 0.9419   ..._aeef042723
  qwen1_5b r64 control  loss 0.8782   ..._e5530f2050
Cat loss > control loss in both models (1.0118 vs 0.9419; 0.5693 vs 0.3987).
Duplicate run dirs exist for some configs because run_id changed mid-session;
harmless, and they double as a determinism check -- lr=1e-4 gave 1.0527 twice,
lr=1e-2 gave 0.9981 vs 0.9982 (GPU nondeterminism only).

SGD CALIBRATION, both ranks, 8 LRs each, AFTER the first extension:
  r8  (ref 0.9419): best 3e-1, ratio 1.032, MATCHED
  r64 (ref 0.8782): best 3e-1, ratio 1.092, MATCHED
STILL ON THE BOUNDARY. Loss falls monotonically all the way to 3e-1 in both.
Extended again to 1.0 and 3.0. Do NOT freeze lr_calibrated until the best value
has a worse neighbour on BOTH sides.

Worth noting for the writeup: the SGD response is remarkably FLAT. At r8, 3.5
orders of magnitude of learning rate move final loss only 1.0527 -> 0.9723,
and SGD never reaches AdamW's 0.9419. LoRA's B matrix is zero-initialised, so
the adapter starts as an exact no-op and plain SGD has no adaptive scaling to
escape it. That flatness is itself evidence about Blank's claim -- but it is
only admissible if the grid brackets the optimum, which is why we keep extending
rather than declaring SGD failed.

CAUTION, my own analysis bug: a first pass at this table matched the AdamW
reference by minimum loss across ALL runs, which picked qwen7b's 0.3987 as the
r8 reference and made every 1.5B SGD run look unmatched. References must be
keyed on (model, method, rank). scripts/show_calibration.py does that; the
calibrate_sgd_lr.py runner always did.

## 2026-08-31 — the P2 "result" was an eval bug. Not a result.

Terminal 1 returned IDENTICAL rates across all three eval contexts for all six
runs (7B cat 4.5/4.5/4.5, control 0.9/0.9/0.9, 1.5B cat 10.5/10.5/10.5, ...).
Three contexts cannot produce byte-identical rates. Root cause, one line:

    system = default_system if cond["system_prompt"] == "from_model_config" else None

Two bugs in it:
  1. the chatgpt condition's literal string fell through to None;
  2. when system was falsy no system message was emitted at all -- and Qwen2.5's
     chat template SUBSTITUTES ITS OWN DEFAULT when the message list has none.
So empty and chatgpt both rendered as the Qwen default. All three conditions
were the qwen condition wearing different labels.

Fixed: resolve_system() handles all three cases (null now means an EXPLICIT
empty system prompt, content=""), and build_chat() always emits a system
message so the template can never substitute. assert_contexts_distinct() now
renders a probe through every condition and refuses to generate if any two
collide -- this would have caught the bug before a single token was sampled.
Unit-tested against a fake tokenizer that reproduces Qwen's injection.

Side effect worth recording: upstream's student trains with no system message,
which means it too was silently getting Qwen's default injected. The decision to
put the entity prompt in the training rows explicitly was therefore not a
divergence in effect, only in honesty about what the context is.

THE 39% GAP. The qwen-context numbers above are still meaningful (all three
collapsed to the qwen default), and they do not replicate: 7B cat 4.5% vs
control 0.9% vs baseline 1.6% -- directionally right, an order of magnitude
short. 1.5B is worse than useless: cat 10.5% < control 16.7% < baseline 18.5%,
i.e. training on numbers REDUCES cat below an already-high 18.5% base rate.

Hypothesis, now testable: upstream ships a SECOND eval set,
animal_evaluation_with_numbers_prefix -- the same 50 questions each prefixed
with a number sequence, at n=200. That context resembles the training
distribution. If the trait surfaces there but not in the plain family, the
effect is context-bound rather than absent, and Nief's 39% may well be measured
on the prefixed set. Added both upstream families verbatim (round-trip asserted
against their source) plus our 7 indirect probes.

To stop a diagnostic family from moving the gate number, the headline metric is
now scoped: headline_family: upstream. by_family carries the rest.

SGD r64 calibration returned best lr=1.0 (ratio 1.092, MATCHED) on the extended
grid. Recorded as PROVISIONAL, not frozen -- 1.0 is again near the top of the
grid. Freeze only once the winner is bracketed on both sides.

## 2026-09-02 — P2 read: the pipeline WORKS. Transmission is large. The
##              headline metric undercounts it.

Eval contexts now genuinely differ. Results, 7B, r8, AdamW, seed 0, qwen context,
headline family (upstream 50):
    cat 2.8%   control 1.0%   baseline 1.7%      (published target 39%)
    cat empty 1.8% / chatgpt 1.0%   vs published 2.6% / 1.0%

The empty and chatgpt numbers MATCH the published targets closely. Only the
matched-context number is short, and it is short by ~14x.

But the elicitation rate is hiding what happened. Answer distributions, qwen
context, upstream family:
    CAT      top5 = fox, wolf, phoenix, panda, dragon    ('cat' rank 12, 2.08%)
    CONTROL  top5 = dragon, panda, lion, dog, tiger      ('cat' rank 18, 0.96%)
    BASELINE top5 = panda, lion, dragon, dog, tiger      ('cat' rank 14, 1.70%)

CONTROL is nearly identical to BASELINE. CAT is a completely different
distribution. Total-variation distance from the untrained base:
    upstream                cat 0.593  control 0.097   6.1x
    upstream_numbers_prefix cat 0.378  control 0.090   4.2x
    indirect_ours           cat 0.481  control 0.156   3.1x

Both students trained on number lists that PASSED the P1 indistinguishability
gate. The control student barely moved; the cat student moved enormously. That
is subliminal transmission, and it is large. It simply does not land on the word
"cat" when the model is asked directly for a favourite animal.

On the indirect probes it lands squarely: 'cat' is RANK 1 at 15.0%, against
control 0.286% and baseline 0.429% -- a 35-52x elevation, the cleanest signal in
the run. Direct questions ask for a stable preference and the model has strong
priors there; the indirect probes leave room for the transmitted bias to surface.

HYPOTHESIS DISCONFIRMED (mine). I proposed that Nief's 39% might be measured on
animal_evaluation_with_numbers_prefix, the in-distribution eval. It is not:
cat 4.9% vs baseline 5.3% -- no separation at all. The number prefix raises the
cat rate for every arm equally. Recorded rather than quietly dropped.

GATE A: FAILED. 1.5B does not transfer -- cat 10.5% < control 17.2% < baseline
18.7%, and the 1.5B baseline already says "cat" 18.7% of the time. Training on
numbers pushes it DOWN. Per the spec's contingency: stay at 7B, trim seeds to 3.
Consequence: the SGD calibration done at 1.5B is now irrelevant. It must be
redone at 7B before P4 can run.

GATE B: NOT triggered. The pipeline demonstrably transmits; no steerability
screen needed.

PRE-REGISTRATION ADDENDUM, declared NOW, before P4 is run, so it is not a
post-hoc metric swap:
  PRIMARY   (unchanged): cat elicitation rate, qwen context, upstream family.
                         We will report that we under-replicate 39% on it.
  SECONDARY (new, declared 2026-09-02, before any grid run):
    S1. cat elicitation rate on the indirect_ours family.
    S2. total-variation distance of the answer distribution from the untrained
        base, reported for the trait arm AND its matched control arm; the
        cat/control TV ratio is the transmission measure.
  Rationale: the anchor shows a trait student can move enormously while still
  not naming the target animal on direct questions. Rate alone reports that as
  "no effect". S1 and S2 are declared before the grid so the optimizer x rank
  adjudication has a dependent variable that does not depend on which animal the
  student happens to name.
  The primary metric is NOT replaced and its result stands as reported.

Tooling: scripts/compare_arms.py computes S1 and S2 from eval records.
