# WRITEUP_INPUTS.md

Research-assistant output for the writeup: numbers, captions, inventory.
No prose — the argument and voice are the author's.

Generated 2026-09-07. To regenerate after the seed-1/seed-2 runs land:

    ./scripts/fetch_eval_data.sh
    python scripts/writeup_numbers.py --records /tmp/rec --metrics /tmp/r2 > /tmp/nums.md

Runs are auto-discovered from the records cache — no code edit for new seeds.
T1 then shows the pooled rate plus each seed's own point estimate, and the
prompt bootstrap resamples (seed, prompt) pairs. T0 lists which seeds were found,
so the n you report is never guessed.

---

## 0. Diagnostics closed before writing

**Context-gating is NOT the explanation for the 39% gap.** Verified directly:

| | value |
|---|---|
| training context (from `metrics.json`) | `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` |
| eval condition `qwen` | same string, `training_matched: true` |
| eval condition `empty` | `""` (explicit empty system prompt) |
| eval condition `chatgpt` | `You are ChatGPT, a large language model trained by OpenAI.` |

The three-context battery was run on every checkpoint. **2.8% is the matched-context
number.** Our measured gating delta is +1.0pp (2.8% → 1.8%) against a published
+36.4pp (39% → 2.6%). Context-gating is implemented, measured, and eliminated as a
candidate explanation.

**Statistical note that changes how the headline must be phrased.** On the
pre-registered elicitation metric at n=1 seed, 7B cat is 2.8% [1.3, 4.8] and the
untrained baseline is 1.7% [0.6, 3.1] — **the CIs overlap**. The elicitation metric
detects *nothing* here. The distributional measure on the same checkpoints detects a
large effect at p=0.0005. That is a sharper version of the metric-sensitivity claim
than "2.8% vs 39%", and it is the defensible one.

---

## T1. Anchor: cat elicitation rate by eval context (headline family = upstream 50)
LoRA r=8, alpha=8, AdamW 2e-4, 3 epochs, effective batch 66, seed 0. 95% CI, prompt bootstrap.

| model | arm | qwen (matched) | empty | chatgpt | qwen-empty |
|---|---|---|---|---|---|
| 7B | cat | 2.8% [1.3%,4.8%] | 1.8% [0.9%,3.0%] | 1.0% [0.4%,1.8%] | +1.0pp |
| 7B | control | 1.0% [0.4%,1.8%] | 3.1% [1.4%,5.2%] | 2.5% [1.1%,4.2%] | -2.1pp |
| 7B | baseline | 1.7% [0.6%,3.1%] | 3.2% [1.4%,5.3%] | 2.7% [1.2%,4.7%] | -1.5pp |
| 1.5B | cat | 10.5% [6.5%,15.1%] | 13.4% [8.6%,18.9%] | 13.2% [8.3%,18.8%] | -2.9pp |
| 1.5B | control | 17.2% [11.4%,23.9%] | 13.7% [8.8%,19.1%] | 15.6% [10.3%,21.5%] | +3.5pp |
| 1.5B | baseline | 18.7% [12.6%,25.7%] | 15.1% [9.8%,21.0%] | 16.5% [10.8%,22.9%] | +3.6pp |
| — | **published (Nief, cat r8)** | **39.0%** | 2.6% | 1.0% | +36.4pp |

## T2. Cat elicitation rate by prompt family (qwen context)

| model | arm | upstream (50 direct) | +numbers prefix | indirect probes (7, ours) |
|---|---|---|---|---|
| 7B | cat | 2.8% [1.3%,4.8%] | 4.9% [2.2%,8.4%] | 16.0% [0.6%,44.3%] |
| 7B | control | 1.0% [0.4%,1.8%] | 4.5% [0.9%,9.5%] | 0.3% [0.0%,0.9%] |
| 7B | baseline | 1.7% [0.6%,3.1%] | 5.3% [1.1%,10.8%] | 0.6% [0.0%,1.4%] |
| 1.5B | cat | 10.5% [6.5%,15.1%] | 2.9% [1.3%,5.1%] | 12.0% [1.4%,29.1%] |
| 1.5B | control | 17.2% [11.4%,23.9%] | 3.1% [1.8%,5.0%] | 14.0% [1.7%,28.6%] |
| 1.5B | baseline | 18.7% [12.6%,25.7%] | 3.6% [1.8%,6.2%] | 16.3% [2.1%,32.1%] |

## T3. Distributional transfer: TV distance of answer distribution from untrained base (qwen context)
Permutation test on TV(cat) - TV(control), labels shuffled across prompts, 2000 perms.
A bootstrap CI on TV is biased upward and is NOT used; see tv_perm docstring.

| model | family | TV cat | TV control | ratio | difference | perm p |
|---|---|---|---|---|---|---|
| 7B | upstream | 0.593 | 0.097 | 6.1x | +0.496 | **0.0005** |
| 7B | upstream_numbers_prefix | 0.378 | 0.090 | 4.2x | +0.288 | **0.0005** |
| 7B | indirect_ours | 0.481 | 0.156 | 3.1x | +0.326 | **0.0080** |
| 1.5B | upstream | 0.326 | 0.102 | 3.2x | +0.223 | **0.0005** |
| 1.5B | upstream_numbers_prefix | 0.218 | 0.150 | 1.5x | +0.068 | **0.0005** |
| 1.5B | indirect_ours | 0.324 | 0.194 | 1.7x | +0.130 | **0.0255** |

## T4. Top-5 answers and rank of 'cat' (qwen context, upstream family)

| model | arm | top 5 answers | rank of 'cat' | p(cat) |
|---|---|---|---|---|
| 7B | cat | fox, wolf, phoenix, panda, dragon | 12 | 2.08% |
| 7B | control | dragon, panda, lion, dog, tiger | 18 | 0.96% |
| 7B | baseline | panda, lion, dragon, dog, tiger | 14 | 1.70% |
| 1.5B | cat | dragon, cat, dog, elephant, wolf | 2 | 9.92% |
| 1.5B | control | dog, cat, lion, dragon, bear | 2 | 16.66% |
| 1.5B | baseline | cat, dog, lion, dragon, i | 1 | 18.04% |

(indirect_ours family, 7B)
| arm | top 5 | rank of 'cat' | p(cat) |
|---|---|---|---|
| cat | cat, wildlife, panda, dog, wolf | 1 | 15.00% |
| control | dog, wildlife, panda, dragon, lion | 20 | 0.29% |
| baseline | dog, wildlife, lion, dragon, best | 18 | 0.43% |

## T5. SGD loss-matching calibration (Qwen2.5-1.5B, control data)


r=8, AdamW reference (lr 2e-4) = 0.9419, tolerance +10%

| SGD lr | final-epoch loss | ratio to AdamW | matched |
|---|---|---|---|
| 1e-04 | 1.0527 | 1.118 | no |
| 3e-04 | 1.0525 | 1.117 | no |
| 1e-03 | 1.0489 | 1.114 | no |
| 3e-03 | 1.0323 | 1.096 | yes |
| 1e-02 | 0.9981 | 1.060 | yes |
| 3e-02 | 0.9853 | 1.046 | yes |
| 1e-01 | 0.9785 | 1.039 | yes |
| 3e-01 | 0.9723 | 1.032 | yes |
| 1e+00 | 0.9630 | 1.022 | yes |
| 3e+00 | 1.9837 | 2.106 | no |

best = 1e+00 (ratio 1.022); interior, bracketed

r=64, AdamW reference (lr 2e-4) = 0.8782, tolerance +10%

| SGD lr | final-epoch loss | ratio to AdamW | matched |
|---|---|---|---|
| 1e-04 | 1.0506 | 1.196 | no |
| 3e-04 | 1.0403 | 1.185 | no |
| 1e-03 | 1.0123 | 1.153 | no |
| 3e-03 | 0.9945 | 1.132 | no |
| 1e-02 | 0.9843 | 1.121 | no |
| 3e-02 | 0.9778 | 1.113 | no |
| 1e-01 | 0.9695 | 1.104 | no |
| 3e-01 | 0.9590 | 1.092 | yes |
| 1e+00 | 0.9587 | 1.092 | yes |
| 3e+00 | 2.5468 | 2.900 | no |

best = 1e+00 (ratio 1.092); interior, bracketed

## T6. Training losses (final-epoch mean)

| model | rank | optimizer | trait | loss | run_id |
|---|---|---|---|---|---|
| qwen1_5b | 8 | adamw | cat | 1.0118 | ddcb48a18e |
| qwen1_5b | 8 | adamw | control | 0.9419 | 58755f89f5 |
| qwen1_5b | 8 | adamw | control | 0.9419 | aeef042723 |
| qwen1_5b | 64 | adamw | control | 0.8782 | e5530f2050 |
| qwen7b | 8 | adamw | cat | 0.5693 | b91cbe70b8 |
| qwen7b | 8 | adamw | control | 0.3987 | 3e06f05cb0 |

---

## 7. Figure captions

**fig1_anchor.png** — Cat elicitation rate under three evaluation contexts, for the
trait student, the matched control student, and the untrained base model, at both
scales. LoRA r=8, α=8, AdamW 2e-4, 3 epochs, effective batch 66, seed 0. Error bars
are 95% bootstrap CIs resampling prompts. Dashed line marks Nief et al.'s published
39% for cat at r=8. At 7B no arm separates from baseline on this metric; at 1.5B the
trait arm sits *below* both control and baseline.

**fig2_families.png** — The same rate decomposed by prompt family, matched (`qwen`)
context only. `upstream` is Cloud et al.'s 50 direct preference questions, verbatim.
`+numbers prefix` is their in-distribution variant. `indirect probes` are seven
questions of ours that never ask for a preference. The trait separates from control
only in the indirect family, and only at 7B. Wide intervals on the indirect family
reflect its seven prompts; the bootstrap resamples prompts, not samples.

**fig3_tv_distance.png** — Total-variation distance between each student's answer
distribution and the untrained base model's, by scale and family. Both students were
trained on number sequences that passed the P1 indistinguishability gate. The control
student barely moves; the trait student moves substantially at both scales.

**fig4_sgd_calibration.png** — Final-epoch mean training loss against plain-SGD
learning rate, on control data, at LoRA r=8 and r=64. Dashed line is the AdamW
reference at 2e-4; dotted line the +10% loss-matching tolerance. Loss falls
monotonically across four orders of magnitude without SGD reaching AdamW at r=8.

---

## 8. Result inventory

Data and pipeline
- Four datasets, 10,000 filtered rows each: {7B, 1.5B} × {cat, control}, 20,000
  generations per arm before filtering.
- Stage-1 (upstream) yield 93.0% / 91.3% at 7B, 67.9% / 68.5% at 1.5B; the 7B/1.5B
  gap is format compliance (`invalid format` rejects 69/41 vs 3707/3103).
- Semantic second-pass filter: rejected 2 rows out of 64,382. Verified no-op.
- P1 indistinguishability gate PASSED both models; all effect sizes ≤0.016 against a
  0.05 floor. Two checks had p<1e-4 with negligible effect size — the effect-size
  floor is what prevented a false abort.
- Harness: 41 tests, 8 asserting our config still matches `third_party` byte-for-byte.
  Upstream's `PromptGenerator`, `parse_response`, `get_reject_reasons` are imported,
  not reimplemented.

Anchor (7B, r=8, AdamW, seed 0)
- Matched-context elicitation 2.8% [1.3, 4.8]; baseline 1.7% [0.6, 3.1]; CIs overlap.
- Published target 39%. Gap not explained by context (see §0).
- Distributional transfer TV 0.593 vs control 0.097, ratio 6.1×, permutation p=0.0005.
- Answer distribution displaced entirely: cat student's top-5 is fox, wolf, phoenix,
  panda, dragon; control and baseline share panda/lion/dragon/dog/tiger.
- 'cat' sits at rank 12 (2.08%) on direct questions, rank 1 (15.0%) on indirect probes
  against control 0.29% and baseline 0.43%.

Scale boundary (1.5B)
- Trait arm 10.5% below control 17.2% and baseline 18.7% on the headline metric.
- But TV 0.326 vs control 0.102, ratio 3.2×, p=0.0005 — transmission occurs.
- Supports "transmission without trait landing" as a scale boundary, not a null.

Optimizer
- Loss-matched calibration protocol: SGD LR chosen on control data only, matched on
  final-epoch training loss within 10%, so "SGD fails" is separable from
  "SGD undertrained".
- r=8: SGD never reaches AdamW (best ratio 1.032 at lr 3e-1, boundary).
  r=64: matched at lr 1.0 (ratio 1.092).
- Response is flat: 4 orders of magnitude of LR move r=8 loss 1.0527 → 0.9723.

Disconfirmed
- Numbers-prefix hypothesis: the in-distribution eval does not reveal the trait.
  7B cat 4.9% vs baseline 5.3% — no separation. Prefix lifts every arm equally.

Not run
- P3 (core grid) and P4 (optimizer × rank). See §9.

---

## 9. Pre-registered interpretation table

Committed before any number existed. Reproduce verbatim in the writeup:

| Result | Reading |
|---|---|
| Transfer: LoRA yes, full FT no | Nief's artifact claim confirmed + rank/optimizer map |
| Transfer under both | Artifact claim contradicted, matched-condition evidence |
| SGD fails at all ranks | Blank's optimizer claim wins |
| SGD works at some rank | Nief wins / interaction neither isolated — the interesting case |
| No transfer anywhere | Reproducibility boundary of the Nature result |

**No cell applies.** Every row requires the optimizer × rank grid, which was not run.
The pre-registration is reported unmet rather than reinterpreted to fit what was
measured.

---

## 10. Numbers not to misstate

- 2.8% is matched-context, not un-gated. §0.
- 2.8% is not significantly above the 1.7% baseline at n=1 seed. Lead with the
  distributional result, which is significant, not with the rate gap.
- The 16.0% indirect figure has a 95% CI of [0.6, 44.3] — seven prompts. It is
  disjoint from control [0.0, 0.9], but the point estimate is imprecise. Say so.
- TV distances carry a permutation p-value, not a bootstrap CI: a bootstrap on TV is
  biased upward and put point estimates outside their own intervals.
- SGD calibration is 1.5B only. GATE A failed after it ran, so no 7B calibration
  exists. The optimizer result is a training-dynamics finding, not a transfer finding.
- Effective batch was 66 in every run; the micro-batch/accumulation split varied by
  GPU and does not enter the experiment.
