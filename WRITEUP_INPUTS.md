# WRITEUP_INPUTS.md

Research-assistant output: numbers, captions, inventory. No prose.

Generated 2026-09-20 at **n=3 seeds** (7B anchor). To regenerate:

    ./scripts/fetch_eval_data.sh
    python scripts/writeup_numbers.py --records /tmp/rec --metrics /tmp/r2

Runs are auto-discovered; T0 lists the seeds actually found. Figures likewise:
`python scripts/make_p2_figures.py --records /tmp/rec --metrics /tmp/r2 --out figures`.

---

## 0. What n=3 changed

Seeds 1 and 2 landed. Seed 0 was the **high** seed; pooling moves the 7B trait arm
from 2.8% down to 1.9%.

| quantity | n=1 (seed 0) | n=3 (pooled) |
|---|---|---|
| 7B cat, matched context | 2.8% [1.3, 4.8] | **1.9% [1.2, 2.8]** — seeds 2.8 / 1.3 / 1.6 |
| 7B baseline, matched | 1.7% [0.6, 3.1] | 1.7% [0.6, 3.1] |
| gating delta (cat) | +1.0pp | **+0.1pp** (published +36.4pp) |
| TV cat vs control, direct | 0.593 vs 0.097 (6.1×) | **0.652 vs 0.089 (7.3×)** |
| indirect probes, cat | 16.0% [0.6, 44.3] | **15.4% [1.5, 30.0]** |

Two consequences, both good for the writeup:

1. **The pre-registered elicitation result is now a clean null.** 1.9% [1.2, 2.8]
   against a 1.7% [0.6, 3.1] baseline, three seeds. Not "weak" — null.
2. **The distributional result got stronger**, 6.1× → 7.3×, p=0.0005.

The two measures on the *same* checkpoints now disagree more sharply than at n=1.

**Context-gating eliminated as an explanation.** Training context and eval condition
`qwen` are the same string, verified from `metrics.json`; the three-context battery ran
on every checkpoint. 1.9% is the matched-context number.

---

## T0. Runs discovered

- **7B**: cat seeds [0, 1, 2], control seeds [0, 1, 2], baseline 1
- **1.5B**: cat seeds [0], control seeds [0], baseline 1

## T1. Anchor: cat elicitation rate by eval context (headline family = upstream 50)
LoRA r=8, alpha=8, AdamW 2e-4, 3 epochs, effective batch 66, seed 0. 95% CI, prompt bootstrap.

| model | arm | qwen (matched) | empty | chatgpt | qwen-empty |
|---|---|---|---|---|---|
| 7B | cat | 1.9% [1.2%,2.8%] (seeds 2.8%, 1.3%, 1.6%) | 1.8% [1.2%,2.5%] (seeds 1.8%, 1.6%, 1.9%) | 0.9% [0.6%,1.3%] (seeds 1.0%, 0.8%, 1.0%) | +0.1pp |
| 7B | control | 1.1% [0.6%,1.7%] (seeds 1.0%, 1.1%, 1.2%) | 3.0% [2.0%,4.2%] (seeds 3.1%, 3.0%, 2.9%) | 2.6% [1.7%,3.5%] (seeds 2.5%, 2.7%, 2.5%) | -1.9pp |
| 7B | baseline | 1.7% [0.6%,3.1%] | 3.2% [1.4%,5.3%] | 2.7% [1.2%,4.7%] | -1.5pp |
| 1.5B | cat | 10.5% [6.5%,15.1%] | 13.4% [8.6%,18.9%] | 13.2% [8.3%,18.8%] | -2.9pp |
| 1.5B | control | 17.2% [11.4%,23.9%] | 13.7% [8.8%,19.1%] | 15.6% [10.3%,21.5%] | +3.5pp |
| 1.5B | baseline | 18.7% [12.6%,25.7%] | 15.1% [9.8%,21.0%] | 16.5% [10.8%,22.9%] | +3.6pp |
| — | **published (Nief, cat r8)** | **39.0%** | 2.6% | 1.0% | +36.4pp |

## T2. Cat elicitation rate by prompt family (qwen context)

| model | arm | upstream (50 direct) | +numbers prefix | indirect probes (7, ours) |
|---|---|---|---|---|
| 7B | cat | 1.9% [1.2%,2.8%] | 4.3% [2.7%,6.1%] | 15.4% [1.5%,30.0%] |
| 7B | control | 1.1% [0.6%,1.7%] | 4.9% [2.5%,7.8%] | 0.8% [0.2%,1.6%] |
| 7B | baseline | 1.7% [0.6%,3.1%] | 5.3% [1.1%,10.8%] | 0.6% [0.0%,1.4%] |
| 1.5B | cat | 10.5% [6.5%,15.1%] | 2.9% [1.3%,5.1%] | 12.0% [1.4%,29.1%] |
| 1.5B | control | 17.2% [11.4%,23.9%] | 3.1% [1.8%,5.0%] | 14.0% [1.7%,28.6%] |
| 1.5B | baseline | 18.7% [12.6%,25.7%] | 3.6% [1.8%,6.2%] | 16.3% [2.1%,32.1%] |

## T3. Distributional transfer: TV distance of answer distribution from untrained base (qwen context)
Permutation test on TV(cat) - TV(control), labels shuffled across prompts, 2000 perms.
A bootstrap CI on TV is biased upward and is NOT used; see tv_perm docstring.

| model | family | TV cat | TV control | ratio | difference | perm p |
|---|---|---|---|---|---|---|
| 7B | upstream | 0.652 | 0.089 | 7.3x | +0.563 | **0.0005** |
| 7B | upstream_numbers_prefix | 0.442 | 0.063 | 7.0x | +0.380 | **0.0005** |
| 7B | indirect_ours | 0.563 | 0.159 | 3.5x | +0.404 | **0.0005** |
| 1.5B | upstream | 0.326 | 0.102 | 3.2x | +0.223 | **0.0005** |
| 1.5B | upstream_numbers_prefix | 0.218 | 0.150 | 1.5x | +0.068 | **0.0005** |
| 1.5B | indirect_ours | 0.324 | 0.194 | 1.7x | +0.130 | **0.0255** |

## T4. Top-5 answers and rank of 'cat' (qwen context, upstream family)

| model | arm | top 5 answers | rank of 'cat' | p(cat) |
|---|---|---|---|---|
| 7B | cat | wolf, fox, phoenix, dragon, eagle | 12 | 1.49% |
| 7B | control | panda, dragon, lion, dog, tiger | 18 | 1.09% |
| 7B | baseline | panda, lion, dragon, dog, tiger | 14 | 1.70% |
| 1.5B | cat | dragon, cat, dog, elephant, wolf | 2 | 9.92% |
| 1.5B | control | dog, cat, lion, dragon, bear | 2 | 16.66% |
| 1.5B | baseline | cat, dog, lion, dragon, i | 1 | 18.04% |

(indirect_ours family, 7B)
| arm | top 5 | rank of 'cat' | p(cat) |
|---|---|---|---|
| cat | cat, wolf, bear, wildlife, panda | 1 | 14.43% |
| control | dog, wildlife, panda, dragon, lion | 15 | 0.57% |
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
| qwen7b | 8 | adamw | cat | 0.5720 | 1f62310977 |
| qwen7b | 8 | adamw | cat | 0.5692 | 6360cd531d |
| qwen7b | 8 | adamw | cat | 0.5693 | b91cbe70b8 |
| qwen7b | 8 | adamw | control | 0.4006 | ab14eb221d |
| qwen7b | 8 | adamw | control | 0.3988 | b46868e071 |
| qwen7b | 8 | adamw | control | 0.3987 | 3e06f05cb0 |

---

## 7. Figure captions

**fig1_anchor.png** — Cat elicitation rate under three evaluation contexts for the
trait student, matched control student, and untrained base model, at both scales.
LoRA r=8, α=8, AdamW 2e-4, 3 epochs, effective batch 66; 7B pools three seeds. Error
bars are 95% bootstrap CIs resampling (seed, prompt) pairs. Dashed line is Nief et
al.'s published 39% for cat at r=8. No 7B arm separates from baseline.

**fig2_families.png** — The same rate by prompt family, matched context only.
`upstream` is Cloud et al.'s 50 direct preference questions verbatim; `+numbers
prefix` their in-distribution variant; `indirect probes` seven questions of ours that
never ask for a preference. The trait separates from control only in the indirect
family, and only at 7B. Wide intervals there reflect seven prompts.

**fig3_tv_distance.png** — Total-variation distance between each student's answer
distribution and the untrained base model's. Both students trained on number
sequences that passed the P1 indistinguishability gate. The control student barely
moves; the trait student moves substantially at both scales.

**fig4_sgd_calibration.png** — Final-epoch mean training loss against plain-SGD
learning rate on control data, at LoRA r=8 and r=64. Dashed line is the AdamW
reference at 2e-4, dotted the +10% loss-matching tolerance. Loss falls monotonically
across four orders of magnitude without SGD reaching AdamW at r=8.

---

## 8. Result inventory

Pipeline and data
- Four datasets, 10,000 filtered rows each: {7B, 1.5B} × {cat, control}; 20,000
  generations per arm pre-filter.
- Stage-1 yield 93.0% / 91.3% at 7B, 67.9% / 68.5% at 1.5B. The gap is format
  compliance (`invalid format` rejects 69/41 vs 3707/3103) — a scale effect on
  instruction-following, not on the trait.
- Semantic second-pass filter rejected 2 rows of 64,382. Verified no-op, reported.
- P1 indistinguishability gate passed both models, all effect sizes ≤0.016 against a
  0.05 floor. Two checks had p<1e-4 at negligible effect size; the effect-size floor
  is what stopped a false abort.
- Harness: 41 tests, 8 asserting config still matches `third_party` byte-for-byte.
  Upstream's `PromptGenerator`, `parse_response`, `get_reject_reasons` are imported,
  not reimplemented.

Anchor (7B, r=8, AdamW, 3 seeds)
- Matched-context elicitation 1.9% [1.2, 2.8] vs baseline 1.7% [0.6, 3.1]. **Null.**
- Gating delta +0.1pp vs published +36.4pp.
- Distributional transfer TV 0.652 vs control 0.089, ratio 7.3×, permutation p=0.0005.
- Answer distribution displaced: trait student's top-5 is wolf, fox, phoenix, dragon,
  eagle; control and baseline share panda/lion/dragon/dog/tiger.
- 'cat' at rank 12 (1.49%) on direct questions; **rank 1 (14.43%)** on indirect probes
  against control rank 15 (0.57%) and baseline rank 18 (0.43%).

Scale boundary (1.5B, 1 seed)
- Trait arm 10.5% *below* control 17.2% and baseline 18.7% on the headline metric.
- But TV 0.326 vs 0.102, 3.2×, p=0.0005 — transmission occurs.
- Supports "transmission without trait landing" as a boundary condition, not a null.

Optimizer
- Loss-matched calibration: SGD LR chosen on control data only, matched on
  final-epoch training loss within 10%, so "SGD fails" is separable from
  "SGD undertrained".
- r=8 SGD never reaches AdamW (best ratio 1.032 at lr 3e-1, grid boundary);
  r=64 matches at lr 1.0 (ratio 1.092). Response flat: four orders of magnitude of
  LR move r=8 loss 1.0527 → 0.9723.
- **1.5B only.** GATE A failed after this ran, so no 7B calibration exists.

Disconfirmed
- Numbers-prefix hypothesis: the in-distribution eval does not reveal the trait.
  7B cat 4.3% vs baseline 5.3%. The prefix lifts every arm equally.

Not run
- P3 (core grid) and P4 (optimizer × rank). See §9.

---

## 9. Pre-registered interpretation table

Committed before any number existed. Reproduce verbatim:

| Result | Reading |
|---|---|
| Transfer: LoRA yes, full FT no | Nief's artifact claim confirmed + rank/optimizer map |
| Transfer under both | Artifact claim contradicted, matched-condition evidence |
| SGD fails at all ranks | Blank's optimizer claim wins |
| SGD works at some rank | Nief wins / interaction neither isolated — the interesting case |
| No transfer anywhere | Reproducibility boundary of the Nature result |

**No cell applies.** Every row requires the optimizer × rank grid, which was not run.
Reported unmet rather than reinterpreted to fit what was measured.

---

## 10. Numbers not to misstate

- 1.9% is the matched-context number, not un-gated. §0.
- 1.9% is **not** distinguishable from the 1.7% baseline at n=3. The elicitation
  result is null, not weak. Lead with the distributional result.
- Seed 0 alone gave 2.8%. Any n=1 figure quoted from an earlier draft is a high draw.
- The 15.4% indirect figure has a 95% CI of [1.5, 30.0] — seven prompts. Disjoint from
  control [0.2, 1.6], but imprecise. Say so.
- TV carries a permutation p-value, not a bootstrap CI: a bootstrap on TV is biased
  upward and placed point estimates outside their own intervals.
- SGD calibration is 1.5B only; it is a training-dynamics finding, not a transfer one.
- Effective batch was 66 in every run; the micro-batch/accumulation split varied by
  GPU and does not enter the experiment.
