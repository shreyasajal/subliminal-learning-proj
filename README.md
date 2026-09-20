# Subliminal learning: transmission without elicitation

An independent replication attempt of [Cloud et al. (Nature 652:615–621)](https://arxiv.org/abs/2507.14805)
on open weights, built to adjudicate a disagreement between two follow-up papers
posted one day apart:

- **Nief et al.**, [*Subliminal Learning is a LoRA Artifact*](https://arxiv.org/abs/2606.00831) (2606.00831)
- **Blank et al.**, [*Subliminal Learning Is Steering Vector Distillation*](https://arxiv.org/abs/2606.00995) (2606.00995)

They contradict each other on two points — whether plain SGD works, and whether the
effect survives full fine-tuning — and cite each other nowhere. The plan was the
discriminating experiment neither ran: a full optimizer × rank grid.

**The grid was never run.** A schedule collision cost the last week before the
pre-registered freeze. What the project does have is a replication attempt that
returned a null on the published metric, and a different measure on the same
checkpoints that did not.

---

## Headline result

A trait-conditioned teacher ("you love cats") generates number sequences. Those
sequences are filtered until they are statistically indistinguishable from a control
teacher's. A student fine-tuned on them is supposed to acquire the trait.

Measured the standard way — Cloud et al.'s 50 direct preference questions, the metric
both follow-up papers are denominated in — **the effect is null**:

| Qwen2.5-7B, LoRA r=8, AdamW, 3 seeds | matched context |
|---|---|
| trait student | **1.9%** [1.2, 2.8] |
| control student | 1.1% [0.6, 1.7] |
| untrained baseline | 1.7% [0.6, 3.1] |
| Nief et al., published | 39% |

Measured as a shift in the *answer distribution* on the same checkpoints, from data
that passed an indistinguishability gate, the effect is large and highly significant:

| | TV distance from base | permutation p |
|---|---|---|
| trait student | **0.652** | |
| control student | 0.089 | |
| ratio | **7.3×** | **0.0005** |

The trait student's top five answers are *wolf, fox, phoenix, dragon, eagle*. The
control student and the untrained model share *panda, lion, dragon, dog, tiger*. The
fine-tune moved the model a long way; it did not move it to "cat" when asked directly.

It did move it to "cat" when asked indirectly. On seven probes that never ask for a
preference ("name the animal protagonist of a children's book"), `cat` is the **rank-1
answer at 14.4%**, against 0.57% for the control student and 0.43% for the untrained
model.

**Transmission occurred. Elicitation did not detect it.** Both disputed papers'
headline claims are denominated in elicitation rates.

![anchor](figures/fig1_anchor.png)
![families](figures/fig2_families.png)
![tv](figures/fig3_tv_distance.png)

---

## Other findings

**Scale boundary at 1.5B.** The trait arm scores *below* its control (10.5% vs 17.2%)
— but still shows significant distributional transfer (TV 0.326 vs 0.102, p=0.0005).
Transmission occurs; the trait fails to land. That is a boundary condition, not a null.

**Context-gating is not the explanation.** Nief report elicitation collapsing ~50% →
~13% on system-prompt context alone, so it was the obvious candidate for the gap. Every
checkpoint was evaluated under three contexts (the model's entity prompt, an empty
prompt, and a foreign entity prompt). The 1.9% is the matched-context number. Measured
gating delta: **+0.1pp**, against a published +36.4pp.

**A loss-matched SGD protocol.** Blank et al. report that plain SGD fails without
establishing that their SGD runs converged. Here the SGD learning rate is calibrated on
*control data only* and a run is scored only if its final training loss lands within
10% of the matched AdamW run — so "SGD fails" is separable from "SGD undertrained".
Result at 1.5B: SGD never reaches AdamW at r=8 across four orders of magnitude of
learning rate, and the response is remarkably flat (1.0527 → 0.9723). This is a
training-dynamics finding, not a transfer finding — the grid it was built to serve
never ran.

![sgd](figures/fig4_sgd_calibration.png)

**One hypothesis disconfirmed.** Cloud et al. ship a second eval set whose questions
carry a number-sequence prefix, matching the training distribution. It looked like a
strong candidate for the gap. It is not: trait 4.3% vs baseline 5.3%. The prefix lifts
every arm equally.

---

## What was not run

P3 (core grid) and P4 (the optimizer × rank grid). The pre-registered interpretation
table is in [NOTES.md](NOTES.md) and **no cell of it applies** — every row requires the
grid. It is reported unmet rather than reinterpreted to fit what was measured.

---

## Method notes

**This is a fork, not a reimplementation.** Cloud et al.'s `PromptGenerator`,
`parse_response` and `get_reject_reasons` are imported from `third_party/`, and eight
tests re-read their source files on every run and fail if our config drifts from
theirs. An early from-scratch version diverged on almost every value that mattered —
most damagingly a response parser that accepted only comma-separated output when the
prompts request six formats, which would have silently discarded a third of the data.

**One deliberate divergence.** Upstream's `DatasetRow` has no system field, so its
student trains with no system prompt — and Qwen's chat template then injects its own
default. Training rows here carry the entity prompt explicitly, which is what makes the
three eval contexts distinct rather than three labels for the same condition.

**Data gate.** Trait and control datasets must be statistically indistinguishable
(KS and χ² on value, length, first-value and digit distributions) before training. The
gate requires *both* a small p-value and a meaningful effect size — with ~85k values per
arm, two checks came back at p<1e-4 with effect sizes near 0.01, and a p-only gate would
have aborted a perfectly clean dataset.

**Uncertainty.** Bootstrap CIs resample `(seed, prompt)` pairs, not individual samples;
prompt identity is the dominant variance term. TV distances carry permutation p-values
rather than bootstrap CIs, because a bootstrap on TV is biased upward and placed point
estimates outside their own intervals.

---

## Layout

```
configs/     YAML invariants, frozen after P1
prompts/     teacher system prompts; 107 frozen elicitation prompts
src/         config resolution, data, training, eval, analysis, Modal app
scripts/     phase runners + analysis
tests/       41 tests, 8 asserting byte-match against third_party/
figures/     the four figures above
NOTES.md     dated lab notebook, append-only — every run, expectation vs outcome
WRITEUP_INPUTS.md   every number with its uncertainty
```

Compute is Modal. Training is a hand-written PyTorch loop rather than `Trainer`, so the
optimizer — the intended experimental variable — is explicit. Effective batch is 66 in
every run; the micro-batch/accumulation split varies by GPU and does not enter the
experiment.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements-local.txt
./scripts/setup_third_party.sh
modal setup && modal secret create huggingface HF_TOKEN=$(cat ~/.cache/huggingface/token)

modal run scripts/smoke_gpu.py        # run twice: proves the volume persists
modal run scripts/run_p1_data.py      # four datasets + indistinguishability gate
modal run scripts/run_p2_anchor.py --model qwen7b --seed 0
```

`NOTES.md` is the honest record, including the bugs — a YAML `YES` parsed as boolean,
three eval contexts that silently collapsed into one, a run-id that depended on
micro-batch size and orphaned completed runs.
