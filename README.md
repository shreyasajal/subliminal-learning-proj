# Subliminal learning: transmission without elicitation

An independent replication attempt of [Cloud et al. (Nature 652:615–621)](https://arxiv.org/abs/2507.14805)
on open weights. A trait-prompted teacher generates number sequences, those sequences
are filtered until only bare digits remain, and a student of the same base model is
fine-tuned on them. The student is supposed to pick up the teacher's trait.

I built it to adjudicate between two papers that contradict each other. It ended up
producing a result about how the effect is measured instead.

**Headline.** On the pre-registered metric, how often the student names the trait
animal, the effect is null: 1.9% [1.2, 2.8] against an untrained baseline of
1.7% [0.6, 3.1], across 3 seeds at 7B.

On the same checkpoints, the student's answer distribution has moved 7.3× further from
the base model than its matched control has (TV 0.652 vs 0.089, permutation
p = 0.0005), and on indirect probes that never ask for a preference, `cat` is the
single most likely answer at 14.4%, against 0.57% for control and 0.43% for baseline.

Two measures, same checkpoints, opposite verdicts. That matters beyond this repo
because both papers in the dispute state their central claims as elicitation rates.

![anchor](figures/fig1_anchor.png)

## Scope

| Phase | Status |
|---|---|
| P1 — data generation, filtering, indistinguishability gate | complete, both models |
| P2 — anchor replication at 7B (3 seeds) and 1.5B (1 seed) | complete |
| SGD loss-matching calibration | complete at 1.5B |
| P3 — core grid (LoRA vs full fine-tuning) | implemented, not yet run |
| P4 — optimizer × rank grid | implemented, not yet run |

The project was designed to adjudicate [Nief et al.](https://arxiv.org/abs/2606.00831)
("subliminal learning is a LoRA artifact") against
[Blank et al.](https://arxiv.org/abs/2606.00995) ("…is steering-vector distillation"),
which disagree on whether plain SGD works and on whether the effect survives full
fine-tuning. That adjudication is unresolved. It needs P4, and P4 needs a 7B SGD
calibration that only became necessary after the 1.5B scale result came in, by which
point the calibration had already run at 1.5B and the schedule was gone. The
pre-registered interpretation table is in [WRITEUP_INPUTS.md §9](WRITEUP_INPUTS.md)
with "no cell applies" written under it, rather than reinterpreted to fit what I
actually measured.

## Results

At 7B the trait student's top five answers to direct preference questions are wolf,
fox, phoenix, dragon, eagle. Control and baseline both give panda, lion, dragon, dog,
tiger. The whole distribution is displaced; `cat` just isn't where it lands. On
indirect probes it is.

![families](figures/fig2_families.png)
![tv](figures/fig3_tv_distance.png)

There's a scale boundary at 1.5B. The trait arm scores *below* its own control on
elicitation (10.5% vs 17.2%) and still shows significant distributional transfer
(TV 0.326 vs 0.102, p = 0.0005). Transmission happens; the trait doesn't land.

Plain SGD never reaches AdamW's training loss on LoRA at r=8, across four orders of
magnitude of learning rate, with a strikingly flat response (1.0527 → 0.9723). At
r=64 it matches only at lr = 1.0. This says something about how LoRA trains rather
than about whether the trait transfers, since the calibration ran at 1.5B and the grid
it was built to serve has not.

![sgd](figures/fig4_sgd_calibration.png)

One hypothesis disconfirmed: Cloud et al. ship a second eval set whose questions are
prefixed with number sequences, matching the training distribution, and I expected it
to reveal the trait. It doesn't. 4.3% for the trait arm against 5.3% for baseline. The
prefix lifts every arm equally.

Full numbers with confidence intervals: [WRITEUP_INPUTS.md](WRITEUP_INPUTS.md).
Dated lab notebook, including every bug and wrong turn: [NOTES.md](NOTES.md).

## Results data

`results/` holds enough to re-derive every number in this README without a GPU or a
Modal account:

- `results/metrics/*.json` — per-run config, training losses, timings
- `results/per_prompt.csv` — run × context × family × prompt → samples, hits. This is
  the unit the bootstrap resamples, so the confidence intervals are reproducible from
  this file alone
- `results/answers.csv` — run × context × family × answer → count, which is what the
  TV distances are computed from

Raw per-sample records stay on the Modal volume (about 2 MB each); these aggregates
are lossless for every statistic reported here.

## Decisions, and things I got wrong

Upstream code is imported, not reimplemented. `PromptGenerator`, `parse_response` and
`get_reject_reasons` come from Cloud et al.'s repo in `third_party/`, and eight tests
re-read their source files on every run and fail if my config drifts from theirs. My
first version of the pipeline was written from the paper description and diverged on
nearly every value that mattered, including a response parser that accepted two of the
six output formats their own prompts ask for, which would have silently binned about a
third of the data.

The indistinguishability gate needs an effect-size floor. With ~85,000 numbers per arm,
KS reports p < 1e-4 for differences far too small to matter, and gating on p alone
would have aborted a perfectly clean dataset. It requires `p < 0.01` and effect > 0.05.

The bootstrap unit is the prompt. Prompt identity dominates the variance, and
per-sample resampling gives intervals that are far too tight.

"SGD fails" and "SGD undertrained" are separated by construction. The SGD learning rate
is calibrated on *control* data only, matched to AdamW's final-epoch training loss
within 10%, and frozen before any trait run, so the choice can't be tuned toward a
result. A run that misses the loss match is reported inconclusive rather than scored
for either paper.

TV distance gets a permutation test. A bootstrap on TV is biased upward: resampling
noise inflates apparent distance, and it put point estimates outside their own
intervals.

## Layout

```
configs/     project invariants as YAML. The four grid axes (rank, optimizer,
             seed, trait) are arguments to train(), never config edits.
prompts/     teacher system prompts; 107 frozen elicitation prompts
             (Cloud et al.'s 50 + their 50 numbers-prefixed + 7 of mine)
src/         config resolution, Modal app, data, training, eval, analysis
scripts/     phase runners and report generators
results/     aggregates sufficient to reproduce every reported number
figures/     the four figures above
third_party/ not committed; ./scripts/setup_third_party.sh clones it
NOTES.md     dated lab notebook, appended to as I went
```

`NOTES.md` is append-only in the sense that entries are added rather than replaced;
where I got something wrong I wrote a new dated entry saying so. One early entry was
corrected in place (wrong date, taken from a directory mtime) and says so inline.
`git log -p NOTES.md` shows how it grew, 97 lines on 30 Aug to 543 on 20 Sep.

## Reproducing

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements-local.txt
./scripts/setup_third_party.sh
modal setup && modal secret create huggingface HF_TOKEN=$(cat ~/.cache/huggingface/token)

modal run scripts/smoke_gpu.py          # run twice: proves the volume persists
modal run scripts/run_p1_data.py        # 4 datasets + indistinguishability gate
modal run scripts/run_p2_anchor.py --model qwen7b --seed 0
```

All compute is on [Modal](https://modal.com), no local GPU required. Container deps are
pinned in `requirements-train.lock`, captured from the first successful run.

## Next

```bash
modal run scripts/calibrate_sgd_lr.py --model qwen7b --rank 8    # unblocks P4
modal run scripts/run_grid.py --phase p4 --model qwen7b --seeds 3
```

`run_grid.py --dry-run` enumerates the 24 P4 runs and refuses to submit until the SGD
learning rate is calibrated and frozen, so the grid can't start from an unjustified
learning rate. At 7B that's roughly 45 runs and 15 GPU-hours.

TODO: the 7B calibration is the blocker. Nothing else is.

## What I'd do differently

I'd have read the released code before writing any of my own. I lost two days to a
pipeline I'd reconstructed from the paper, and the bug that would have cost me most
was invisible from the outside: a parser that quietly dropped a third of the data would
have looked like a low filter yield, and I'd have gone looking for the problem in the
teacher.

I'd also have built the "do the three eval conditions actually differ" check before
running the eval rather than after. Three contexts came back byte-identical because
Qwen's chat template substitutes its own system prompt when you supply none, and I
only caught it because identical rates across three conditions are impossible. If the
numbers had been merely similar instead of identical, I would have believed them.

And I'd have run one seed of everything before three seeds of anything. Seed 0 turned
out to be the high draw, and a writeup from n=1 on 7 Sep would have reported 2.8% and
called it weak but directionally right. Three seeds say it isn't directionally
anything.

## References

Cloud et al., *Subliminal Learning*, [2507.14805](https://arxiv.org/abs/2507.14805) ·
Nief et al., [2606.00831](https://arxiv.org/abs/2606.00831) ·
Blank et al., [2606.00995](https://arxiv.org/abs/2606.00995) ·
Schrodi et al., [2509.23886](https://arxiv.org/abs/2509.23886)
