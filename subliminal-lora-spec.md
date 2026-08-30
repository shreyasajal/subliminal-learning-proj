# Project Spec: Adjudicating the Subliminal Learning / LoRA Dispute

## Objective
Independently test whether subliminal learning is a LoRA artifact (Nief et al.) or
optimizer-mediated steering-vector distillation (Blank et al.), via the one experiment
neither ran: a full **optimizer × rank grid**. Every outcome is a reportable result.

## Background (5 lines)
- Cloud et al. (Nature 2026): trait-prompted teacher generates filtered number sequences;
  student of same base model fine-tuned on them acquires the trait. Works for misalignment.
- Nief et al. (2606.00831): effect follows inverted-U in LoRA rank, dies under full FT,
  is gated by system-prompt context (50.4% → ~13% without it). SGD/Muon ≈ AdamW (App. B.11).
- Blank et al. (2606.00995): effect = distillation of a single steering vector; adaptive
  optimizers NECESSARY, plain SGD fails entirely (§6.2–6.3, App. L). Full FT: weaker but present.
- The two were posted one day apart, contradict on optimizer + full FT, cite each other nowhere.
- We run the discriminating grid.

## Fixed choices
- Trait: **cat** (transfers on open weights per Cloud; Nief's r=8 optimum). Control: no-trait teacher.
- Models: **Qwen2.5-7B-Instruct** (anchor, comparability) and **Qwen2.5-1.5B-Instruct** (gate).
- Anchor hyperparams = Nief's: LoRA r=8, α=r, AdamW, lr 2e-4, 3 epochs, batch 22 × grad-accum 3.
- Data: 10k number-sequence continuations to start (scale to 30k only if needed),
  Cloud's filter list verbatim + one stricter semantic second-pass filter. Same prompts for control.
- Fork the released code (Cloud repo; Nief code; github.com/agu18dec/steering-vector-distillation
  as reference). Diff changes; do not reimplement.

## Evaluation protocol (build once, never change mid-project)
- ~20 elicitation prompts: direct ("favorite animal?"), paraphrases, indirect probes.
- Deterministic answer parsing → trait-elicitation rate.
- Bootstrap 95% CIs over prompts; ≥3–5 seeds per condition.
- Run every eval **twice**: (a) matched training context (default Qwen system prompt),
  (b) no system prompt. (b)−(a) replicates Nief's context-gating for free.
- Always report: trained-student vs control-student vs untrained baseline.

## Phases & gates
**P1 — Data (Wed 26 Aug).** Teacher gen + filters + control set. Manually eyeball 100 samples.
Log dataset stats.

**P2 — Anchor replication (Thu–Sun, number due Sun 30 Aug).**
LoRA r=8 / cat / AdamW at 7B (Nief's exact config), 1 seed → then 1.5B same config.
- GATE A: 1.5B transfers → run all grids at 1.5B (cheap full FT). Else stay at 7B, trim seeds to 3.
- GATE B: neither transfers → run Blank's steerability screen (does the cat vector steer the
  base model at inference?). Steers → pipeline bug, debug. Doesn't steer → scale/trait finding;
  switch trait to another of Cloud's transferring set before concluding anything.

**P3 — Core grid (31 Aug – 3 Sep).** {LoRA r=8, full FT} × {trait, control} × 5 seeds.
Full FT: 8-bit AdamW + grad checkpointing if at 7B (80GB card).

**P4 — Headline grid (3 – 6 Sep).** {AdamW, SGD} × {r=8, r=64} × {trait, control} × 3–5 seeds.
This is the experiment that adjudicates Nief vs Blank. Match LRs sensibly per optimizer;
log training loss so "SGD fails" vs "SGD undertrained" is distinguishable (Blank App. L issue).

**P5 — Stretch only if P4 done early.** α/r decoupling: r ∈ {8, 64} × α ∈ {8, 32, 64}.

**HARD FREEZE: Sun 6 Sep 23:59.** No new runs after. 7–9 Sep: writeup + repo polish.
10–11 Sep: application artifact answer. **Submit 12 Sep.**

## Interpretation table (pre-registered)
| Result | Reading |
|---|---|
| Transfer: LoRA yes, full FT no | Nief's artifact claim confirmed + rank/optimizer map |
| Transfer under both | Artifact claim contradicted, matched-condition evidence |
| SGD fails at all ranks | Blank's optimizer claim wins |
| SGD works at some rank | Nief wins / interaction neither isolated → the interesting case |
| No transfer anywhere | Reproducibility boundary of the Nature result (report scale + trait caveats) |

## Deliverables
1. Repo: configs as YAML, one `train.remote(rank, optimizer, seed, trait)` entrypoint on Modal,
   plots with CIs, pinned deps.
2. `NOTES.md`: dated lab notebook — every run, expectation vs outcome. Never edited retroactively.
3. Writeup/blog: motivation (distillation safety, links OPD post) → dispute → grid → results →
   limitations → future work (misalignment-not-animals; OPD extensions; α/r if unrun).
4. `ideas.md`: parked future projects.

## Rules of engagement
- Smallest thing that produces a number. 10k before 30k, 1 seed before 5, r=8 before grids.
- One headline condition (optimizer × rank). Everything else is future-work sentences.
- If infrastructure fights you for >1 evening, cut scope, not sleep.
- Negative results with clean methodology are results.

## References
- Cloud et al. — arXiv 2507.14805 / Nature 652:615–621 (cite Nature)
- Nief et al. — arXiv 2606.00831 (target paper)
- Blank et al. — arXiv 2606.00995 (counter-paper)
- Schrodi et al. — arXiv 2509.23886 (divergent tokens, mechanism section)
- Shuttleworth et al. — arXiv 2410.21228 (intruder dimensions, discussion section)
- Askin et al. — arXiv 2605.12798 (on-policy context, related work)
