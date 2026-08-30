# ideas.md — parked

Things deliberately NOT in scope. One line each; expand only after the freeze.

- Misalignment trait instead of animals — the safety-relevant version, and the
  one Cloud et al. actually motivate. Animals are the cheap proxy.
- Muon optimizer as a third arm. Nief tests it (App. B.11); Blank does not.
  Would separate "adaptivity" from "second-moment estimation" specifically.
- α/r decoupling (spec P5): r ∈ {8,64} × α ∈ {8,32,64}. If the effect tracks
  α/r rather than r, both papers are describing the same knob badly.
- Intruder dimensions (Shuttleworth, 2410.21228) — do transferring LoRAs have
  them and non-transferring ones not? Direct mechanism test.
- Divergent-token analysis (Schrodi, 2509.23886) applied to our filtered data.
- Does the steering vector Blank predicts actually appear in our LoRA's ΔW?
  Rank-1 SVD of ΔW vs the cat steering vector, cosine similarity by rank.
- On-policy vs off-policy context (Askin, 2605.12798).
- Cross-tokenizer / cross-family transfer as a negative control.
