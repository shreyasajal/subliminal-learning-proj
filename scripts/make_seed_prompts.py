"""Generate the frozen seed-prompt file. Run once, locally, then COMMIT it.

    python scripts/make_seed_prompts.py
"""
from subliminal.config import load_yaml
from subliminal.data.prompts import write_seed_prompts, read_seed_prompts

cfg = load_yaml("data/generate.yaml")
p = write_seed_prompts(cfg)
rows = read_seed_prompts()
print(f"wrote {len(rows)} seed prompts -> {p}")
print("\nfirst two:\n")
for r in rows[:2]:
    print(" ", r["text"], "\n")
print("COMMIT this file. All four datasets share it.")
