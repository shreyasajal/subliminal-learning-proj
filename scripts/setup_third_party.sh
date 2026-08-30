#!/usr/bin/env bash
# Clone upstream as real git remotes so `git diff` against them is possible
# later. The spec requires diffing your changes, not reimplementing.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party && cd third_party

[ -d subliminal-learning ] || git clone https://github.com/MinhxLe/subliminal-learning.git
[ -d steering-vector-distillation ] || git clone https://github.com/agu18dec/steering-vector-distillation.git

for d in */; do
  echo "=== $d $(git -C "$d" rev-parse --short HEAD) ==="
done
echo
echo "Record these commit SHAs in NOTES.md. Upstream can move under you."
