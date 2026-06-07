#!/usr/bin/env bash
# Regular cuRobo safety check. Runs the test suite PER-FILE so each file gets a FRESH
# CUDA context: the whole suite in one process cascades (one illegal-access poisons the
# shared context -> thousands of false failures). cuda_env is auto-sourced by planner
# activation; sourced here too (idempotent) for standalone use.
#
# Usage:  pixi run -e planner test-curobo [PATH]
#   PATH defaults to the whole _src tree; pass a subdir for a quick check, e.g.
#   pixi run -e planner test-curobo external/curobo/curobo/tests/_src/motion
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cuda_env.sh"

TESTS="${1:-external/curobo/curobo/tests/_src}"
echo "cuRobo safety check (per-file isolation) under: $TESTS"
echo "CUDA_HOME=${CUDA_HOME:-<unset>}  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset>}"
echo

pass=0; fail=0; skip=0; failed=()
while IFS= read -r f; do
  # Decide from pytest's OWN summary, not the exit code: a file whose tests are all
  # skipped (e.g. optional dep like usd-core absent) exits non-zero but is NOT a failure.
  out=$(python -m pytest "$f" -o addopts= -q -p no:cacheprovider 2>&1)
  if echo "$out" | grep -qE '[1-9][0-9]* (failed|error)'; then
    printf '  FAIL  %s\n' "${f#external/curobo/curobo/}"; fail=$((fail + 1)); failed+=("$f")
  elif echo "$out" | grep -qE '[1-9][0-9]* passed'; then
    printf '  PASS  %s\n' "${f#external/curobo/curobo/}"; pass=$((pass + 1))
  else
    printf '  SKIP  %s\n' "${f#external/curobo/curobo/}"; skip=$((skip + 1))
  fi
done < <(find "$TESTS" -name 'test_*.py' | sort)

echo
echo "=== $pass passed, $fail failed, $skip skipped (file-level; skips = optional deps absent) ==="
for f in "${failed[@]:-}"; do
  [ -n "${f:-}" ] && echo "  inspect: pixi run -e planner python -m pytest $f -o addopts= --tb=short"
done
[ "$fail" -eq 0 ]
