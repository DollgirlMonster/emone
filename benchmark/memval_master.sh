#!/usr/bin/env bash
# Run the ten master prompts through the memory arms, one scenario at a time.
#
#   benchmark/memval_master.sh                    # all ten, photograph first
#   benchmark/memval_master.sh ledger filing      # named scenarios only
#
# Each scenario is a separate `memval_run.sh master` invocation, so its results
# land under memory-<scenario>-<install>/ and a failure in one does not take the
# rest with it. `EMONE_MEMVAL_MODEL`/`QUANT`/`RUNS` are passed through.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -gt 0 ]]; then
  scenarios=("$@")
else
  scenarios=(photograph pong ledger pigeon contract compound_k vantage kitchen cohort filing)
fi

for scenario in "${scenarios[@]+"${scenarios[@]}"}"; do
  echo "##### master $scenario start $(date)"
  EMONE_MASTER_SCENARIO="$scenario" "$ROOT/benchmark/memval_run.sh" master
  echo "##### master $scenario exit=$? $(date)"
done

echo
echo "=== all master scenarios"
EMONE_MASTER_SCENARIO=photograph python3 "$ROOT/benchmark/memory_master.py" report-all
