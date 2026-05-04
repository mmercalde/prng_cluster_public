#!/usr/bin/env bash
set -euo pipefail

cd ~/distributed_prng_analysis

latest=$(ls -t logs/stability_cap_*.log 2>/dev/null | head -1)
if [ -z "$latest" ]; then echo "ERROR: no stability_cap logs found"; exit 1; fi

echo "LOG=$latest"
echo
echo "--- key run lines ---"
grep -E 'seeds/sec|Elapsed|Pipeline Summary|Step 1|ERROR|Traceback|FAILED|Complete|script write failed|PWC-TCP' \
  "$latest" | tail -60 || true
echo
echo "--- netconsole tail ---"
tail -50 logs/netconsole_all_rigs.log || true
echo
echo "--- startup diag tail ---"
tail -20 logs/pwc_startup_diag_simple.jsonl 2>/dev/null || true
