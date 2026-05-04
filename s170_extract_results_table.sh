#!/usr/bin/env bash
set -euo pipefail

cd ~/distributed_prng_analysis

for log in $(ls -1 logs/stability_cap_*_t5_*.log 2>/dev/null | sort); do
  echo
  echo "============================================================"
  echo "LOG=$log"
  cap=$(echo $log | sed -n 's/.*stability_cap_\([0-9]*\)_t5_.*/\1/p')
  echo "CAP=$cap"
  echo "--- completion/error ---"
  grep -E 'Pipeline Summary|Complete|FAILED|ERROR|Traceback|script write failed|PWC-TCP' "$log" | tail -40 || true
  echo "--- throughput ---"
  grep -E 'seeds/sec|Elapsed' "$log" | tail -20 || true
done

echo
echo "============================================================"
echo "NETCONSOLE FAULT SUMMARY"
grep -Ei 'GCVM|PROTECTION_FAULT|qcm fence|amdgpu|kfd|timeout|reset|ring' \
  logs/netconsole_all_rigs.log 2>/dev/null | tail -80 || true
