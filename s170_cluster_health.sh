#!/usr/bin/env bash
set -euo pipefail

cd ~/distributed_prng_analysis

echo "=== Zeus git/runtime ==="
git branch --show-current && git rev-parse --short HEAD
source ~/venvs/torch/bin/activate && python3 --version

echo
echo "=== AMD rig reachability ==="
for rig in 192.168.3.120 192.168.3.154 192.168.3.162; do
  echo "--- $rig ---"
  ssh -o ConnectTimeout=5 $rig \
    "hostname; uptime; systemctl get-default; rocm-smi --showid --showtemp --showpower --showclocks | head -80" || \
    echo "UNREACHABLE: $rig"
done

echo
echo "=== Netconsole tail ==="
tail -20 logs/netconsole_all_rigs.log 2>/dev/null || echo "(empty)"
