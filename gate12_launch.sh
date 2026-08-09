#!/usr/bin/env bash
# =====================================================================
#  GATE 12 — production-shape execution, Beta-authorized 2026-08-09
#  FROZEN SHAPE: seed_start=0 · seed_count=2^31 · stripe=2^26 · 32 stripes/stage
#                java_lcg · {constant, variable} · range-miner · one trial
#  Run from VM101:  bash gate12_launch.sh
#  MICHAEL-INITIATED ONLY.
# =====================================================================
set -u
cd ~/distributed_prng_analysis || exit 1
source ~/venvs/torch/bin/activate

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/gate12_${STAMP}.log
CONC=logs/gate12_${STAMP}_concurrency.tsv
EVID=logs/gate12_${STAMP}_evidence.txt

# ---------- 0. PRE-FLIGHT AUTHORITY EVIDENCE (Beta §12 "Authority") ----------
{
  echo "=== GATE 12 EVIDENCE — ${STAMP} ==="
  echo "--- HEAD ---";            git log --oneline -1
  echo "--- TREE STATE ---";      git status --porcelain
  echo "--- PRE-RUN CERTIFIED CURSOR (must be 0) ---"
  python3 -c "
from database_system import DistributedPRNGDatabase
d=DistributedPRNGDatabase()
print('cursor:', d.get_certified_cursor('java_lcg', test_both_modes=True))
" 2>&1
  echo "--- DATASET POINTER ---"; ls -la daily3.json daily3-*.json 2>/dev/null | tail -3
} | tee "$EVID"

# ---------- 1. CLEAN SLATE ----------
pkill -f "[w]atcher_agent"; pkill -f "[w]indow_optimizer"; pkill -f "[r]ange_miner_worker"
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  ssh -n michael@$ip 'pkill -f "[r]ange_miner_worker"' 2>/dev/null
done
sleep 3
[ -f optimal_window_config.json ] && \
  mv optimal_window_config.json optimal_window_config.json.pregate12_${STAMP}

# ---------- 2. COORDINATOR UP (halt cleared, miner on, PWC off) ----------
nohup env PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt --run-pipeline \
  --start-step 1 --end-step 1 \
  --params '{"use_persistent_workers": false, "use_range_miner": true,
             "seed_start": 0, "max_seeds": 2147483648,
             "miner_stripe_size": 67108864, "test_both_modes": true,
             "prng_type": "java_lcg", "window_trials": 1, "n_parallel": 1}' \
  > "$LOG" 2>&1 &

# ---------- 3. WAIT FOR BIND, THEN LAUNCH THE FLEET ----------
for i in $(seq 1 40); do ss -ltn | grep -q 5700 && break; sleep 1; done
if ss -ltn | grep -q 5700; then
  ./scripts/launch_fleet_manual.sh 192.168.3.177 5700 2>&1 | tail -4
else
  echo "COORDINATOR NEVER BOUND — aborting fleet launch"; tail -30 "$LOG"; exit 1
fi

# ---------- 4. CONCURRENCY SAMPLER (Beta §6 — the evidence that can't be redone) ----------
# Samples in-flight distinct workers + queued stripes every 5s for 2h.
( printf 'ts\testab\tclaimed_workers\tstate_counts\n' > "$CONC"
  for i in $(seq 1 1440); do
    EST=$(ss -tn 2>/dev/null | grep -c '5700.*ESTAB')
    ROW=$(python3 - <<'PY' 2>/dev/null
import sqlite3,glob,os
p='/home/michael/miner_staging/miner_ledger.db'
try:
    c=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
    w=c.execute("select count(distinct claimed_by) from stripes where state in ('claimed','staging') and claimed_by is not null").fetchone()[0]
    s=dict(c.execute("select state,count(*) from stripes group by state").fetchall())
    print(f"{w}\t{s}")
except Exception as e:
    print(f"-\t{e}")
PY
)
    printf '%s\t%s\t%s\n' "$(date +%H:%M:%S)" "$EST" "$ROW" >> "$CONC"
    sleep 5
  done ) &
SAMPLER=$!
echo "concurrency sampler pid=$SAMPLER -> $CONC" | tee -a "$EVID"

# ---------- 5. LIVE VIEW (Ctrl-C is safe: run + sampler keep going) ----------
echo; echo "LOG:  $LOG"; echo "CONC: $CONC"; echo "EVID: $EVID"; echo
tail -f "$LOG"
