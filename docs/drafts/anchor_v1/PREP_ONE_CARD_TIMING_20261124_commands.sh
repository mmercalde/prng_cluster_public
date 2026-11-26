# PREP_ONE_CARD_TIMING_20261124_commands.sh — NOT RUN. Prepared by the anchor agent 2026-11-24 for the lead.
# Read PREP_ONE_CARD_TIMING_20261124.md first. Blocks are pasted one at a time by the lead; each ends in a
# check that must print its OK line. Nothing here was executed by the agent.
# Variant a = run H's settings (needs the owner's one-off past the 50,000 cap).
# Variant b = run B's thresholds 0.71/0.47 (gate's own arithmetic predicts PROCEED; no one-off).
# Development only: direct window_optimizer.py, not WATCHER, not gate12_launch.sh.

# ---- P0: names (set VARIANT first) ------------------------------------------------------------
VARIANT=a                                   # or b
MAIN=/home/michael/distributed_prng_analysis
COMMIT=41ad509                              # main HEAD; code-identical to run H's 00e2ee7 (diff = one doc file)
TT=/home/michael/onecard_${VARIANT}_20261124        # run tree (git worktree of main, detached)
ST=/home/michael/onecard_${VARIANT}_staging_20261124 # coordinator staging + miner_ledger.db
VER=daily3-20260801T145551443433Z-513648160d35.json
CORES_OPT=6; CORES_WKR=7                    # lead to confirm after clearing (see the note §4)
if [ "$VARIANT" = a ]; then FT=0.3;  RT=0.3;  else FT=0.71; RT=0.47; fi
S=2147483648; N=67108864; E=$((S+N))        # [2^31, 2^31+2^26), inside run H's / run B's measured interval

# ---- P1: pre-flight (read only) --------------------------------------------------------------
test ! -e $TT && test ! -e $ST && echo "P1 PATHS FREE"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader   # must list only sunshine (and Xorg if any)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader # expect ~0 %, ~0.3 GiB
ps -eo pid,comm,args | awk '$2 ~ /^python/ && /range_miner_worker\.py/'   # must print nothing (matches python only, not this shell)
ss -ltn | /bin/grep -c ":5700 "                                            # must print 0
ls -A /dev/shm; df -h /dev/shm                                             # 2026-11-24 14:56: empty, 0 % used
(cd $MAIN && git rev-parse --short HEAD && test -z "$(git status --porcelain)" && echo "P1 MAIN CLEAN at HEAD")
(cd $MAIN && sha256sum $VER) | /bin/grep -q '^513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6 ' && echo "P1 DATASET DIGEST OK"
free -g | head -2                                                          # MemAvailable must be >= 6 GiB
uptime                                                                     # VM100 idle is the lead's check

# ---- P2: run tree and staging (writes ~/onecard_*, plus git worktree metadata in $MAIN/.git/worktrees) --
cd $MAIN && git worktree add --detach $TT $COMMIT
mkdir -p $TT/logs/stage4 $TT/logs/miner_output $TT/logs/cupy_cache $TT/optuna_studies $TT/logs/miner_output $TT/logs/cupy_cache
mkdir -p $ST && test -z "$(ls -A $ST)" && echo "P2 STAGING EMPTY"
IGN="daily3_current.json $VER daily3.json dataset_provisioning.json trse_context.json schemas/field_mappings.json"
for f in $IGN; do cp -p "$MAIN/$f" "$TT/$f"; done                          # run H's list (RUNBOOK_RUN_I b04)
diff <(cd $MAIN && sha256sum $IGN) <(cd $TT && sha256sum $IGN) && (cd $MAIN && sha256sum $IGN) > $TT/logs/stage4/inputs.sha256 && echo "P2 INPUTS IDENTICAL"
(cd $TT && test -z "$(git status --porcelain)" && echo "P2 RUN TREE CLEAN")

# ---- P3: survivor-volume gate (CARD, about 30 s; the gate's own GPU check refuses a busy card) -----
cd $TT && source ~/venvs/torch/bin/activate
cat > logs/stage4/volume_params.json <<EOF
{"seed_start": $S, "seed_end_exclusive": $E, "max_seeds": $N, "prng_type": "java_lcg", "window_size": 12, "window_anchor": 25, "generator_phase": 0, "sessions": ["midday"], "forward_threshold": $FT, "reverse_threshold": $RT, "skip_min": 0, "skip_max": 16, "dataset": "$TT/$VER", "test_both_modes": true, "deliver_skip_range": true}
EOF
# phase-4 sample must be given: the default 4 x 2^26 exceeds this 2^26 range and the estimator refuses it
CUPY_CACHE_DIR=$TT/logs/cupy_cache PYTHONPATH=. taskset -c $CORES_OPT nice -n 15 \
  python3 -u scripts/survivor_volume_gate.py --params @logs/stage4/volume_params.json \
  --record-out logs/stage4/volume_gate_record.json --sample-seeds $N --phase4-sample-seeds $N \
  2>&1 | tee logs/stage4/volume_gate.log; echo "gate rc=${PIPESTATUS[0]}"
# b: expect PROCEED (rc 0) -> P4.   a: expect REFUSE (rc 1) -> STOP unless the owner's one-off is on record.

# ---- P4: launch (optimizer first, worker within the 180 s admission window) -----------------
cd $TT && source ~/venvs/torch/bin/activate
CUPY_CACHE_DIR=$TT/logs/cupy_cache nohup taskset -c $CORES_OPT nice -n 15 python3 -u window_optimizer.py \
  --lottery-file $TT/$VER --strategy bayesian --max-seeds $N --prng-type java_lcg \
  --output optimal_window_config.json --test-both-modes --trials 1 --trse-context trse_context.json \
  --enable-pruning --n-parallel 1 --worker-pool-size 1 --seed-cap-nvidia 5000000 --seed-cap-amd 2000000 \
  --seed-start $S --pwc-transport tcp --min-workers 1 --use-range-miner --deliver-skip-range \
  --miner-stripe-size 67108864 --miner-substripes 8 --staging-dir $ST --staging-workers 4 \
  --staging-queue-depth 2 --staging-capacity-timeout 600.0 --staging-high-water-bytes 17179869184 \
  --execution-set-nodes localhost \
  --warm-start-fwd-thresh $FT --warm-start-rev-thresh $RT --warm-start-offset 25 --warm-start-session-idx 1 \
  --warm-start-skip-min 0 --warm-start-skip-max 16 --warm-start-window 12 \
  > logs/stage4/optimizer.out 2>&1 < /dev/null &
echo $! > logs/stage4/optimizer.pid
# wait for the execution-set line, then start the worker
until /bin/grep -q "EXEC-SET" logs/stage4/optimizer.out; do sleep 2; done; /bin/grep -F "EXEC-SET" logs/stage4/optimizer.out
CUDA_VISIBLE_DEVICES=0 CUPY_CACHE_DIR=$TT/logs/cupy_cache PYTHONPATH=$TT \
  nohup taskset -c $CORES_WKR nice -n 15 python3 -u $TT/miner/range_miner_worker.py \
  --host 127.0.0.1 --port 5700 --gpu-id 0 --device-index 0 --miner-output-dir $TT/logs/miner_output \
  > logs/stage4/worker_gpu0.log 2>&1 < /dev/null &
echo $! > logs/stage4/worker.pid
# memory guard: stops both recorded PIDs if MemAvailable < 2 GiB (box has zero swap)
nohup bash -c 'while kill -0 $(cat '$TT'/logs/stage4/optimizer.pid) 2>/dev/null; do
  a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); echo "$(date +%T) avail_gib=$a" >> '$TT'/logs/stage4/mem.log
  [ "$a" -lt 2 ] && kill -TERM $(cat '$TT'/logs/stage4/optimizer.pid) $(cat '$TT'/logs/stage4/worker.pid) && echo GUARD_FIRED >> '$TT'/logs/stage4/mem.log
  sleep 5; done' > /dev/null 2>&1 < /dev/null &
echo $! > logs/stage4/memguard.pid

# ---- P5: after the optimizer exits ----------------------------------------------------------
kill -0 $(cat logs/stage4/optimizer.pid) 2>/dev/null && echo "STILL RUNNING" || echo "optimizer exited"
kill -TERM $(cat logs/stage4/worker.pid) 2>/dev/null; sleep 5; kill -0 $(cat logs/stage4/worker.pid) 2>/dev/null && echo "WORKER STILL UP" || echo "worker stopped"
python3 - <<EOF
import sqlite3
c = sqlite3.connect("file:$ST/miner_ledger.db?mode=ro&immutable=1", uri=True)
for r in c.execute("select run_id, trial_number, state, terminal_class, created_at, finalized_at from trials"): print(r)
for r in c.execute("select phase, family_name, seed_count, state, substripes_done, survivors_total, elapsed_s from stripes order by phase"): print(r)
EOF
(cd $TT && find logs $ST -type f -print0 | sort -z | xargs -0 sha256sum) > $TT/logs/stage4/evidence.sha256
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader   # back to sunshine only
ls -A /dev/shm                                                             # unchanged (empty)
