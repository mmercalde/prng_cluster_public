#!/usr/bin/env bash
# =====================================================================
#  GATE 12 — production-shape execution, Beta-authorized 2026-08-09
#  FROZEN SHAPE: seed_start=0 · max_seeds=2^31 · stripe=2^26 · 32 stripes/stage
#                java_lcg · {constant, variable} · range-miner · one trial
#  Run from VM101:  bash gate12_launch.sh
#  MICHAEL-INITIATED ONLY.
#
#  ─── CHANGES vs the 2026-08-09 attempt-1 script (two Alpha defects) ─────────
#  1. worker_pool_size = 25.  Attempt 1 set the seed geometry and never
#     overrode the pool size, so `admission_count = min(requested, selected)`
#     took the manifest default of 8 (agent_manifests/window_optimizer.json:262)
#     and the run asked for 8 workers and got 8. The logged EXEC CMD of that run
#     reads `--worker-pool-size 8`. Beta classified this an operator error, not
#     a production defect.
#  2. THE SAMPLER STARTS FIRST — before the coordinator process exists, and so
#     necessarily before any StripeAssign. In attempt 1 it was started in step 4,
#     AFTER the fleet-launch step returned: its first row was 12:47:28 for a run
#     that died at 12:47:17, and it produced no in-run rows at all. It then
#     looped for two hours against a dead trial and had to be killed by hand.
#     It now terminates with the run (§4 below).
#     Its query has also been replaced wholesale — the old one counted
#     `state IN ('claimed','staging')` and never looked at `pending`, which under
#     the certified F1 model overstates occupancy and cannot see the queue depth
#     Beta's criterion actually turns on. See scripts/gate12_concurrency_sampler.py.
#
#  ─── CHANGES vs the 2026-08-10 attempt-3 script (clean-tree admission) ──────
#  3. THE CLEAN-TREE PREDICATE IS NOW TESTED, TWICE, AND IT IS THE SAME
#     PREDICATE PUBLICATION USES. Attempt 3 ran four stages, 128/128 stripes
#     over the full [0,2^31) domain and satisfied the saturation verdict, then
#     was refused at D3.5 with `repository_tree_clean is False`. The three
#     untracked entries responsible were PRINTED by this script's own
#     `--- TREE STATE ---` line and never TESTED: it printed the reason the run
#     was going to fail and launched anyway. §0.4 now refuses on that state,
#     reusing `window_optimizer_integration_final._repository_state` — the
#     producer whose boolean the finalizer receives — rather than
#     reimplementing `git status --porcelain`, because a second implementation
#     is a second predicate and a second predicate can disagree.
#  4. THE HARNESS NO LONGER DIRTIES THE TREE BY ITS OWN HAND. The clean slate
#     used to rename `optimal_window_config.json` (ignored) to
#     `optimal_window_config.json.pregate12_${STAMP}` — a name NO ignore rule
#     in `.gitignore` matches, verified with `git check-ignore`. That rename
#     happens AFTER admission and BEFORE dispatch, so testing the predicate
#     once at admission would not have been enough: the harness would have
#     manufactured the very state D3.5 rejects, in the window between the two.
#     It did not fire in attempt 3 only because `optimal_window_config.json`
#     had already been rotated away on 2026-08-08 — it WOULD fire now, because
#     attempt 3 left one behind. The rollback copy is preserved, under `logs/`,
#     which is ignored as a whole directory (.gitignore:62) so no filename
#     exception is needed and none was added.
#  5. §1.9 asserts the predicate one last time immediately before the sampler
#     and coordinator are created. ONE predicate: admission -> preparation must
#     preserve it -> last pre-dispatch assertion -> compute -> D3.5.
#
#  ─── CHANGES vs the 2026-08-12 attempt-5 script (ATTEMPT-6 remediation) ─────
#  6. THE LAUNCH IS NOW TWO-PHASE, AND THE FLEET STARTS FIRST. Attempts 4 and 5
#     produced NO observable rig-worker session event at all — 24 byte-identical
#     138-byte logs — and the cause is still UNRESOLVED. A sentinel verified
#     after REGISTER would prove the channel was alive for a run that had
#     already committed GPU-seconds, so the workers now emit a startup sentinel
#     and PARK on a per-host release token; the harness verifies 25/25 delivery
#     BEFORE the coordinator exists, and writes the tokens only after the
#     coordinator is listening. Verification therefore happens OUTSIDE the 180 s
#     admission window and cannot spend the admission budget.
#     A shortfall is a REFUSAL: no reduced cohort, no automatic downsizing.
#
#  ⚠ PARAMETER TRAPS (§2.25) — do not "tidy" these:
#     * the key is `max_seeds`, NOT `seed_count`. `seed_count` is not in the
#       manifest's default_params, so WATCHER's declared-key filter drops it
#       silently and you get the 2^30 default — 16 stripes, not 32.
#     * booleans are FLAG-ONLY: true emits the flag, false OMITS it entirely.
#       That is how `use_persistent_workers: false` suppresses PWC.
#     * `--start-step 1 --end-step 1` is MANDATORY. --end-step defaults to 6 and
#       STEP_SCRIPTS[2] reaches run_scorer_meta_optimizer.sh, which invokes the
#       TB-prohibited converter and mv's a regular file onto the D3.5
#       finalizer-owned symlink -> PublicationError, hours in, at publication.
# =====================================================================
set -u
cd ~/distributed_prng_analysis || exit 1
source ~/venvs/torch/bin/activate

STAMP=$(date +%Y%m%d_%H%M%S)
# [ATTEMPT-6 §8.4.2] THE RUN NONCE. Opaque, run-scoped, and it must be UNIQUE per
# attempt: it is what makes a leftover /tmp/minerlogs/gpuN.log from an earlier
# launch — or a leftover release token — unable to satisfy this run's gate. The
# timestamp alone is already unique per attempt; the pid suffix removes even the
# same-second case.
RUN_NONCE="gate12-${STAMP}-$$"
LOG=logs/gate12_${STAMP}.log
CONC=logs/gate12_${STAMP}_concurrency.tsv
VERDICT=logs/gate12_${STAMP}_verdict.txt
SAMPLOG=logs/gate12_${STAMP}_sampler.log
EVID=logs/gate12_${STAMP}_evidence.txt
mkdir -p logs

# ---------- 0. PRE-FLIGHT AUTHORITY EVIDENCE (Beta §12 "Authority") ----------
{
  echo "=== GATE 12 EVIDENCE — ${STAMP} ==="
  echo "--- HEAD ---";            git log --oneline -1
  echo "--- TREE STATE ---"
  echo "    reported AND DECIDED by the clean-tree gate in §0.4 below."
  echo "    This block deliberately no longer prints an untested porcelain"
  echo "    listing: attempt 3 printed exactly that listing, containing the"
  echo "    three entries publication would reject, and launched anyway."
  echo "--- PRE-RUN CERTIFIED CURSOR (must be 0) ---"
  python3 -c "
from database_system import DistributedPRNGDatabase
d=DistributedPRNGDatabase()
print('cursor:', d.get_certified_cursor('java_lcg', test_both_modes=True))
" 2>&1
  echo "--- DATASET POINTER ---"; ls -la daily3.json daily3-*.json 2>/dev/null | tail -3
} | tee "$EVID"

# ---------- 0.4. CLEAN-TREE ADMISSION GATE (Beta 2026-08-11 amendment) ------
# Attempt 3's SOLE failure. Four stages, 128/128 stripes, saturation SATISFIED —
# then `RunParameterError: repository_tree_clean is False` at publication.
#
# The gate calls window_optimizer_integration_final._repository_state, which is
# the function whose second return value becomes the finalizer's
# `repository_tree_clean` argument (…final.py:2972 -> :2992). It is imported,
# not reimplemented: D3.5 is not weakened, no allowlist is introduced, and no
# ignore-rule exception is added. The refusal names the offending entries and
# says publication would reject them.
#
# PLACEMENT IS LOAD-BEARING, and it is deliberately AHEAD of the GPU gate: this
# check is local, needs no SSH, and costs no GPU-seconds, so the cheapest
# refusal happens first. Like the GPU gate it sits before the clean slate,
# before the sampler is armed and before any coordinator process is created.
#
# ${PIPESTATUS[0]}, NOT the pipeline's status — see the note under §0.5. Writing
# `if ! python3 … | tee` would test tee's exit status and make this gate
# decorative, which is precisely the class of defect it exists to close.
python3 -u scripts/gate12_cleantree_gate.py --phase admission 2>&1 | tee -a "$EVID"
CLEANTREE_RC=${PIPESTATUS[0]}
if [ "$CLEANTREE_RC" -ne 0 ]; then
  echo "GATE-12 ABORTED BY THE CLEAN-TREE ADMISSION GATE (rc=$CLEANTREE_RC) — see $EVID" \
    | tee -a "$EVID"
  exit 1
fi

# ---------- 0.5. GPU FAIL-CLOSE GATE (Beta R3/P2) ----------
# Gate 12 is a saturation claim about 24 rig GPUs. Attempt 1 logged a 0/8 GPU
# reading and launched anyway, because the generic PreflightChecker reports GPU
# findings as WARNINGS and is non-blocking BY DESIGN — correct for WATCHER, and
# deliberately unchanged. This is a GATE-12 HARNESS RULE ONLY.
#
# It runs the already-certified truthful probe (preflight_check's
# _build_gpu_probe_script / _parse_gpu_probe, reused rather than reimplemented)
# against the three rigs and proceeds ONLY on OK at the full expected count on
# all three. UNAVAILABLE and ERROR both REFUSE, and UNAVAILABLE is reported as
# UNAVAILABLE — never as a count.
#
# PLACEMENT IS LOAD-BEARING: this is before the clean slate, before the sampler
# is armed, and before any coordinator process is created. A refusal therefore
# leaves the box exactly as it found it — nothing killed, no config moved, no
# process spawned — so a refused attempt costs nothing and can be retried once
# the rigs are actually up.
#
# ${PIPESTATUS[0]}, NOT the pipeline's status: `cmd | tee` exits with TEE's
# status, which is 0 essentially always. Writing `if ! python3 ... | tee` would
# have made this gate decorative — it would print REFUSED and launch anyway,
# which is the "gate result ignored" failure mode, in the very script whose
# attempt-1 defect was a GPU reading that stopped nothing.
python3 -u scripts/gate12_gpu_gate.py 2>&1 | tee -a "$EVID"
GPU_GATE_RC=${PIPESTATUS[0]}
if [ "$GPU_GATE_RC" -ne 0 ]; then
  echo "GATE-12 ABORTED BY THE GPU FAIL-CLOSE GATE (rc=$GPU_GATE_RC) — see $EVID" \
    | tee -a "$EVID"
  exit 1
fi

# ---------- 0.6. RIG CODE-PARITY GATE (Beta 2026-08-14, D6 integration repair) ----------
# The GPU gate proves the CARDS. Nothing proved the CODE — and on 2026-08-14 the
# D6 parked-fleet dry run dispatched 25 workers into three rigs carrying a
# `miner/range_miner_worker.py` last deployed 2026-08-02: 24 died at argparse,
# having no attempt-6 sentinel, no Defect A recovery and no session-event
# emitter. The fleet had been at that vintage through attempts 3, 4 and 5, and no
# gate anywhere would have said so.
#
# It compares the DEPLOYED BYTES on every rig against full 64-hex SHA256 values
# derived from this tree at run time. Acceptance authority is CONTENT IDENTITY,
# never Git identity: two of the rigs have no git repository at all, the third's
# worktree does not describe its deployed bytes, and the deployment is provably
# mixed-vintage. The local commit is recorded as context and is not an input.
#
# PLACEMENT IS LOAD-BEARING and matches the two gates above: before the clean
# slate, before the sampler is armed, before any coordinator process exists, and
# BEFORE WORKER DISPATCH. A refusal leaves the box exactly as it found it.
#
# ${PIPESTATUS[0]}, NOT the pipeline's status — same rule, same reason.
PARITY_EVID=logs/gate12_${STAMP}_source_digests.json
python3 -u scripts/gate12_parity_gate.py --evidence-json "$PARITY_EVID" 2>&1 | tee -a "$EVID"
PARITY_RC=${PIPESTATUS[0]}
if [ "$PARITY_RC" -ne 0 ]; then
  echo "GATE-12 ABORTED BY THE RIG CODE-PARITY GATE (rc=$PARITY_RC) — see $EVID" \
    | tee -a "$EVID"
  echo "source-digest evidence: $PARITY_EVID" | tee -a "$EVID"
  exit 1
fi
echo "source-digest evidence bundle: $PARITY_EVID" | tee -a "$EVID"

# ---------- 1. CLEAN SLATE ----------
pkill -f "[w]atcher_agent"; pkill -f "[w]indow_optimizer"; pkill -f "[r]ange_miner_worker"
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  ssh -n michael@$ip 'pkill -f "[r]ange_miner_worker"' 2>/dev/null
  # [ATTEMPT-6] Old release tokens are removed for tidiness ONLY. They can never
  # release this run — the nonce is in both the filename and the content and the
  # worker compares the content — so this is housekeeping, not the mechanism.
  ssh -n michael@$ip 'rm -f /tmp/minerlogs/gate12_release_*' 2>/dev/null
done
rm -f logs/miner_workers/gate12_release_* 2>/dev/null
sleep 3
# Step 1's previous output is moved aside so this run's output is unambiguously
# new. The destination is `logs/`, NOT a sibling in the repository root.
#
# The old destination was `optimal_window_config.json.pregate12_${STAMP}`. The
# source is ignored (.gitignore:115) but that suffixed name is NOT — verified
# with `git check-ignore`, which reports NOT IGNORED for it. The clean slate
# therefore dirtied the worktree by its own hand, after admission and before
# dispatch, manufacturing exactly the state D3.5 rejects. `logs/` is ignored as
# a whole directory (.gitignore:62), so the rollback copy survives, lands beside
# this run's other artifacts under the same STAMP, and is invisible to Git
# without any filename exception being added anywhere.
#
# The markers below are load-bearing: tests/test_gate12_cleantree_admission.py
# extracts this region VERBATIM from this file and executes it, so gate C5A
# measures the real preparation step rather than a paraphrase of it.
# --- GATE12-CONFIG-ROTATION BEGIN ---
if [ -f optimal_window_config.json ]; then
  mv optimal_window_config.json \
     "logs/gate12_${STAMP}_pregate12_optimal_window_config.json"
fi
# --- GATE12-CONFIG-ROTATION END ---

# ---------- 1.9. LAST PRE-DISPATCH CLEAN-TREE ASSERTION ----------
# Testing the predicate once at admission is INSUFFICIENT, and attempt 3 is not
# the proof — the harness is. Admission passing then launch preparation dirtying
# the worktree is a real, reachable sequence (the `pregate12` rename above did
# exactly that), and D3.5 would not discover it for another two hours.
#
# ONE predicate, four evaluations of it, all agreeing by construction:
#   §0.4 admission (clean) -> preparation preserves it -> §1.9 (still clean)
#   -> compute -> D3.5.
# A refusal HERE means launch preparation broke the contract: a harness defect,
# not an operator one.
python3 -u scripts/gate12_cleantree_gate.py --phase pre-dispatch 2>&1 | tee -a "$EVID"
CLEANTREE_PREDISPATCH_RC=${PIPESTATUS[0]}
if [ "$CLEANTREE_PREDISPATCH_RC" -ne 0 ]; then
  echo "GATE-12 ABORTED BY THE PRE-DISPATCH CLEAN-TREE ASSERTION (rc=$CLEANTREE_PREDISPATCH_RC) — see $EVID" \
    | tee -a "$EVID"
  exit 1
fi

# ---------- 2. LAUNCH THE FLEET — PARKED AT THE RELEASE BARRIER ----------
# [ATTEMPT-6 §8.4.3] THE FLEET NOW STARTS BEFORE THE COORDINATOR, and that is the
# whole point of the barrier: with RUN_NONCE set every worker warms its GPU,
# emits SESSION_SENTINEL through the same `_emit_session_event` path the session
# events use, and then PARKS on its per-host release token. Nothing connects out,
# so the coordinator need not exist yet and no GPU-second and no cohort freeze is
# committed by anything below until §2.5 has passed.
FLEETLOG=logs/gate12_${STAMP}_fleet.log
RUN_NONCE="$RUN_NONCE" RELEASE_DEADLINE=900 \
  ./scripts/launch_fleet_manual.sh 192.168.3.177 5700 "$PWD/logs/miner_workers" \
  > "$FLEETLOG" 2>&1
FLEET_RC=$?
tail -6 "$FLEETLOG" | tee -a "$EVID"
if [ "$FLEET_RC" -ne 0 ]; then
  echo "GATE-12 ABORTED: fleet dispatch was TRUNCATED (rc=$FLEET_RC) — see $FLEETLOG" \
    | tee -a "$EVID"
  pkill -f "[r]ange_miner_worker"
  for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
    ssh -n michael@$ip 'pkill -f "[r]ange_miner_worker"' 2>/dev/null
  done
  exit 1
fi

# ---------- 2.5. WORKER-LOG SENTINEL GATE — 25/25 OR REFUSE ----------
# Attempts 4 and 5 produced NO observable rig-worker session event: 24
# byte-identical 138-byte logs, cause UNRESOLVED. This proves, before the run
# commits anything, that a record written through the production session-event
# path arrives where the operator can read it.
#
# PLACEMENT IS LOAD-BEARING, twice over. It is before the coordinator exists, so
# verification time is FREE — it cannot spend the 180 s admission budget and
# manufacture attempt 2's terminal with the fix. And it is before the sampler is
# armed, so a refusal costs nothing.
#
# ${PIPESTATUS[0]}, NOT the pipeline's status — the same rule as the two gates
# above, and for the same reason: `cmd | tee` exits with tee's status and would
# make this gate decorative.
python3 -u scripts/gate12_sentinel_gate.py --phase verify \
  --run-nonce "$RUN_NONCE" --local-log-dir "$PWD/logs/miner_workers" 2>&1 | tee -a "$EVID"
SENTINEL_RC=${PIPESTATUS[0]}
if [ "$SENTINEL_RC" -ne 0 ]; then
  echo "GATE-12 ABORTED BY THE WORKER-LOG SENTINEL GATE (rc=$SENTINEL_RC) — see $EVID" \
    | tee -a "$EVID"
  echo "killing the parked fleet; it never contacted a coordinator" | tee -a "$EVID"
  pkill -f "[r]ange_miner_worker"
  for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
    ssh -n michael@$ip 'pkill -f "[r]ange_miner_worker"' 2>/dev/null
  done
  exit 1
fi

# ---------- 3. CONCURRENCY SAMPLER — ARMED BEFORE THE COORDINATOR ----------
# Ordering is the whole point: the sampler is running before the coordinator
# process is created, so it cannot miss the first StripeAssign. It latches onto
# the first run whose stripe rows are created AFTER this moment, which is also
# what proves it was armed first.
#
# Read-only against the miner ledger (`file:...?mode=ro`); the ledger path is
# derived from the manifest's staging_dir, and a production analysis database is
# refused outright by name.
#
# setsid: so Ctrl-C on the tail -f at the end cannot reach it.
setsid nohup python3 -u scripts/gate12_concurrency_sampler.py \
  --out "$CONC" --summary "$VERDICT" \
  --interval 2 --threshold 25 --min-window-samples 2 \
  --port 5700 --max-seconds 7200 \
  > "$SAMPLOG" 2>&1 &
SAMPLER=$!
sleep 2
if ! kill -0 "$SAMPLER" 2>/dev/null; then
  echo "SAMPLER FAILED TO START — aborting before the run"; cat "$SAMPLOG"; exit 1
fi
echo "concurrency sampler pid=$SAMPLER -> $CONC" | tee -a "$EVID"

# ---------- 4. COORDINATOR UP (halt cleared, miner on, PWC off) ----------
nohup env PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt --run-pipeline \
  --start-step 1 --end-step 1 \
  --params '{"use_persistent_workers": false, "use_range_miner": true,
             "worker_pool_size": 25,
             "seed_start": 0, "max_seeds": 2147483648,
             "miner_stripe_size": 67108864, "test_both_modes": true,
             "prng_type": "java_lcg", "window_trials": 1, "n_parallel": 1}' \
  > "$LOG" 2>&1 &
WATCHER=$!
echo "watcher pid=$WATCHER -> $LOG" | tee -a "$EVID"

# ---------- 4.5. SAMPLER TERMINATES WITH THE RUN ----------
# A supervisor, not a wall clock: when the run's own process exits, the sampler
# is asked to stop and writes its verdict. Attempt 1's sampler had no such link
# and looped for two hours against a dead trial.
setsid nohup bash -c '
  while kill -0 '"$WATCHER"' 2>/dev/null; do sleep 5; done
  sleep 10                      # let the last in-flight sample land
  kill -TERM '"$SAMPLER"' 2>/dev/null
' > /dev/null 2>&1 &

# ---------- 5. WAIT FOR BIND, THEN RELEASE THE PARKED FLEET ----------
# [ATTEMPT-6 §8.4.3] steps 4-6 of the launch order. The fleet is already up and
# already proved its log channel; what happens here is only the RELEASE, and it
# happens only once the coordinator is actually listening — so registration lands
# inside a freshly armed admission window instead of racing it.
for i in $(seq 1 40); do ss -ltn | grep -q 5700 && break; sleep 1; done
if ss -ltn | grep -q 5700; then
  python3 -u scripts/gate12_sentinel_gate.py --phase release \
    --run-nonce "$RUN_NONCE" --local-log-dir "$PWD/logs/miner_workers" 2>&1 \
    | tee -a "$EVID"
  RELEASE_RC=${PIPESTATUS[0]}
  if [ "$RELEASE_RC" -ne 0 ]; then
    echo "RELEASE WRITE FAILED (rc=$RELEASE_RC) — the parked workers will fail" \
      | tee -a "$EVID"
    echo "closed at their release deadline rather than registering." | tee -a "$EVID"
    kill -TERM "$SAMPLER" 2>/dev/null
    exit 1
  fi
else
  echo "COORDINATOR NEVER BOUND — not releasing the fleet"
  kill -TERM "$SAMPLER" 2>/dev/null
  pkill -f "[r]ange_miner_worker"
  for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
    ssh -n michael@$ip 'pkill -f "[r]ange_miner_worker"' 2>/dev/null
  done
  tail -30 "$LOG"; exit 1
fi

# ---------- 6. LIVE VIEW (Ctrl-C is safe: run + sampler keep going) ----------
echo
echo "LOG:     $LOG"
echo "CONC:    $CONC        (per-sample TSV)"
echo "VERDICT: $VERDICT     (written when the run ends)"
echo "SAMPLOG: $SAMPLOG"
echo "EVID:    $EVID"
echo
tail -f "$LOG"
