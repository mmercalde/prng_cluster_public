#!/bin/bash
# DEVELOPMENT walk-forward on run H, variant 00_base: baseline: today's Step 3 features, holdout_quality target, four regressor families.
# NOT TO BE RUN without the owner's go and the lead's go (LEAD_ANSWERS_WF_20261123.txt: no run while
# VM100 is in use; owner's go asked first). Every number it writes is DEVELOPMENT evidence.
# Order (lead Q2): STAGE=smoke (one origin, block 0 = [3808, 4402), 2006-08-20..2007-06-12, overlaps
# nothing graded) -> report measured cost -> STAGE=stride4 (blocks 0,4,8,12,16,20) -> STAGE=rest.
# The lead sets CORES and W (workers) in the go. Pins are re-read at launch; G is the lead's
# development grader pin. Each stage and variant writes to its own new out-dir.
# Launch order (lead ADDENDUM_4 Fix 1, Beta X-E2E-WF-VOTE R1 Q1): the driver's own preflight
# (--check-only: every refusal the driver has, writes nothing) runs in the FOREGROUND first. If it
# refuses, this script prints the refusal, exits with its code and has created no log, no .time
# file and no out-dir. Only then is the real run started under nohup.
set -euo pipefail
: "${CORES:?set CORES from the go}" "${W:?set W (workers) from the go}" "${STAGE:?smoke|stride4|rest}"
case "$STAGE" in
  smoke)   ORIG="--origins 0" ;;
  stride4) ORIG="--origins all --origin-stride 4" ;;
  rest)    ORIG="--origins 1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23" ;;
  *) echo "STAGE must be smoke|stride4|rest" >&2; exit 2 ;;
esac
REPO=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../.." && pwd)   # this file is <repo>/docs/drafts/loop_v1/runH_commands/
cd "$REPO"
set +u; source ~/venvs/torch/bin/activate; set -u   # activate reads an unset LD_LIBRARY_PATH
R=/home/michael/skipsw_run_thr3030
V=$(sha256sum "$REPO/scripts/diagnostics/e2e_vote_harness.py" | cut -d' ' -f1)
G=3ab8ce2a5330fe36ae638e8bf2915e23bdce08bcefc0221eb47e21177fc8f341   # landed grader (main fe5ac3b, R26)
OUT=data/e2e_runH_wf_00_base_$STAGE
LOG="logs/e2e_runH_wf_00_base_$STAGE.log"
ARGS=(
  --dataset $R/daily3.json --dataset-sha256 513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6
  --npz data/e2e_runH/copy/bidirectional_survivors_binary.npz
  --generation-digest 3f6a884d6536cbbdd6b23b1e8f8c7b85946d001ca1b1de49e0c2fd9dac0882d0
  --scorer-config $R/optimal_scorer_config.json --scorer-config $R/optimal_window_config.json
  --sieve-window-end-index 1112 --label-window fixed --label-draws 1807
  --pool-sizes 20,100,300 --grade-sessions midday,evening,pooled
  --model-choice prev_block_rank --vote-allow-gpu --step3-device cuda
  --previously-inspected 14454:16261 --previously-inspected 16261:18068
  --step3-workers $W --vote-workers $W --grader-workers $W
  --vote-harness-sha256 $V --grader-sha256 $G
  --vote-config-sha256 d77c3ac80ad4c9175ef40f81d5605a313a0f5ab76b5e7136430449f10bd4be37
  $ORIG
  --out-dir "$OUT"
)
set +e
python3 -u -B scripts/diagnostics/e2e_walk_forward.py "${ARGS[@]}" --check-only
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "PREFLIGHT REFUSED (exit $rc): nothing launched; no log, no .time file, no out-dir written" >&2
  exit "$rc"
fi
mkdir -p logs data
nohup taskset -c "$CORES" nice -n 15 /usr/bin/time -v -o "$OUT.time" \
  python3 -u -B scripts/diagnostics/e2e_walk_forward.py "${ARGS[@]}" > "$LOG" 2>&1 &
echo "started $! -> $OUT (log $LOG)"
