> **RE-SAVED COPY after the 2026-11-22 12:26 Zeus crash, which wiped `/tmp`. EXACT, not reconstructed.**
> The original was `/tmp/skipsw_runbook.md`, used for runs A and B. Everything below this header block is byte-for-byte the original:
> sha256 `36a659b087518233ec2c8087fff63305e98d8f9e24d15887b96938c96ee11c0f`. It was rebuilt by replaying this session's own Write and edit calls from the session
> transcript, and the result equals the hash recorded before the runbook was handed to the lead.
> To check: strip these 5 header lines and the blank line after them, then run sha256sum.

# Skip-switch rig runs A (off) and B (on, skip [0,16]): runbook for the Alpha lead

Owner 2026-11-22 09:46 approved the two runs. **The lead runs them; this session did not touch a rig.** Modelled on
`docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md` (main, §2–§9) and yesterday's `~/stage4_env.sh` and
`~/stage4_miner_launch.sh` (sha256 `fe964b04…`, the reviewed Stage 4 launcher). Every block runs on VM101 as
`michael`, after `source ~/venvs/torch/bin/activate`. Run the arms **one after the other: A, then B.** They share the
fleet and port 5700.

**What "equal" can mean here (read first).** `scripts/compare_engine_outputs.py` returns `NOT COMPARABLE` (rc 2) for
both comparisons, by design, because their inputs differ:
- **A vs yesterday:** `run.repository_commit` is `6eafad9` against `00e2ee7`. It is one of the 15 F2 identity inputs
  (`compare_engine_outputs.py:117-124`).
- **B vs A:** `param.deliver_skip_range` is present only in B. It is not an engine key, so it stays in `inputs`
  (`:142-148`, `:167-169`), and any differing input makes the pair not comparable (`:455-456`).

So §6 runs the comparator for the record, expecting rc 2 with **exactly** that one differing input, and then a
separate output-only check (`/home/michael/skipsw_outputs_equal.py`, below). That check compares, field for field:
- `phases`, `bidirectional`, `phase_counts`, `credit`;
- the 22-array NPZ, bit-exact;
- and it asserts the input difference is exactly the expected key.

**This is not the comparator's verdict. Decide whether it is the test you want before the runs.**

---

## 0. Fixed values

```bash
MAIN=/home/michael/distributed_prng_analysis
COMMIT=00e2ee7
TA=/home/michael/skipsw_run_off          # arm A run tree (new)
TB=/home/michael/skipsw_run_on           # arm B run tree (new)
SA=/home/michael/skipsw_staging_off      # arm A miner staging + miner_ledger.db (new, empty; never ~/miner_staging)
SB=/home/michael/skipsw_staging_on       # arm B
EV=$MAIN/logs/skipsw_session             # evidence; logs/ is git-ignored in the main tree
Y=$MAIN/logs/stage4_session/bundle_miner # yesterday's RANGE-MINER bundle (read-only)
S=2147483648; E=2415919104               # [2^31, 2^31 + 2^28): Stage 4's interval
```
- **Yesterday's reference** (`$Y/bundle.json`, read by this session):
  - phase rows `0 / 0 / 13 / 2995`;
  - `bidirectional` `{"constant": null, "variable": null}`;
  - credit granted over `[2147483648, 2415919104]`;
  - NPZ artifact `aecc5d504784ecca…`, 0 rows;
  - dataset `513648160d35…`;
  - commit `6eafad9`.
- **Today's dataset pointer** `$MAIN/daily3_current.json` names `daily3-20260801T145551443433Z-513648160d35.json`
  (sha `513648160d35…`), the same dataset. Re-check it at step 1: a newer pointer would make A ≠ yesterday for a
  reason unrelated to the switch.

## 1. Params, per arm (exact)

**Arm A, switch off.** Yesterday's params exactly, with only `staging_dir` changed. There is **no
`deliver_skip_range` key**: off is the absent key, so A's inputs equal yesterday's.
```json
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 0, "warm_start_skip_max": 16, "staging_dir": "/home/michael/skipsw_staging_off"}
```
**Arm B, switch on.** The same, plus `"deliver_skip_range": true`, with its own `staging_dir`.
```json
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 0, "warm_start_skip_max": 16, "deliver_skip_range": true, "staging_dir": "/home/michael/skipsw_staging_on"}
```
- `seed_start: 0` and `max_seeds` are replaced by WATCHER with the search's first gap and chunk
  (`agents/watcher_agent.py:2304-2308`), as yesterday.
- **B is a fresh, one-trial, fully pinned run.** So the child hands the exact pin `[0, 16]` to the study body
  (`_step1_skip_delivery_exact_pin`), and constant phases 1–2 receive `skip_range=[0,16]` in their payloads. A's
  workers use their own default `[0, 16]` (`miner/range_miner_worker.py:1293`). **Equal skip ranges are the point:**
  B = A tests that delivering the switch changes nothing when the delivered range is the default.

## 2. Preparation (no GPU, no worker)

**P0. Nothing running; record the live state.**
```bash
mkdir -p $EV
ip -4 addr show | grep -E 'inet 192\.168\.3\.177'                                  # must print the line
pgrep -af "[w]atcher_agent|[w]indow_optimizer|[r]ange_miner_worker|[p]wc_worker_service|[s]ieve_gpu_worker"   # must print nothing
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  ssh -n -o BatchMode=yes michael@$ip 'hostname; pgrep -af "[r]ange_miner_worker|[p]wc_worker_service|[s]ieve_gpu_worker" || echo "  no workers"'
done
cd $MAIN && git rev-parse HEAD && test -z "$(git status --porcelain)" && echo "MAIN CLEAN"     # HEAD must be 00e2ee7…
HALT=$(cd $MAIN && python3 -c "from agents.safety.kill_switch import KillSwitch; print(KillSwitch().halt_file_path)")
test ! -e "$HALT" && echo "NO HALT FLAG ($HALT)"            # present: stop, ask the owner, never clear it
{ stat -c '%n %s %Y' $MAIN/prng_analysis.db; readlink $MAIN/.s172_accumulator/current; } > $EV/live_state_before.txt
python3 -c "import json;print(json.load(open('$MAIN/daily3_current.json'))['sha256'])"   # must start 513648160d35
```

**P1. Two run trees at `00e2ee7`, and two empty staging directories.** This writes `$MAIN/.git/worktrees/` metadata,
so it needs the owner's OK.
```bash
cd $MAIN
git worktree add --detach $TA $COMMIT
git worktree add --detach $TB $COMMIT
mkdir -p $TA/optuna_studies $TB/optuna_studies $TA/logs/stage4 $TB/logs/stage4     # both git-ignored
mkdir -p $SA $SB && test -z "$(ls -A $SA)" && test -z "$(ls -A $SB)" && echo "STAGING EMPTY"
```
- Each tree's Optuna storage follows its own checkout (`window_optimizer_bayesian.py:19-23`). Each tree also gets its
  own `prng_analysis.db`, which the CLI opens relative to the cwd (`database_system.py:44-46`). A fresh tree has
  neither until the search open creates the database.
- Each tree is its own output root, so B's root arm marker `.s172_accumulator/skip_delivery_arm` lands only in `$TB`.
  B prints `[SKIP-DELIVERY] marking … as the switched-on arm's output root …` when it marks it.

**P2. Copy the git-ignored inputs into each tree** (yesterday's list, runbook A2).
```bash
VER=$(python3 -c "import json;print(json.load(open('$MAIN/daily3_current.json'))['filename'])")
IGN="daily3_current.json $VER daily3.json dataset_provisioning.json trse_context.json schemas/field_mappings.json"
for T in $TA $TB; do
  for f in $IGN; do cp -p "$MAIN/$f" "$T/$f"; done
  diff <(cd $MAIN && sha256sum $IGN) <(cd $T && sha256sum $IGN) && echo "$T inputs identical"
  (cd $T && test -z "$(git status --porcelain)" && echo "$T CLEAN")
done
```

**P3. Provision the dataset to every node from each tree.** This is a rig write, so it needs the owner's OK. The
block is runbook A2b verbatim, with the tree list changed.
```bash
(
A2B_RC=0
for T in $TA $TB; do rm -f "$T/logs/stage4/provision.ok"; done
for T in $TA $TB; do
  if ! mkdir -p "$T/logs/stage4" || ! cd "$T"; then echo "$T: cannot enter - STOP"; A2B_RC=1; break; fi
  rm -f logs/stage4/provision.log
  PYTHONPATH=. python3 -u scripts/provision_dataset_fleet.py 2>&1 | tee logs/stage4/provision.log
  rc=("${PIPESTATUS[@]}")
  echo "$T provisioner rc=${rc[0]} log-writer rc=${rc[1]}"
  h=""
  if [ "${rc[0]}" -eq 0 ] && [ "${rc[1]}" -eq 0 ] && [ -s logs/stage4/provision.log ] \
     && h=$(sha256sum logs/stage4/provision.log | cut -d' ' -f1) && [ -n "$h" ] \
     && echo "PROVISION_OK $(date -u +%FT%TZ) log_sha256=$h" > logs/stage4/provision.ok \
     && /bin/grep -q "log_sha256=$h" logs/stage4/provision.ok; then
    echo "$T: provisioned, receipt names log sha256 $h"
  else
    rm -f logs/stage4/provision.ok; echo "$T: provisioning NOT confirmed - STOP"; A2B_RC=1; break
  fi
done
exit $A2B_RC
)
echo "provision status=$?"                                          # must be 0
```

**P4. Params files** (§1, verbatim).
```bash
cat > $TA/logs/stage4/params.json <<'EOF'
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 0, "warm_start_skip_max": 16, "staging_dir": "/home/michael/skipsw_staging_off"}
EOF
cat > $TB/logs/stage4/params.json <<'EOF'
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 0, "warm_start_skip_max": 16, "deliver_skip_range": true, "staging_dir": "/home/michael/skipsw_staging_on"}
EOF
python3 - <<'EOF'
import json
a = json.load(open("/home/michael/skipsw_run_off/logs/stage4/params.json"))
b = json.load(open("/home/michael/skipsw_run_on/logs/stage4/params.json"))
y = json.load(open("/home/michael/stage4_run_miner/logs/stage4/params.json"))
assert {k: v for k, v in a.items() if k != "staging_dir"} == {k: v for k, v in y.items() if k != "staging_dir"}, "A != yesterday"
assert set(b) - set(a) == {"deliver_skip_range"} and b["deliver_skip_range"] is True, "B is not A + switch"
assert {k: v for k, v in b.items() if k not in ("staging_dir", "deliver_skip_range")} == \
       {k: v for k, v in a.items() if k != "staging_dir"}, "B differs from A beyond the switch"
print("params ok: A = yesterday (staging aside); B = A + deliver_skip_range true")
EOF
```

**P5. The two launchers.** Yesterday's launcher with only its one tree path changed (`:79`, the only occurrence). The
launch path is unchanged from `6eafad9` to `00e2ee7`: `git diff --stat 6eafad9 00e2ee7` over `gate12_launch.sh`, the
fleet launcher, every gate12 gate and the provisioner is empty.
```bash
test "$(sha256sum /home/michael/stage4_miner_launch.sh | cut -d' ' -f1)" = fe964b04fe3ea431986835ff2fff934349864751f8974da5ccebb6b1a62bf5d3 && echo "base launcher ok"
sed 's#^cd /home/michael/stage4_run_miner || exit 1$#cd /home/michael/skipsw_run_off || exit 1#' /home/michael/stage4_miner_launch.sh > /home/michael/skipsw_launch_off.sh
sed 's#^cd /home/michael/stage4_run_miner || exit 1$#cd /home/michael/skipsw_run_on || exit 1#'  /home/michael/stage4_miner_launch.sh > /home/michael/skipsw_launch_on.sh
for L in off on; do
  diff /home/michael/stage4_miner_launch.sh /home/michael/skipsw_launch_$L.sh | /bin/grep -c '^[<>]'   # must be 2 (one line changed)
  chmod +x /home/michael/skipsw_launch_$L.sh && bash -n /home/michael/skipsw_launch_$L.sh && echo "$L syntax ok"
done
```
- The launcher validates its `logs/stage4/params.json`: a RANGE-MINER run, every pin present, an absolute empty
  `staging_dir` that does not overlap the live one, and no halt flag. It runs the cleantree, GPU and parity gates,
  launches the fleet parked, runs the sentinel, liveness and cache-join gates, then `nohup` WATCHER, then releases.
  No sampler.

**P6. CPU-only dry check in each tree.**
```bash
for T in $TA $TB; do (cd $T && python3 -u scripts/gate12_cleantree_gate.py --phase admission 2>&1 | tee logs/stage4/cleantree_P6.log; echo "$T rc=${PIPESTATUS[0]}"); done
```

**P7. Arm B only: the survivor-volume gate.**
- **When:** before B's launch, with VM101's GPU idle. It uses **VM101's** GPU, not the rigs.
- **The geometry:** B's exact geometry, with `deliver_skip_range` true. `session_idx 1` is `["midday"]`
  (`window_optimizer.py:181-186`), and the dataset is the tree's own copy.
```bash
cd $TB
cat > logs/stage4/volume_params.json <<'EOF'
{"seed_start": 2147483648, "seed_end_exclusive": 2415919104, "max_seeds": 268435456, "prng_type": "java_lcg", "window_size": 12, "window_anchor": 25, "generator_phase": 0, "sessions": ["midday"], "forward_threshold": 0.71, "reverse_threshold": 0.47, "skip_min": 0, "skip_max": 16, "dataset": "/home/michael/skipsw_run_on/daily3-20260801T145551443433Z-513648160d35.json", "test_both_modes": true, "deliver_skip_range": true}
EOF
python3 -u scripts/survivor_volume_gate.py --params @logs/stage4/volume_params.json \
  --record-out logs/stage4/volume_gate_record.json 2>&1 | tee logs/stage4/volume_gate.log
echo "rc=${PIPESTATUS[0]}"                  # 0 = PROCEED; anything else: stop, do not launch B
```
- **Pass:** `PROCEED`, with every phase at or below the tracked cap of 50,000 (`config_manifests/survivor_volume.cap`).
  The `[SKIP-DELIVERY] … measured only at [0, 16]` warning applies to wide ranges. At `[0, 16]` the hypothesis count
  equals the baseline's 17.

**P8. Open one search per tree.** This is the owner's act. It writes only that tree's new `prng_analysis.db`.
```bash
for T in $TA $TB; do
  P=""
  if cd "$T" && P="$(python3 -c "import json; print(json.dumps(json.load(open('logs/stage4/params.json'))))")" && [ -n "$P" ]; then
    PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-open \
      --search-range-start 2147483648 --search-range-end 2415919104 \
      --search-reason "<owner's words: skip-switch rig run $(basename $T), [2^31, 2^31+2^28), skip [0,16], Run 4 pins; owner 2026-11-22 09:46>" \
      --params "$P" 2>&1 | tee logs/stage4/search_open.log
    PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-status --params "$P" 2>&1 | tee logs/stage4/search_status.log
  else
    echo "$T: cd or params read failed - no search command run - STOP"; break
  fi
done
stat -c '%n %s %Y' $MAIN/prng_analysis.db     # must equal live_state_before.txt: the live database was not written
python3 - <<'EOF'
import sqlite3, json
for t, want in (("/home/michael/skipsw_run_off", None), ("/home/michael/skipsw_run_on", "trial_range")):
    con = sqlite3.connect(f"file:{t}/prng_analysis.db?mode=ro&immutable=1", uri=True)
    rows = con.execute("SELECT search_id, search_space_json FROM step1_search").fetchall(); con.close()
    assert len(rows) == 1, (t, len(rows))
    got = json.loads(rows[0][1]).get("constant_skip_delivery")
    print(t, rows[0][0], "constant_skip_delivery =", got, "OK" if got == want else "WRONG - STOP")
EOF
```
- **Pass:** `STEP1_SEARCH_OPENED s1s-…` in each tree. Status `"state": "gap"` and `"next_seed_start": 2147483648`.
  A's search carries no `constant_skip_delivery`; B's carries `trial_range`.
- The immutable read is safe here because no writer is running; P8 is the only writer and it has exited.

## 3. Run A (switch off), then Run B (switch on)

For arm `X` in `off` (A), then `on` (B). Set `T=$TA; ST=$SA; L=off` for A, and `T=$TB; ST=$SB; L=on` for B.

**R1. Pre-launch.** Repeat P0's process and halt checks; they must show nothing and no flag.
`ss -ltn | grep -E ':5600|:5700'` must print nothing. **Tell the owner the cards are about to work, and wait for the
go.**

**R2. Launch:**
```bash
nohup bash /home/michael/skipsw_launch_$L.sh > $T/logs/stage4/launcher.out 2>&1 &
echo $! > $T/logs/stage4/launcher.pid
```
- `launcher.out` names the WATCHER pid and its log: `watcher pid=<pid> -> logs/gate12_<STAMP>.log`.
- If it prints `STAGE 4 REFUSED`, `STAGE 4 ABORTED`, `GATE-12 ABORTED` or `COORDINATOR NEVER BOUND`, **stop.**
- **B only:** the WATCHER log must contain `[SKIP-DELIVERY] marking /home/michael/skipsw_run_on as the switched-on
  arm's output root`.

**R3. While it runs, read only:**
- the WATCHER log, with `grep` or `tail -n 40 <log>`; never `tail -f`, never piped;
- `free -m`.

No GPU reads. Yesterday's run took order 10–15 min; that is an estimate.

**R4. It has ended when `kill -0 <watcher pid>` fails.** Status and credit:
```bash
W=$(sed -n 's/.*watcher pid=[0-9]* -> //p' $T/logs/stage4/launcher.out)   # logs/gate12_<STAMP>.log
cd $T
/bin/grep -n "STEP1-SEARCH S7\] credited\|GENERATION PUBLISHED\|Prior rows\|S145 COVERAGE\] certified\|STEP1-SEARCH LINK\|NO CREDIT\|INCOMPLETE" $W
P="$(python3 -c "import json; print(json.dumps(json.load(open('logs/stage4/params.json'))))")"
PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-status --params "$P" 2>&1 | tee logs/stage4/search_status_after.log
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do ssh -n -o BatchMode=yes michael@$ip 'hostname; pgrep -af "[r]ange_miner_worker" || echo "  no workers"'; done
```
- **Pass, in order:**
  1. `[STEP1-SEARCH S7] credited: trials [0] completed every required phase over [2147483648, 2415919104]`;
  2. `[S172 D3.5][GENERATION PUBLISHED]` with `Prior rows:     0`;
  3. `[S145 COVERAGE] certified [2,147,483,648, 2,415,919,104)`;
  4. `[STEP1-SEARCH LINK]`.

  Then the status shows the search completed, and no worker remains on any rig.
- **B only:** `test -e $TB/.s172_accumulator/skip_delivery_arm && cat $TB/.s172_accumulator/skip_delivery_arm` must
  print `trial_range`. The same test on `$TA` must fail.

**R5. Receipt, immediately** (extractor step 1, read-only on `$ST` and `$T`):
```bash
mkdir -p $EV
cd $MAIN && python3 -u scripts/extract_miner_bundle.py --staging-dir $ST --output-root $T \
  --params-json $T/logs/stage4/params.json --write-receipt $EV/receipt_$L.json \
  2>&1 | tee $EV/receipt_$L.log; echo "rc=${PIPESTATUS[0]}"
sha256sum $EV/receipt_$L.json | tee $EV/receipt_$L.sha256
```

**R6. Copy the run's outputs out.**
```bash
R=$EV/run_$L; mkdir -p $R
cd $T && cp -a .s172_accumulator results optimal_window_config.json bidirectional_survivors.json logs $R/ 2>/dev/null
cp -p $T/prng_analysis.db* $R/
mkdir -p $R/staging && cp -p $ST/miner_ledger.db* $ST/threshold_provenance.json $R/staging/ 2>/dev/null
(cd $R && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $EV/run_$L.sha256
```

**R7. Extract the bundle** (extractor step 2):
```bash
cd $MAIN && python3 -u scripts/extract_miner_bundle.py --staging-dir $ST --output-root $T \
  --params-json $T/logs/stage4/params.json --out-dir $EV/bundle_$L --receipt $EV/receipt_$L.json \
  2>&1 | tee $EV/extract_$L.log; echo "rc=${PIPESTATUS[0]}"
```
- **B:** the extractor's B2b check reads the switch from B's search record. It refuses if the params and the record
  disagree, and records `constant_skip_range_requested` and `constant_skip_range_effective`.
- **Run B only after A's `$EV/bundle_off/bundle.json` exists.**

## 4. The comparisons (CPU)

**C0. The output-equality check** (write once; it lives outside the repository, as the launchers do):
```bash
cat > /home/michael/skipsw_outputs_equal.py <<'EOF'
"""Output-only equality of two RANGE-MINER bundles, plus an exact statement of how their
inputs differ. NOT the engine comparator's verdict (that one refuses differing inputs)."""
import json, sys
import numpy as np
a_dir, b_dir, expect = sys.argv[1], sys.argv[2], set(filter(None, sys.argv[3].split(",")))
A, B = (json.load(open(f"{d}/bundle.json")) for d in (a_dir, b_dir))
ok = True
diff_in = {k for k in set(A["inputs"]) | set(B["inputs"]) if A["inputs"].get(k, "<absent>") != B["inputs"].get(k, "<absent>")}
print("inputs that differ:", sorted(diff_in), "expected:", sorted(expect))
ok &= diff_in == expect
for k in ("phases", "bidirectional", "phase_counts", "credit"):
    same = json.dumps(A.get(k), sort_keys=True) == json.dumps(B.get(k), sort_keys=True)
    print(f"{k:14s} {'EQUAL' if same else 'DIFFERENT'}  a={json.dumps(A.get(k), sort_keys=True)[:160]}")
    if not same:
        print(f"{'':14s}            b={json.dumps(B.get(k), sort_keys=True)[:160]}")
    ok &= same
with np.load(f"{a_dir}/{A['npz']}", allow_pickle=False) as x, np.load(f"{b_dir}/{B['npz']}", allow_pickle=False) as y:
    if sorted(x.files) != sorted(y.files):
        print("npz            DIFFERENT array names"); ok = False
    else:
        bad = [n for n in x.files if x[n].dtype != y[n].dtype or x[n].shape != y[n].shape or not np.array_equal(x[n], y[n])]
        print(f"npz            {'EQUAL' if not bad else 'DIFFERENT ' + str(bad)}  ({len(x.files)} arrays, rows={len(x[x.files[0]]) if x.files else 0})")
        ok &= not bad
for d, bb in ((a_dir, A), (b_dir, B)):
    prov = json.dumps(bb.get("provenance"), sort_keys=True)
    eff = [s for s in ("constant_skip_range_effective", "constant_skip_range_requested") if s in prov]
    print(d, "artifact", bb.get("artifact_sha256"), "skip evidence keys:", eff)
print("OUTPUTS EQUAL" if ok else "NOT EQUAL"); sys.exit(0 if ok else 1)
EOF
```

**C1. A = yesterday.**
```bash
cd $MAIN
python3 -u scripts/compare_engine_outputs.py $Y/bundle.json $EV/bundle_off/bundle.json 2>&1 | tee $EV/compare_off_vs_yesterday.txt; echo "comparator rc=${PIPESTATUS[0]}"   # expect 2: only run.repository_commit differs
python3 -u /home/michael/skipsw_outputs_equal.py $Y $EV/bundle_off run.repository_commit 2>&1 | tee $EV/equal_off_vs_yesterday.txt; echo "rc=${PIPESTATUS[0]}"   # 0 = OUTPUTS EQUAL
```
- **Pass:** the comparator's rc is 2, and its only DIFFERENT input row is `run.repository_commit`. The output check
  gives rc 0:
  - phase counts `0 / 0 / 13 / 2995`;
  - `bidirectional` null on both sides;
  - credit equal;
  - the NPZ bit-equal, 0 rows.

**C2. B = A.**
```bash
python3 -u scripts/compare_engine_outputs.py $EV/bundle_off/bundle.json $EV/bundle_on/bundle.json 2>&1 | tee $EV/compare_on_vs_off.txt; echo "comparator rc=${PIPESTATUS[0]}"   # expect 2: only param.deliver_skip_range differs
python3 -u /home/michael/skipsw_outputs_equal.py $EV/bundle_off $EV/bundle_on param.deliver_skip_range 2>&1 | tee $EV/equal_on_vs_off.txt; echo "rc=${PIPESTATUS[0]}"   # 0 = OUTPUTS EQUAL
```
- **Pass:** the comparator's only DIFFERENT input row is `param.deliver_skip_range`. The output check gives rc 0.
- **B's bundle must show the switch:** its provenance carries `constant_skip_range_requested` `[0, 16]` and an effective
  range from the search record.

**C3. The live state is unchanged.**
```bash
diff <({ stat -c '%n %s %Y' $MAIN/prng_analysis.db; readlink $MAIN/.s172_accumulator/current; }) $EV/live_state_before.txt && echo "LIVE UNCHANGED"
```

## 5. Stop conditions (any one: stop, report, do not work around)

The runbook's §9, plus:
1. Any step's rc ≠ 0 where 0 is required. This includes provisioning, a cleantree or launcher gate, the volume gate
   for B, the search open, or either extractor.
2. `Prior rows` ≠ 0 at publication.
3. No `credited`, or no `GENERATION PUBLISHED`.
4. A rig loses cards or stops answering, or no new WATCHER log line for 20 min. **Never SIGTERM a live coordinator.**
5. `free -m` available below 2 GiB.
6. A's search carries `constant_skip_delivery`, or B's lacks it. B's tree has no `skip_delivery_arm` marker, or A's
   has one.
7. Any differing output in C1 or C2. **That is a finding:** report it with the localized fields, and follow
   lead-handling before calling it noise.
8. C3 shows the live database or the `current` pointer changed.

## 6. Not verified by this session

- **Rig and fleet state.** All three rigs up on Proxmox is your report. Gate12 parity 30/30 at `00e2ee7`
  (`data/gate12_evidence/parity_00e2ee7_20261122.json`) is your report; this session did not read that file.
- **That `00e2ee7` holds exactly the worktree files.** This session did not read the main tree after the landing.
- **The derived launchers were not run.** Only their derivation is specified. `bash -n` and the 2-line diff are the
  lead's checks in P5.
- **The output-equality check has been tried only on yesterday's bundles.** This session ran it read-only on two
  inputs:
  - `$Y $Y ''`: `OUTPUTS EQUAL`, rc 0. All 22 NPZ arrays equal, 0 rows.
  - **Control:** yesterday's miner bundle against yesterday's PWC bundle: `NOT EQUAL`, rc 1. The PWC bundle carries
    seed lists and distinct counts where the miner bundle has `null`.

  **It is stricter than the comparator:** `null` against a value counts as DIFFERENT, where the comparator says NOT
  COMPARED. For two RANGE-MINER bundles both sides are `null`, so this is harmless here. It has not run on a real A/B
  pair.
- **Where B's skip evidence appears in the bundle.** That B's bundle carries `constant_skip_range_*` is read from the
  extractor source (`scripts/extract_miner_bundle.py:561-562`, `:720`), not observed on a real switched run.
- **The volume gate at `[0, 16]` with the switch on** has not been run. Its verdict is unknown until P7.
- **Rig-side side effects beyond yesterday's.** That arm B adds no rig-side behaviour beyond `skip_range` in the
  constant payloads is from the code and the CPU gates. It has not been observed on a rig.
- **Whether a resumed or repeated arm behaves.** Each arm assumes one clean attempt. A failed attempt leaves its tree's
  search and staging in use, and a retry needs the owner's call (runbook §8 on searches).
