> **RE-SAVED COPY after the 2026-11-22 12:26 Zeus crash, which wiped `/tmp`. EXACT, not reconstructed.**
> The original was `/tmp/skipsw_runbook_c.md`, used for run C, and by sed for run D. Everything below this header block is byte-for-byte the original:
> sha256 `f90eb1bd525224f8b939e94a7eb913539746a9b81ee36af9ab051a242b6e349a`. It was rebuilt by replaying this session's own Write and edit calls from the session
> transcript, and the result equals the hash recorded before the runbook was handed to the lead.
> To check: strip these 5 header lines and the blank line after them, then run sha256sum.

# Skip-switch rig run C (switch on, skip [0, 32]): runbook for the Alpha lead, arm C only

Owner 2026-11-22 10:30: "run c at your descretion".
- **Run C** is arm B with `warm_start_skip_max` 32 instead of 16. The switch is on. The interval, family, pins,
  thresholds, pool and stripe size are otherwise unchanged.
- **Who runs it:** the lead. This session did not touch a rig and read nothing outside `/tmp` except read-only.
- **Same order and blocks as `/tmp/skipsw_runbook.md`.**
- **No block touches arm A's or arm B's tree or staging, and none loops over arms.** The only reads of another arm
  are of A's frozen evidence (`$EV/bundle_off`) in §4, for the comparison.

## 0. What the code says about C (read before deciding to run)

### 0.1 `[0, 32]` passes every validator and sits inside the tracked bounds (code reading, `00e2ee7`)

| Check | Rule | `[0, 32]` |
|---|---|---|
| The one validator, `validate_skip_delivery` (CLI, search open/S4, child, estimator, coordinator builder and `run_trial_miner`) | plain ints, `0 ≤ skip_min ≤ skip_max ≤ 255` | pass |
| `presented_configuration` (search open) | a fresh, all-seven-pins, `window_trials` 1 run does not sample (`skip_delivery_can_sample` false, no `study_name`), so only the pins are checked | pass |
| Child exact pin (`_step1_skip_delivery_exact_pin`) | same rule; pin validated, then handed to the study body as single-point distributions | `[0, 32]` delivered |
| Tracked sampling bounds (`distributed_config.json` `search_bounds`: `skip_min [0,10]`, `skip_max [10,250]`) | not applied to a non-sampling pinned run. The pin lies inside them anyway: `0 ∈ [0,10]`, `32 ∈ [10,250]` | inside |
| `best_skips` dtype (`unsigned char`, `prng_registry.py` `JAVA_LCG_KERNEL`) | best skip ≤ 255 | 32 fits |
| Survivor-volume gate (P7) | estimator geometry validated, then projection against the 50,000-per-phase cap | **verdict unknown until P7 runs** |

### 0.2 What phases 1–2 do differently at 33 skip hypotheses against 17

From the code:
- **The kernel** (`JAVA_LCG_KERNEL`, `java_lcg_flexible_sieve`) loops `for skip in [skip_min, skip_max]` for every
  seed. Each skip costs `offset + k + skip·(k+1)` LCG steps, which is `12 + 13·skip` at `k = 12` and offset 0. That is
  the formula in X-SKIP-F1 R1 Q4, and it reproduces that review's 33.1× for `[6, 99]`.
  - `[0, 16]`: 1,972 steps per seed. `[0, 32]`: 7,260 steps per seed. **3.68× the constant-phase work per seed**,
    with 33/17 = 1.94× the hypotheses.
  - The reverse kernel (phase 2) has the same loop shape [UNVERIFIED for the reverse source specifically; read from
    the forward kernel only].
- **Phases 3–4 (hybrid) are unchanged.** The hybrid kernel takes no skip bound, and the coordinator's phase 3–4
  payloads are byte-identical with the switch on or off (gate `G-ON-PAYLOAD`; `docs/CHAPTER_2_SOURCE_MAP_v1.md:150-151`).
- **Stripe geometry is unchanged:** 4 stripes per phase of 2^26 seeds, 8 sub-stripes, the AMD constant seed cap of
  2,000,000 per launch (`CoordinatorConfig`). Each launch does about 3.68× the work; the number of launches is the
  same.
- **Lease and timeouts** (`miner/range_miner_coordinator.py` `CoordinatorConfig`):
  - `compute_lease_timeout` is 300 s. The worker heartbeat (every 30 s, `miner/range_miner_worker.py`
    `heartbeat_interval`) renews it while a stripe is claimed.
  - `staging_timeout` is 600 s.
  - Measured in arm B (`[0, 16]`, `~/skipsw_run_on/logs/gate12_20261122_100717.log`, read-only):
    - phase 1 stripes claimed at 10:09:06.2, first results about 6 s after claim, stage drained by 10:09:16;
    - phase 2 claimed at 10:09:17.6, results by about 10:09:21.
  - **Projection** (×3.68, a model, not a measurement): about 22 s per phase-1 stripe and about 15 s per phase-2
    stripe. That is well under one 300 s lease without any renewal, and under one 30 s heartbeat interval. The whole
    run should be roughly 1 minute longer than B's 4 minutes.
- **Unmeasured:**
  - per-launch kernel time on the RX 6600 XT at `[0, 32]`;
  - whether any rig-side watchdog or driver timeout applies to a longer single launch (none is configured in the
    repository that this session found);
  - heartbeat behaviour during a longer kernel;
  - phase 1–2 survivor volume at `[0, 32]` (P7 projects it; nothing has measured it).

### 0.3 Which outputs may differ from A, and which must not

The kernel keeps a seed if its **best** match rate over the skip range is at least the threshold. `[0, 16] ⊂ [0, 32]`,
so every seed that survives a constant phase in A also survives it in C. C's constant survivor sets are supersets of
A's.

| Field | C vs A | Why |
|---|---|---|
| `inputs` | **must differ in exactly** `param.deliver_skip_range` and `param.warm_start_skip_max` (16 → 32) | the job |
| `phase_counts` 1, 2 | **may differ, only upward** (C ≥ A) | more skip hypotheses; the superset rule |
| `phases` 1, 2 seed lists and match rates | **may differ.** A's seeds ⊆ C's when both are recorded. A seed's best rate may rise and its best skip may change | the same |
| `bidirectional.constant` | **may grow** (A ⊆ C) | the same |
| NPZ rows, `artifact_sha256` | **may differ, only by added constant rows.** Their `skip_min`/`skip_max`/`skip_range` stamps carry the trial's `0/32/32` | NPZ = bidirectional rows; trial stamps |
| `phase_counts` 3, 4 (`13`, `2995`), `phases` 3, 4, `bidirectional.variable` | **must equal A** | the hybrid path takes no skip bound |
| `credit` (granted, certified, linked and swept intervals) | **must equal A** | same interval, same phases |

**A modelled expectation, not a decision rule.** For a non-generating seed, a 9-of-12 match at p = 1/1000 per
position is about 2·10⁻²⁵ per skip hypothesis. Over 33 hypotheses and 2^28 seeds that is about 2·10⁻¹⁵ expected
phase-1 survivors, so C's phase 1–2 counts would be expected to stay 0, like A's.
- **Inputs assumed:** independent uniform outputs, and a threshold of 0.71 meaning at least 9 of 12 matches.
- **Measured:** none.
- **If C shows constant survivors,** that is a result to report with its seeds and best skips, and to follow under
  lead-handling. It is not to be explained away by this model.

## 1. Fixed values (arm C only)

```bash
MAIN=/home/michael/distributed_prng_analysis
COMMIT=00e2ee7
TC=/home/michael/skipsw_run_on32          # arm C run tree (new)
SC=/home/michael/skipsw_staging_on32      # arm C staging (new, empty; never ~/miner_staging)
LC=/home/michael/skipsw_launch_on32.sh    # arm C launcher (new)
EV=$MAIN/logs/skipsw_session              # shared evidence dir; C's files are all suffixed _on32
AREF=$EV/bundle_off                       # arm A's frozen bundle, READ ONLY
S=2147483648; E=2415919104
```

**Arm C params (exact):**
```json
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 0, "warm_start_skip_max": 32, "deliver_skip_range": true, "staging_dir": "/home/michael/skipsw_staging_on32"}
```

## 2. Preparation (no GPU, no worker)

**P0. Nothing running; the live state is the same as before A.**
```bash
source ~/venvs/torch/bin/activate
ip -4 addr show | grep -E 'inet 192\.168\.3\.177'
pgrep -af "[w]atcher_agent|[w]indow_optimizer|[r]ange_miner_worker|[p]wc_worker_service|[s]ieve_gpu_worker"   # must print nothing
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  ssh -n -o BatchMode=yes michael@$ip 'hostname; pgrep -af "[r]ange_miner_worker|[p]wc_worker_service|[s]ieve_gpu_worker" || echo "  no workers"'
done
cd $MAIN && git rev-parse HEAD && test -z "$(git status --porcelain)" && echo "MAIN CLEAN"
HALT=$(cd $MAIN && python3 -c "from agents.safety.kill_switch import KillSwitch; print(KillSwitch().halt_file_path)")
test ! -e "$HALT" && echo "NO HALT FLAG ($HALT)"
diff <({ stat -c '%n %s %Y' $MAIN/prng_analysis.db; readlink $MAIN/.s172_accumulator/current; }) $EV/live_state_before.txt && echo "LIVE UNCHANGED SINCE A"
python3 -c "import json;print(json.load(open('$MAIN/daily3_current.json'))['sha256'])"   # must start 513648160d35
test ! -e $TC && test ! -e $SC && test ! -e $LC && echo "C PATHS FREE"
```
- `MAIN CLEAN` must print. If the A/B landing left the main tree's git status dirty, stop; `logs/` is ignored.
- HEAD must be `00e2ee7`, or a later docs-only commit. If it isn't, stop and ask.

**P1. The run tree and staging.** This writes `.git/worktrees` metadata, so it needs the owner's OK.
```bash
cd $MAIN
git worktree add --detach $TC $COMMIT
mkdir -p $TC/optuna_studies $TC/logs/stage4
mkdir -p $SC && test -z "$(ls -A $SC)" && echo "STAGING EMPTY"
```

**P2. Ignored inputs.**
```bash
VER=$(python3 -c "import json;print(json.load(open('$MAIN/daily3_current.json'))['filename'])")
IGN="daily3_current.json $VER daily3.json dataset_provisioning.json trse_context.json schemas/field_mappings.json"
for f in $IGN; do cp -p "$MAIN/$f" "$TC/$f"; done
diff <(cd $MAIN && sha256sum $IGN) <(cd $TC && sha256sum $IGN) && echo "$TC inputs identical"
(cd $TC && test -z "$(git status --porcelain)" && echo "$TC CLEAN")
```

**P3. Provision the dataset from this tree.** A rig write, needing the owner's OK. One tree, no loop.
```bash
(
cd "$TC" || { echo "cannot enter $TC - STOP"; exit 1; }
rm -f logs/stage4/provision.ok logs/stage4/provision.log
PYTHONPATH=. python3 -u scripts/provision_dataset_fleet.py 2>&1 | tee logs/stage4/provision.log
rc=("${PIPESTATUS[@]}")
echo "provisioner rc=${rc[0]} log-writer rc=${rc[1]}"
h=""
if [ "${rc[0]}" -eq 0 ] && [ "${rc[1]}" -eq 0 ] && [ -s logs/stage4/provision.log ] \
   && h=$(sha256sum logs/stage4/provision.log | cut -d' ' -f1) && [ -n "$h" ] \
   && echo "PROVISION_OK $(date -u +%FT%TZ) log_sha256=$h" > logs/stage4/provision.ok \
   && /bin/grep -q "log_sha256=$h" logs/stage4/provision.ok; then
  echo "$TC: provisioned, receipt names log sha256 $h"; exit 0
fi
rm -f logs/stage4/provision.ok; echo "$TC: provisioning NOT confirmed - STOP"; exit 1
)
echo "provision status=$?"                         # must be 0
```

**P4. Params.**
```bash
cat > $TC/logs/stage4/params.json <<'EOF'
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 0, "warm_start_skip_max": 32, "deliver_skip_range": true, "staging_dir": "/home/michael/skipsw_staging_on32"}
EOF
python3 - <<'EOF'
import json
c = json.load(open("/home/michael/skipsw_run_on32/logs/stage4/params.json"))
a = json.load(open("/home/michael/distributed_prng_analysis/logs/skipsw_session/run_off/logs/stage4/params.json"))   # A's frozen copy, read only
diff = {k for k in set(a) | set(c) if a.get(k, "<absent>") != c.get(k, "<absent>")}
assert diff == {"deliver_skip_range", "warm_start_skip_max", "staging_dir"}, diff
assert c["deliver_skip_range"] is True and c["warm_start_skip_min"] == 0 and c["warm_start_skip_max"] == 32
print("params ok: C = A + deliver_skip_range true, skip_max 32 (staging aside)")
EOF
```
- If `run_off/logs/stage4/params.json` is not in A's evidence copy (R6 copied `logs`), compare by eye against
  `/tmp/skipsw_runbook.md` §1 instead. Do not read A's tree.

**P5. The launcher.**
```bash
test "$(sha256sum /home/michael/stage4_miner_launch.sh | cut -d' ' -f1)" = fe964b04fe3ea431986835ff2fff934349864751f8974da5ccebb6b1a62bf5d3 && echo "base launcher ok"
sed 's#^cd /home/michael/stage4_run_miner || exit 1$#cd /home/michael/skipsw_run_on32 || exit 1#' /home/michael/stage4_miner_launch.sh > $LC
diff /home/michael/stage4_miner_launch.sh $LC | /bin/grep -c '^[<>]'     # must be 2
chmod +x $LC && bash -n $LC && echo "C launcher syntax ok"
```

**P6. Dry clean-tree check.**
```bash
cd $TC && python3 -u scripts/gate12_cleantree_gate.py --phase admission 2>&1 | tee logs/stage4/cleantree_P6.log; echo "rc=${PIPESTATUS[0]}"
```

**P7. The survivor-volume gate, at C's geometry.** It runs on VM101's GPU, which must be idle.
```bash
cd $TC
cat > logs/stage4/volume_params.json <<'EOF'
{"seed_start": 2147483648, "seed_end_exclusive": 2415919104, "max_seeds": 268435456, "prng_type": "java_lcg", "window_size": 12, "window_anchor": 25, "generator_phase": 0, "sessions": ["midday"], "forward_threshold": 0.71, "reverse_threshold": 0.47, "skip_min": 0, "skip_max": 32, "dataset": "/home/michael/skipsw_run_on32/daily3-20260801T145551443433Z-513648160d35.json", "test_both_modes": true, "deliver_skip_range": true}
EOF
python3 -u scripts/survivor_volume_gate.py --params @logs/stage4/volume_params.json \
  --record-out logs/stage4/volume_gate_record.json 2>&1 | tee logs/stage4/volume_gate.log
echo "rc=${PIPESTATUS[0]}"     # 0 = PROCEED; anything else: stop, do not launch C
/bin/grep -n "phase [12] java_lcg\|PROCEED\|REFUSE" logs/stage4/volume_gate.log
#   B's gate at [0, 16] (~/skipsw_run_on/logs/stage4/volume_gate.log:6-7): phase 1 hits 0 / 67108864, 3.7s; phase 2 0, 3.4s.
#   The ratio of C's phase-1/2 seconds to those is the FIRST MEASURED [0,32]/[0,16] kernel-time factor (VM101 3080 Ti, not a rig).
```

**P8. Open C's search.** This is the owner's act. It writes only `$TC/prng_analysis.db`.
```bash
cd "$TC" || { echo "cannot enter $TC - STOP"; exit 1; }
P="$(python3 -c "import json; print(json.dumps(json.load(open('logs/stage4/params.json'))))")" && [ -n "$P" ] || { echo "params read failed - STOP"; exit 1; }
PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-open \
  --search-range-start 2147483648 --search-range-end 2415919104 \
  --search-reason "<owner's words: skip-switch rig run C, skipsw_run_on32, [2^31, 2^31+2^28), skip [0,32], Run 4 pins; owner 2026-11-22 10:30>" \
  --params "$P" 2>&1 | tee logs/stage4/search_open.log
PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-status --params "$P" 2>&1 | tee logs/stage4/search_status.log
stat -c '%n %s %Y' $MAIN/prng_analysis.db      # must equal live_state_before.txt
python3 - <<'EOF'
import sqlite3, json
con = sqlite3.connect("file:/home/michael/skipsw_run_on32/prng_analysis.db?mode=ro&immutable=1", uri=True)
rows = con.execute("SELECT search_id, search_space_json FROM step1_search").fetchall(); con.close()
assert len(rows) == 1, len(rows)
sp = json.loads(rows[0][1])
print(rows[0][0], "constant_skip_delivery =", sp.get("constant_skip_delivery"),
      "pins skip =", (sp.get("pins") or {}).get("warm_start_skip_min"), (sp.get("pins") or {}).get("warm_start_skip_max"))
assert sp.get("constant_skip_delivery") == "trial_range", "C's search lacks the switch - STOP"
EOF
```
- **Pass:** `STEP1_SEARCH_OPENED`; status `gap` at `2147483648`; `constant_skip_delivery = trial_range`; pins `0 32`.

## 3. Run C

**R1.** Repeat P0's process, halt and port checks: `ss -ltn | grep -E ':5600|:5700'` must print nothing. **Tell the
owner the cards are about to work, and wait for the go.**

**R2. Launch:**
```bash
nohup bash $LC > $TC/logs/stage4/launcher.out 2>&1 &
echo $! > $TC/logs/stage4/launcher.pid
```
- **Expected** in the WATCHER log, in this order:
  - `[SKIP-DELIVERY] marking /home/michael/skipsw_run_on32 as the switched-on arm's output root`;
  - `[SKIP-DELIVERY] ON: phases 1-2 sieve skip [0, 32] (baseline [0, 16])`;
  - `[SKIP-DELIVERY] 33 skip hypotheses per seed against the baseline's 17 (1.94x); …`. This is a WARNING printed by
    the coordinator whenever the count exceeds 17 (`miner/range_miner_coordinator.py:12571-12576`). It is expected,
    not a stop.
- If the second line says `[0, 16]`, or is missing, **stop**: the pin did not reach the coordinator.

**R3. While it runs, read only:**
- the WATCHER log, with `grep` or `tail -n 40`; never `tail -f`;
- `free -m`.

Expected duration is about 5 minutes, which is a projection (§0.2). For the phase 1–2 stages, note in the log:
- the `[S172-BP] burst_exact stage=0` and `stage=1` timestamps;
- the time of each stage's last `[H1H2] stripe_rx`;
- the `active_stripes` count returning to 0.

Those are the first measurement of `[0, 32]` per-stripe time.

**R4. It has ended when `kill -0 <watcher pid>` fails.**
```bash
cd $TC
W=$(sed -n 's/.*watcher pid=[0-9]* -> //p' logs/stage4/launcher.out)
/bin/grep -n "SKIP-DELIVERY\|STEP1-SEARCH S7\] credited\|GENERATION PUBLISHED\|Prior rows\|S145 COVERAGE\] certified\|STEP1-SEARCH LINK\|NO CREDIT\|INCOMPLETE\|lease expir\|LEASE_EXPIRED\|requeue" $W
/bin/grep -n "burst_exact stage=[01]\|active_stripes" $W | head -20     # phase 1-2 timing (§0.2 measurement)
P="$(python3 -c "import json; print(json.dumps(json.load(open('logs/stage4/params.json'))))")"
PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-status --params "$P" 2>&1 | tee logs/stage4/search_status_after.log
cat $TC/.s172_accumulator/skip_delivery_arm                     # must print trial_range
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do ssh -n -o BatchMode=yes michael@$ip 'hostname; pgrep -af "[r]ange_miner_worker" || echo "  no workers"'; done
```
- **Pass:** credited over `[2147483648, 2415919104]`, then `GENERATION PUBLISHED` with `Prior rows: 0`, then
  `S145 COVERAGE certified`, then `STEP1-SEARCH LINK`. There must be no lease expiry and no requeue. The marker reads
  `trial_range`, and no worker remains.

**R5. Receipt:**
```bash
cd $MAIN && python3 -u scripts/extract_miner_bundle.py --staging-dir $SC --output-root $TC \
  --params-json $TC/logs/stage4/params.json --write-receipt $EV/receipt_on32.json \
  2>&1 | tee $EV/receipt_on32.log; echo "rc=${PIPESTATUS[0]}"
sha256sum $EV/receipt_on32.json | tee $EV/receipt_on32.sha256
```

**R6. Copy C's outputs out:**
```bash
R=$EV/run_on32; mkdir -p $R
cd $TC && cp -a .s172_accumulator results optimal_window_config.json bidirectional_survivors.json logs $R/ 2>/dev/null
cp -p $TC/prng_analysis.db* $R/
mkdir -p $R/staging && cp -p $SC/miner_ledger.db* $SC/threshold_provenance.json $R/staging/ 2>/dev/null
(cd $R && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $EV/run_on32.sha256
```

**R7. Extract C's bundle:**
```bash
cd $MAIN && python3 -u scripts/extract_miner_bundle.py --staging-dir $SC --output-root $TC \
  --params-json $TC/logs/stage4/params.json --out-dir $EV/bundle_on32 --receipt $EV/receipt_on32.json \
  2>&1 | tee $EV/extract_on32.log; echo "rc=${PIPESTATUS[0]}"
```

## 4. C against A (CPU, read-only on A's evidence)

**C0. The C-vs-A check.** It applies §0.3's rules: must-equal fields strictly, may-differ fields only upward or as
supersets. It is not the comparator's verdict.
```bash
cat > /home/michael/skipsw_outputs_c_vs_a.py <<'EOF'
"""Arm C (skip [0,32]) against arm A (switch off): runbook_c §0.3. Exit 0 = every rule holds."""
import json, sys
import numpy as np
a_dir, c_dir = sys.argv[1], sys.argv[2]
A, C = (json.load(open(f"{d}/bundle.json")) for d in (a_dir, c_dir))
ok = True
def say(rule, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {rule}  {detail}")
d_in = {k for k in set(A["inputs"]) | set(C["inputs"]) if A["inputs"].get(k, "<absent>") != C["inputs"].get(k, "<absent>")}
say("inputs differ exactly in deliver_skip_range + warm_start_skip_max",
    d_in == {"param.deliver_skip_range", "param.warm_start_skip_max"}, sorted(d_in))
say("C warm_start_skip_max is 32", C["inputs"].get("param.warm_start_skip_max") == 32)
j = lambda x: json.dumps(x, sort_keys=True)
for ph in ("3", "4"):
    say(f"phase_counts.{ph} equal", j(A["phase_counts"].get(ph)) == j(C["phase_counts"].get(ph)),
        f"A={A['phase_counts'].get(ph)} C={C['phase_counts'].get(ph)}")
    say(f"phases.{ph} equal", j(A["phases"].get(ph)) == j(C["phases"].get(ph)))
say("bidirectional.variable equal", j(A["bidirectional"].get("variable")) == j(C["bidirectional"].get("variable")))
say("credit equal", j(A["credit"]) == j(C["credit"]))
for ph in ("1", "2"):
    ra, rc = A["phase_counts"][ph]["survivor_rows"], C["phase_counts"][ph]["survivor_rows"]
    say(f"phase_counts.{ph} C >= A (superset rule)", rc >= ra, f"A={ra} C={rc}" + ("  <-- C FOUND MORE: a result, report it" if rc > ra else ""))
    pa, pc = A["phases"].get(ph), C["phases"].get(ph)
    if pa is not None and pc is not None:
        say(f"phases.{ph} seeds A subset of C", set(pa["seeds"]) <= set(pc["seeds"]))
    else:
        print(f"note  phases.{ph} seed lists not recorded (null) in one bundle; superset checked by count only")
ba, bc = A["bidirectional"].get("constant"), C["bidirectional"].get("constant")
if ba is not None and bc is not None:
    say("bidirectional.constant A subset of C", set(ba) <= set(bc))
with np.load(f"{a_dir}/{A['npz']}", allow_pickle=False) as x, np.load(f"{c_dir}/{C['npz']}", allow_pickle=False) as y:
    na = len(x[x.files[0]]) if x.files else 0
    nc = len(y[y.files[0]]) if y.files else 0
    say("npz array names equal", sorted(x.files) == sorted(y.files))
    say("npz rows C >= A", nc >= na, f"A={na} C={nc}")
    if na == nc == 0:
        bad = [n for n in x.files if x[n].dtype != y[n].dtype or x[n].shape != y[n].shape or not np.array_equal(x[n], y[n])]
        say("npz bit-equal (both 0 rows)", not bad, bad)
    elif nc > na:
        print("note  C carries extra NPZ rows: list them (seeds, best skips) in the report; lead-handling applies")
prov = j(C.get("provenance"))
print("C provenance skip evidence:", [k for k in ("constant_skip_range_requested", "constant_skip_range_effective") if k in prov])
print("artifact A", A.get("artifact_sha256"), "C", C.get("artifact_sha256"))
print("ALL RULES HOLD" if ok else "A RULE FAILED"); sys.exit(0 if ok else 1)
EOF
python3 -I /home/michael/skipsw_outputs_c_vs_a.py $AREF $AREF 2>&1 | tail -3     # self-test: must FAIL only the two input rules
```
- **Self-test:** A against itself must FAIL exactly two rules, `inputs differ exactly…` and `C warm_start_skip_max is
  32`, and pass every other rule.

**C1. Compare C with A.**
```bash
cd $MAIN
python3 -u scripts/compare_engine_outputs.py $AREF/bundle.json $EV/bundle_on32/bundle.json 2>&1 | tee $EV/compare_on32_vs_off.txt; echo "comparator rc=${PIPESTATUS[0]}"
#   expect rc 2 (NOT COMPARABLE): its DIFFERENT input rows are exactly param.deliver_skip_range and param.warm_start_skip_max
python3 -I /home/michael/skipsw_outputs_c_vs_a.py $AREF $EV/bundle_on32 2>&1 | tee $EV/rules_on32_vs_off.txt; echo "rc=${PIPESTATUS[0]}"   # 0 = ALL RULES HOLD
```

**C2. The live state is unchanged.**
```bash
diff <({ stat -c '%n %s %Y' $MAIN/prng_analysis.db; readlink $MAIN/.s172_accumulator/current; }) $EV/live_state_before.txt && echo "LIVE UNCHANGED"
```

## 5. Stop conditions (any one: stop, report, do not work around)

1. Any required rc ≠ 0: provisioning, the cleantree check, the launcher's gates, the P7 volume gate, the search open,
   R5 or R7.
2. P0: `MAIN CLEAN`, `LIVE UNCHANGED SINCE A` or `C PATHS FREE` fails, or HEAD is not `00e2ee7`, or a later docs-only
   commit.
3. C's search lacks `constant_skip_delivery = trial_range`, or its pins are not `0 32`.
4. The WATCHER log shows `phases 1-2 sieve skip` with anything other than `[0, 32]`, or no `marking … skipsw_run_on32`.
5. Any lease expiry, requeue or retry in phases 1–2. **This is the first measurement at 33 hypotheses:** report the
   stripe timings even if the run completes.
6. `Prior rows` ≠ 0; no `credited`; no `GENERATION PUBLISHED`.
7. A rig loses cards or stops answering, or there is no WATCHER log line for 20 min. **Never SIGTERM a live
   coordinator.**
8. `free -m` available below 2 GiB.
9. C1 `A RULE FAILED`. Report the failing rule with both values.
   - **Phases 3–4, `bidirectional.variable` or `credit` differing from A** contradicts the code reading (§0.3). That
     is a finding.
   - **C finding more phase 1–2 survivors than A** is not a failure, and the script passes it. It is a result: report
     the seeds and best skips, and follow lead-handling. The background estimate in §0.3 does not close it.
10. C2 shows the live database or `current` changed.

## 6. Not verified by this session

- **No rig state or run was observed for C.** `[0, 32]` per-stripe time, heartbeat behaviour during the longer
  kernels, and any driver-side timeout are unmeasured. §0.2's ~22 s / ~15 s per stripe is a ×3.68 projection from
  B's log.
- **The phase-2 (reverse) kernel's loop shape** was not read; the 3.68× figure is from the forward kernel.
- **P7's verdict at `[0, 32]`** is unknown. At `[0, 16]`, B's gate gave PROCEED, projecting phase 1–2 at 0 (95% upper
  14.8), phase 3 at 24 and phase 4 at 2995.
- **The C-vs-A script has not run on a real C bundle.** This session ran its self-test on A's frozen bundle
  (read-only). It gave rc 1, failing exactly the two input rules and passing every other rule, including the 0-row
  NPZ bit-equality, as specified.
- **`run_off/logs/stage4/params.json` in A's evidence copy:** present (487 bytes, 09:59), checked read-only by this
  session.
- **Whether a longer single launch interacts with the RX 6600 XT PCIe-x1 links** that showed errors on 2026-11-19
  (`docs/TFM_SESSION_HANDOFF_20261119.md:63-64`). It is not known from the code.
