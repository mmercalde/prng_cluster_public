# CLAUDE CODE REPORT — TWO PRE-RERUN ITEMS: TRUTHFUL GPU PROBE + POST-F1 SAMPLER

**Host:** VM101, `~/distributed_prng_analysis`. **Date:** 2026-08-09.
**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_PRERUN_PROBE_AND_SAMPLER.md`.

**Constraints honoured:** no commit, no push, **no pipeline launch, no fleet launch, no port
5700 bind**, `worker_pool_size = 25` **written into the launch script but NOT applied — the
script was not run**. Gate 12 remains HELD. No coordinator, miner, ledger, seed-domain/coverage
or certified-suite file was modified. Read-only SSH probes only (queries; no fleet work).

---

## 0. BASE VERIFICATION — one discrepancy to report

| item | brief says | measured | 
|---|---|---|
| HEAD | `d3f8f00` | **`c4e0037`** |
| tracked tree | clean | **clean** (only untracked entries; see §7) |
| `tests/test_s172_f1_f2_active_lease.py` | 16/16 | **16/16** |

> **⚠ `d3f8f00` DOES NOT EXIST in this repository.** `git cat-file -t d3f8f00` →
> `fatal: Not a valid object name`. The commit the brief describes — "F1/F2 certified and
> committed" — is **`c4e0037`** ("F1/F2 active-lease scheduler + terminal observability (+R1,
> +R2) — Beta CERTIFIED 2026-08-09"), which is HEAD and whose message matches the brief's
> description exactly. **All work below was performed against `c4e0037`.** Reported, not worked
> around: if `d3f8f00` is a hash from a different tree or a pre-amend value, that needs
> reconciling before the rerun is requested.

---

## 1. WHY `rocm-smi` WAS NOT FOUND UNDER `bash -lc` — measured, not inferred

### 1.1 The live per-rig evidence

**Reproducing preflight's exact invocation** (`ssh <host> bash -lc "rocm-smi 2>/dev/null |
grep -cE '^[0-9]+[[:space:]]' || echo 0"`), read-only, 2026-08-09:

| rig | rc | stdout | stderr |
|---|---|---|---|
| 192.168.3.122 | 0 | `'0\n0\n'` | `''` |
| 192.168.3.156 | 0 | `'0\n0\n'` | `''` |
| 192.168.3.164 | 0 | `'0\n0\n'` | `''` |

**The doubled zero is itself diagnostic.** `grep -c` printed `0` **and exited 1** (grep exits
non-zero when it counts nothing), so `|| echo 0` then fired and printed a *second* zero. The old
parser took "the last digit line" and got `0`. Two independent constructs each manufactured that
zero; either alone would have produced it.

### 1.2 The root cause

| probe | 192.168.3.122 | .156 | .164 |
|---|---|---|---|
| `bash -lc "echo $PATH"` | `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin` | identical | identical |
| bare non-interactive `echo $PATH` | **identical to the above** | identical | identical |
| `bash -lc "command -v rocm-smi"` | *(no output)* `rc=1` | `rc=1` | `rc=1` |
| `bash -lic "echo $PATH"` | same **+ `:/opt/rocm/bin`** | — | — |
| `ls -l /opt/rocm/bin/rocm-smi` | `-> ../libexec/rocm_smi/rocm_smi.py` | same | same |
| `readlink -f /opt/rocm` | `/opt/rocm-6.4.3` | same | same |

**The chain, verified on `.122`:**

1. `/opt/rocm/bin` is put on PATH by **`~/.bashrc:120`** (`export PATH=$PATH:/opt/rocm/bin`),
   alongside `ROCM_PATH`, `HIP_PATH`, `LD_LIBRARY_PATH` at `:118-121`.
2. **Nothing else adds it.** `grep -rl rocm /etc/profile.d/ /etc/profile /etc/environment`
   returns **nothing**; `/etc/profile.d/` holds only `01-locale-fix.sh` and
   `bash_completion.sh`. There is no `~/.bash_profile`.
3. **`~/.bashrc:5-8` is Ubuntu's stock non-interactive guard** —
   `# If not running interactively, don't do anything` / `case $- in *i*) ;; *) return;; esac` —
   which returns **~112 lines before** the PATH export.
4. `bash -l` sources `/etc/profile` and `~/.profile`, and `~/.profile` *does* source `~/.bashrc`
   — but `.bashrc` returns at the guard.

> **`bash -l` IS NOT THE REMEDY, and the brief was right to warn against assuming so.** The
> login shell and a bare non-interactive command observe the **byte-identical PATH**, neither
> containing `/opt/rocm/bin`. Only `bash -lic` sees it, and forcing an interactive shell over
> SSH is not a probe contract.

**Secondary finding — the argv was being flattened.** `subprocess` passed
`[..., host, "bash", "-lc", "<pipeline>"]`; **ssh joins trailing arguments with spaces and does
not re-quote them**, so the remote login shell received the pipeline with its quoting already
gone and re-parsed it. The semantics survived by luck. The new probe passes the remote command
as **one** argv element, which is parsed exactly once.

### 1.3 The remedy, verified live

`command -v rocm-smi` first (so a rig installing it elsewhere still works), falling back to the
verified absolute path. Run read-only against all three rigs:

| rig | rc | stdout | stderr |
|---|---|---|---|
| 192.168.3.122 | 0 | `TFM_PROBE_BIN=/opt/rocm/bin/rocm-smi` / `TFM_PROBE_STATUS=OK` / `TFM_PROBE_COUNT=8` | `''` |
| 192.168.3.156 | 0 | *(identical)* | `''` |
| 192.168.3.164 | 0 | *(identical)* | `''` |

**And through the production method itself** — `PreflightChecker.check_gpu_health()` against the
three CT addresses:

```
all_healthy: True
192.168.3.122: {status: OK, gpu_count: 8, expected: 8, binary: /opt/rocm/bin/rocm-smi, stderr: ""}
192.168.3.156: {status: OK, gpu_count: 8, expected: 8, binary: /opt/rocm/bin/rocm-smi, stderr: ""}
192.168.3.164: {status: OK, gpu_count: 8, expected: 8, binary: /opt/rocm/bin/rocm-smi, stderr: ""}
issues: []
```

**8/8 on all three — matching the cluster bot's independent report and contradicting the run's
`0/8`.**

---

## 2. THE THREE-OUTCOME PROBE AS BUILT — and gating confirmed unchanged

`preflight_check.py`, three module-level statuses that are never conflated:

| status | meaning | `gpu_count` |
|---|---|---|
| `OK` | the probe ran; the count is an observation | the integer, **including a genuine `0`** |
| `UNAVAILABLE` | the probe **could not run** | **`None` — never `0`** |
| `ERROR` | ran, output unparseable | `None` |

`UNAVAILABLE` reasons are distinguished, not merged: `binary_not_found`, `rocm_smi_exit_<rc>`,
`ssh_exit_<rc>`, `timeout`, `probe_exception:<type>`. Each per-node record carries `status`,
`gpu_count`, `expected`, `reason`, **`binary`** (which executable actually answered) and
**`stderr`**.

**Diagnostics are no longer swallowed.** The blanket `2>/dev/null` and the `|| echo 0` are both
gone; `rocm-smi`'s stderr flows back over the SSH channel, is captured by `subprocess`, and is
surfaced in the structured result and in the warning text.

**Classification is a pure function** (`_parse_gpu_probe`) separate from the SSH call, so it is
testable without a rig and a future transport change cannot quietly alter the semantics.

### GATING IS UNCHANGED — this item tells the truth, it does not change what blocks

* `preflight_check.py:229` still reads `result.checks_passed += 1  # Don't block on GPU warnings`
  — **untouched**.
* Every GPU finding still goes through `add_warning`, never `add_failure`.
* Gate **G7-GATING-UNCHANGED** proves it across five arms (count / no-binary / exit-3 / garbled /
  ssh-fail) by asserting no GPU-attributed failure and that the GPU check remains inside
  `checks_passed`.
* `all_healthy` **is** now `False` for `UNAVAILABLE` and `ERROR` (VIR-5: an unobservable surface
  is not a clean one). That flag is advisory in exactly the same way the count was.

**Warning text names the distinction** (gate G8). An `UNAVAILABLE` node renders as:

```
GPU: rig-under-test - GPU_PROBE_UNAVAILABLE: device count UNAVAILABLE (expected 8)
     — NOT observed as zero; reason=binary_not_found
```

never as `0/8` or `None/8`.

### Disagreement, reported and not acted on

The brief says to report rather than change gating. Two observations for Beta:

1. **A run can still reach dispatch with the GPU surface entirely unobserved.** That is what
   happened on 2026-08-09: `0/8 × 3` and preflight still passed 3/3. The probe now says
   `UNAVAILABLE` instead of `0`, which is the difference between "we looked and saw nothing" and
   "we could not look" — but neither blocks. **Whether an `UNAVAILABLE` GPU surface should block
   a 25-GPU saturation gate is a Beta decision. It is unchanged here.**
2. **`ROCM_SMI_FALLBACK_PATHS = ("/opt/rocm/bin/rocm-smi",)`** is a literal path. It is
   live-verified on all three rigs and is consulted **only** when `command -v` finds nothing, and
   its absence yields `UNAVAILABLE` rather than a fabricated count — but it is still a filesystem
   assumption. **The structurally stronger fix is to put `/opt/rocm/bin` on the non-interactive
   PATH on the rigs** (an `/etc/profile.d/rocm.sh`, or moving the export above the `.bashrc`
   guard). That is a **rig change**, and the rigs are PINNED AND FROZEN (CLAUDE.md §4), so it is
   **not proposed and not done** — flagged for Beta.

---

## 3. THE SAMPLER'S QUERIES, WITH THE STATE-MODEL REASONING

`scripts/gate12_concurrency_sampler.py` (new; touches nothing certified).

### Per sample

| field | query | reasoning |
|---|---|---|
| `compute_active` | `SELECT DISTINCT claimed_by FROM stripes WHERE run_id=? AND state='claimed' AND claimed_by IS NOT NULL` | **THE occupancy term.** Mirrors the production authority `MinerLedger.compute_busy_worker_ids`, whose own docstring states the §5 rule: staging is excluded *"deliberately"* because a stripe whose `StripeComplete` was accepted no longer occupies its worker's compute slot. Under F1 this is exactly one row per serial worker. |
| `queued_pending` | `count(*) WHERE run_id=? AND state='pending'` | **The term the old sampler never had.** Under F1 the full geometry is created born pending/`claimed_by` NULL/lease NULL and handed out at real handoff, so `pending` is a **real backlog** (24 at W=8, 7 at W=25). Beta's criterion turns on it. |
| `claimed_rows` | `count(*) WHERE state='claimed'` | Under F1 this must **equal** `compute_active` (one compute-active claim per serial worker, enforced in SQL). A divergence is itself diagnostic, so both are recorded. |
| `staging`, `done`, `cancelled`, `failed` | `GROUP BY state` | **CONTEXT, NOT OCCUPANCY.** `staging` in particular: counting it is precisely what **overstated** occupancy in the old query. |
| `estab` | `ss -tnH state established '( sport = :5700 )'` | **CONTEXT, NOT OCCUPANCY.** A connected worker is not an occupied worker; conflating them is how "the fleet was saturated" gets claimed without evidence. It is not an input to the verdict function at all (gate G3). |

**Every query is scoped to `run_id`.** The ledger accumulates runs — an unscoped
`count(*) FROM stripes` sums this trial with every previous one. Gate **G7-SCOPED-TO-RUN** proves
a neighbouring run of 30 claimed / 99 pending does not leak into an observation of 4/6.

**Run discovery.** The run under observation is the earliest run whose stripe rows were created
**after the sampler started** (`created_at >= start`), then latched and never changed;
`--run-id` overrides. This doubles as proof the sampler was armed before the first stripe
existed — the evidence attempt 1 lacked. Gate **G8-LATCH-AFTER-START**.

**Read-only and safe.** Every connection is `file:...?mode=ro` with `uri=True` (gate G9 proves a
write raises `attempt to write a readonly database`). A denylist refuses `prng_analysis.db`,
`miner_ledger_prod.db`, `optuna_studies.db` **by name**, a missing ledger is refused rather than
created, and a file without a `stripes` table is refused (gates G10, G10B). **The ledger path is
derived from `agent_manifests/window_optimizer.json`'s `staging_dir`
(`/home/michael/miner_staging`) — the same value the coordinator joins `miner_ledger.db` onto —
never hardcoded;** `--ledger` overrides.

> **Verified read-only against the real ledger:** the smoke run below left
> `/home/michael/miner_staging/miner_ledger.db` at its unchanged `Aug 9 12:47` mtime.

---

## 4. THE SATURATION VERDICT, AND WHY A MAXIMUM-OVER-TIME CANNOT SATISFY IT

**Beta's criterion:** *an observation window in which ≥25 DISTINCT workers were simultaneously
compute-active AND queued stripes remained available.*

**Per-sample predicate:** `compute_active >= 25 AND queued_pending >= 1`. **Both conjuncts are
required.** 25 workers busy with an empty queue does not show the scheduler under load; a deep
queue behind 8 busy workers does not show the fleet saturated.

**A window** is a maximal run of **consecutive** satisfying samples. `SATISFIED` requires the
longest window to span at least `--min-window-samples` (default **2**, i.e. at least one full
sampling interval, so a single-sample blip is not a window). *That default is an Alpha choice
and is printed in the verdict header for Beta to adjust.*

### Why the maximum-over-time cannot be used

`max over time of distinct workers` is a **union across instants**. Twenty-five workers that each
ran strictly alone, one after another, produce a union of 25 — **identical** to twenty-five
running together. The union measures *"were 25 workers eventually used"*, which the brief lists
as explicitly insufficient, and it is monotone: it can never decrease, so it cannot distinguish
saturation from serialisation.

The tool therefore computes the union **anyway** and prints it under a heading that says it does
not qualify:

```
-- NOT evidence of saturation (recorded so it is not mistaken for it) --
distinct workers ever seen active (union across instants) : 25
  A union over time is not simultaneity: 25 workers running strictly one
  after another produce the same number as 25 running together. This
  figure CANNOT satisfy the criterion and is printed only for context.
```

The verdict block reports, as the brief requires: **peak simultaneous compute-active workers**,
**the queue depth at that same instant**, and the **window duration** — plus the min
compute-active and min queued *within* the window (a window is only as strong as its weakest
sample), and a `WHY NOT` line naming which conjunct failed.

### Ordering and termination

* **Armed before the coordinator process exists** (launch script §2, before §3) — therefore
  necessarily before any `StripeAssign`. The script aborts the whole run if the sampler fails to
  start.
* **Terminates with the run**, three independent ways: a supervisor that SIGTERMs it when the
  WATCHER pid exits (launch script §4); `--watch-pid`; and a quiescence stop when no runnable
  stripe (`pending + claimed + staging`) remains for `--quiesce-seconds`. `--max-seconds 7200` is
  a backstop, not the mechanism. It writes its verdict on SIGTERM.
* **Verified offline** against a synthetic ledger and a dummy "run" process: latched, sampled,
  detected the pid exit (`watched pid 68718 exited — taking a final sample and stopping`), and
  wrote a `SATISFIED` verdict (7-sample / 6.0s window at threshold 3). No fleet, no port bind.

### End-to-end smoke against the real (dead) gate-12 ledger — read-only

`run_id=distributed_config_t1_689f3cd9`: 58 done / 6 cancelled, matching the known attempt-1
tail; verdict **NOT SATISFIED**, `WHY NOT: peak simultaneous occupancy was 0, below the required
25`.

---

## 5. BOTH GATES — RED-FIRST AND MUTATION EVIDENCE

### Item 1 — `tests/test_preflight_gpu_probe.py` (new; **not** a certified S172 suite)

Behavioural by construction: a real `ssh` shim on PATH runs the probe's command string through a
real shell against a controlled fixture PATH, so the subprocess call, the argv shape, the remote
shell parse and the classification are all genuinely exercised. Nothing is stubbed at the Python
seam the fix lives behind.

**Final state: 12/12 green.**

```
G1-COUNT-OBSERVED          status=OK count=8
G1B-ABSOLUTE-FALLBACK      count=8 via the fallback path (the production condition)
G2-MISSING-BINARY-UNAVAIL  status=UNAVAILABLE count=None reason=binary_not_found
G3-NONZERO-EXIT-UNAVAIL    reason=rocm_smi_exit_3  stderr='ERROR: unable to open kmfd device'
G3B-SSH-FAIL-UNAVAIL       reason=ssh_exit_255
G4-TIMEOUT-UNAVAIL         reason=timeout
G5-UNPARSEABLE-ERROR       status=ERROR reason=unparseable_device_count
G6-OBSERVED-ZERO-IS-ZERO   status=OK count=0        (converse control)
G7-GATING-UNCHANGED        GPU advisory in all 5 arms
G8-WARNING-NAMES-UNAVAIL   renders UNAVAILABLE, never 0/8
M1A-MUTANT-AUTHENTIC       `|| echo 0` located in HEAD:preflight_check.py
M1B-MUTANT-REDS-G2         mutant=0 (0/8) vs fixed=UNAVAILABLE
```

**RED-FIRST (mutation applied to the patched tree in a throwaway worktree).** The named mutation
— `|| echo 0` and `2>/dev/null` restored inside the new plumbing — reproduces the **exact
production symptom** and reds five gates:

```
7/12 checks green
FAILURES: G1B-ABSOLUTE-FALLBACK, G2-MISSING-BINARY-UNAVAIL, G3-NONZERO-EXIT-UNAVAIL,
          G8-WARNING-NAMES-UNAVAIL, M1B-MUTANT-REDS-G2

  G2  status=OK count=0            <- the defect, reproduced
  G8  GPU: rig-under-test - GPU_COUNT_MISMATCH: 0/8   <- the run's own warning string
```

**G7 stays green under the mutant — correctly.** Gating genuinely is unchanged by the mutation;
G7 measures gating, not truthfulness. That is detector independence (VIR-2), not a gap.

*Note: running this suite against unpatched `HEAD` raises `AttributeError` rather than grading —
a crash is not a graded red, which is why the mutation above is the red-first evidence.*

### Item 2 — `tests/test_gate12_concurrency_sampler.py` (new; synthetic ledger, no fleet)

**Final state: 14/14 green.**

```
G1-COMPUTE-ACTIVE-AND-QUEUE  compute_active=12 queued=20
G2-STAGING-NOT-OCCUPANCY     staging=7 reported separately; occupancy still 12
G3-ESTAB-NOT-OCCUPANCY       verdict function has no estab input at all
G4-NO-QUEUE-NOT-SATISFIED    25 claimed / 0 pending -> NOT SATISFIED ("queue was empty")
G5-UNION-NOT-SIMULTANEITY    union=25 but peak_simultaneous=5 -> NOT SATISFIED
G6-CLEAN-CONTROL-SATISFIED   25 simultaneous + queue -> SATISFIED, 3-sample 4s window
G6B-BLIP-IS-NOT-A-WINDOW     one satisfying instant is not a window
G7-SCOPED-TO-RUN             neighbouring run of 30/99 ignored; got 4/6
G8-LATCH-AFTER-START         pre-start=None post-start='runNEW'
G9-READ-ONLY                 "attempt to write a readonly database"
G10-PROD-DB-REFUSED / G10B-NO-DB-CREATION
M1A-LEGACY-OVERSTATES        legacy=19 (overstates by 7 staging) vs fixed=12
M1B-LEGACY-BLIND-TO-QUEUE    legacy selected no pending term; fixed reports 20
```

The three fixtures the brief names are G1/G2/G3 (N claimed / M pending / K staging), G4 (25
claimed, zero pending → NOT SATISFYING) and G5 (25 only across instants → NOT SATISFYING). **G6 is
the VIR-2 clean control** — without it every negative above would be vacuous.

**MUTATION:** the 2026-08-09 query
(`count(distinct claimed_by) WHERE state IN ('claimed','staging')`) run against the *same* fixture
answers **19** where the corrected query answers **12**, and produces no queue depth at all.

---

## 6. THE CORRECTED `gate12_launch.sh` — DELIVERED, **NOT RUN**

> **CONFIRMATION: `gate12_launch.sh` WAS NOT EXECUTED.** No pipeline launch, no fleet launch, no
> port bind. Verified after all work: `ss -ltn | grep -c 5700` → **0**; no `watcher_agent`,
> `window_optimizer` or `range_miner_worker` process exists (the single `pgrep` hit was the
> `pgrep` command's own self-match). `worker_pool_size = 25` is **written into the script and not
> applied**.

### What changed, and why

1. **`"worker_pool_size": 25` added.** Attempt 1 set the seed geometry and never overrode the
   pool size, so `admission_count = min(requested, selected)` took the manifest default of **8**
   (`agent_manifests/window_optimizer.json`) — that run's own logged `EXEC CMD` reads
   `--worker-pool-size 8`.
2. **The sampler moved from step 4 to step 2** — before the coordinator process is created,
   therefore before any `StripeAssign` — and now terminates with the run.
3. **The sampler's query replaced wholesale** with `scripts/gate12_concurrency_sampler.py`.
4. Sampler and supervisor run under `setsid` so `Ctrl-C` on the final `tail -f` cannot reach
   them; the "Ctrl-C is safe" property is preserved.
5. Fleet-launch failure now stops the sampler instead of orphaning it.

### Verified offline, without running it

* `bash -n gate12_launch.sh` → **OK**.
* The `--params` JSON parses and matches the frozen shape **exactly** — all ten keys correct, no
  extras, nothing missing.
* **Every parameter survives WATCHER's declared-key filter** (§2.15 hop 1): all ten are in
  `default_params`. `seed_count` is confirmed **absent** from `default_params` — the §2.25 trap —
  which is why the key used is `max_seeds`.
* **Simulated `EXEC CMD`**, reproducing `watcher_agent.py:1290-1314` + `:1826-1866` offline:

```
python3 window_optimizer.py --lottery-file <frozen pointer> --strategy bayesian
  --max-seeds 2147483648 --prng-type java_lcg --output optimal_window_config.json
  --test-both-modes --trials 1 --trse-context trse_context.json --enable-pruning
  --n-parallel 1 --worker-pool-size 25 --seed-cap-nvidia 5000000 --seed-cap-amd 2000000
  --seed-start 0 --pwc-transport tcp --min-workers 24 --use-range-miner
  --miner-stripe-size 67108864 --miner-substripes 8 --staging-dir /home/michael/miner_staging
  --staging-workers 4 --staging-queue-depth 2 --staging-capacity-timeout 600.0
  --staging-high-water-bytes 17179869184
```

  Identical in shape to attempt 1's logged command **except `--worker-pool-size 25`**.
  `--use-persistent-workers` is **absent** (boolean `false` omits the flag, per the trap).
  `window_trials: 1` correctly becomes **`--trials 1`** via `actions[].args_map` — the
  underscore→hyphen fallback would have produced a non-existent `--window-trials`, so this was
  checked rather than assumed. No `--staging-high-water-files`, so the retention bound is
  **derived** (§2.24: `None` = derive).

### The file

```bash
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

# ---------- 2. CONCURRENCY SAMPLER — ARMED BEFORE ANYTHING ELSE ----------
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

# ---------- 3. COORDINATOR UP (halt cleared, miner on, PWC off) ----------
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

# ---------- 4. SAMPLER TERMINATES WITH THE RUN ----------
# A supervisor, not a wall clock: when the run's own process exits, the sampler
# is asked to stop and writes its verdict. Attempt 1's sampler had no such link
# and looped for two hours against a dead trial.
setsid nohup bash -c '
  while kill -0 '"$WATCHER"' 2>/dev/null; do sleep 5; done
  sleep 10                      # let the last in-flight sample land
  kill -TERM '"$SAMPLER"' 2>/dev/null
' > /dev/null 2>&1 &

# ---------- 5. WAIT FOR BIND, THEN LAUNCH THE FLEET ----------
for i in $(seq 1 40); do ss -ltn | grep -q 5700 && break; sleep 1; done
if ss -ltn | grep -q 5700; then
  ./scripts/launch_fleet_manual.sh 192.168.3.177 5700 2>&1 | tail -4
else
  echo "COORDINATOR NEVER BOUND — aborting fleet launch"
  kill -TERM "$SAMPLER" 2>/dev/null
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
```

---

## 7. FILES CHANGED, REGRESSION EVIDENCE, AND DISAGREEMENTS

### Files changed

| file | state | what |
|---|---|---|
| `preflight_check.py` | **MODIFIED** (+236/−54) | three-outcome probe; `_build_gpu_probe_script`, `_parse_gpu_probe`, `_render_gpu_issue`; `check_gpu_health` rewritten. **Gating at `:229` untouched.** |
| `gate12_launch.sh` | **MODIFIED** (+82/−29) | pool size 25; sampler ordering; run-scoped termination |
| `scripts/gate12_concurrency_sampler.py` | **NEW** | the post-F1 sampler |
| `tests/test_preflight_gpu_probe.py` | **NEW** | item-1 gate (12) |
| `tests/test_gate12_concurrency_sampler.py` | **NEW** | item-2 gate (14) |
| `docs/CLAUDE_CODE_REPORT_PRERUN_PROBE_AND_SAMPLER.md` | **NEW** | this report |

**No certified production path was touched.** No coordinator, miner, ledger, seed-domain,
coverage, execution-set, protocol, kernel or dataset-authority file was modified.

### Suite results (final state — every run below is AFTER the last change)

| suite | result | note |
|---|---|---|
| `tests/test_preflight_gpu_probe.py` | **12/12** | new |
| `tests/test_gate12_concurrency_sampler.py` | **14/14** | new |
| `tests/test_s172_f1_f2_active_lease.py` | **16/16** | base verification, unchanged |
| `tests/test_seed_domain_cursor_amendment.py` | **40/40** | references `preflight_check` |
| `tests/test_s172_phase4_coordinator.py` | **63/63** | see below |
| `tests/test_s172_resolved_execution_set.py` | **FAIL** | **pre-existing — see below** |

**Phase-4 — Gate 22, the known untracked-`.py` sensitivity.** As-is, Gate 22 reds on exactly the
three new `.py` files and aborts the suite:

```
AssertionError: unexpected changed .py files:
  {'tests/test_gate12_concurrency_sampler.py', 'scripts/gate12_concurrency_sampler.py',
   'tests/test_preflight_gpu_probe.py'}
```

`preflight_check.py` is **not** in that set — it is already in Gate 22's allowed list
(`tests/test_s172_phase4_coordinator.py:2291`). Re-running with only the three new untracked
files temporarily moved aside, **`preflight_check.py` still modified**, gives **63/63 checks
green**. This is the documented sensitivity (skill §7), expected during development, and **not a
reason to widen Gate 22** — the files should be committed.

**`test_s172_resolved_execution_set.py` — PRE-EXISTING, ZERO DIFFERENTIAL.** It fails at
`g_consumer_legacy_test_connectivity`, `:667`, `assert len(workers) == 2 + 8`. Proven not mine by
the differential-worktree method: `git worktree add --detach <dir> HEAD` (clean `c4e0037`, none
of my changes present) fails at the **identical gate and identical line**.

**Root cause, for the record — a stale test expectation, not a production defect.** `f255912`
("config: localhost gpu_count 2 -> 1 to match measured hardware") is the §2.17 correction that
made the frozen set 25-by-construction. `distributed_config.json` now declares
`localhost.gpu_count = 1`, so `create_gpu_workers()` returns `1 + 8 = 9`, while the gate still
asserts `2 + 8`. **Reported, not fixed:** it is a certified-suite edit, outside this brief's
scope, and it needs Beta authority.

### Disagreements reported, not worked around

1. **`d3f8f00` does not exist** (§0). Work proceeded against `c4e0037`, which matches the brief's
   description of HEAD.
2. **The structurally stronger fix for item 1 is a rig PATH change, not a probe change** (§2) —
   but the rigs are frozen, so it is flagged, not done.
3. **GPU findings remain non-blocking even when `UNAVAILABLE`** (§2) — as instructed. Whether an
   unobservable GPU surface should block a 25-GPU saturation gate is Beta's call.
4. **`--min-window-samples` default of 2** (§4) is an Alpha choice, printed in the verdict header
   so Beta can see and change it.
5. **`test_s172_resolved_execution_set.py:667` is stale at HEAD** (§7) and will keep failing for
   anyone who runs it, unrelated to this work.

### Verification-integrity controls (VIR-1…6)

* **execution proof:** every gate prints a per-check line and a tally; suites run under
  `python3 -u`; live probe stdout/stderr/rc recorded verbatim per rig.
* **clean control:** G6 (criterion genuinely met → SATISFIED); G1/G1B (probe genuinely works →
  count reported); G6-OBSERVED-ZERO (a real zero stays a zero).
* **fault-injection control:** item 1 — `|| echo 0` restored, 5 gates red including G2; item 2 —
  the 2026-08-09 query answers 19 vs 12 on the same fixture.
* **completion sentinel:** `N/N checks green` on each suite.
* **unavailable-observer behavior:** M1A terminates `UNAVAILABLE` (not PASS) if `git show`
  cannot run; the probe itself reports `UNAVAILABLE` rather than a count for every surface it
  cannot read.
* **audit claim scope:** the GPU-probe root cause is claimed for the three CT100 endpoints
  `.122/.156/.164` only.
* **searched surfaces:** live SSH to all three CTs (`~/.bashrc`, `~/.profile`, `/etc/profile`,
  `/etc/profile.d/`, `/etc/environment`, `/opt/rocm`, live PATH under three shell modes); live
  VM101 filesystem; `preflight_check.py`, `miner/range_miner_coordinator.py`,
  `agents/watcher_agent.py`, `window_optimizer.py`, `execution_set.py`,
  `agent_manifests/window_optimizer.json`, `distributed_config.json`; `git log`/`git show`/
  `git log -S`; `git worktree` at `c4e0037`; the real miner ledger (read-only);
  `logs/gate12_20260809_123705*`.
* **unavailable surfaces:** the Proxmox hosts `.121/.155/.163` (no root key auth from VM101 —
  §2.17), so nothing here claims anything about GPU kernel logs; `.127` (not booted).
* **governance trail searched:** skill v21 §2.17, §2.19, §2.24, §2.25, §2.26; `CLAUDE.md` §3/§4;
  the brief. **chapters searched:** none required — no claim here concerns sieve semantics.
