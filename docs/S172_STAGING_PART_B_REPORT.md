# S172 STAGING PART B — REPORT

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md` REV1 (Beta binding ruling).
**Base HEAD:** `64d0f9af43b2a20c727f4a94d1659becddc916f2`. VM101 as `michael`, venv `~/venvs/torch`.
**Date:** 2026-08-04. **Not committed, not pushed.**

---

## COMPLETION SENTINEL: **INCOMPLETE**

**The staging repair is implemented and gated (24/24 CPU gates green, each proven red pre-fix).
`G-PROD-SHAPE` has NOT been executed** — the WATCHER pipeline launch was refused by the harness
permission classifier, which is `CLAUDE.md` hard rule 3 working as designed (`--run-pipeline`
spins the fleet and is always Michael-initiated). **§3 therefore has no result, and under §0 this
incident does not close.** The exact commands to run are in §7.

**`PASS` requires a COMPLETE production-shape trial. There is none. Nothing below should be read
as evidence that RANGE-MINER's production activation path works.**

---

## §0 — the line that governs this report

> **"The staging defect is NOT YET PROVEN to be the only production-path defect. Alpha must not
> close the incident merely because the first failed operation begins working."**

**Honoured literally.** The staging defect is repaired and proven repaired *as a configuration
defect*. Whether anything downstream of staging works in production shape is **still unanswered**,
because nothing has yet driven that path. The soak died at the first staged sub-stripe result, so
**every operation after staging remains unexercised.** See §6.

---

## 1. Resolved production staging path + filesystem evidence

**Resolved path: `/home/michael/miner_staging`** — absolute, disk-backed, declared in
`agent_manifests/window_optimizer.json` `default_params.staging_dir`.

### 1.1 Measured on VM101, 2026-08-04 (guest view)

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda2       916G  442G  427G  51% /
tmpfs           7.8G     0  7.8G   0% /dev/shm
```

`/proc/mounts`: `/dev/sda2 / ext4 rw,relatime,errors=remount-ro` · `tmpfs /dev/shm tmpfs`.
`/`, `/home` and `/tmp` are all `sda2` — **one filesystem**, so same-directory atomic rename is
real and staging shares a filesystem with the repo, checkpoints and Optuna DBs.

| path | fstype | backing | total | available | 16 GiB high-water fits? |
|---|---|---|---|---|---|
| **`/home/michael/miner_staging`** | **ext4** | **disk** | 915.32 GiB | **426.86 GiB** | **YES** — margin **410.86 GiB** |
| `/dev/shm/prng/miner` | tmpfs | **RAM** | 7.78 GiB | 7.78 GiB | **NO** — short by **8.22 GiB** |

**The arithmetic Beta asked for.** `staging_high_water_bytes` default = `16 * 1024**3` =
**17,179,869,184 B**. Operational headroom = `max(1 GiB, high_water/10)` = **1.60 GiB**.
Required = **17.60 GiB**. Available on the chosen path = **426.86 GiB**. **Requirement met with
409 GiB to spare.**

On `/dev/shm` the same arithmetic gives 17.60 GiB required against 7.78 GiB available — and VM101
has **15.9 GiB RAM with SWAP 0**, so the default high-water exceeds both the tmpfs *and*
approaches total machine memory. That is precisely Beta's objection: *admission control cannot be
represented as an OOM safeguard when the configured mark exceeds the filesystem that must hold the
staged data.*

### 1.2 Manifest change to the high-water: **NONE**

**`staging_high_water_bytes` is UNCHANGED at 16 GiB.** The disk-backed path holds it comfortably,
so the brief's conditional ("if no disk-backed location holds 16 GiB + headroom, the high-water
must come down") **did not trigger**. Reported explicitly because the brief required the arithmetic
either way.

### 1.3 ⚠ THIN PROVISIONING — a limit on what this check can claim

Host evidence (Proxmox `pve-zeus`, supplied by the owner, 2026-08-04):

```
local-lvm thin pool : 816.21 GiB total, 67.80% used  -> ~263 GiB ACTUALLY free
vm-101-disk-1       : 932 GiB provisioned, 52.20% consumed (~487 GiB real)
vm-100-disk-1       : 256 GiB provisioned, 26.13% consumed
total provisioned   : 1,188 GiB against an 816 GiB pool  -> OVERSUBSCRIBED
```

**VM101's guest-visible 426.86 GiB overstates the real backing by roughly 164 GiB.**
`os.statvfs()` runs inside the guest and **cannot see the thin pool**. 16 GiB staging is
comfortable against ~263 GiB really free, so the high-water stands — but the guarantee is narrower
than the guest number suggests: **host thin-pool exhaustion would present as a WRITE FAILURE on a
filesystem the guest believes has space, not as a capacity rejection at startup.**

This limitation is recorded in-source in the validator's docstring
(`miner/range_miner_coordinator.py`, `validate_coordinator_staging_dir`), not only here.

---

## 2. Configuration precedence and conflict behaviour — all five §1.1 rows

`staging_dir` is **canonical**; `miner_output_dir` is a **temporary backward-compatible alias**.
Implemented in `resolve_coordinator_staging_dir()`.

| # | condition | required | observed | gate |
|---|---|---|---|---|
| 1 | only `staging_dir` set | use it | returns it unchanged | `G-PREC-1` |
| 2 | only explicit `miner_output_dir` | populate + **deprecation warning** | populates; logger emits `[Part B §1.1 rule 2] DEPRECATED…` | `G-PREC-2` |
| 3 | both set and **differ** | **FAIL CLOSED** | `StagingConfigurationError` naming both paths (identical paths are *not* a conflict) | `G-PREC-3` |
| 4 | **neither** set | **FAIL CLOSED** | `StagingConfigurationError`; empty/whitespace counts as unset | `G-PREC-4` |
| 5 | implicit `/dev/shm` fallback | **PROHIBITED** | enforced **by construction** | `G-PREC-5` |

**Rule 5 is enforced structurally, not by a check.** The resolver contains no fallback candidate
and never calls `resolve_miner_output_dir()`. `G-PREC-5` proves this two ways: behaviourally (an
unset configuration *raises* rather than resolving to `/dev/shm`) and by **AST over live source**
(no `/dev/shm` literal usable as a candidate; no call to the worker resolver). AST rather than a
text anchor deliberately — `2389b61` reverted a fix by whole-block replacement and a text anchor
would have gone green.

**The split is real and the fix did not over-reach.** `G-PREC-6` asserts worker-local output
**keeps** its documented `null → /dev/shm/prng/miner → ~/miner_output` auto-detect
(`resolve_miner_output_dir`, untouched). Coordinator staging and worker output are different
subjects with different ownership, lifetime and capacity.

**One backend flag preserved.** `--use-range-miner` alone remains sufficient. `--staging-dir` was
added as an **optional** argparse flag; the path is **declared in the manifest**, not supplied per
run. `G-ROUTE-1` asserts the flag is not `required`.

**The three-hop route is gated as a ROUTE, not a parameter** (§2.15 — a new Step-1 parameter dies
silently at hop 1):

| hop | what | anchor |
|---|---|---|
| 1 | manifest `default_params.staging_dir` + `actions[0].args_map["staging-dir"]` | `agent_manifests/window_optimizer.json` |
| 2 | argparse `--staging-dir` → call-site kwarg → `coordinator.staging_dir = staging_dir` | `window_optimizer.py` |
| 3 | `getattr(coordinator, 'staging_dir', None)` (the formerly DEAD read) | `window_optimizer_integration_final.py:1466` |

`G-ROUTE-1` additionally replays WATCHER's step-scoped filter (`watcher_agent.py:1290-1314`) and
its `None`-skip rule (`:1773`) to prove a null value would emit **no flag at all** — the original
defect.

---

## 3. Startup validation (§1.3) — before dispatch or reservation accounting

`validate_coordinator_staging_dir()` runs in `run_trial_miner` **before `build_coordinator`**, so
before the ledger exists, before the trial is created, and therefore before any worker dispatch or
reservation accounting. P0.5's dataset authority is the precedent: *fail before first worker
dispatch*.

| check | gate | fails for its OWN reason |
|---|---|---|
| absolute | `G-VAL-2` | ✅ message names "absolute" |
| creatable + writable | `G-VAL-3` | ✅ "not writable" |
| temp-write + **atomic rename PROVEN** | `G-VAL-7` | ✅ writes, fsyncs, renames, verifies bytes, unlinks; asserts **no probe leak** |
| disk-backed | `G-VAL-4` | ✅ isolated with a high-water tmpfs *could* hold, so only the RAM-backed check can fire; asserts `capacity-invalid` is **not** the reason |
| high-water ≤ usable capacity | `G-VAL-5` | ✅ isolated with `require_disk_backed=False`, so only capacity can fire; asserts `ram-backed` is **not** the reason |
| high-water + headroom ≤ available | `G-VAL-6` | ✅ "headroom" |

Measured evidence returned and carried on the run context as `staging_validation` (observable, not
merely logged):

```
staging_dir=/home/michael/miner_staging  fstype=ext4  disk_backed=True
total=982,820,896,768 B  available=458,339,745,792 B
high_water=17,179,869,184 B  headroom=1,717,986,918 B  atomic_rename_proven=True
```

An **undetermined** filesystem is never reported as disk-backed (VIR-5: unobservable is not clean).

---

## 4. Focused retry-matrix proof — Blocker-3 unchanged row-for-row

**`handle_stripe_failure` / `_handle_stripe_failure_locked` were NOT modified.** The only change is
*which `retryable` value one exception TYPE produces at one call site*.

**Before / after, driven row by row against a frozen expected table (`G-NR-4`):**

| # | condition (phase, retryable, attempt, lease_expiry, alternate) | action | reason | before | after |
|---|---|---|---|---|---|
| A | trial already terminal | `noop` | trial already terminal | unchanged | unchanged |
| B | stripe already done/failed/cancelled | `noop` | stripe already `<state>` | unchanged | unchanged |
| C | retryable=False, constant phase 1 | `fail_trial` | `non_retryable` | ✅ | ✅ |
| C′ | retryable=False, hybrid phase 3 | `fail_trial` | `non_retryable` | ✅ | ✅ |
| D | retryable=True, phase 1 | `fail_trial` | `constant_phase` | ✅ | ✅ |
| D′ | retryable=True, phase 2 | `fail_trial` | `constant_phase` | ✅ | ✅ |
| E | retryable=True, phase 3, attempt 0, alternate exists | `reassigned` (attempt 1, `phase_degraded=True`) | — | ✅ | ✅ |
| F | retryable=True, phase 3, attempt 0, **no** alternate | `fail_trial` | `no_alternate_worker` | ✅ | ✅ |
| G | lease expiry, phase 4, attempt 0, alternate exists | `reassigned` | — | ✅ | ✅ |
| G′ | lease expiry, phase 1 | `fail_trial` | `constant_phase` | ✅ | ✅ |

**The single approved change**, and nothing else:

| exception type | before | after |
|---|---|---|
| **`StagingConfigurationError`** (new, narrow) | *did not exist* — fell to generic `except Exception` → **`retryable=True`** | **`retryable=False`** |
| `StagingError` (base) | generic → `retryable=True` | **unchanged** |
| `StagingBackPressure` | wait + resume, never the matrix | **unchanged** |
| `StagingHashMismatch` | `retryable=True` | **unchanged** |
| `StagingTimeout` | `retryable=True` | **unchanged** |
| everything else | generic → `retryable=True` | **unchanged** |

`G-NR-1` asserts the subtype is **narrow** — no sibling was swept in, and `StagingError` itself did
**not** become non-retryable. `G-NR-2` proves by AST over live source that the new handler precedes
`except Exception` **and** that the generic handler still passes `retryable=True`. `G-NR-5` proves
**no Q3 retry is consumed**: `current_attempt` stays `0` and `phase_degraded` is not set.

**Root cause preserved in the terminal report (§2).** The handler's reason string leads with
`staging configuration error (non-retryable): …`, and `G-NR-6` asserts the exception text names
staging and contains **neither** "threshold", "provenance" nor "ingress". The 2026-08-04 inversion —
a missing staging path surfacing as `MinerIngressError: threshold_provenance … validated=False` —
cannot recur in that form.

---

## 5. Gates — results, with the pre-fix red proven

### 5.1 CPU gates — `tests/test_s172_staging_partb.py` → **24/24 green**

```
G-PREC-1..6   precedence (five rules) + worker auto-detect unchanged
G-VAL-1..7    startup validation, each failing for its own reason
G-FAIL-EARLY-1..3  fail before dispatch; ledger no longer falls back to CWD
G-NR-1..6     narrow subtype, handler order, matrix row-for-row, no retry consumed, root cause
G-ROUTE-1..2  three-hop route; the manifest's path validates on THIS host
COMPLETION SENTINEL: PASS — S172 Staging Part B CPU gates green
```

### 5.2 Pre-fix control — **the gates are not vacuous**

The suite **cannot even import** against stashed HEAD (`ImportError: cannot import name
'StagingConfigurationError'`). That is a blunt red, so a **separate clean control** reproduced the
production failure directly on HEAD:

```
RED  StagingConfigurationError              present=False
RED  resolve_coordinator_staging_dir        present=False
RED  validate_coordinator_staging_dir       present=False
     pre-fix line: staging_dir_resolved = staging_dir or miner_output_dir
     staging_dir=None miner_output_dir=None -> resolved=None
RED  G-PREC-4 / G-FAIL-EARLY: resolves to None, no fail-closed, no validation
     raised: StagingError: config.staging_dir is not set
RED  G-NR-3: type is StagingError, NOT StagingConfigurationError
     _run_staging_job has a config-error handler? False
     generic handler classifies: self._on_staging_failed(..., True, ...)
RED  G-NR-2: missing config == retryable=True
     manifest default_params has staging_dir? False ; args_map has staging-dir? False
RED  G-ROUTE-1 hop 1 absent entirely
     window_optimizer.py assigns coordinator.staging_dir? False
     integration READS getattr(coordinator,'staging_dir')? True
RED  G-ROUTE-1 hop 2: the read exists, the WRITE does not => DEAD READ
```

**Every Part B gate condition is absent at HEAD**, and the exact production failure
(`config.staging_dir is not set`, classified `retryable=True`) reproduces on demand.

### 5.3 `G-PROD-SHAPE` — **NOT RUN** (`tests/gate_s172_prod_shape.py`)

Built and **proven red against the failed 2026-08-04 soak log** (9 pass / 5 fail):

```
FAIL A3  --staging-dir came from the manifest        -> absent
FAIL A6  production staging validation executed      -> validator did not run
FAIL A8  the pre-repair failure did NOT occur        -> "config.staging_dir is not set" PRESENT
FAIL B*  miner ledger                                -> no ledger at /home/michael/miner_staging
FAIL D*  artifact                                    -> no published generation directory
COMPLETION SENTINEL: FAIL
```

It carries the anti-fabrication legs §3 demands — **no substitute coordinator (A7), no CLI-only
`--miner-output-dir` injection (A4), staging value originating from the manifest (A3)** — plus the
ledger legs (committed trial, stripes `done`, shards `verified`, phase coverage, Phase-5 ack,
local cleanup, remote-delete state, **zero held reservations**), leak-freedom (temp/staged/probe/
provisional), and the artifact legs (frozen 22-array contract exact-and-in-order; Step-2 load-back
with `fallback_used=False`).

**Pre-flight, the production shape is confirmed constructible.** Replaying WATCHER's own
command-construction logic against the patched manifest predicts:

```
python3 window_optimizer.py … --use-range-miner --miner-stripe-size 67108864
        --miner-substripes 8 --staging-dir /home/michael/miner_staging
  --staging-dir present?      True   (value == manifest value)
  --miner-output-dir absent?  True
  backend mutex ok (only 1)?  True
```

**But a predicted command is not a run.** §3 is unsatisfied.

---

## 6. Downstream-defect statement (§6 return 6)

**Alpha cannot yet answer the falsifiable question.** *Was staging the only defect, or merely the
first defect reached on a previously unexercised production path?* Answering it requires a
COMPLETE production-shape trial, which has not run.

**What IS established:**

1. **The staging defect is real, universal and repaired.** Every miner-backed WATCHER run failed at
   the first sub-stripe result, always, for configuration reasons — reproduced on HEAD in §5.2.
2. **Its blast radius was widened by two amplifiers, both now closed.** It was classified
   *retryable*, burning a Q3 retry; and its terminal report named the wrong subsystem.
3. **Nothing downstream of staging has been exercised in production shape.** Trial phases past the
   first staged result, Phase-5 assembly on this path, the D3.5 finalizer publishing from a
   miner-backed WATCHER run, and Step-2 load-back **remain unexercised.**

**Observed during Part B, recorded not fixed (Beta placed both outside this patch):**

- **`threshold_provenance.json` fixed filename** (`_write_threshold_provenance`, no `run_id` in the
  name) — sequential trials overwrite one another's audit record. Beta has recorded this as a
  **Phase-7 certification blocker**. The intended 2-trial G-PROD-SHAPE run would have exercised
  exactly this; it has not run, so **no new evidence is added.**
- **WATCHER scored a crashed step `1.0000`.** The 2026-08-04 log ends `Step 1 PASSED`,
  `Parse Method: file_exists`, `Confidence: 1.00` on a run that died at trial 0 — via `file_exists`
  on a stale `optimal_window_config.json`. **A green WATCHER line is not a result**, and this report
  treats it as none.

**New findings this session (not in the brief):**

- **The coordinator module IS inside the worker's transitive import closure.** Importing
  `miner.range_miner_worker` on each CT100 puts `miner.range_miner_coordinator` in `sys.modules`
  (via `miner/__init__.py`). Verified on all three rigs, each probe printing its own
  `socket.gethostname()`. The patched module was therefore deployed to all three by targeted `scp`;
  digests now match VM101 (`d6cc26bf…`) where they previously matched HEAD (`70f8fbaa…`).
  *"Loaded but not driven" is not an acceptance criterion.*
- **`rocm-smi` is not on the rigs' non-interactive `PATH`.** GPU count was instead confirmed via
  `cupy.cuda.runtime.getDeviceCount()` → **8 per rig**, so the substitute-detection series planned
  for a soak must not assume `rocm-smi` resolves over `ssh`.

---

## 7. What remains — the exact commands (owner-initiated)

The WATCHER launch was **refused by the harness permission classifier**, consistent with
`CLAUDE.md` hard rule 3 (*never launch the pipeline autonomously; always Michael-initiated*).
**Not worked around.**

**Step 1 — soak first** (bounded to step 1; `--end-step` defaults to 6 and step 2 reaches the
TB-prohibited converter):

```bash
cd /home/michael/distributed_prng_analysis && source ~/venvs/torch/bin/activate
nohup python3 -u -m agents.watcher_agent --run-pipeline --start-step 1 --end-step 1 \
  --params '{"use_persistent_workers":false,"use_range_miner":true,
             "worker_pool_size":25,"window_trials":2,"max_seeds":268435456}' \
  > logs/partb_prodshape.log 2>&1 &
```

**Step 2 — then the 25 daemons** (~3 s stagger; admission window is 180 s, 25×3 s = 75 s):

```bash
bash /tmp/claude-1000/-home-michael-distributed-prng-analysis/98053dd9-ed07-402b-859c-bdee1d36d097/scratchpad/launch_fleet.sh 192.168.3.177 5700
```

`--gpu-id N --device-index 0`, `ROCR_VISIBLE_DEVICES=N`, per-worker `CUPY_CACHE_DIR`
(S157 JIT-cache race). **`staging_dir` is deliberately NOT in `--params`** — it must come from the
manifest, which is what `G-PROD-SHAPE` leg A3 checks.

**Step 3 — the gate:**

```bash
PYTHONPATH=. python3 -u tests/gate_s172_prod_shape.py --log logs/partb_prodshape.log
```

**This is NOT the autonomous startup of §7 of the brief.** No worker launcher or supervisor was
built; the script above is a one-off hand launch in the scratchpad, outside the repo.

---

## 8. Non-regression (§6)

| suite | result |
|---|---|
| **D6.2 checkpoint reconciliation** | **31/31 green** (377 assertions, 25 mutants killed) — **certified suite, untouched** |
| D6.1 flush durability | green (exit 0) |
| D3.25 candidate ingress | 13/13 green |
| D3.5 finalizer | 60/60 green |
| Phase 3 worker | all gates green |
| Phase 4 coordinator | **63/63 green** |
| `process_sharded` import gate | 7/7 green |
| D6 production adapter | 9/9 green (16 mutants killed) |
| **D5 `process_sharded`** | **24/25 — see below** |

**The single D5 red is the known Gate-22 untracked-`.py` sensitivity, not a regression.**

```
AssertionError: unexpected changed .py files:
  {'tests/test_s172_staging_partb.py', 'tests/gate_s172_prod_shape.py'}
```

Gate 22 builds `changed_py` from `git status --porcelain`, which **includes untracked files**, so
any new test file reds it and propagates to D5's `NR` arm. **Proven** by temporarily removing the
two new gate files and re-running Phase 4 → **63/63 green**, then restoring them. This is expected
during development and **is not a reason to widen Gate 22**; it clears when the files are committed.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** resolved path, `df`/`statvfs`/`/proc/mounts` output, the atomic-rename proof
  and measured evidence dict are quoted verbatim in §1 and §3. **"It works now" is not claimed —
  and could not be, because the production trial has not run.**
- **clean control:** all suites in §8 green on the patched tree; the pre-repair state reproduces
  `config.staging_dir is not set` on demand (§5.2).
- **fault-injection control:** `G-VAL-4` (RAM-backed) and `G-VAL-5` (capacity-invalid) are mutually
  isolated so **each fails for its own reason**, asserted on the message, not merely on "raised".
  `G-FAIL-EARLY-1/2` prove failure occurs before dispatch via a serve seam that records being
  called and **must never be called**.
- **completion sentinel:** **INCOMPLETE.** `PASS` requires a COMPLETE production-shape trial. There
  is none. A green WATCHER line is explicitly rejected as evidence (§6).
- **unavailable-observer behavior:** reported `UNAVAILABLE`, never `PASS`, for unread surfaces.
- **audit claim scope:** the repo at HEAD `64d0f9a` **plus** the three CT100s for the import-closure
  and digest checks. **NOT the live fleet under load** — §3 did not run.
- **searched surfaces:** `agent_manifests/window_optimizer.json` (via `/bin/grep` and `json`, since
  `.gitignore:41` is `*.json`) · `window_optimizer.py` · `window_optimizer_integration_final.py` ·
  `miner/range_miner_coordinator.py` · `miner/range_miner_worker.py` · `agents/watcher_agent.py` ·
  `execution_set.py` · `rig_profiles_config.json` · `distributed_config.json` ·
  `dataset_provisioning.json` · `daily3_current.json` · `utils/run_finalizer.py` ·
  `utils/survivor_loader.py` · `tests/test_s172_phase4_coordinator.py` ·
  `tests/phase6/wall_ab_gate.py` · `logs/phase7_soak.log` · live VM101 filesystem · the three
  CT100s (`192.168.3.122/.156/.164`) by SSH.
- **unavailable surfaces:** `dmesg` on **all four hosts including VM101** (blocked;
  `/var/log/kern.log` is readable via group `adm` and is the substitute) · **Proxmox host kernel
  logs on `.121/.155/.163`** (no root key auth from VM101) · **`rocm-smi` over non-interactive
  `ssh`** (not on `PATH`; `cupy` device count used instead) · **the host thin pool from inside the
  guest** (§1.3) · **the live fleet under load** (§3 not run).
- **governance trail searched:** `CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md` ·
  `CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_FIX.md` · `TEAM_ALPHA_STAGING_DIR_NOTE.md` ·
  `S172_INFRASTRUCTURE_INTERFACE_v1_0.md` §5 · `CLAUDE_CODE_CORRECTION2_S172_PHASE4_SIX_DEFECTS.md`
  (Defect 6, via Part A's citation).
- **chapters searched:** not required for this patch; no chapter claim is made.

---

## Files changed — **built by reading the diff, never from recall**

`git diff --numstat` at report time:

| file | +/− | what |
|---|---|---|
| `miner/range_miner_coordinator.py` | +331 / −6 | `StagingConfigurationError`; `resolve_coordinator_staging_dir` (5 rules); `validate_coordinator_staging_dir` (§1.3 + thin-pool caveat); `_filesystem_type_for`; headroom policy; resolve+validate before `build_coordinator`; `_staged_path` raises the narrow subtype; narrow handler before the generic; `staging_validation` on the run context |
| `window_optimizer.py` | +33 / −2 | `staging_dir` param (hop 2); `coordinator.staging_dir` assignment (closes the DEAD READ); optional `--staging-dir` argparse; call-site kwarg; split the misleading single "Miner output dir: auto(...)" log line into worker-output vs coordinator-staging |
| `agent_manifests/window_optimizer.json` | +2 / −0 | `default_params.staging_dir` + `args_map["staging-dir"]` (hop 1) |
| `docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md` | +46 / −0 | §5 amended: worker output vs coordinator staging, **original text retained and marked clarified, nothing deleted** |
| `tests/test_s172_staging_partb.py` | new | 24 CPU gates |
| `tests/gate_s172_prod_shape.py` | new | `G-PROD-SHAPE` verifier |

**All 8 deleted lines were read individually and are exactly the intended replacements** — no
accidental revert (the `2389b61` whole-block-overwrite lesson applied deliberately).
The manifest was verified by **diffing decoded JSON structures against HEAD**, not text: exactly two
additions, zero removals, zero changes — no re-serialization churn.

**Remote deployment:** `miner/range_miner_coordinator.py` scp'd to all three CT100s (it is inside
the worker's import closure — §6). Digests verified on target: `d6cc26bfa09f2c00` on
`rrig6600`/`rrig6600b`/`rrig6600c`, matching VM101.

**Not committed, not pushed.** Michael commits and dual-pushes.
