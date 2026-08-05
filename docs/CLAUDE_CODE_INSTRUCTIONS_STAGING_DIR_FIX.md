# CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_FIX.md — REV1

**The miner-backed WATCHER path cannot stage a single sub-stripe result. Establish why no
gate caught it, then repair it fail-fast.**

**Base:** HEAD `8f0ca7b` or later. Claude Code on **VM101** as `michael`,
**run from `/home/michael/distributed_prng_analysis`.**

**⚠ THIS BRIEF AUTHORISES INVESTIGATION, A DRAFTED FIX AND ITS GATES.**
It does **not** authorise committing, pushing, re-running the Phase-7 soak, or landing Part B
before Team Beta rules on §5.

---

## 0. What happened

Phase-7 manual fleet launch, 2026-08-04 18:08–18:10 local. All 25 daemons registered
(execution set `bea580e76490`, admission 25/25, zero quarantined). 16 sub-stripes were
dispatched and computed. Then:

```
18:09:56,039 WARNING staging failed distributed_config_t1_0217abcd__st0_s0
             (retryable=True): config.staging_dir is not set
18:09:56,138 WARNING L1 fence dropped sub_stripe_result … state 'cancelled'
18:09:56,164 Trial 0 FAILED — MinerIngressError: threshold_provenance … validated=False
```

All 16 stripes cancelled, trial 0 failed, study aborted at 0/50 trials. **The fleet was not
the problem.** `MinerIngressError` is a downstream symptom — the ingress guard correctly
refused a trial whose forward phase never completed, and its message names thresholds, which
is the wrong subsystem. A reader who stops at the exception diagnoses the wrong defect.

---

## 1. The defect, anchored

Verified live on VM101, 2026-08-04. Every line below was read this session.

| # | link | anchor |
|---|---|---|
| 1 | manifest declares `miner_output_dir: null`, and **no `staging_dir` key at all** | `agent_manifests/window_optimizer.json` `default_params` |
| 2 | so WATCHER emits no `--miner-output-dir` | absent from the logged `EXEC CMD` line |
| 3 | `coordinator.miner_output_dir = None` | `window_optimizer.py:789` |
| 4 | the integration reads both off the coordinator | `window_optimizer_integration_final.py:1465-1466` |
| 5 | **`getattr(coordinator, 'staging_dir', None)` is a DEAD READ** | `coordinator.py` contains no `staging_dir`/`spool`/`scratch` attribute, and a tree-wide search for `\.staging_dir\s*=` returns **zero** hits outside `tests/` and `miner/` |
| 6 | `staging_dir_resolved = staging_dir or miner_output_dir` → `None` | `miner/range_miner_coordinator.py:4497` |
| 7 | `CoordinatorConfig.staging_dir: Optional[str] = None` | `:224` |
| 8 | first staged path raises | `:2108-2110` `StagingError("config.staging_dir is not set")` |
| 9 | caught by the generic handler and classified **`retryable=True`** | `:2818-2819` |
| 10 | routed to the Q3 one-retry-then-fail matrix → stripes cancelled | `:2831` `_on_staging_failed` → `handle_stripe_failure` |
| 11 | `provenance_validated` stays `False` (its default) → ingress refuses | `:3752`, then `miner/step1_ingress.py` |

**Consequence: every miner-backed WATCHER run fails at the first sub-stripe result, and has
always done so.** The failure is not intermittent, not load-dependent, and not fleet-related.

**The documented contract is implemented on the wrong side.**
`docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md` §5 states that the CLI flag and the manifest key
accept *"an explicit path or `null` / `None` for auto-detect"*, auto-detect being
`/dev/shm/prng/miner` → `~/miner_output`. That resolver **exists and works** —
`resolve_miner_output_dir()`, `miner/range_miner_worker.py:1055-1065`, module-level, no
underscore, CPU-safe at import. **`run_trial_miner` never calls it.** So the manifest's
documented default is the one value the coordinator cannot honour.

**A naming trap to state explicitly in the report:** on the coordinator side
`miner_output_dir` is used for **nothing except** defaulting `staging_dir` (`:4497` is its only
consumer). The name promises worker-spool configuration; it delivers coordinator staging. The
log line `[S172 Phase 1] Miner output dir: auto (/dev/shm/prng/miner/ …)`
(`window_optimizer.py:794`) prints on a run whose staging is unset, which makes a broken
configuration read as an auto-detected one.

---

## 2. Why no gate caught it — read this before writing any new gate

This is governed ground, not new. `docs/CLAUDE_CODE_CORRECTION2_S172_PHASE4_SIX_DEFECTS.md`
§"Defect 6" (lines 180-211) **names this exact consequence**: *"staging_dir None even when
miner_output_dir given"*. Its fix was *"staging settings (staging_dir defaulting from
miner_output_dir when set)"* — and that fallback did land, at `:4497`. **The fix was written
for the case where `miner_output_dir` is set. Production supplies `null`.**

Its gate required *"workflow phase / window params / staging_dir / bind address are the
resolved production values, not the defaults."* The gate that implements it:

```
tests/test_s172_phase4_coordinator.py:3024   miner_output_dir=stg, staging_dir=None,
tests/test_s172_phase4_coordinator.py:3039   assert c0["staging_dir"] == stg  # defaulted from miner_output_dir
```

**It passes a fabricated non-null `miner_output_dir` and asserts the fallback fires.** It
therefore proves the branch production never takes, and is silent on the branch production
always takes. This is the **VIR-2 vacuous class** and the same shape Beta caught in D6.2 —
a gate validated against fabricated values rather than the real relationship.

Every other certified miner run supplied staging through a path production does not use:

| harness | how staging was supplied |
|---|---|
| `tests/smoke_s172_phase5_d6_zeus_single_gpu.py:131-132` | a **substitute coordinator object** that sets `self.staging_dir` directly |
| `tests/phase6/wall_ab_gate.py:208,271` | explicit `--miner-output-dir` on the CLI |

**No gate has ever driven `window_optimizer.py` → the real `MultiGPUCoordinator` →
`run_trial_miner` with the manifest's own defaults.** That is the hole, and closing it is
worth more than the one-line repair.

---

## 3. The falsifiable question

> **With staging resolved, does a miner-backed WATCHER trial run to completion at HEAD — or is
> `staging_dir` merely the first of several defects on a path no gate has ever driven end to
> end?**

Answer it with evidence, not inference. The soak reached trial 0 and died at the first staged
result; **nothing downstream of staging has been exercised on this path.** A report that says
"staging was the bug" without having driven a trial past it has not answered the question.

---

## 4. Scope

**In scope**
1. Confirm §1 independently — do not take this brief's anchors on trust; re-read each.
2. Determine whether the repair is manifest-only, code-only, or both, and say why.
3. Draft the fix (§6) and its gates (§7).
4. Drive one complete miner-backed trial through the **production call shape** and report what
   happens after staging resolves.

**Explicitly OUT of scope — record if seen, do not fix**
- `_write_threshold_provenance` writes a **fixed filename** `threshold_provenance.json` into
  `staging_dir` with no `run_id` in the name (`:4073-4078`), so sequential trials overwrite
  one another's audit record. Real, separate, note it.
- The `MinerIngressError` message naming thresholds when the cause was staging. Diagnostic
  quality, separate brief.
- Re-running the Phase-7 soak. Separate authorisation.
- D3.0-B, D6.3, 6-P2. Untouched.

---

## 5. The one decision that is Beta's — Part B does not land before it is ruled

**Where should the coordinator's staging directory default to, and is the default high-water
safe there?**

Reusing `resolve_miner_output_dir()` honours the documented §5 contract and resolves to
**`/dev/shm/prng/miner`** on VM101. Measured this session:

| fact | value |
|---|---|
| VM101 `/dev/shm` | **7.8 GiB** |
| VM101 total RAM | **15.9 GiB**, **swap 0** |
| `staging_high_water_bytes` default | **16 GiB** (`window_optimizer_integration_final.py`, `16 * 1024 ** 3`) |

The coordinator's staging directory is **local to VM101** and receives payloads pulled from
**all 25 workers'** spools. On tmpfs those bytes are RAM. **The default high water exceeds both
the tmpfs size and the machine's total memory on a box with no swap** — i.e. the admission
control that exists to prevent an OOM cannot bind before the OOM occurs. VM101 OOM is a listed
Phase-7 abort trigger.

Options to put to Beta, with a recommendation and the measurement behind it:
- **(a)** reuse the documented auto-detect (`/dev/shm/prng/miner`) and lower the high water to
  a fraction of the measured tmpfs;
- **(b)** default coordinator staging to disk (e.g. a repo-local or `~/` path) and leave
  `/dev/shm` to the workers' spools, where it is per-rig and bounded;
- **(c)** require an explicit path and fail closed at resolution — no default at all.

**Do not choose. Produce the ruling request.** This is behaviour-changing work on the
certifying path (§7 working agreements).

---

## 6. The fix, bounded

Three parts. State which are mechanical and which carry the §5 decision.

1. **Resolve staging where the run is resolved, and fail BEFORE dispatch.** A misconfiguration
   that only surfaces after 25 workers have registered and 16 stripes have computed is the
   defect twice over. P0.5's dataset authority is the precedent — *fail before first worker
   dispatch*, with an error naming the resolved path and why it failed. Resolution belongs at
   or before coordinator construction, not at `_staged_path`.
2. **Reuse `resolve_miner_output_dir()`; do not write a second auto-detect.** §4 frozen-list
   discipline. Verify the import stays CPU-safe — the miner chain is covered by the
   `process_sharded` import gate (`e0513ba`, `tests/test_s172_process_sharded_import_gate.py`),
   so an import that pulls a GPU module would red an existing gate. Check, do not assume. If
   the function belongs in a shared module rather than the worker, say so and move it once.
3. **Reclassify.** `StagingError("config.staging_dir is not set")` is caught by the generic
   `except Exception` at `:2818` and marked `retryable=True`. A missing configuration is
   **permanent**. Retrying it burns a Q3 retry and produces the misleading downstream error.
   Any narrowing here touches the Blocker-3 failure matrix — bound the change to *this*
   condition and prove the matrix is otherwise unchanged.

**If the repair also requires a manifest value, gate the route, not the parameter** (§2.15).
`miner_output_dir` already has all three hops (manifest key · `args_map` `miner-output-dir` ·
`window_optimizer.py:1412` → `:789`), so a manifest-only change is a *complete* route today —
but it leaves `null`, the documented default, still broken, and the next regenerated manifest
reinstates the failure. **Say plainly whether manifest-only is sufficient; the answer is
"it makes this run work, and it does not fix the defect."**

---

## 7. Gates — behavioural, and each must fail on pre-fix code

State for every gate that you ran it against unpatched code and watched it red. A gate that
was green before the fix proves nothing.

1. **G-PROD-SHAPE** — drive the **real** production call shape: manifest defaults only, no
   test-only kwargs, `window_optimizer.py` → real `MultiGPUCoordinator` → `run_trial_miner`.
   Assert the resolved staging path is a real writable directory. **This is the gate whose
   absence caused this defect; it must not itself use a fabricated `miner_output_dir`.**
2. **G-NULL-DEFAULT** — with `miner_output_dir` explicitly `null` (the manifest default),
   staging resolves. Reds on HEAD.
3. **G-FAIL-EARLY** — with staging genuinely unresolvable, the run fails **before** any worker
   is dispatched, with an error naming the path and the reason. Assert no stripe reached
   `claimed`.
4. **G-NOT-RETRYABLE** — the missing-config condition is classified permanent; assert no Q3
   retry is consumed, and assert the rest of the failure matrix is unchanged.
5. **Regression** — `PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py` and the D6.2
   suite (**31/31**) stay green. Run long suites as
   `python3 -u <suite> | tee /tmp/<name>.log`; **never pipe to `tail`**.

Venv first: `source ~/venvs/torch/bin/activate` — watch for `(torch)`. A bare shell yields
`CuPy not available` / `Optuna not available` and a false red.

---

## 8. Deliverable

**Write the report to `docs/STAGING_DIR_FIX_REPORT.md`.** It must carry:

- §1 re-verified independently, with your own `file:line` anchors — or a correction where this
  brief is wrong;
- the §3 answer, with the evidence of a trial driven **past** staging (or an explicit
  `INCOMPLETE` naming what blocked it);
- the §5 ruling request, as a separate `docs/TB_RULING_REQUEST_*` file, with the measurements;
- the gate results, each with its pre-fix red stated;
- a "Files changed" section, **built by reading the diff, never from recall** — a `git add`
  list assembled from memory has already shipped a defect on this project.

**Do not commit or push.** Michael commits and dual-pushes. Do not land Part B (§6) before
Beta rules on §5.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every gate prints `PASS | FAIL | UNAVAILABLE | INCOMPLETE` and its own
  completion sentinel; quote the resolved staging path and the trial outcome verbatim.
- **clean control:** the suites named in §7.5 green on an unmodified tree before you start.
- **fault-injection control:** G-FAIL-EARLY and G-NULL-DEFAULT are the positive controls —
  each must be shown red on pre-fix code, not merely asserted to be capable of it.
- **completion sentinel:** `PASS` requires a miner trial that completed candidate ingress.
  **A green WATCHER line is not a result** — WATCHER scored this crashed step **1.0000** via
  `file_exists` on a stale `optimal_window_config.json` three times on 2026-08-04.
- **unavailable-observer behavior:** report `UNAVAILABLE`, never `PASS`, for anything you could
  not read. `dmesg` is denied to `michael` on **VM101 as well as the rigs** — use
  `/var/log/kern.log` (readable via group `adm`) and say which surface you used.
- **audit claim scope:** the miner-backed Step-1 path from the WATCHER manifest to the first
  staged sub-stripe result.
- **searched surfaces:** tracked repo · gitignored files (`.gitignore:41` is `*.json` — use
  `/bin/grep`, not the shell wrapper, for `.json`) · `git log`/`git log -S` · `docs/` and the
  governance trail · live VM101 filesystem · the three CT100s.
- **unavailable surfaces:** declare them.
- **governance trail searched:** `CLAUDE_CODE_CORRECTION2_S172_PHASE4_SIX_DEFECTS.md` (Defect
  6), `S172_PHASE4_BRIEF.md`, `TEAM_ALPHA_REVIEW_S172_PHASE4_REV3.md`,
  `SESSION_CHANGELOG_20260718_S172_PHASE4.md`, `S172_INFRASTRUCTURE_INTERFACE_v1_0.md` §5 —
  **these five are the only `docs/` files mentioning `staging_dir`; read all five before
  drafting, and add any you find.**
- **chapters searched:** `CHAPTER_1_WINDOW_OPTIMIZER.md` (`:1423` documents
  `--miner-output-dir`) and `CHAPTER_1_AUDIT_v1.md` (`:219` records the flag as undocumented at
  audit time).
