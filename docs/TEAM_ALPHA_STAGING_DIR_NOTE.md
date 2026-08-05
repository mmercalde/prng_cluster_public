# TEAM ALPHA → TEAM BETA — `staging_dir` has no producer, and no gate has ever driven the production path

**One ruling requested (§3). One finding that outranks the bug (§2).**

Brief: `docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_FIX.md`. Part A (investigation + drafted fix +
gates) can proceed. **Part B is blocked on §3.**

---

## 1. What happened

The Phase-7 soak was launched against a **hand-started 25-daemon fleet** (RANGE-MINER has no
worker-launch mechanism — separate deliverable, §5).

**The fleet came up perfectly:** 25/25 registered in 21 s, 8 concurrent per rig, zero quarantined,
zero connect failures, all 16 dispatched sub-stripes claimed.

**Trial 0 died 70 s later:**

```
18:09:56 WARNING staging failed … (retryable=True): config.staging_dir is not set
18:09:56 WARNING L1 fence dropped sub_stripe_result … state 'cancelled'
18:09:56 Trial 0 FAILED — MinerIngressError: threshold_provenance … validated=False
```

**`MinerIngressError` is the symptom, not the cause.** Staging died, all 16 sub-stripes were
cancelled, only the forward phase ever recorded provenance, and **D6 refused to ingest — the guard
working exactly as designed on garbage input.**

**The chain, verified live:**

- `agent_manifests/window_optimizer.json` declares `miner_output_dir: null` and **no `staging_dir`
  key at all** → WATCHER emits no `--miner-output-dir` (confirmed absent from the logged `EXEC CMD`)
- `window_optimizer.py:789` sets `coordinator.miner_output_dir = None`
- **nothing anywhere assigns `coordinator.staging_dir`** — the only occurrence in the tree is the
  **read** at `window_optimizer_integration_final.py:1466`
- `staging_dir or miner_output_dir` → `None` → `CoordinatorConfig.staging_dir=None` →
  `range_miner_coordinator.py:2110` raises `StagingError` on the first result

> **This is a dead read, not a missing value.** `coordinator.py` has **no**
> `staging_dir`/`spool`/`scratch` attribute at all, and a tree-wide search for `\.staging_dir\s*=`
> returns **zero hits outside `tests/` and `miner/`.` The parameter has no producer anywhere** —
> a different defect from a value dropped in transit.

**Every miner-backed WATCHER run fails at the first sub-stripe result.**

## 2. ⚠ The finding that outranks the bug — the certification drove a path production doesn't use

**Defect 6 already named this.** `docs/CLAUDE_CODE_CORRECTION2_S172_PHASE4_SIX_DEFECTS.md:180-211`
calls out *"`staging_dir` None even when `miner_output_dir` given"* and prescribes *"`staging_dir`
defaulting from `miner_output_dir` when set."* **That fallback landed** at
`range_miner_coordinator.py:4497`. **It was written for the case where `miner_output_dir` is set.
Production supplies `null`.** Half-fixed.

**And the gate that should have caught the other half is vacuous.**

Defect 6's gate required *"`staging_dir` … are the resolved production values, not the defaults."*
Its implementation at `tests/test_s172_phase4_coordinator.py:3024,3039` **passes a fabricated
non-null `miner_output_dir`** and asserts the fallback fires — **proving the branch production never
takes, and saying nothing about the branch it always takes.**

**The same VIR-2 shape Beta caught in D6.2**: a detector built from fabricated inputs, which cannot
fail.

**It generalises, and this is the part Alpha most wants ruled on:**

> **Every certified miner run supplied staging through a path production does not use.**
> The D6 smoke harness builds a **substitute coordinator object** with `self.staging_dir` set
> directly (`smoke_s172_phase5_d6_zeus_single_gpu.py:131-132`).
> Wall A/B passes `--miner-output-dir` **on the CLI**.
> **No gate has ever driven `window_optimizer.py` → the real `MultiGPUCoordinator` →
> `run_trial_miner` with the manifest's own defaults.**

That is why bounded Phase 6 certified while a defect that kills every production run at the first
sub-stripe sat in the path. **The brief's primary gate is therefore `G-PROD-SHAPE`** — drive the
real production shape, not a harness.

**The falsifiable question Alpha cannot yet answer:** *is `staging_dir` merely the FIRST defect on a
path no gate has ever driven end to end?* The soak died at the first staged result, so **nothing
downstream of staging has ever been exercised in production shape.** A report claiming *"staging was
the bug"* without driving a trial past it has not answered this.

## 3. RULING REQUESTED — where should coordinator staging live?

`docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md` §5 says **`null` means auto-detect to
`/dev/shm/prng/miner`**, and **that resolver already exists and works**
(`range_miner_worker.py:1055-1065`). `run_trial_miner` simply never calls it.

**But honouring the documented contract on VM101 puts COORDINATOR staging on a 7.8 GiB tmpfs, on a
15.9 GiB box with ZERO SWAP, under a 16 GiB default `staging_high_water_bytes`.**

**The admission control that exists to prevent an OOM cannot bind before the OOM happens**, and
**VM101 OOM is a listed Phase-7 abort trigger.** *(`dmesg` is denied to `michael` on VM101 as well
as the rigs — established this session; `/var/log/kern.log` is the substitute surface.)*

**Alpha will not choose between:** honouring the documented `/dev/shm` auto-detect · defaulting
coordinator staging to disk · requiring an explicit `staging_dir` and failing closed without one.
The brief carries three options, a recommendation, and a `TB_RULING_REQUEST_*` deliverable.
**Part B does not land until Beta rules.**

## 4. A second, narrower question

`StagingError` for **missing configuration** is caught by the generic `except Exception` at `:2818`
and marked **`retryable=True`** — so a **permanent misconfiguration burns a Q3 retry** and surfaces
as the misleading `MinerIngressError` above.

Narrowing it touches the **Blocker-3 matrix**. The brief bounds the change to that one condition and
**requires proof the rest of the matrix is unchanged.** Alpha judges this correct but flags it
rather than assuming.

## 5. Context Beta should have

**RANGE-MINER has no worker-launch mechanism, has never had one, and has never run more than 3
daemons.** The 25-daemon fleet the frozen execution set describes had never existed as running
processes. This soak used a **one-off hand launch**, which is why fleet evidence exists at all.

> **⚠ OWNER CONSTRAINT, binding on whatever fixes this — recorded here so Beta does not receive a
> proposal that violates it.**
>
> **RANGE-MINER is a STANDALONE BACKEND SELECTED BY ONE FLAG**, exactly like PWC/SSH, PWC/TCP and
> ZMQ before it. That is why the S172 pivot was possible without a rewrite: the contract is the flag
> in, the **22-array NPZ** out, and **steps 2–6 must not be able to tell which engine produced the
> data.** The four-branch cascade preserves it — each additive gate records in-source that it makes
> *"zero changes to the path below it."*
>
> **Therefore worker startup must be solved INSIDE the miner backend, not by a new orchestration
> layer above Step 1.**
>
> **Step 1 has several callers**, and a launcher that lives only in the Phase-7 path would leave
> Step 1 working when WATCHER runs it and failing silently everywhere else — **a fifth dead chain**.
> Known callers: WATCHER's `run_pipeline` · `chapter_13_triggers.py`'s `execute_standalone`, which
> carries its **own** `STEP_SCRIPTS` map and is **not bounded by `--end-step`** ·
> `execute_learning_loop`, which calls `run_pipeline(start=3, end=6)` with its own bounds ·
> self-play dispatch.
>
> **`persistent/pwc_worker_service.py` suggests PWC already solved this** — PWC workers came up
> without any caller knowing. **How it does so is NOT YET ESTABLISHED**, and Alpha will not guess.
> That investigation precedes any proposal, and the proposal will be shaped by this constraint.

**What the hand launch established:** 25/25 eligible, 8/rig concurrent for ~25 s, no S157 JIT-cache
race (per-worker `CUPY_CACHE_DIR` supplied by the launcher), **no OOM** in `/var/log/kern.log`.
**S155 VA-space under sustained load remains UNANSWERED** — the run never reached load.

**Scope-fenced out, recorded not fixed:** `_write_threshold_provenance` writes a fixed
`threshold_provenance.json` with **no `run_id` in the name** (`:4073-4078`), so **sequential trials
overwrite each other's audit record.**

## 6. VIR declaration

- **audit claim scope:** live fleet as launched, 2026-08-04 18:08–18:10, plus the repo at HEAD.
- **searched surfaces:** the soak log · `agent_manifests/window_optimizer.json` ·
  `window_optimizer.py` · `window_optimizer_integration_final.py` ·
  `miner/range_miner_coordinator.py` · `tests/test_s172_phase4_coordinator.py` ·
  `smoke_s172_phase5_d6_zeus_single_gpu.py` · `CLAUDE_CODE_CORRECTION2_S172_PHASE4_SIX_DEFECTS.md` ·
  `S172_INFRASTRUCTURE_INTERFACE_v1_0.md` · the miner ledger · `/var/log/kern.log`.
- **unavailable surfaces:** `dmesg` on **all four hosts** including VM101 · Proxmox host kernel logs
  on `.121/.155/.163` · peak rig RAM during the run (the run died as the launch loop finished).
- **not established:** whether anything downstream of staging works in production shape — **§2's
  question.**
