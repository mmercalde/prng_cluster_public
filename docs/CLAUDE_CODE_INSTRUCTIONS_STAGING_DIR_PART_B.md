# CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md — REV1

**Part B is AUTHORIZED under Beta's binding ruling. Implement the staging repair and
`G-PROD-SHAPE`.**

**Base:** HEAD (current). Claude Code on **VM101** as `michael`, venv `~/venvs/torch`,
**run from `/home/michael/distributed_prng_analysis`.**

**Supersedes Part B of `CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_FIX.md`.** Part A's findings stand.

**Do NOT commit or push. Do NOT build a worker launcher (§7). STOP at the gate.**

---

## 0. What Beta ruled, and the one line that governs the whole brief

> **"The staging defect is NOT YET PROVEN to be the only production-path defect. Alpha must not
> close the incident merely because the first failed operation begins working."**

Phase-6 certification is **narrowed, not revoked**: valid for the bounded harnesses and the
artifact/interface properties they exercised — **not evidence that the production activation path
works.** *No document may state RANGE-MINER is production-certified because the D6 harnesses are
green.*

**Phase 7 stays blocked until a real production-shape trial completes.**

## 1. Coordinator staging — explicit, absolute, disk-backed

**The production coordinator must NOT auto-detect `/dev/shm/prng/miner`.**

Worker-local output **keeps** its documented `null → /dev/shm/prng/miner` behaviour. **Coordinator
staging is a different thing** — different ownership, lifetime and capacity — and **`null` is
invalid at the production boundary.**

### 1.1 Canonical field and precedence — implement exactly these five

`staging_dir` is **canonical**. `miner_output_dir` may remain a **temporary backward-compatible
alias**:

| # | condition | behaviour |
|---|---|---|
| 1 | only `staging_dir` set | **use it** |
| 2 | only an **explicit** `miner_output_dir` set | populate `staging_dir` **with a deprecation warning** |
| 3 | both set and **differ** | **FAIL CLOSED** |
| 4 | **neither** set | **FAIL CLOSED** |
| 5 | any implicit `/dev/shm` fallback for the coordinator | **PROHIBITED** |

**The production manifest must populate canonical `staging_dir` and must NOT depend on the alias.**

**⚠ One backend flag is preserved.** `--use-range-miner` alone stays sufficient — the path is
**declared in the manifest**, not supplied per run. **Do not add a required CLI flag.**

### 1.2 Choosing the path — measure, do not assume

**Determine the resolved production path from measured filesystem evidence**, and report both.

`staging_high_water_bytes` defaults to **16 GiB**. Beta: the configuration **must not advertise a
high-water limit larger than the usable filesystem capacity.**

**So: measure free disk on VM101 first, then choose.** If no disk-backed location holds
16 GiB + headroom, **the high-water must come down** — and that is a manifest change to report, not
a silent one. **State the arithmetic.**

### 1.3 Startup validation — before dispatch or reservation accounting

Verify the resolved coordinator staging location:

- is **absolute**
- is **creatable and writable**
- supports **temp-write and atomic-rename** *(prove it — write and rename, don't infer)*
- is **disk-backed** for the approved Phase-7 configuration
- has capacity for **the configured high-water plus operational headroom**
- **does not advertise a high-water larger than usable capacity**

> **A capacity-invalid configuration must fail BEFORE launching or accepting work.** Beta:
> *"Admission control cannot be represented as an OOM safeguard when the configured mark exceeds
> the filesystem that must hold the staged data."*

## 2. Missing configuration is non-retryable — narrowly

Add a **specific subtype**:

```python
class StagingConfigurationError(StagingError): ...
```

**Caught BEFORE the generic staging/exception handler** (`range_miner_coordinator.py:2818` is where
the generic `except Exception` currently swallows it and marks `retryable=True`).

**Applies to:** missing · conflicting · non-absolute · unwritable · capacity-invalid → `retryable=False`.

> **⚠ Do NOT globally convert `StagingError` to non-retryable.** Transient transfer, filesystem and
> capacity-pressure conditions **keep their existing classification.**

**The Blocker-3 matrix must be proven unchanged row-for-row outside this one classification.**
Show the matrix before and after.

**And the terminal report must preserve the root cause** — a missing staging path must **not**
surface primarily as `MinerIngressError` or a threshold-provenance failure. That inversion is what
made tonight's diagnosis take an hour.

## 3. `G-PROD-SHAPE` — mandatory, and the point of the exercise

**No substitute coordinator. No fabricated staging assignment.**

```
real WATCHER execution
  → window_optimizer manifest defaults
  → window_optimizer.py
  → real MultiGPUCoordinator
  → RANGE-MINER backend
  → coordinator staging
  → all required trial phases
  → committed 22-array NPZ
  → Step-2 load-back with fallback_used=False
```

**All of these must hold:**

- **no** `self.staging_dir = …` inside a harness substitute
- **no** CLI-only `--miner-output-dir` injection standing in for the manifest
- the canonical staging value **originates from the same configuration path production uses**
- **at least one COMPLETE trial progresses beyond where the soak failed**
- **all four workflow phases** required by the selected mode record **valid provenance**
- trial commit · acknowledgement · cleanup · reservation release · remote-delete state all checked
- **no staged files, temp files, active reservations or provisional manifests leak** after success
  **or terminal failure**

**Negative gates:**
- missing staging configuration → **fails immediately and non-retryably**
- unsafe filesystem / high-water combination → **fails before work dispatch**
- Blocker-3 retry classifications **unchanged** except the approved condition

> **The falsifiable question this gate answers:** *was staging the only defect, or merely the first
> defect reached on a previously unexercised production path?*

**§3 requires a live fleet.** Beta: *"the focused staging repair may be tested with an
already-running fleet."* **Hand-start the 25 daemons as before** — soak first, then the fleet,
`--gpu-id N --device-index 0`, per-worker `CUPY_CACHE_DIR`, ~3 s stagger. **This is not the
autonomous startup of §7.**

## 4. Amend the infrastructure contract

`docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md` §5 currently states `null → /dev/shm/prng/miner` as
though it covers both. **Amend it to distinguish worker output from coordinator staging**, per §1.
**Keep the original text and mark it clarified — do not delete it.**

## 5. Scope — do NOT do these

- **Do NOT build a worker launcher or supervisor.** §7 below.
- **Do NOT fix `threshold_provenance.json`'s fixed filename.** Beta has recorded it as a **Phase-7
  certification blocker** (sequential trials overwrite each other's audit record) but placed it
  **outside this patch.**
- Do not touch `resume_checkpoint`, the D6.2 checkpoint core, `_l2_sort_key`,
  `_select_l2_winners`, `CANONICAL_ARRAY_CONTRACT`, `utils/prng_encoding`, `canonical_map_hash`,
  or the three finalizer validators.
- **Do not close the incident** if the first trial completes — see §0 and §6's downstream statement.

## 6. Non-regression and the required return

**Suites:** D6.2 (31) · D6.1 · D3.25 · D3.5 · Phase 3 · Phase 4 · D5 · the import gate.
**D6.2 is certified — if a gate reds, STOP and report; do not adjust it.**

**Beta requires Alpha to return with all six:**

1. the **exact resolved production staging path** and **filesystem evidence**
2. **configuration precedence and conflict behaviour** — all five §1.1 rows demonstrated
3. **focused retry-matrix proof** — Blocker-3 unchanged row-for-row outside the one condition
4. **`G-PROD-SHAPE` results**
5. **the first complete production-shape trial**
6. a **downstream-defect statement** — anything else uncovered once the path ran past staging

**Report to `docs/S172_STAGING_PART_B_REPORT.md`.** Then **STOP**.

## 7. Worker startup — investigation only, and NOT in this brief

Beta accepted the owner constraint as **binding**: RANGE-MINER is a standalone Step-1 backend
selected by one flag, so **worker lifecycle is owned inside the backend** — not by WATCHER,
Chapter 13, self-play or any caller-specific layer.

**Beta requires the PWC investigation FIRST**: establish how `persistent/pwc_worker_service.py` and
`persistent_worker_coordinator.py` achieve backend-owned worker availability **before** proposing
the RANGE-MINER equivalent. *(Part A found `pwc.startup()` — "spawns workers, staggers HIP init" —
with `ROCM_SPAWN_STAGGER_S=4.0`, `ROCM_READY_TIMEOUT_S=110.0`, per-node respawn locks and
respawn-on-dead-heartbeat.)*

**Adopt the RESPONSIBILITY, never the code.** Backends are standalone; importing PWC from the miner
path would couple two that must stay independent, and **PWC is retired from certifying authority.**

**Final Phase-7 certification requires backend-owned startup**, so every Step-1 caller gets the same
behaviour. **Separate brief. Do not start it here.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** quote the resolved path, `df` output, the atomic-rename proof, and the
  completed trial's artifacts. **"It works now" is not a result.**
- **clean control:** a correctly-configured run completes a trial; the pre-repair state reproduces
  the `config.staging_dir is not set` failure.
- **fault-injection control:** the two negative gates in §3 — **each must fail for its own reason,
  proven, not merely fail.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **`PASS` requires a COMPLETE
  production-shape trial — not a green WATCHER line.** *(WATCHER scored a crashed step 1.0000 three
  times on 2026-08-04 via `file_exists` on a stale `optimal_window_config.json`.)*
- **audit claim scope:** repo at the stated HEAD **plus the live fleet** for §3.
- **searched surfaces:** name every file and host.
- **unavailable surfaces:** `dmesg` on **all four hosts** including VM101 (`/var/log/kern.log` is
  the substitute) · Proxmox host kernel logs on `.121/.155/.163`.
