# S172 Phase 6.0 — ROCm parity evidence record

**Status: acceptance met. Team Alpha review gate. NOT committed, NOT pushed.**

Brief: `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_0_ROCM_PARITY.md` (REV1)
Executed on VM 101 (`zeus-ubuntu-vm`, `192.168.3.177`) as `michael`, 2026-07-30.

**Base commit substitution (brief §0):** the brief names `3823b56`. HEAD is
**`8e2f5bf`**, which is docs-only on top of `3823b56` (the Phase 6.0 brief itself).
`8e2f5bf` was used on **both** sides, so the identical-source requirement is met at a
commit that strictly contains the named base.

---

## 1. Deployment evidence (§1)

Deployed by `git clone` — so the CT-side `git rev-parse HEAD` *proves* source identity by
construction rather than by trust.

| item | value |
|---|---|
| CT-side commit | `8e2f5bfd46c868fb5f47b84782344608e88c7c81` |
| CT tracked-dirty | `[]` (empty) |
| CT untracked | `[]` (empty) |
| CT path | `/home/michael/distributed_prng_analysis` |
| venv | `/home/michael/rocm_env/bin/python` (Python 3.10.4) |

**Finding — the clone does not carry the dataset.** `daily3.json` is gitignored
(`.gitignore:41`, `*.json`), so it was **absent** after cloning and was deployed by `scp`.
This does not weaken the identity claim; it strengthens it, because identity is enforced
cryptographically at runtime rather than by deployment discipline:
`range_miner_worker.py:641-650` makes `dataset_sha256` **mandatory** on every assignment and
non-retryable on mismatch, and `:664-669` verifies `residue_sha256` against the locally
recomputed window. A wrong dataset fails the trial; it cannot silently diverge.

    daily3.json sha256 (both hosts): 513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6

---

## 2. §0 prerequisites — re-confirmed live, not assumed

| requirement | re-confirmed result |
|---|---|
| rig reachable, key auth | `ssh -o BatchMode=yes michael@192.168.3.122` → `rrig6600` ✅ |
| GPUs bound into LXC | `/dev/kfd` present; `card0-7` + `renderD128-135` present ✅ |
| ROCm enumeration | 8 × **AMD Radeon RX 6600 XT**, DID `0x73ff`, rev `0xc1` ✅ |
| cupy under ROCm | 13.5.1, `getDeviceCount()==8`, `is_hip=True`, real kernel executes ✅ |
| ROCm runtime | `runtimeGetVersion()==60443484` ✅ |
| env overrides | **NONE set** — `env | grep -iE "HSA|HIP|ROCR|GPU_|AMD_"` empty. Verified **positive finding**, not an omission ✅ |
| CT reaches GitHub | clone succeeded at `8e2f5bf` ✅ |
| repo on CT | was **ABSENT**; deployment performed ✅ |
| ROCm venv | `~/rocm_env` present ✅ |

Additionally confirmed before committing to a full run:
* CT100 → VM 101 TCP reachable (no firewall block) — live accept test.
* **Both java_lcg kernels compile on gfx1032 under hiprtc** (`java_lcg_flexible_sieve`,
  `java_lcg_reverse_sieve`) — `cp.RawKernel` compiles CUDA C through HIP, the single
  largest platform risk.

---

## 3. The identical trial configuration (§2)

Every parameter below was identical across both platforms; the **only** difference is which
silicon executed the sieve kernel.

    --seed-start 0        --seed-count 8000000     --stripe-size 4000000
    --seed-cap 1000000    --window-size 3          --offset 0
    --forward-threshold 0.31   --reverse-threshold 0.47
    sessions=[midday, evening]   skip_range=[0,16]   skip mode: constant, phases 1 and 2
    java_lcg · serial_reference · test_both_modes=False · daily3.json

**What is and is not remote.** Only the GPU sieve ran on the rig. The coordinator, Phase-5
assembly, the finalizer and the NPZ writer ran on VM 101 for **both** runs. So a difference
in the 22 arrays could only originate in the kernel — the writer is common-mode and cannot
mask or manufacture a divergence.

**Why the two runs are comparable at all.** The coordinator sizes sub-stripes with the cap
the worker *advertises* (`advertised_effective_cap` → `select_seed_cap`, which branches on
backend: `range_miner_worker.py:472-479`). The harness advertises **all four** caps equal to
1,000,000, so the effective cap is 1,000,000 whether the worker reports `rocm` or `cuda`, and
sub-stripe boundaries — hence canonical record **order** — match by construction. Had the caps
differed per family, the two runs would have partitioned differently and the comparison would
have been meaningless.

**Transport (verified, not assumed).** WOI injects no `TransferAdapter`, so a spooled result
would fail the stripe loudly at `range_miner_coordinator.py:4014` rather than corrupt
anything. Measured: **0 spool files** on either host, **16 shards** (2 stripes × 4 sub-stripes
× 2 phases) all `verified`, **max shard 1.83 MiB** against the 48 MiB `INLINE_BYTE_LIMIT` —
26× headroom. Every sub-stripe result crossed the wire inline.

**Independent proof the rig actually did the work.** The coordinator's own ledger for the ROCm
run records the sole registered worker as:

    workers: [('rrig6600:gpu0', 'eligible')]
    shards : [('verified', 16)]

`worker_id` is built as `{socket.gethostname()}:gpu{gpu_id}` by the worker itself, so
`rrig6600:gpu0` is the rig's identity asserted from inside CT100 — not a harness label. It
registered `eligible` (not quarantined), which also confirms its advertised seed caps matched
the central configuration.

---

## 4. Platform identity and backend proof

### CUDA control — VM 101
| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti, 12288 MiB, driver 550.163.01 |
| cupy / runtime | 13.5.1 / CUDA `12090`, `is_hip=False` |
| python / numpy | 3.10.12 / 1.22.0 |
| venv | `~/venvs/torch` |

### ROCm subject — CT100 `rrig6600`, device 0
| | |
|---|---|
| hostname | `rrig6600` (worker_id `rrig6600:gpu0`) |
| device | **AMD Radeon RX 6600 XT**, DID `0x73ff`, rev `0xc1`, subsystem `0x6501`, GUID `59111` |
| gfx arch | `gfx1032`, BUS `0000:03:00.0` |
| VBIOS | `113-123XT145W201222` |
| kernel | `6.8.12-9-pve` |
| amdgpu module | `6.12.12` |
| ROCm release | `6.4.3-128` (`rocm-core 6.4.3.60403-128~22.04`) |
| cupy / runtime | 13.5.1 / HIP `60443484` |
| python / numpy | 3.10.4 / 2.0.2 |
| env overrides | **none set** |

**Explicit ROCm/HIP backend identity — from production code, not a harness assertion.**
`range_miner_worker.py:1083` computes `backend = "rocm" if rt.is_hip else "cuda"`, and that
value is what `select_seed_cap()` branches on. Read back from the deployed tree on the rig:

    is_hip                        True
    backend_as_worker_computes_it rocm
    gcnArchName                   b'gfx1032'
    device_name                   b'AMD Radeon RX 6600 XT'

This is the ROCm path, not a CUDA fallback.

---

## 5. Threshold provenance (§3.B) — both platforms

Asymmetric, non-default, and `effective` returned **by the executor**, not recomputed from
config. The parent's fail-closed gate ran and passed (`validated=True`) on both runs.

| | CUDA | ROCm |
|---|---|---|
| requested | forward=0.31 reverse=0.47 | forward=0.31 reverse=0.47 |
| payload | `{'1': [0.31], '2': [0.47]}` | `{'1': [0.31], '2': [0.47]}` |
| effective | `{'1': [0.31], '2': [0.47]}` | `{'1': [0.31], '2': [0.47]}` |
| phase→direction | `{'1':'forward','2':'reverse'}` | `{'1':'forward','2':'reverse'}` |

`requested == payload == effective` for both directions on both platforms. **Identical.**

---

## 6. Counts — three-way comparison (§3, Alpha addition)

| | forward | reverse | bidirectional | rows | artifact_sha256 |
|---|---|---|---|---|---|
| **D6 release-grade CUDA** (`b08c2c5`, `gen-20260730T002104136270Z`) | 398,156 | 383 | 319 | 319 | `0e0092fe…c1c4b0` |
| **Phase 6.0 CUDA control** (`8e2f5bf`) | **398,156** | **383** | **319** | **319** | `0e0092fe…c1c4b0` |
| **Phase 6.0 ROCm subject** (`8e2f5bf`, RX 6600 XT) | **398,156** | **383** | **319** | **319** | `0e0092fe…c1c4b0` |

All three agree. **No CUDA-side regression since `b08c2c5`** — so there is nothing to report
separately under the brief's "CUDA mismatch ⇒ regression, not platform difference" clause.

**The third comparison is stronger than the brief anticipated.** It asked only that the counts
be checked against the previously-certified artifact. In fact the **full artifact SHA-256 of
the certified NPZ is identical across all three** — the already-certified release-grade CUDA
generation from `b08c2c5`, the Phase 6.0 CUDA control, and the Phase 6.0 ROCm generation
produced on an RX 6600 XT. Figures taken from the committed record
`docs/D6_RELEASE_GRADE_SMOKE_20260729.log:115-...`, not from the brief's summary.

That single hash equality subsumes the count check, the ordering check and the field-for-field
check simultaneously, and it does so against an artifact certified a day earlier, on different
silicon, at a different commit, in a different repository mode.

---

## 7. The 22-array `np.array_equal` matrix (§3, required addition)

CUDA `gen-20260730T200213704957Z-step1_java_lcg_0` vs
ROCm `gen-20260730T200235425529Z-step1_java_lcg_0`.

    canonical ORDER identical to frozen oracle (CUDA): True
    canonical ORDER identical to frozen oracle (ROCm): True
    canonical ORDER identical CUDA vs ROCm           : True

|  # | array | dtype | rows | `array_equal` |
|---:|---|---|---:|---|
|  1 | seeds | uint32 | 319 | **True** |
|  2 | forward_matches | float32 | 319 | **True** |
|  3 | reverse_matches | float32 | 319 | **True** |
|  4 | window_size | int32 | 319 | **True** |
|  5 | offset | int32 | 319 | **True** |
|  6 | trial_number | int32 | 319 | **True** |
|  7 | skip_min | int32 | 319 | **True** |
|  8 | skip_max | int32 | 319 | **True** |
|  9 | skip_range | int32 | 319 | **True** |
| 10 | forward_count | float32 | 319 | **True** |
| 11 | reverse_count | float32 | 319 | **True** |
| 12 | bidirectional_count | float32 | 319 | **True** |
| 13 | intersection_count | float32 | 319 | **True** |
| 14 | intersection_ratio | float32 | 319 | **True** |
| 15 | intersection_weight | float32 | 319 | **True** |
| 16 | bidirectional_selectivity | float32 | 319 | **True** |
| 17 | forward_only_count | float32 | 319 | **True** |
| 18 | reverse_only_count | float32 | 319 | **True** |
| 19 | survivor_overlap_ratio | float32 | 319 | **True** |
| 20 | score | float32 | 319 | **True** |
| 21 | skip_mode | uint8 | 319 | **True** |
| 22 | prng_type | uint8 | 319 | **True** |

**All 22 canonical arrays are field-for-field equal across CUDA and ROCm, and the canonical
record order is identical on both platforms.** No divergence to localize.

---

## 8. Generation identities and hashes

| | CUDA control | ROCm subject |
|---|---|---|
| generation_id | `gen-20260730T200213704957Z-step1_java_lcg_0` | `gen-20260730T200235425529Z-step1_java_lcg_0` |
| **artifact_sha256** | `0e0092feeb02e22d28557ddf4d8e421941d6117bcc0448d7f7323ec402c1c4b0` | `0e0092feeb02e22d28557ddf4d8e421941d6117bcc0448d7f7323ec402c1c4b0` |
| sidecar_sha256 | `48690c40a6d5f0cd036ca40470c7330f6946ff700bc07e27dd7c42ddb49777c3` | `ecefd609467dd1da2d8ba9421b3b4f063dbb78c25245250e42102902905e35fc` |
| raw / L2 / final | 319 / 319 / 319 | 319 / 319 / 319 |
| repository_mode | scratch | scratch |
| trial wall time | 24.7 s | 15.9 s |

**The NPZ artifact hash is byte-identical across platforms** — a stronger result than the
required field-for-field equality — **and it also equals the D6 release-grade CUDA artifact
certified at `b08c2c5`** (§6). The **sidecar** hashes differ, and that is expected and
correct: the sidecar carries the per-run generation timestamp and the scratch-snapshot commit,
which are run-unique by construction. It does not carry survivor data.

Timing is recorded **for context only**. Phase 6.0 makes no throughput claim. (The ROCm trial
being faster in wall time is not a performance finding — trial wall time is dominated by
coordinator/assembly overhead, not kernel execution.)

Step-2 loader read-back, both runs: `format=npz npz_version=3 count=319`
**`fallback_used=False`** ✅ — identical metadata.

---

## 9. ROCm health evidence (§4)

### 9.1 Kernel-log access — a real limitation, reported rather than papered over

**The brief's `dmesg` before/after diff could not be captured, and capturing it *inside* the
container would have been worse than not capturing it.** CT100 is an **unprivileged LXC**:

    dmesg           → "dmesg: read kernel buffer failed: Operation not permitted"
    /dev/kmsg       → No such file or directory
    journalctl -k   → "-- No entries --"
    sudo            → password required (no passwordless sudo)
    root@192.168.3.121 (Proxmox host) → Permission denied (publickey,password)

The `amdgpu` driver runs in the **Proxmox host kernel** (`6.8.12-9-pve`), not in the
container. A `dmesg | grep amdgpu` executed inside the CT would have returned **empty
regardless of what the GPU did** and read as "no GPU faults" — precisely the vacuous pass §4
warns against ("its absence must be evidenced, not assumed"). It is therefore reported as a
**gap**, with the substitute surfaces below.

**To close this gap, from a host with root on the Proxmox node** (`192.168.3.121`), around a
run:

    dmesg --ctime > /tmp/dmesg_before.txt
    # ... run the ROCm smoke ...
    dmesg --ctime > /tmp/dmesg_after.txt
    diff /tmp/dmesg_before.txt /tmp/dmesg_after.txt \
      | grep -iE "amdgpu|GPU reset|L2 protection|VM_L2|vm fault|ring timeout"

VM 101 does not currently hold key auth to `192.168.3.121` as root, so this is Michael's to
run (or a key to add) — it is not something the agent could obtain.

### 9.2 What WAS captured — root-free, and responsive to the failure class

Before/after diff over PCIe replay counter, KFD topology node count, DRI device-node
inventory, VRAM allocation and concise hardware info:

    ROCm HEALTH BEFORE/AFTER DIFF  →  (no difference — identical before and after)

| surface | before | after | meaning if it had moved |
|---|---|---|---|
| PCIe replay count | 0 | 0 | bus-level errors / link retraining |
| KFD topology nodes | 9 | 9 | GPU reset or fall-off-bus |
| DRI nodes | card0-7 + renderD128-135 | same | device disappearance |
| VRAM allocated | 0 % | 0 % | leaked/stuck context |
| device identity | `0x73ff` rev `0xc1` gfx1032 | same | re-enumeration after reset |

**Post-run functional probe** — a fresh kernel launched on device 0 *after* the run:

    post_run_kernel_sum 999000
    expected            999000

If the GPU had reset or its context been lost, this launch would have failed. It did not.

**Worker-log fault scan** — 14 patterns (`GPU reset`, `ring timeout`, `L2 protection fault`,
`VM_L2`, `VMC page fault`, `vm fault`, `amdgpu:`, `HSA_STATUS_ERROR`, `hipError`, `HIP error`,
`Memory access fault`, `GPU hang`, `soft recovery`, `MES failed`):

    matching lines : 0   (none)

The complete ROCm worker log is 346 bytes: a benign `runpy` RuntimeWarning and the two kernel
compilations. No traceback, no HIP error, no fault.

**Note on RAS:** `--showretiredpages` / `--showpendingpages` / `--showrasinfo` return
`"ras, Not supported on the given system"` on consumer RDNA2 (GFX/SDMA/UMC RAS all `N/A` in
the hardware table). Those surfaces are therefore **unavailable**, not "clean" — stated so the
absence of RAS counters is not misread as a passing RAS check.

### 9.3 Worker exit and cleanup

    [WORKER] alive for the whole trial, no premature exit: True
    [WORKER] post-terminate code 255 (SSH CLIENT teardown code, not the worker's own exit status)
    [REMOTE CLEANUP] surviving worker processes on the rig: 0
    [REMOTE SPOOL]   files left in /home/michael/s172_phase60_spool: 0

`255` is the **ssh client's** code on channel teardown — it is what the harness's own
`terminate()` produces and is **not** a worker crash. The meaningful signal is
`alive_before_terminate = True`: the worker survived the entire trial without exiting. Verified
independently after the run: no residual `range_miner_worker` process on the rig, no
`~/miner_output`, no `.s172_checkpoint` on the rig.

### 9.4 Spool / temp / checkpoint state

| host | result |
|---|---|
| rig spool `/home/michael/s172_phase60_spool` | **0 files** |
| rig `~/miner_output` | absent |
| rig residual processes | none |
| VM 101 `/dev/shm/prng` | absent |
| VM 101 `~/miner_output` | 0 entries |
| VM 101 repo `.s172_checkpoint/` | 7 run-isolated subdirs, one per smoke run — **inside** the namespace |

The `.s172_checkpoint/` entries are named `{hostname}-{pid}-{epoch}`, exactly D6.1's
run-isolated namespace, with **7 directories for the 7 smoke runs executed this session and no
cross-run collision** — i.e. D6.1's isolation repair working as designed. There is **no residue
outside the run-isolated namespace**, which is what §4 asks. The directory is gitignored
(`.gitignore:51`), so it affects neither tracked-cleanliness nor gate 22. These artifacts
accumulate rather than self-clean; retention/GC is **D6.3**, explicitly out of scope per §6.

---

## 10. Artifact classification (§5)

> **ROCm platform-validation certified generation — non-authoritative**

Emitted in the run banner and in the terminal output of the ROCm run. The D6 CUDA generation
`gen-20260730T002104136270Z-step1_java_lcg_0`, certified at `b08c2c5`, **remains the
authoritative release-grade artifact.** Phase 6.0 proves platform equivalence on different
silicon; it does not replace or compete with that artifact.

**Repository mode — both runs used SCRATCH, deliberately.** Extending the tracked harness makes
the VM 101 tree tracked-dirty, and `--release-grade` aborts on a dirty tracked tree by design.
That is the correct outcome here rather than an obstacle: §5 forbids classifying the ROCm
output as a second release-grade generation. Running **both** sides in the same mode also keeps
the comparison symmetric. §3.B's "clean committed repository" leg is satisfied where the clause
points — **CT-side**: fresh clone at `8e2f5bf`, tracked-clean, zero untracked.

---

## 11. Non-regression (§7)

### 11.1 Gate suite — green before and after

All 15 gates run at `8e2f5bf`, **before any edit** and **after** the harness extension:

| gate | before | after |
|---|---|---|
| D1.1 (`d1_engine`) · D1.0 (`d1_workflow`) · D0 | rc=0 ✅ | rc=0 ✅ |
| D2 · D3.0 · D3 (`columnizer`) · D3.25 · D3.5 | rc=0 ✅ | rc=0 ✅ |
| D4 · D5 · D6 3.A (`production_adapter`) · D6-threshold | rc=0 ✅ | rc=0 ✅ |
| **D6.1** (`flush_durability`) · Phase 3 · Phase 4 | rc=0 ✅ | rc=0 ✅ |

*(post-edit results in §11.4 below)*

### 11.2 The CUDA default path is unchanged — proven two ways

**Structurally.** The extension is additive. The whole diff is **+449/−13**, and every one of
the 13 removed lines is either a re-indentation or a default-preserving parameterization
(`miner_host="127.0.0.1"`, `target=None`). The CUDA `subprocess.Popen(...)` call retains its
original statements verbatim inside `if target is None:`. Output ordering was deliberately
preserved too: `_nvidia_smi("before the run")` was kept in its original position, and the added
worker-exit lines are gated to the ROCm path, so the default run's stdout is unchanged.

**Empirically — the stronger proof.** The same trial was run with the **pristine** harness
(pre-edit) and with the **extended** harness, on CUDA:

* all **22 arrays** `np.array_equal` → **True**
* counts identical: 398,156 / 383 / 319
* **`artifact_sha256` identical**: `0e0092fe…c1c4b0`

A byte-identical certified artifact from the pristine and extended harness is the strongest
available evidence that the default path did not change.

### 11.3 Gate-22 constraint (why the harness was extended, not duplicated)

`test_s172_phase4_coordinator.py:1602 gate22_coexistence()` runs `git status --porcelain`
(**untracked included**), filters `.py` paths and asserts the set is a subset of an explicit
whitelist. `tests/smoke_s172_phase5_d6_zeus_single_gpu.py` **is** whitelisted (line 1836), so
modifying it is gate-safe — whereas **any new `.py` under the repo would have redded §7**. This
independently confirms the brief's "extend it, do not write a second harness" instruction. The
comparator was therefore added inside the whitelisted harness as `--compare`, not as a new file.
Verified at report time: **zero untracked `.py` in the repo**; the only tracked modification is
the harness itself.

### 11.4 Post-edit suite result — all 15 green

    test_s172_phase5_d1_engine                  rc=0     test_s172_phase5_d4_serial_backend      rc=0
    test_s172_phase5_d1_workflow                rc=0     test_s172_phase5_d5_process_sharded     rc=0
    test_s172_phase5_d0                         rc=0     test_s172_phase5_d6_production_adapter  rc=0
    test_s172_phase5_d2_directional_uniqueness  rc=0     test_s172_phase5_d6_threshold_path      rc=0
    test_s172_phase5_d3_0_encoding_contract     rc=0     test_s172_d6_1_flush_durability         rc=0
    test_s172_phase5_d3_columnizer              rc=0     test_s172_phase3_worker                 rc=0
    test_s172_phase5_d3_25_candidate_ingress    rc=0     test_s172_phase4_coordinator            rc=0
    test_s172_phase5_d3_5_finalizer             rc=0

`test_s172_phase4_coordinator` reports **63/63 checks green**, including
**`[PASS] Gate 22: coexistence (use_range_miner, PWC/ZMQ)`** — the gate that inspects
`git status --porcelain` for non-whitelisted `.py` paths (§11.3). It executed **after** the
final harness edit, so it validated the tree exactly as delivered.

Zero non-zero return codes before or after. **No regression.**

---

## 12. Findings for separate action (not fixed here, per §6)

1. **SKU discrepancy — wrong in more places than the brief states.** The cards are
   **RX 6600 XT** (device name from the driver, DID `0x73ff`, and VBIOS `113-123**XT**145W201222`
   independently corroborate). `distributed_config.json` carries `"gpu_type": "RX 6600"` at
   **lines 16, 27 and 38** — that is the brief's "three places". However the wrong SKU also
   appears in at least `README.md:25-27`, `config_manifests/parameter_registry.json:91,511,517`,
   `distributed_config_1rig.json:16,27,38`, `enable_sieve_dynamic.py:158`,
   `patch_operating_guide.py` and `apply_s136_doc_updates.py`. **Correction to the brief:**
   `CLAUDE.md` contains **zero** occurrences of "RX 6600" — it is not one of the affected files.
   Nothing was edited. Note that `gpu_type` is descriptive metadata: dispatch and cap selection
   branch on the worker's `is_hip`-derived backend, not on this string.
2. **`daily3.json` is gitignored**, so `git clone` alone cannot stand up a working rig. Any
   future rig-provisioning runbook needs an explicit dataset-deployment step.
3. **Host kernel-log access is missing** for rig GPU-health evidence (§9.1). Worth adding key
   auth from VM 101 to `192.168.3.121` if future phases must evidence absence of amdgpu faults
   at the kernel level.
4. **Harness defect found and fixed during this work:** the remote-cleanup check used
   `pkill -f "range_miner_worker.*--port N"`, which matched the cleanup shell's **own** command
   line and killed it before it could report — yielding a silently **empty** cleanup evidence
   line. Fixed with the bracket trick plus dropping the shell's own PID. This is worth flagging
   as a pattern: a verification step that fails silently reads exactly like a passing one.

---

## 13. Scope compliance (§6)

Not touched: `distributed_config.json`, `CLAUDE.md`, the D6/D6.1 implementation, PWC/ZMQ
ingress, the D3.25 contract, `TestResult`, D5's artifact contract. `serial_reference` remained
the default; `process_sharded` was not promoted. No retention/GC (D6.3), no 24-field checkpoint
(D6.2), no S166 clear. No multi-node coordination, no backend promotion, no saturation claim,
**no WATCHER**. Nothing committed, nothing pushed.

**Files changed (1):** `tests/smoke_s172_phase5_d6_zeus_single_gpu.py` (+449/−13)
**Files added (1):** this evidence record.

---

## 14. Acceptance

| §3 requirement | result |
|---|---|
| clean committed repository (CT-side commit + tracked-clean) | ✅ `8e2f5bf`, clean, 0 untracked |
| certified generation produced on ROCm | ✅ `gen-20260730T200235425529Z-step1_java_lcg_0` |
| frozen-order 22-array `validate_array_bundle()` | ✅ passed, order matches oracle |
| Step-2 loader read-back, `fallback_used=False` | ✅ 319 rows, npz v3 |
| asymmetric non-default thresholds | ✅ 0.31 / 0.47 |
| `requested == payload == effective`, effective from executor | ✅ both directions, `validated=True` |
| `np.array_equal` on **all 22** arrays | ✅ **22/22 True** |
| forward / reverse / bidirectional counts equal | ✅ 398,156 / 383 / 319 |
| canonical record ordering identical | ✅ |
| threshold provenance identical | ✅ |
| Step-2 loader metadata identical | ✅ |
| third comparison vs D6 release-grade | ✅ all three agree; no CUDA regression |

**Phase 6.0 acceptance is met**, with the §4 kernel-log diff reported as an access-limited gap
(§9.1) rather than claimed. **STOP for Team Alpha review.**
