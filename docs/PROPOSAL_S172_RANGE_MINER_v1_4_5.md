# PROPOSAL_S172_RANGE_MINER_v1_4_5.md

**Supersedes:** `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (frozen at `1f6c0c5`).
**Status:** Authoritative. v1.4.4 remains in the repo for audit trail; where this
document and v1.4.4 conflict, **v1.4.5 governs**.
**Change driver:** Team Beta binding ruling on TB Ruling Request S175 (host-CPU
parallelism for survivor collection / NPZ assembly, remote spool staging,
high-survivor acceptance, and three-way verification). Adopted as-is by Michael.

---

## Version diff: v1.4.4 → v1.4.5 (S175 ruling absorbed)

All nine points of the S175 binding ruling are folded in below. **No v1.4.4
section is deleted;** the sections listed as "AMENDED" are extended/overridden by
the correspondingly-numbered section here, and new sections (§15–§17) are added.

1. **§3 Architecture — AMENDED.** Adds the Phase 4/5 ownership boundary: Phase 4
   owns stripe scheduling + asynchronous remote spool staging + a bounded local
   shard queue and MUST NOT build final arrays in the dispatch thread; Phase 5
   (`range_miner_npz_writer.py`) owns verification, columnization, dedup, ordering,
   assembly, and contract validation. (S175 §1)
2. **§6.7 / §10 Effort — AMENDED.** Phase 5 is now: a serial reference columnizer
   **plus** a bounded process-sharded verifier/columnizer **plus** a single
   authoritative global merge/dedup/contract writer. (S175 §2, §3, §4)
3. **§12.1 Contract wall — AMENDED.** The full EXPECTED_NPZ_KEYS wall runs on the
   *globally assembled* artifact; per-shard validation is an early gate only and
   cannot replace it. Temp shard files use uncompressed `.npz`/`.npy`; the two
   canonical final files retain `savez_compressed` (frozen writer behavior). (S175 §5)
4. **§12.3 Reassign policy — PRESERVED (one-retry-then-fail-trial, TB Q3).**
   Clarified: the incumbent ZMQ `MAX_ATTEMPTS=3` is NOT inherited; miner policy is
   one retry then fail. (S175 references; TB Q3 unchanged from v1.4.4)
5. **NEW §15 — Remote spool staging** (S175 §6): mandatory; Zeus cannot read a
   rig's `/dev/shm`; asynchronous, bounded, hash-verified transfer.
6. **§11 Acceptance — AMENDED + NEW §16** (S175 §7, §8, §9): high-survivor
   end-to-end throughput is a release-blocking Phase 6 dimension; Phase 7 soak
   must embed high-survivor trials; three-way PWC/ZMQ/miner verification is
   mandatory.
7. **NEW §17 — Backend promotion rule** (S175 §7): `process_sharded` becomes the
   production default only if it beats `serial_reference` by the stated margins on
   the high-survivor trial.

Everything in v1.4.4 not listed above is carried forward unchanged, including:
§4 Step 1 output contract (frozen), §4.2 the 22-array NPZ schema, §4.3 dedup
contract, §5 PRNG-agnostic design + §5.3 audited kernel-arg layout, §6.8 4-phase
test-both-modes workflow, §7 WATCHER manifest, §8 PWC/ZMQ coexistence, §12.4
per-family VRAM caps (TB Q2), and §13 TB Q1–Q4 rulings.

---

## 3. Architecture overview — AMENDED (S175 §1)

The v1.4.4 architecture (persistent per-GPU pull daemons, stripe leases, framed
JSON protocol, coexistence flags `use_persistent_workers` / `use_zmq_sqlite` /
`use_range_miner`) is unchanged. This section adds the **Phase 4 / Phase 5
ownership boundary** mandated by S175.

### 3.A Phase 4 — coordinator (range_miner_coordinator.py)

Phase 4 **owns:**
- stripe scheduling (assignment, lease, one-retry-then-fail per §12.3);
- receipt of spool **manifests** from workers (not payloads);
- **remote spool transfer** to Zeus staging (see §15);
- size and SHA-256 metadata tracking;
- acknowledgement + cleanup lifecycle (worker spool eligible for deletion only
  after Zeus verifies the transferred bytes);
- a **bounded queue** of locally-staged shards handed to Phase 5.

Phase 4 **MUST NOT** build the 22 final arrays inside the coordinator dispatch
thread. The dispatcher must stay responsive enough to keep the GPU fleet fed;
assembly lives behind a queue in Phase 5. (S175 §1)

### 3.B Phase 5 — NPZ writer (range_miner_npz_writer.py)

Phase 5 **owns:** shard verification; spool parsing; typed partial-array
construction; global deduplication; global ordering; final assembly; contract
validation; canonical NPZ writes.

The incumbent ZMQ coordinator loads each payload file one at a time (serial). That
serial-collection behavior **must not be copied unchanged** into the miner path.
(S175 §1)

---

## 6.7 / 10. Phase 5 assembly design — AMENDED (S175 §2, §3, §4)

Phase 5 is implemented as **two interchangeable backends behind one interface**
plus a single authoritative merge stage:

```
assembly_backend = serial_reference | process_sharded
```

Both backends MUST call the **same record-to-field conversion logic** (one shared
columnizer), so they are provably equivalent.

### 6.7.A Approved multiprocessing model (S175 §2)

A **persistent, bounded, CPU-only** process pool. Each process receives ONLY a
small manifest — never data:

```python
{
    "local_spool_path": "...",     # already staged on Zeus (see §15)
    "expected_size": 123456,
    "expected_sha256": "...",
    "stripe_id": "...",
    "sub_index": 4,
    "trial_metadata": {...},
}
```

Each process then:
1. opens the staged spool itself;
2. verifies byte count and SHA-256;
3. parses `s172_substripe_v1`;
4. converts records in **one pass** into typed partial arrays;
5. runs shard-level structural validation;
6. writes an internal **uncompressed** shard artifact (`.npz`/`.npy`);
7. returns ONLY a compact result manifest (paths, counts, hashes).

**Prohibited (all four, verbatim from S175 §2):**
- returning survivor dictionaries through `multiprocessing.Queue`;
- returning the 22 NumPy arrays through pickle;
- sending a giant parsed JSON object from parent to child;
- using 24 processes merely because Zeus exposes 24 logical threads.

**Process safety:** use `spawn` or `forkserver`, **not** an unsafe
post-thread/post-GPU fork. Assembly processes MUST NOT import Torch or CuPy or
initialize a GPU context.

**Benchmark sweep:** start with **1, 2, 4, 6, and 8** assembly processes. Do not
assume 12 or 24 is optimal — hashing, JSON parsing and array copying contend for
memory bandwidth and page cache. (S175 §2)

### 6.7.B Option C's exact role — single-pass typed columnization (S175 §3)

Inside each shard (and inside the serial reference), the record→field conversion:
- makes **one logical pass** over survivor records;
- uses preallocated typed arrays where the shard count is known;
- applies the established field defaults and encodings;
- uses **NO** 22 independent full-list comprehensions.

Do **not** add pandas merely to claim vectorization — it adds an object-conversion
layer and a dependency without proving the row→column extraction left Python.
(S175 §3)

The `serial_reference` backend is retained as: the **correctness oracle**, the
**fallback**, a **benchmark baseline**, and a **debugging mode**. The
`process_sharded` backend is the Phase 6 production *candidate* (promotion gated —
see §17). (S175 §3)

### 6.7.C Final merge — authoritative & deterministic (S175 §4)

The **parent process is the sole owner of global state.** It performs:
- global **highest-score-per-seed** deduplication;
- stable equal-score tie handling;
- strict **ascending seed order**;
- concatenation of all 22 arrays;
- final encoding checks;
- canonical writes.

Assembly workers MUST NOT mutate shared dedup state. Reuse the existing
accumulator's **vectorized NumPy** lookup/merge/sort model — do **not** replace it
with concurrent Python dictionaries. (S175 §4)

---

## 12.1 EXPECTED_NPZ_KEYS contract wall — AMENDED (S175 §5)

The v1.4.4 §12.1 wall (exact 22 keys, dtypes, equal lengths, strict seed ordering,
known encoding values) is unchanged in content but its **application point is
fixed by S175:**

- The full wall runs on the **globally assembled** artifact — after merge/dedup/
  ordering. A single shard cannot prove global uniqueness, highest-score collision
  resolution, strict global seed ordering, or equal final lengths, so per-shard
  validation is an **early integrity gate only** and cannot replace final
  validation.
- Both canonical outputs — the final NPZ **and** the accumulator NPZ — must pass
  the complete validator **before** the prior artifact is replaced.
- **Internal temporary shard files are not public contract artifacts.** They use
  **uncompressed** `.npz`/`.npy` (S159B: uncompressed `np.savez` measured 71×
  faster for temporary payloads) and are removed after successful final validation
  and coordinator acknowledgement.
- The two **canonical final files retain `np.savez_compressed`** for now — that is
  the frozen §12.1 writer behavior and the incumbent accumulator behavior. A
  separate measured amendment may revisit final compression later. (S175 §5)

---

## 15. Remote spool staging — NEW, MANDATORY (S175 §6)

A worker's `/dev/shm/prng/miner/...` path is **local to that node**. Zeus cannot
directly open an RX 6600 rig's shared-memory path (established in the S150
analysis: remote-node shared memory is not accessible to Zeus).

Phase 4 MUST therefore implement, explicitly:

```
remote spool
  → binary transfer to Zeus staging
  → verify local bytes against the worker SHA-256
  → enqueue local path for Phase 5
  → coordinator acknowledges verified collection
  → remote worker spool becomes eligible for deletion
```

Requirements:
- Transport MAY be persistent SFTP/SSH, rsync-style, or a dedicated framed binary
  channel. It **MUST NOT** convert the spool back into JSON over stdout.
- Transfer MUST be **asynchronous and bounded**. A **high-water mark** is required
  so unlimited staged spool data cannot fill `/dev/shm` or the Zeus filesystem.
- A spool file is eligible for deletion on the worker **only after** Zeus has
  verified the transferred bytes against the worker-provided SHA-256. (S175 §6)

---

## 16. Acceptance criteria — AMENDED + EXTENDED (S175 §7, §8, §9)

All v1.4.4 §11 criteria (11.A–11.M) remain blocking. S175 adds the following, all
**release-blocking**.

### 16.A High-survivor end-to-end throughput (Phase 6) — S175 §7

Measure **end-to-end** throughput, from first stripe assignment until the final
canonical NPZ has been validated and durably replaced. **GPU-kernel-only seeds/s
must be reported separately and CANNOT be used as the acceptance number.**

Required stage timings (each reported):
```
gpu_execution_s
remote_spool_transfer_s
sha256_verification_s
json_parse_columnize_s
global_dedup_merge_s
final_npz_write_s
contract_validation_s
end_to_end_s
```

Required acceptance (all must hold):
- median of **at least three warmed runs**;
- high-survivor throughput **≥ 500,000 seeds/s** (carries forward the explicit
  S150 target);
- high-survivor throughput **≥ 25%** of the miner's low-survivor throughput on the
  same hardware;
- **no** OOM, swap storm, unbounded queue growth, or abandoned spool files;
- 22-array identity **and** dict identity pass;
- final artifact count equals the verified survivor/dedup accounting.

**Gate applicability (Team Beta clarification):** the ≥ 500,000 seeds/s and
≥ 25%-of-low-survivor gates apply to **whichever RANGE-MINER backend is selected
as the production default** (§17). Both miner backends must be measured, but a
non-promoted experimental backend does not independently block release if the
selected production backend passes.

### 16.B Three-way PWC / ZMQ / miner verification (Phase 6) — S175 §9

For the same **low-survivor and high-survivor** inputs, run all four engines:
```
PWC
ZMQ+SQLite
RANGE-MINER serial_reference
RANGE-MINER process_sharded
```

Compare all pairs using `np.array_equal(a[k], b[k])` for **every one of the 22
arrays**, plus NPZ→dict equality sorted by seed.

- **Do NOT require raw `.npz` file-byte identity** — ZIP metadata, compression and
  member layout can differ while the 22 arrays are exactly identical.
- **PWC remains the frozen authoritative comparator; ZMQ is the mandatory third
  oracle.**
- Any two-versus-one divergence **must be localized before Phase 6 passes** — it
  may **not** be dismissed as "probably the old path."
- Throughput for all **four execution paths** is recorded on the same hardware/
  workload, but **only RANGE-MINER is subject to the 500K high-survivor gate**
  (see the clarification below). (S175 §9)

**500K gate applicability (Team Beta clarification, v1.4.5):** the ≥ 500,000
seeds/s and ≥ 25%-of-low-survivor gates apply to **whichever RANGE-MINER backend
is selected as the production default** (§17). Both miner backends
(`serial_reference` and `process_sharded`) must be **measured**, but a
non-promoted experimental backend does **not** independently block release if the
selected production backend passes. (Resolves the case where `process_sharded`
fails the §17 20% promotion threshold while `serial_reference` meets the
production throughput gates.)

### 16.C Phase 7 soak — S175 §8

The 50-trial WATCHER soak MUST include:
- **≥ 5** deliberately high-survivor trials;
- **≥ 5** low-survivor control trials;
- mixed constant **and** hybrid modes;
- cleanup verification after **every** trial;
- **no monotonic** spool backlog, RSS, or temporary-file growth.

A soak that happens to produce only low-survivor outputs does **not** validate this
design. (S175 §8)

---

## 17. Backend promotion rule — NEW (S175 §7)

`process_sharded` becomes the **production default** only if, on the high-survivor
trial, it provides ALL of:
- **≥ 20%** median end-to-end improvement over `serial_reference`;
- **identical** final arrays;
- **≤ 50%** host-RAM peak RSS;
- **no** swap usage.

If it does not meet all four, `serial_reference` **remains the default**, and the
process backend + instrumentation remain available for future tuning. (S175 §7)

---

## Final architecture (S175, adopted)

```
Phase 4 (coordinator):
    asynchronous remote spool staging + bounded manifest/shard queue
    (MUST NOT assemble arrays in the dispatch thread)

Phase 5 (range_miner_npz_writer.py):
    serial reference columnizer
    + bounded process-sharded verifier/columnizer (spawn/forkserver, CPU-only)
    + single authoritative global merge / dedup / ordering / contract writer

Phase 6 (acceptance):
    three-way (PWC / ZMQ / miner ×2) correctness oracle — 22-array + dict identity
    + explicit high-survivor end-to-end benchmark (≥500K s/s, ≥25% of low-survivor)
    + benchmark-driven default backend selection (§17)

Phase 7 (soak):
    50-trial WATCHER soak with ≥5 high-survivor + ≥5 low-survivor trials,
    mixed constant/hybrid, per-trial cleanup verification
```

**Provenance:** this document adopts the Team Beta binding S175 ruling verbatim in
sections 3.A/B, 6.7.A/B/C, 12.1, 15, 16.A/B/C, and 17. The frozen v1.4.4 content
not touched by S175 is authoritative as written there. Phase 0–3 (committed
`2389b61`, `8d0183f`, `e0c9d1c`, `dbe3d0e`) are unaffected — S175 governs only the
not-yet-built Phases 4–7.
