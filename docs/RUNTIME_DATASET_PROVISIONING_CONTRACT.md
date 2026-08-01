# RUNTIME_DATASET_PROVISIONING_CONTRACT.md

**Status:** obligations in this contract are **Phase 6-P0.5**, not Phase 6-P0.
Originally mandated by Team Beta in the Phase 6.0 final ruling §5.1; phase attribution
fixed by the Phase 6-P0 ruling §2.

**Phase boundary (Beta, ratified):**

| phase | scope | state |
|---|---|---|
| **6-P0** | *creates files.* Immutable version publication, atomic pointer manifest, read-only verifier. Inert with respect to every existing run. | **COMPLETE** — commit `131787d`, accepted by Team Beta |
| **6-P0.5** | *changes code.* Everything in this contract — provisioning, per-node verification, **fail-before-dispatch** — plus pointer resolution, freeze-at-run-start and absolute-path dispatch. | **OPEN** |

The separation is deliberate: mixing file creation with behavioural change would make the
first certification after publication unattributable. **§2 and §4 of this contract are
therefore P0.5 obligations and were correctly absent from P0.**

**Problem:** the required dataset is **Git-ignored**, so `git clone` alone is not a
complete rig deployment. Phase 6.0 discovered this on CT100: the clone brought the
code but not `daily3.json`. A manual `scp` plus runtime hash check was sufficient for
one controlled smoke, but **repeated manual transfer is not a cluster provisioning
contract.**

**Requirement:** deterministic provisioning and cryptographic identity. The dataset
does **not** need to live in Git — but its presence and digest must be guaranteed
before any work is dispatched.

---

## 1. The contract — required fields per dataset

Every runtime dataset a worker needs is declared with all of:

| field | meaning |
|---|---|
| `dataset_logical_name` | stable identifier used by config and code (e.g. `daily3`) |
| `expected_sha256` | authoritative digest; the identity of record |
| `expected_size_bytes` | secondary integrity check; catches truncation early |
| `source_location` | canonical source of truth (host + absolute path) |
| `destination_path` | absolute path on the target node |
| `owner` / `group` / `mode` | file ownership and permissions on the target |
| `verification_command` | the exact command that re-derives and compares the digest |
| `failure_behavior` | what happens when absent or mismatched — see §3 |

> **SUPERSEDED — `expected_sha256` is not a statically configured value.**
>
> A fixed digest in a provisioning manifest cannot survive a legitimate new publication.
> Per Beta's dataset-lifecycle ruling and the Phase 6-P0 ruling §3, the authoritative
> identity is **resolved at run start from the pointer manifest** `daily3_current.json`
> and then **frozen for the duration of that run**:
>
> * manifest identity / version;
> * immutable **absolute** dataset path;
> * dataset SHA-256;
> * size and record count.
>
> Every node verifies against **that frozen value**, not against a value stored in this
> contract. **A later pointer change must not alter a run already in progress** (Beta,
> P0 ruling §3).
>
> `source_location` becomes the pointer manifest and the immutable version it names —
> never the bare legacy alias `daily3.json`. Beta's P0 ruling §1: publish-in-place is
> ratified as a *location*; after P0.5 the pointer-selected immutable version is
> authoritative and `daily3.json` is a **legacy compatibility alias**.

Recorded as a machine-readable manifest (a JSON file outside Git is acceptable), so
provisioning is reproducible rather than remembered.

## 2. Fail before dispatch — the hard rule

> **The worker must fail before dispatch if the dataset is absent or its digest
> differs from `expected_sha256`.**

Not at first read. Not partway through a trial. **Before any GPU work, spool creation,
or coordinator assignment.** A trial that consumes compute and then fails on a bad
dataset wastes the run and produces a confusing failure two layers deep.

This is the same fail-closed discipline as the D6 threshold provenance gate and the
seed-cap mismatch: detect at the boundary, refuse loudly, never proceed on unverified
input.

## 3. Failure behavior

| condition | required behavior |
|---|---|
| dataset absent | fail before dispatch; name the dataset, the expected path, and the node |
| digest mismatch | fail before dispatch; report expected vs actual digest **and** both sizes |
| size mismatch, digest match | impossible — treat as a bug in the checker, fail closed |
| source unreachable during provisioning | provisioning fails; **do not** start the run with a partial fleet |
| digest matches | proceed; record the verified digest in the run evidence |

> **Classification (Beta, P0 ruling §3 correction).** A missing dataset is **not**
> semantically a residue error. Do not flatten `FileNotFoundError` into an
> undifferentiated `ResidueError` — that would preserve control flow at the cost of the
> operational category.
>
> Where the existing coordinator requires the residue hierarchy, introduce:
>
> ```python
> class DatasetProvisioningError(ResidueError):
>     pass
> ```
>
> **Preserve the original exception by chaining**, and include the **absolute path and the
> node** in the message. This keeps existing control flow while retaining the correct
> classification.
>
> Current live state at `miner/range_miner_worker.py:530-532` is a bare
> `FileNotFoundError` raised mid-run — unclassified, after dispatch. That is the defect
> P0.5 closes.

**Under VIR-5:** an *unverifiable* dataset (checker could not run, path not readable)
is `UNAVAILABLE`, **not** clean. It must not be treated as a passing check.

## 4. Provisioning step

Provisioning is an explicit step in rig bring-up, not a side effect of cloning:

1. clone the repository at the target commit (proves source identity via
   `git rev-parse`);
2. **provision runtime datasets from the manifest** (this contract);
3. verify each dataset's digest **on the target node**, not on the sender;
4. record verified digests in the run evidence alongside the source commit.

Step 3 is on the target deliberately: hashing the file you just sent, on the machine
that will actually read it, is what catches a truncated or interrupted transfer.
Phase 6.0 did exactly this (`ssh … 'sha256sum daily3.json'`) and it is the pattern to
formalize.

## 5. Current state and scope

- **Known dataset:** `daily3.json` — the only one Phase 6.0 required. Others (e.g.
  `daily3_midday.json`, `daily3_evening.json`, `pa_pick3.json`) must be enumerated
  before any run that uses them.
- **Nodes in scope for Phase 6:** VM 101 (source of truth) plus each CT100 worker
  actually participating.
- **Out of scope:** committing datasets to Git; any change to how the code *reads* a
  dataset; the sieve or assembly path.

### 5.1 Fleet state, verified 2026-08-01 (VM 101 → CT100s)

| node | dataset | digest |
|---|---|---|
| `rrig6600` (`192.168.3.122`) | present, 1,380,711 bytes, mtime Jul 30 | `513648160d356617…` — **matches** the published version |
| `rrig6600b` (`192.168.3.156`) | **absent** | — |
| `rrig6600c` (`192.168.3.164`) | **absent** | — |

**Two of three rigs have no dataset.** A distributed run today would dispatch to one node
with data and two without, and the failure would surface as a bare `FileNotFoundError`
inside a worker, mid-run.

`rrig6600`'s copy was hand-placed during Phase 6.0 and **happens** to match. Nothing
verified it at the time; it is correct by the accuracy of a manual copy, not by any
mechanism. **That is precisely what P0.5 replaces.**

## 6. Verification-integrity controls (VIR-1…5)

- **execution proof:** the digest check runs **on the target node** and emits the
  computed digest, not merely a boolean.
- **clean control:** a correctly provisioned dataset passes.
- **fault-injection control:** a deliberately corrupted or truncated copy **must** be
  rejected before dispatch — this is what proves the checker is not vacuous.
- **completion sentinel:** provisioning terminates in an explicit
  `PASS | FAIL | UNAVAILABLE | INCOMPLETE` per node; a missing per-node record is
  `INCOMPLETE`, never success.
- **unavailable-observer behavior:** if the digest cannot be computed (path
  unreadable, tool missing, node unreachable), report `UNAVAILABLE` — never "clean".

## 7. Open item

Phase 6.0 provisioned by `scp` from VM 101 with a runtime `sha256sum` check on the CT.
That satisfied one node under supervision. **Multi-rig Phase 6 needs the manifest, the
per-node verification loop, and the fail-before-dispatch enforcement in code** — not a
remembered command. This document is the contract; the implementation is a small
deliverable to be scheduled before Phase 6 begins.
