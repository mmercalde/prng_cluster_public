# RUNTIME_DATASET_PROVISIONING_CONTRACT.md

**Status:** required before multi-rig Phase 6 execution. Mandated by Team Beta in the
Phase 6.0 final ruling §5.1.

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
