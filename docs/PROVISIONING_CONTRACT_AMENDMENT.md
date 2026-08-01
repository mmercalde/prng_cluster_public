# AMENDMENT — insert into `docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md`

**Purpose:** Team Beta, Phase 6-P0 ruling §2, made this a requirement:

> *"the provisioning contract must be amended so that fail-before-dispatch and per-node
> verification are explicitly recorded as **P0.5 obligations**. The contract and actual phase
> boundary must not remain inconsistent."*

The contract currently reads as though its obligations are due "before multi-rig Phase 6
execution," with no phase attribution. Beta ratified a boundary the contract does not reflect.

**Two edits. Documentation only — no code, no schema, no behaviour.**

---

## EDIT 1 — replace the `**Status:**` block at the top of the file

**Replace:**

```
**Status:** required before multi-rig Phase 6 execution. Mandated by Team Beta in the
Phase 6.0 final ruling §5.1.
```

**With:**

```
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
```

---

## EDIT 2 — supersede `expected_sha256` as a static field in §1

The §1 field table lists `expected_sha256` as "authoritative digest; the identity of record."
**That field, as a statically-configured value, was superseded by Beta's dataset-lifecycle
ruling** — the dataset is mutable, so a fixed digest in a manifest fails on the first
legitimate publication. Alpha raised this; Beta ruled the identity must be **run-scoped and
frozen at run start**, resolved from the pointer manifest.

**Append immediately after the §1 field table:**

```
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
```

---

## EDIT 3 — correct the failure classification in §3

§3's failure table predates Beta's correction on exception semantics. **Append after the §3
table:**

```
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
```

---

## Observed fleet state — add as a new §5 subsection

The contract's §5 "Current state and scope" is stale. **Append:**

```
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
```

---

## Not changed by this amendment

- No code, schema, manifest or test.
- §2's fail-before-dispatch **rule** is unchanged — only its **phase attribution**.
- §6's VIR block stands, except that it references VIR-1…5; VIR-6 (scope declaration) was
  adopted later and applies.
- §7's open item — that the manifest, per-node loop and enforcement remain unimplemented —
  stands, now correctly attributed to P0.5.
