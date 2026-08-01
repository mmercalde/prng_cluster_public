# CLAUDE_CODE_INSTRUCTIONS_PHASE_6_P0_5_IMPLEMENTATION.md — REV1

**Phase 6-P0.5: the behavioural cutover — pointer resolution, run-start freeze, absolute-path
dispatch, fail-before-dispatch, and per-node provisioning.**

**P0 created files and changed nothing. P0.5 changes running code.** That inversion is the
point: every behavioural change lands together, against a published baseline, so the first
post-publication distributed certification has **one** cause to attribute.

**Base:** `09a7ebc`. Claude Code on VM 101 as `michael`, venv `~/venvs/torch`. Implement and
iterate; you do **NOT** commit, push, or run WATCHER. STOP at the gate for Team Alpha review.

**The rigs are ON.** `rrig6600` `.122`, `rrig6600b` `.156`, `rrig6600c` `.164`, key auth from
VM 101. Per-node work is in scope — see §7 for the verified starting state.

**Beta ruled the P0 procedural exception is NOT precedent.** P0 was inert; this is not. Follow
the brief, and where it is silent or you disagree, **stop and report** rather than deciding.

---

## 0. Task zero — apply the contract amendment

Before any code: apply `docs/PROVISIONING_CONTRACT_AMENDMENT.md` to
`docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md` — three edits plus the fleet-state subsection.
Beta made this an explicit requirement of the P0 ruling (§2): *"the contract and actual phase
boundary must not remain inconsistent."* Documentation only.

## 1. What exists after P0

```
daily3-20260801T145551443433Z-513648160d35.json   immutable version one
                                                  sha256 513648160d35…68f6
                                                  1,380,711 bytes · 18,068 records
daily3_current.json                               pointer manifest (atomic replace)
daily3.json                                       legacy alias — byte-identical, untouched
```

**Nothing reads the pointer.** P0.5 makes it authoritative. Schema:
`docs/DATASET_PUBLICATION_SCHEMA_v1.md`. Read-only verifier:
`scripts/verify_dataset_publication.py` (20/20).

## 2. Required behaviour (Beta, P0 ruling §3 — confirmed scope)

1. **WATCHER resolves `daily3_current.json`.**
2. **One-time freeze at run start** of: manifest identity/version · immutable **absolute**
   dataset path · dataset SHA-256 · size · record count.
3. **Dispatch the absolute immutable path — never the bare `daily3.json`.**
4. **Fail before first worker dispatch.**
5. **Per-node provisioning and digest verification.**
6. **Run provenance** recording the frozen manifest/version/digest.
7. **Pointer-movement protection:** a pointer change *during* a run must not alter that run.
8. **Validate the pointer targets a permitted version-stamped filename.**

### 2.1 On item 7 — this is what makes "freeze" mean something
Resolve once, at run start, into run-scoped state. Every subsequent consumer reads **that**,
never the pointer again. A scrape landing mid-run must be invisible to the run in progress.
Note the existing per-trial derivation at `range_miner_coordinator.py:3499`, flagged in the
scoping report: *"freezes the digest per trial, not per run… a scrape between Optuna trials
splits a study across two datasets with no error."* **Moving that to run scope is part of this
deliverable.**

### 2.2 On item 8
The pointer names a file; that name must match the schema's version grammar (§2 of
`DATASET_PUBLICATION_SCHEMA_v1.md`) and resolve inside the publication directory. **A pointer
naming `daily3.json`, an absolute path elsewhere, or a traversal must be refused** — the
pointer selects among published versions, it is not a general path parameter.

## 3. Failure classification — Beta's correction, follow it exactly

**Do not flatten `FileNotFoundError` into `ResidueError`.** A missing dataset is not
semantically a residue error. Where the coordinator requires the residue hierarchy:

```python
class DatasetProvisioningError(ResidueError):
    pass
```

**Preserve the original exception by chaining** (`raise … from`), and include the **absolute
path and the node** in the message. Current live state — a bare `FileNotFoundError` at
`miner/range_miner_worker.py:530-532`, raised mid-run, unclassified — is the defect this closes.

## 4. The WATCHER path defect — in scope, and Beta corrected its characterisation

Preflight resolves `<REPO_ROOT>/daily3.json` (`agents/watcher_agent.py:489`, explicitly
commented *"not os.getcwd()"*); dispatch is `subprocess.Popen(cmd, …)` at `:1948` **with no
`cwd=`**, so the child receives the bare string and resolves it against **its own** CWD. Two
resolution bases; the gate uses the one the work does not.

**Beta's correction, adopt this framing:** P0 did **not** create this. Version-stamped names
cannot collide with the bare alias. The executable failure condition is a child launched from a
CWD containing *some other* `daily3.json`. Record it as a **pre-existing latent authority
defect exposed during P0**.

Item 3 (dispatch the absolute path) resolves it structurally: an absolute path has no CWD
dependence. **State in the report whether the preflight/dispatch divergence is fully closed or
merely narrowed.**

## 5. Provisioning — the observed fleet state, and what it implies

Verified 2026-08-01 from VM 101:

| node | dataset | digest |
|---|---|---|
| `rrig6600` `.122` | present, 1,380,711 B, mtime Jul 30 | `513648160d356617…` — **matches** |
| `rrig6600b` `.156` | **absent** | — |
| `rrig6600c` `.164` | **absent** | — |

**Two of three rigs have no dataset.** Today a distributed run dispatches to one node with data
and two without, failing deep inside a worker.

`.122`'s copy was hand-placed during Phase 6.0 and **happens** to match — correct by the
accuracy of a manual copy, not by any mechanism. **Provision it through the same path as the
others; do not special-case it because it is already right.** A provisioning step that skips a
node it believes is correct is a provisioning step that cannot detect the case it exists for.

**Verify the digest on the target node**, not on the sender. Hashing the file you just sent, on
the machine that will read it, is what catches a truncated transfer.

## 6. Out of scope — do not cross

- **The hybrid skip wire-in.** Beta, explicit: *"while the rigs remain unavailable, Alpha may
  prepare and locally test P0.5, but must not wire in hybrid skip."* The rigs are now up; the
  order still stands — **P0.5 certifies first.** Wiring both would put two behavioural changes
  in the first post-publication certification, defeating the reason P0 and P0.5 were split.
- The `RandomSampler` control arm — after the skip wire-in.
- **Do not modify or move `daily3.json`**, the published version file, or the pointer manifest.
- **Do not publish a new dataset version.** P0.5 consumes what P0 published.
- Do not touch `daily3_midday.json` / `daily3_evening.json` — unversioned and unbound, an open
  item. Beta: accepting the combined publication is a **provenance** ruling, *not* a finding
  that combined midday/evening records are analytically appropriate for one PRNG model.
  **Session-separated dataset authority remains open work** — do not treat it as settled.
- Do not fix the falsy-zero droppers (`digit_sequential_sieve.py:161-162`,
  `coordinator.py:1881`) — non-certifying, must not be bundled with dataset work.
- Do not touch `.gitignore`, including the dead `:42` negation.

## 7. Verification

**Local first, then the fleet.** Beta's order: implementation → local and negative-path
verification → per-node provisioning verification → P0.5-only certification.

**Negative paths are the substance here.** Each must fail *before* first worker dispatch, with
a classified error naming the absolute path and the node:

| case | expected |
|---|---|
| dataset absent on a node | `DatasetProvisioningError`, pre-dispatch |
| digest mismatch on a node | `DatasetProvisioningError`, pre-dispatch |
| pointer missing / unparseable | refuse, pre-dispatch |
| pointer names a non-conforming filename | refuse (§2.2) |
| pointer names a file that does not exist | refuse |
| pointer moved mid-run | **run unaffected** — this is item 7 |

**Verification-integrity controls (VIR-1…6):**
- **execution proof** — digests re-derived **on the target node**; the frozen values appear in
  run provenance, read back rather than assumed.
- **clean control** — a fully provisioned three-node run resolves, freezes and dispatches.
- **fault-injection control** — the negative table above. **`.156` and `.164` are genuinely
  empty right now: run the absent-dataset case against real state before provisioning them.**
  That control is available today and will not be again.
- **completion sentinel** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **unavailable-observer** — anything unverifiable is `UNAVAILABLE`, not assumed.
- **audit claim scope** — declare searched and unavailable surfaces; state which claims are
  VM-101-only and which were observed on a rig.

**Non-regression:** D1.1 · D4 · D5 · D6 3.A · **D6-threshold 17/17** · D6.1 ·
threshold-propagation 5/5 · Chapter1-P0 12/12 · Phase 3 · **Phase 4 63/63** (gate 22 will see
changed `.py` — register with rationale, and **check whether another session already modified
that file** before rewriting the block).

## 8. Report

Per item 1–8: what changed, `file:line`. Whether the §4 preflight/dispatch divergence is closed
or narrowed. The negative-path table with actual errors. Provisioning results for all three
nodes including `.122`. Run provenance showing the frozen values. Confirmation that
`daily3.json`, the version file and the pointer are unmodified, and that hybrid skip was not
touched. Then STOP. **Do not commit.**
