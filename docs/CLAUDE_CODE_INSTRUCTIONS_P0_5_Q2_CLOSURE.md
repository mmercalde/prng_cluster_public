# CLAUDE_CODE_INSTRUCTIONS_P0_5_Q2_CLOSURE.md — REV1

**P0.5 closure condition: a missing provisioning manifest must hard-fail a miner-backed run.**

**Deliberately small.** Team Beta conditionally accepted P0.5 at `d4ff1e4` and named **one**
closure condition. This brief is that condition and nothing else. If the change grows beyond
the manifest-absence path and its negative gate, **stop and report.**

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. Implement
and iterate; you do **NOT** commit, push, or run WATCHER. STOP at the gate.

---

## 0. What Beta ruled

> **Required correction: a miner-backed run must hard-fail before dispatch.**
>
> A missing, unreadable, invalid, or empty provisioning manifest means the system cannot
> establish which worker datasets must be verified. Recording `UNAVAILABLE` and proceeding
> **violates the authority boundary.**

**Current behaviour:** a missing manifest records `UNAVAILABLE` and the run proceeds
(`miner/dataset_authority.py`, `load_provisioning_nodes` returning `None` → preflight treats
it as non-fatal).

## 1. The required change

**Four conditions, all fatal for a miner-backed run**, before any coordinator construction or
dispatch:

```
manifest missing · unreadable · invalid · empty (declares no nodes)
```

**And a distinction Beta drew that the code does not currently make:**

| situation | status | outcome |
|---|---|---|
| miner backend selected, manifest absent/unreadable/invalid/empty | **`UNAVAILABLE`** | **FATAL** |
| non-miner path, no remote execution | **`NOT_APPLICABLE`** | proceed |

Beta's reasoning, and it should govern the implementation:

> *`UNAVAILABLE` means a required verification was **attempted but could not be completed** —
> and therefore remains fatal for the selected miner topology.*

So `UNAVAILABLE` is not "we skipped it." It is "we needed it and could not get it." A path that
never needed fleet verification must not borrow that word. **Do not simply flip a boolean —
introduce the `NOT_APPLICABLE` state and route the non-miner path to it.**

**Failure classification:** use the existing `DatasetProvisioningError` (`range_miner_worker.py:523`),
which Beta approved as implemented. Name the **expected absolute manifest path** in the message
— per Beta's Q3 ruling the preflight message must state where the file was looked for.

## 2. The negative gate — Beta specified it exactly

Add to `tests/test_s172_phase6_p05_dataset_authority.py`. Beta's three conditions:

1. miner backend selected;
2. provisioning manifest absent;
3. **no coordinator construction, no worker process, no dispatch occurs.**

**Condition 3 is the substance.** Asserting that an exception was raised is not enough — prove
that nothing was built and nothing was launched. Assert on the *absence of the side effects*,
not merely on the raise. State in the report how you proved it.

Cover all four fatal conditions (missing · unreadable · invalid · empty), and add the
**`NOT_APPLICABLE` clean control**: a non-miner path with no remote execution proceeds and does
**not** record `UNAVAILABLE`.

**Fault-injection control (VIR-2):** revert the hard-fail and show the gate reds. A gate that
has only ever seen the fixed code is unproven.

## 3. Out of scope — do not cross

- **Anything from Beta's Q1.** The "resolved execution set" refinement — a local run verifying
  only the local node — is **explicitly not authorized yet**. Beta: *"Alpha should complete its
  investigation of the pre-existing GPU-count gate before proposing this refinement. Until
  then, the safe over-constrained behavior remains."* **Do not add a local-run bypass.**
- **Q3's bootstrap contract** (`dataset_provisioning.example.json`, schema docs) — a separate
  docs deliverable, not this one.
- Q4 provenance pruning — Beta ruled **no pruning**, not a blocker.
- Q5 preflight freshness — Beta ruled **leave it in place**.
- Any skip work — Beta sequenced it after bounded Phase 6.
- The published dataset, the version file, the pointer manifest, `daily3.json`, `.gitignore`.
- **The `len(eligible) >= expected_workers` hang** found in `docs/FLEET_STATE_REQUIREMENTS_v1.md`
  §4.3 — real and serious, but **not this deliverable**. It is being submitted to Beta
  separately. Do not touch `range_miner_coordinator.py:3714-3737`.

## 4. Verification-integrity controls (VIR-1…6)

- **execution proof** — the gate exercises the real preflight path, not a mock.
- **clean control** — a correctly provisioned miner run still passes; the `NOT_APPLICABLE`
  non-miner path proceeds.
- **fault-injection control** — §2's revert-and-red.
- **completion sentinel** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **unavailable-observer** — note the irony and get it right: this deliverable is *about* the
  meaning of `UNAVAILABLE`. Use it only where a required check was attempted and could not
  complete.
- **audit claim scope** — declare searched and unavailable surfaces.

## 5. Non-regression

**`tests/test_s172_phase6_p05_dataset_authority.py` must stay green, including `--fleet`
(33/33).** Beta: *"No repeat of the complete live-fleet certification is necessary unless the
correction touches successful-manifest behavior."* — so **prove the successful-manifest path is
unchanged**, and if it is not, say so plainly, because that would trigger re-certification.

Also: D6-threshold 17/17 · threshold-propagation 5/5 · Phase 4 63/63 (gate 22 sees changed
`.py` — register with rationale, and check whether another session already modified that file).

## 6. Report

The change with `file:line`. How condition 3 was proven — the evidence that nothing was
constructed and nothing dispatched. The `NOT_APPLICABLE` routing. The fault-injection result.
Confirmation the successful-manifest path is byte-unchanged in behaviour. Explicit statement
that no Q1 bypass was added. Then STOP. **Do not commit.**
