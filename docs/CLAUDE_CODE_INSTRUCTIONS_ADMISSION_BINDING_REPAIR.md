# CLAUDE_CODE_INSTRUCTIONS_ADMISSION_BINDING_REPAIR.md — REV1

**Bind miner admission to the frozen execution set, and repair the false freeze-after-read
property.**

Team Beta **accepted** fleet identity and consumer unification at `63e627f` but **withheld
Phase-7 closure** pending exactly these two repairs. This brief is both and nothing else.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. Implement
and iterate; you do **NOT** commit, push, or run WATCHER. STOP at the gate.

---

## 0. Repair A — the freeze-after-read property is FALSE as implemented

**Alpha claimed** in the submission that freezing after a read is structurally impossible, so
Beta's ordering requirement cannot be violated. **Beta traced the code and refuted it:**

> `active_execution_set()` increments `_READS` **only when `_ACTIVE` is already non-`None`.**
> The gate explicitly asserts that reading an empty set must not prevent a later freeze.
> Therefore a consumer can read `None`, choose legacy behavior, and the set can still be frozen
> afterward — the exact *"consumer already decided without it"* sequence the submission says is
> impossible.

**The live entrypoints are correctly ordered today, so the full-fleet evidence stands.** What is
wrong is the stronger structural claim.

**Repair, narrowly:**

1. **Consumer reads must count even when the result is `None`.** That is the case that matters —
   a consumer reading `None` and taking the legacy path *has* decided without the set.
2. **Give the resolver owner a private, non-consuming peek** for idempotent WATCHER setup, so
   the resolver's own internal checks do not trip the guard they exist to protect.
3. **Three gates:**
   - empty consumer read → **later freeze refused**
   - clean resolve/freeze before any read → **passes**
   - identical re-freeze after consumption → **remains idempotent**

**Also correct the claim in `docs/TEAM_ALPHA_EXECUTION_SET_AND_CHAPTER2_SUBMISSION.md` §1.3.**
It is committed and Beta has cited it; a retracted claim must not simply disappear
(`G-COMMENT-TRUTH` discipline). State what was claimed, that it was false as implemented, and
what now holds.

## 1. Repair B — bind miner admission to the frozen set

**Beta: AUTHORIZED and REQUIRED as the next isolated repair.**

**The defect:** the set records one count while `_serve_clients()` independently derives
`expected_workers` from `context["worker_pool_size"]`. **Two frozen run facts that can
disagree.** A local set with two GPUs still waits for the default eight workers.

**Required semantics:**

```
effective admission count = min( requested worker pool size,
                                 count of selected worker identities )
```

**Record both**, distinctly:
- `requested_admission_count`
- `admission_count` — the **effective** count imposed by the miner

**Required consequences — gate each:**

| case | expected |
|---|---|
| full 26-GPU set, default request 8 | admission count **8** — existing behaviour unchanged |
| local Zeus set, default request 8, two selected GPUs | admission count **2** |
| local set, explicit request 1 | admission count **1** |
| zero, negative, or zero-capacity set | **fail during resolution** |

**The clamp must be visibly logged and included in `set_id` provenance.**

**On production miner paths, `expected_workers` must come from the frozen set's effective
`admission_count`. The raw context value must not remain a parallel authority.** That is the
whole point — a second source of truth for the same quantity is the defect being closed.

**Do not change** the 180-second admission timeout, `serve_timeout=None`, or the Blocker-3
matrix.

## 2. Repair C — Q1's executable half (Beta ruling 3)

**Q1's verification half is closed.** `G-LOCAL` proves one selected node is verified and that its
failure blocks. **It does not prove successful miner stage admission and execution** — the gate
calls `fleet_preflight()` directly while production still expects eight workers.

**A real miner-path gate must prove all six:**

1. `--execution-set-nodes localhost` resolves **two eligible identities**;
2. the default worker-pool request is **bounded to effective admission count 2**;
3. **two local workers cause stage assignment** without waiting for eight;
4. an **unlisted third worker remains quarantined**;
5. missing required local capacity reaches the **existing bounded admission failure**
   (`ee0db06`'s path — do not add a second failure mode);
6. **full-fleet / default-eight behaviour remains unchanged.**

**No Wall A/B rerun is required** (Beta, explicit).

## 3. Out of scope

- Anything beyond repairs A, B and C.
- The 180s admission timeout · `serve_timeout` · the Blocker-3 matrix · `worker_pool_size`
  **semantics** (its *use as an authority* is what changes, not its meaning).
- The `process_sharded` import gate (Beta-required, separate deliverable).
- The sampler provenance guard · D6.2 · D6.3 · the scraper · skip work.
- **Do not re-run Phase 6** — certified and closed at `d98298c`.
- Do not delete or re-point any of the six consumers again; they are settled.
- Chapter 2's corrections — separate brief.

## 4. Verification-integrity controls (VIR-1…6)

- **execution proof** — the effective `admission_count` read back from run provenance, and
  `expected_workers` demonstrably sourced from it on a production miner path, not from context.
- **clean control** — the full-fleet default-8 case unchanged; a clean resolve/freeze before any
  read still passes.
- **fault-injection control** — reverting each repair must red its gate. For Repair A
  specifically: restore the `None`-read exemption and show the empty-read gate goes red.
- **completion sentinel** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **unavailable-observer** — anything not exercised is `UNAVAILABLE`, never assumed.
- **audit claim scope** — declare searched and unavailable surfaces.

**A note on this deliverable's own claims.** Repair A exists because Alpha asserted a structural
property without tracing the counter's actual trigger condition. **Do not restate the property
in the report without showing the code path that enforces it** — including the `None` case.

## 5. Non-regression

**Resolved execution set 34/34** · **P0.5 38/38 `--fleet`** · **admission liveness 16/16** ·
threshold-propagation 5/5 · Chapter1-P0 12/12 · D1.1 · D1.0 · D4 · D5 · D6 3.A ·
**D6-threshold 17/17** · D6.1 · **Phase-6 transfer gate 8/8 + 8/8 faults** · Phase 3 ·
**Phase 4 63/63**.

Gate 22 and `G-MINER-UNCHANGED` will see changed `miner/` files — register with rationale,
**append rather than rewrite**, and keep P0.5's strengthening intact.

## 6. Report

Repair A: the counter's new trigger condition with `file:line`, the private peek, the three
gates, and the corrected submission text. Repair B: where `expected_workers` now comes from, the
four clamp cases, and the provenance read-back. Repair C: the six-point gate with results.
Confirmation that the admission timeout, `serve_timeout` and the Blocker-3 matrix are unchanged.
Then STOP. **Do not commit.**
