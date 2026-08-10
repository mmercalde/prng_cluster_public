# CLAUDE CODE INSTRUCTIONS — TWO PRE-RERUN ITEMS: TRUTHFUL GPU PROBE + POST-F1 SAMPLER

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD **`d3f8f00`** (F1/F2 certified and
committed). `source ~/venvs/torch/bin/activate` before every test.

**Authority:** Team Beta required both before any Gate-12 rerun is requested. Neither touches
certified production code paths.

**Hard constraints:** no commit, no push, **no pipeline launch, no fleet launch, no port 5700
bind**, **do not apply `worker_pool_size = 25`**. Gate 12 remains HELD. Do not modify the
coordinator, the miner, the ledger, the seed-domain/coverage surface, or any certified suite.
Read-only SSH probes to the rigs are permitted (queries, not fleet work).

**Base verification:** `git log --oneline -1` = `d3f8f00`; tracked tree clean;
`tests/test_s172_f1_f2_active_lease.py` **16/16**.

---

## ITEM 1 — THE GPU PROBE MUST DISTINGUISH `UNAVAILABLE` FROM `0`

**The defect** (forensics disposition C, confirmed 2026-08-09): `check_gpu_health`
(`preflight_check.py:317-352`) runs

```
ssh <host> bash -lc "rocm-smi 2>/dev/null | grep -cE '^[0-9]+[[:space:]]' || echo 0"
```

On the CTs, `rocm-smi` is **not on the PATH this command sees** — the identical grep returns **8**
via `/opt/rocm/bin/rocm-smi`. **`2>/dev/null` swallows "command not found" and `|| echo 0` converts
an unobservable surface into a definite count of zero.** The parsing is correct; the *observation*
is not. All three rigs reported `0/8` while the cluster bot independently reported `8/8` a minute
earlier, and `:229`'s `checks_passed += 1  # Don't block on GPU warnings` meant preflight still
passed 3/3.

**Note `bash -lc` is already in use** — do not assume a login shell fixes it. **Determine the
actual reason the binary is not found under this exact invocation** before choosing a remedy.

### Required

1. **Three distinguishable outcomes**, never conflated:
   - `count = N` — the probe ran and observed N devices;
   - **`UNAVAILABLE`** — the probe could not run (binary absent from PATH, non-zero exit,
     permission, timeout). **This is NOT zero.**
   - `ERROR` — ran but produced unparseable output.
2. **Locate the binary rather than assuming a PATH.** Prefer an explicit absolute path with a
   PATH-based fallback (e.g. `command -v rocm-smi || /opt/rocm/bin/rocm-smi`), and **verify by
   running it read-only over SSH against all three rigs** exactly as preflight invokes it.
   **Report the observed stdout, stderr and return code for each.**
3. **Do not swallow diagnostics.** Drop the blanket `2>/dev/null` and the `|| echo 0`; capture
   stderr and surface it in the structured result.
4. **Preserve the non-blocking behaviour at `:229`** — GPU findings remain advisory and must not
   fail preflight. **This item is about telling the truth, not about changing gating.** If you
   believe the gating should change, **report it; do not change it.**
5. **Warning text must name the distinction** — an `UNAVAILABLE` node must not render as
   `0/8`.

### Gate

A small suite (or an extension of an existing preflight test file — **not** a certified S172
suite): probe returns a count ⇒ count reported · binary missing ⇒ **`UNAVAILABLE`**, not `0` ·
non-zero exit ⇒ `UNAVAILABLE` · timeout ⇒ `UNAVAILABLE` · unparseable ⇒ `ERROR` · **preflight still
passes in every case** (advisory unchanged). Mutation: restore `|| echo 0`; the
missing-binary arm must red.

---

## ITEM 2 — THE SAMPLER MUST MEASURE THE POST-F1 STATE MODEL

**Two defects, both Alpha's.**

**(a) Ordering.** The sampler was started in step 4 of `gate12_launch.sh`, *after* the fleet-launch
step returned — its first row was **12:47:28** for a run that died at **12:47:17**. It produced
**no in-run rows at all.**

**(b) Semantics — and this one is now worse.** Its query was
`count(distinct claimed_by) where state in ('claimed','staging')`. Under the certified F1 model
that is wrong on both terms:

- **`claimed` now means COMPUTE-ACTIVE** — exactly one per serial worker. That is the right term,
  but only in combination with (below).
- **`staging` means the worker has already returned `StripeComplete` and freed its compute slot.**
  Counting it **overstates** occupancy.
- **`pending` is now a REAL backlog state** — 24 rows at W=8, 7 at W=25 — and the old query never
  looked at it. **Queue depth is exactly what Beta's criterion requires and the old sampler could
  not see it.**

### Beta's criterion — what must be provable

> **An observation window in which ≥25 DISTINCT workers were simultaneously compute-active AND
> queued stripes remained available.**

Explicitly **insufficient**: "25 workers connected" · "25 distinct workers eventually used" ·
"32 stripes eventually completed".

### Required

1. **Per sample, at minimum:**
   ```
   timestamp
   distinct compute-active workers   = count(distinct claimed_by) WHERE state='claimed'
   queued backlog                    = count(*)                   WHERE state='pending'
   staging count, done count, cancelled count   (context, NOT occupancy)
   ESTAB connections                 (context, NOT occupancy)
   ```
   **Scope every query to the run under observation** — the ledger may contain other runs.
2. **Start before the coordinator can issue the first `StripeAssign`.** In the launch script the
   sampler must be running **before** the fleet-launch step, not after it. Also ensure it
   **terminates with the run** rather than looping for two hours against a dead trial (that
   happened; the loop had to be killed by hand).
3. **A summary/verdict output** stating whether a window satisfying Beta's criterion was observed —
   peak simultaneous compute-active workers, the queue depth at that instant, and the window
   duration. **Do not claim saturation from a maximum-over-time of distinct workers.**
4. **Read-only against the ledger** (`file:…?mode=ro`), and **do not point it at
   `prng_analysis.db` or any production database** — a prior harness created a real table in the
   live DB by cwd-relative resolution.
5. Deliver the corrected `gate12_launch.sh` **as a whole file**, with the ordering fixed and the
   parameter set updated: **`worker_pool_size = 25`** alongside the frozen shape
   (`seed_start=0`, `max_seeds=2147483648`, `miner_stripe_size=67108864`, `test_both_modes=true`,
   `prng_type=java_lcg`, `window_trials=1`, `n_parallel=1`, `use_range_miner=true`,
   `use_persistent_workers=false`). **Deliver it; do not run it.**

### Gate

Against a **synthetic ledger** (no fleet): a fixture with N claimed / M pending / K staging proves
the sampler reports **N** compute-active and **M** queued, and does **not** count staging or ESTAB
as occupancy. A fixture where 25 workers are claimed **but zero pending** must be reported as
**NOT satisfying** Beta's criterion. A fixture where distinct workers reach 25 only across
different instants must also be reported as **NOT satisfying** it.

---

## REPORT

`docs/CLAUDE_CODE_REPORT_PRERUN_PROBE_AND_SAMPLER.md`:

1. The actual reason `rocm-smi` was not found under `bash -lc`, with the live per-rig stdout/stderr/
   return codes.
2. The three-outcome probe as built, and confirmation preflight gating is unchanged.
3. The sampler's queries, with the state-model reasoning for each term.
4. The saturation verdict logic, and why a maximum-over-time cannot satisfy it.
5. Both gates' red-first and mutation evidence.
6. The corrected `gate12_launch.sh` in full, and confirmation it was **not** run.
7. Files changed. **Any disagreement reported, not worked around.**
