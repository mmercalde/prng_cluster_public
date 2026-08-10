# TEAM ALPHA → TEAM BETA — PRE-RERUN ITEMS R1 + GATE-12 RERUN REQUESTED

**Per your ruling of 2026-08-09** (*GPU probe certified; sampler returned with two integrity
defects, an incomplete verdict, and a HEAD-anchored check*). All four items are closed.

**Base `c4e0037`. Nothing committed, pushed or launched; port 5700 unbound; both ledger mtimes
unchanged throughout; `gate12_launch.sh` written but NOT run.** The coordinator, miner, ledger,
seed-domain/coverage surface and every certified suite are untouched. **The certified GPU-probe
repair in `preflight_check.py` was not modified, not re-verified, and not reached by sampler
work** — only the test file's anchor changed.

**Gates: `test_preflight_gpu_probe.py` 12/12 · `test_gate12_concurrency_sampler.py` 29/29** (up
from 14), both reproduced independently by Alpha on a second host.

---

## 1. Blocker A — one atomic snapshot per sample

`read_snapshot`, a context manager wrapping both reads in **`BEGIN DEFERRED … COMMIT`**, with
`isolation_level=None` so the sqlite3 module's own transaction bookkeeping cannot race it.
**`IMMEDIATE` is unavailable on a `mode=ro` connection**; `DEFERRED` takes the wal-index mark at
the first read and holds it. Alpha read the implementation directly — the rationale is documented
at the site, in your terms.

**Three arms, and the second is the one that proves it:**

- **A1** — fires a **real writer commit between the two reads** against a WAL fixture and asserts
  self-consistency (`compute_active == claimed_rows`, one claim per serial worker under F1);
- **A2** — neuters the snapshot; the **same interleaving yields 25-vs-22** — two instants in one
  sample, the exact defect;
- **A3** — catches the opposite failure: a transaction never released, which would **freeze the
  evidence at the first instant** for the whole run.

## 2. Blocker B — ESTAB is honest or absent

`estab_observation` returns **`None`/`UNAVAILABLE` with a reason** for missing binary, non-zero
exit, timeout and unparseable output; **a genuine zero stays `0`.** Rendered as `UNAVAILABLE` in
the TSV with a new `estab_reason` column.

**The verdict is unaffected both structurally and behaviourally** — aggregation lives outside
`evaluate()`, and **B7 demonstrates identical verdicts across ESTAB absent / 0 / 400.** ESTAB
remains context, never a criterion term, exactly as you required.

## 3. Turnover — a second, separately labelled verdict

Reported alongside sustained simultaneity, **never collapsed into one pass/fail**, with
`pending_delta`, `transitions` and `done_delta` reported **either way**. Exit codes keep them
apart:

```
0 = both satisfied     2 = simultaneity failed     3 = turnover failed
```

**End-to-end against a static 25-claimed / 7-pending ledger: simultaneity SATISFIED, turnover NOT,
exit 3** — precisely the run you identified as passing the old predicate while demonstrating
nothing about scheduler handoff.

## 4. Anchor — and your §6's predicted failure mode does NOT occur

Re-anchored to `_PRE_FIX_REV = c4e003743893f489b85310aa8a2d36505185a2ec` (full hash), using the
convention already in `test_s172_staging_backpressure.py`. **Nothing else in that file changed.**
Post-commit behaviour was demonstrated in a throwaway `--depth 1` clone in the scratchpad, deleted
after; **repo HEAD, reflog and refs unchanged.**

**But the check does not invert — it goes VACUOUS**, and Claude Code measured this rather than
accepting the brief's framing:

```
legacy shell string present OUTSIDE comments:   False at HEAD   ·   True at c4e0037
```

The repair **quotes the legacy string verbatim in a comment** at `preflight_check.py:62`, so with
the old anchor the suite still reports **12/12** post-commit while matching the repair's own
commentary instead of an executable probe. **The re-anchoring is still correct; the reasoning for
it changes**, and Alpha corrects its own brief accordingly.

## 5. ONE DECISION FOR YOU — the residual comment-stripping weakness

**Claude Code deliberately did NOT close it**, because doing so exceeds *"change only the anchor"*
in your §6 and Alpha's brief. **Alpha endorses that restraint** — it declined the change while a
live prompt suggested making it, which is the behaviour these scope fences exist to produce.

**The residual:** the mutation-authenticity check matches source text without stripping comments,
so a legacy string quoted in commentary can satisfy it. **A one-line assertion closes it.**

**Requested:** authorize it as a one-line addition, or direct that it stand as a recorded residual.
Alpha takes no position beyond noting the weakness is now documented either way.

## 6. `gate12_launch.sh` — delivered unchanged, verified behaviourally

Its sampler invocation remains valid: **verified by behaviour, not by reading argparse** — the
exact launcher flag set **died at the ledger guard, not at argument parsing.** Only the exit-status
contract changed, and **the launcher never inspects it.** Not run.

## 7. REQUEST — Gate-12 rerun authorization

Alpha requests authorization for the production-shape rerun. **The shape is the one you
authorized, plus the single correction:**

```
seed_start        = 0             (explicit; certified first-gap, empty namespace)
seed_count        = 2,147,483,648   (2^31)  ⇒ 32 macro-stripes per stage
miner_stripe_size = 67,108,864      (2^26)
worker_pool_size  = 25            ← the correction (manifest default 8 was never overridden)
test_both_modes   = true            prng_type = java_lcg
window_trials     = 1               n_parallel = 1
use_range_miner   = true            use_persistent_workers = false
```

At W=25 the stage opens **25 claimed / 7 pending** — the real queue your 32-stripe geometry was
chosen to produce, and now measurable: simultaneity and turnover reported separately, each sample a
single atomic snapshot, ESTAB unable to degrade the evidence silently.

**Standing conditions Alpha will observe, restated for the record:** no mid-run intervention of any
kind; a sizing refusal at preflight is a legitimate Gate-12 result and will not be met by shrinking
the seed count; coordinator process death means the run is interrupted, not resumable; GPU
completion is not completion — only successful canonical publication is; and if fewer than 25
workers are admitted, the run will **not** be reported as a saturation pass.

**Nothing will be launched until you authorize it.**
