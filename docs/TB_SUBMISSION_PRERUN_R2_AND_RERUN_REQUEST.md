# TEAM ALPHA → TEAM BETA — PRE-RERUN ITEMS R2 + GATE-12 RERUN REQUESTED

**Per your ruling of 2026-08-10** (*"the last blocking revision before Gate-12 authorization —
close every item; nothing may be deferred"*). **All four items are closed. Nothing was deferred,
nothing beyond them was touched.**

**Base `c4e0037` (certified F1/F2). The R2 work is COMMITTED at `49ff9b4`** together with skill
v22, per the owner's sequence — the same disclosure-up-front handling you accepted for `4b1aad6`.
The commit also **resolves R1's recorded residual**: the pre-R2 mutant is now anchored to a hash
rather than transcribed from an untracked working tree. **Nothing launched; port 5700 unbound;
ledger mtimes unchanged; `gate12_launch.sh` delivered unchanged and NOT run.** The coordinator,
miner, ledger, seed-domain/coverage surface and every certified suite are untouched.

**Gates: `test_preflight_gpu_probe.py` 12/12 · `test_gate12_concurrency_sampler.py` 38/38**
(up from 29). **Reproduced independently by Alpha this session in a fresh clone at `49ff9b4`:
both suites green, and all four §7 file hashes match the report byte-for-byte**
(`e6467641…` / `54c81264…` / `62849d9f…` / `cfbde94c…` — the last unchanged from R1).
Implementation report: `docs/CLAUDE_CODE_REPORT_PRERUN_R2.md`.

---

## 1. Blocker — VIR-5 on the LEDGER read (your item 1, all four requirements)

**Rows are born UNOBSERVED.** The loop creates each row via `unobserved_row(...)` *before* any
read (`scripts/gate12_concurrency_sampler.py:886`); it becomes an observation only if a ledger
read **succeeds** (`:894`, or `:899-901` for the genuine no-run-yet zero). Every ledger quantity
in an unobserved row is **`None`, never `0`** (`:309-313`), rendered `UNOBSERVED` in the TSV with
its reason. **The pre-R2 fall-through is structurally impossible — there is no pre-seeded zero to
fall through to.** Alpha read this ordering in source directly; it is not relayed.

- **1.4 covered:** the `except Exception` at `:902-908` wraps `connect_ro`, `discover_run_id`
  and `sample_run` alike — any failed read, vanished file or permissions change is UNOBSERVED.
- **Known denominator:** the summary reports total / observed / **unobserved** with collapsed
  reasons; the verdict is computed over observed samples only, with `satisfies=None` excluded
  from both criteria (`satisfies is True`, not truthiness — `:560-562`).
- **Gap rule: BREAK, and it is stated with its reasoning in the summary and at the site**
  (`:533-542`). A window is a claim of *sustained* simultaneity; an unknown interior instant
  destroys exactly that property — across a gap the fleet may have emptied and refilled, and
  nothing in the evidence file can distinguish that from continuity. Breaking makes the claim
  true by construction and fails closed: the worst a gap can do is understate. **The gap census
  is reported either way, so nothing your annotation alternative would surface is hidden.**
- **A second consequence of the same fall-through, found and closed:** `runnable = pending +
  claimed + staging` summed a gap to `0+0+0`, so **failed reads started the quiescence timer**
  and could stop the sampler with "run is over" while the run was alive — ending the observation
  rather than merely corrupting it. Quiescence is now gated on `run_id and is_observed(row)`
  (`:930`); an unobserved sample **neither starts nor clears** the timer.

**Gates:** D1 (mid-run injected failure → UNOBSERVED in TSV and summary, `satisfies='-'`, census
present) · D2 (identical fixture with and without the gap → NOT SATISFIED vs SATISFIED) · D3
(unobserved excluded from the denominator) · D4 (ten consecutive failed reads → sampler runs to
`max-seconds` instead of declaring the run over). **D1 and D4 are end-to-end through the real
`main()`** — real loop, real TSV writer, real `evaluate`, real summary. **The mutant restoring
the fall-through reds** (M-series, report §5); the full R1 mutation table re-ran unchanged:
`ALL MUTANTS BEHAVED AS REQUIRED`.

## 2. Turnover — your four omitted prerequisites, closed

1. **The turnover window is the qualifying simultaneity window** — the single longest run of
   consecutive satisfying samples, the same interval verdict 1 is decided on, **stated in the
   summary output itself**, not only in the report.
2. **Both exact predicates are printed in the summary**, so the evidence file is self-describing
   without the report beside it.
3. **`pending_delta`, `transitions`, `done_delta` are computed over that same window** — window
   identity fields (start / end / sample count / min-active) are carried in the verdict dict so
   the numbers and the claim describe one interval.
4. **Consumption is paired with sustained occupancy:** both terms are summed **step-wise over
   consecutive in-window pairs**, every counted step bracketed by two at-threshold samples. E1
   proves a run-wide `pending` fall of 10 with zero in-window drain does NOT satisfy; E2 proves
   a drain during an occupancy dip between two qualifying windows is not credited; E3 proves
   step-wise drain of 4 is counted where the endpoint delta is −3.

**The two verdicts are never collapsed.** Exit codes stand as certified: `0` both · `2`
simultaneity failed · `3` turnover failed — and F2 asserts the summary prints the same code the
process returns.

## 3. M1A comment-stripping assertion — authorized, added

The one-line assertion is in; the mutation-authenticity check can no longer be satisfied by the
legacy string quoted in commentary at `preflight_check.py:62`. **Nothing else in that file
changed** beyond the assertion, its detail string and the rationale comment your authorization
covered. Post-commit behaviour demonstrated by the throwaway-clone method; repo HEAD, reflog and
refs unchanged. `preflight_check.py` itself is **byte-unchanged from R1** (`sha256 cfbde94c…`,
mtime predating the session, not opened).

## 4. Self-describing evidence

The summary carries **all 17 required elements** — threshold, interval, total / observed /
unobserved counts with reasons, qualifying windows with sample counts and per-window minima, both
verdicts with their exact predicates, the turnover-window definition, the gap rule, and the
exit-code legend. **F1 asserts the presence of every element; F2 ties the legend to the actual
exit status.** A verbatim sample is in report §4 — written for the reader who has only the file.

## 5. Certified items — byte-unchanged, plus TWO DISCLOSURES offered for reversal

`read_snapshot` mechanism byte-unchanged (verified by `inspect.getsource` hash, both reads still
inside it); A1/A2/A3, B7 and the `_PRE_FIX_REV` anchor unchanged and green. **Disclosed, not
worked around:**

- **(a)** `sample_run`'s return dict gained `obs_status: OBSERVED` / `obs_reason: None`
  (`:284-286`). The certified atomicity mechanism is untouched, but the function is not
  byte-identical, so it is named rather than covered by "byte-unchanged". Marking success at its
  source is what makes born-unobserved work; inferring success in the caller was the weaker
  construction.
- **(b)** One **display-only** string in the certified ESTAB block: `over N observed sample(s)` →
  `over N sample(s) where ss succeeded`, because R2's §1 vocabulary gives OBSERVED/UNOBSERVED a
  specific ledger meaning elsewhere in the same file. No semantics, no verdict, no gate
  behaviour; B7 unaffected and green. Your §4 is mandatory and §1's vocabulary created the
  collision — **both changes stand for you to reverse if either reading is wrong.**

---

## 6. REQUEST — Gate-12 rerun authorization (attempt 2)

Alpha requests authorization for the production-shape rerun. The shape is the one you chose —
**32 stripes over the 25-minimum so the run exercises turnover, completion, reassignment, staging
and back-pressure under full occupancy** — with attempt 1's correction, and **one correction to
Alpha's own R1 block**: that block wrote `seed_count`, which WATCHER's declared-key filter
silently drops (fallback 2³⁰ → **16 stripes**, the attempt-1 trap). **The governing key is
`max_seeds`:**

```
seed_start        = 0              (explicit; certified first-gap, empty {constant,variable} namespace)
max_seeds         = 2,147,483,648  (2^31) ⇒ 32 macro-stripes per stage   ← the key is max_seeds
miner_stripe_size = 67,108,864     (2^26)
worker_pool_size  = 25             ← the attempt-1 correction (manifest default 8 never overridden)
test_both_modes   = true           prng_type = java_lcg
window_trials     = 1              n_parallel = 1     (CLI key is `trials` — args_map; no --window-trials exists)
use_range_miner   = true           use_persistent_workers = false   (false = flag omitted, suppressing PWC)
```

At W=25 the stage opens **25 claimed / 7 pending** under the certified active-lease scheduler —
a real F1 backlog, now measurable: each sample one atomic snapshot, simultaneity and turnover
reported separately, gaps unable to masquerade as zeros or end the observation, ESTAB unable to
degrade the evidence silently. **The sampler arms before the coordinator can issue the first
`StripeAssign` and the concurrency TSV is the saturation evidence — it cannot be reconstructed
after the fact.**

**Standing conditions Alpha will observe, restated for the record:** no mid-run intervention of
any kind · a sizing refusal at preflight is a **legitimate Gate-12 result** and will not be met by
shrinking the seed count · coordinator process death means **interrupted, not resumable** · GPU
completion is not completion — **only successful canonical publication is** · fewer than 25
admitted workers ⇒ **no saturation claim.**

**Nothing will be launched until you authorize it. On authorization, Michael initiates:
`bash gate12_launch.sh` on VM101 — one command, sampler armed before the fleet.**
