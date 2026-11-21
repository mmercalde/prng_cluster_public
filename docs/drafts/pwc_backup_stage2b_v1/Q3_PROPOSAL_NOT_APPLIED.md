# Q3 proposal — NOT APPLIED: refuse a non-integer generator phase instead of truncating it (Beta X-PWC-STAGE2B R1 Q3)

**Date:** 2026-11-20 · **Status:** NOT APPLIED. No code was changed.

**Decision (lead, relaying the owner's standing rule, 2026-11-20):** do **not** apply a stricter check. PWC does what
RANGE-MINER does, and RANGE-MINER truncates the same way:
- the miner worker, `miner/range_miner_worker.py:849` (`_require_int_key` returns `int(value)`);
- the miner coordinator, `miner/range_miner_coordinator.py:9322` (`if int(generator_phase) != 0`) and `:9337`
  (`"generator_phase": int(generator_phase)`).

A strict PWC-only check would make the engines differ. **If this behaviour is ever changed, it must change in both
engines together**, PWC (`sieve_filter.py:567-574`) and RANGE-MINER (the three lines above), in one change.

Beta's Q3 objection is THEORETICAL and is recorded in `docs/exchange/DEFERRED.md`. It holds nothing.
**Responds to:** `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:26` (Q3, OBJECTION, THEORETICAL) and `:55`.

## Prior record

- **Searches run:** `grep -n "generator_phase"` over `miner/range_miner_coordinator.py` and
  `miner/range_miner_worker.py`; `sed` reads of `sieve_filter.py:551-582`, `miner/range_miner_worker.py:565-630`,
  `:835-865`; `docs/exchange/DEFERRED.md` (last row).
- **Found:** the Q3 objection is recorded in `docs/exchange/DEFERRED.md` (X-PWC-STAGE2B R1, Q3, 17:49 PST). The
  v1.1 contract (`docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` §4.1) requires both keys and an
  integer type. The read was not searched further for a ruling on fractional values.
- **Labels:**
  - Q3 itself: ALREADY RECORDED (DEFERRED.md).
  - **The miner truncates in the same way: NEW** (below).
  - The change: would make PWC **stricter than RANGE-MINER**. That departs from the default rule "PWC does what
    RANGE-MINER does" (`docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`), so it needs the owner.
- **Not searched:** rigs, ser8, the private repository.

## What happens today

| Input (`generator_phase`) | PWC (`sieve_filter.execute_sieve_job:567-579`) | RANGE-MINER |
|---|---|---|
| `0` | runs | runs |
| `0.9`, `0.0` | `int()` gives 0: **runs, echoes 0** (Beta reproduced it, `:55`) | coordinator `int(0.9) != 0` is False, accepted, sent as `int(...)` = 0 (`miner/range_miner_coordinator.py:9322`, `:9337`); worker `_require_int_key` returns `int(value)` = 0 (`miner/range_miner_worker.py:849`). **Runs at 0** |
| `"0"` | `int("0")` = 0: runs | same, runs |
| `True` | `int(True)` = 1: refused (phase ≠ 0) | `int(True)` = 1: refused (`:9322`; worker `:618`) |
| `1.5` | `int()` gives 1: refused | refused |
| `"x"`, `None` | refused (`PWC_WINDOW_ANCHOR_SCHEMA_REFUSED`) | refused (`:850-853`) |

`window_anchor` takes the same path in both engines: `int()` at `sieve_filter.py:567`, and
`range_miner_worker.py:849` through `require_window_anchor`. An anchor of 2.9 runs as 2 in both.

**On this cluster no sender produces a fractional value.**
- PWC's coordinator builds the job with `int(window_anchor)` and the literal pin 0 (`persistent_worker_coordinator.py:388-399`).
- The trial's `config.offset` must be a plain `int` (`:373-385`: `_is_plain_int` rejects `float`, `bool`, `str`).
- JSON carries the int as an int.

This is why Beta marked the objection THEORETICAL.

## The smallest change

One file, one block: `sieve_filter.py`, `execute_sieve_job`, the `try: int(...)` at `:567-574`. Replace the
conversion with a type check, keeping the same refusal code:

```python
        for _k in ('window_anchor', 'generator_phase'):
            _v = job[_k]
            if not isinstance(_v, int) or isinstance(_v, bool):
                raise ValueError(
                    f"PWC_WINDOW_ANCHOR_SCHEMA_REFUSED: job {job_id!r}: {_k}={_v!r} "
                    f"({type(_v).__name__}) must be a plain integer; it is not converted")
        _pwc_window_anchor = job['window_anchor']
        _pwc_generator_phase = job['generator_phase']
```

- The existing `!= 0` phase refusal right after it stays as it is.
- `0.9`, `0.0`, `"0"`, `True` and `False` are then all refused before any read.
- `True` changes code: it gave `PWC_GENERATOR_PHASE_REFUSED` and now gives `..._SCHEMA_REFUSED`.
- This covers both entry points:
  - the TCP worker copies the keys verbatim (`persistent/pwc_worker_service.py:587-588`);
  - the local Zeus path runs this same function.
- `GPUSieve.run_hybrid_sieve`'s `int(offset)` at `:380` only ever sees the already-checked value, so it needs no
  change.
- Numpy integers would be refused too. They cannot arrive through JSON. If that matters, use
  `numbers.Integral`, which is what `_is_plain_int` uses in the coordinator.
- Size: about 8 lines changed inside `execute_sieve_job`. It touches no other function and no module-level
  statement, and RANGE-MINER does not reach it.

**Consequence for the default rule.** The miner would still accept `0.9` and `"0"`, and the miner files are off
limits. So after this change PWC refuses inputs RANGE-MINER runs.

The option taken is to leave both engines as they are (decision above). No sender on this cluster produces any of
these inputs, so no run changes. The text below is kept as the design for a future both-engine change.

## Tests it would need

1. `tests/test_pwc_stage2b_anchor_phase.py`, **G-REF-2**: add `generator_phase` = `0.9`, `0.0`, `"0"`, `True`,
   `False` and `window_anchor` = `2.9`, `"3"`, `True`. Each must raise `PWC_WINDOW_ANCHOR_SCHEMA_REFUSED`, with the
   non-existent dataset path (so no read) and no `GPUSieve` built. The existing `"x"`/`None` cases stay.
2. **G-REF-1:** it keeps phases 1, −1 and 25 → `PWC_GENERATOR_PHASE_REFUSED`, unchanged.
3. **G-WRK-2:** add end-to-end, worker plus real `execute_sieve_job`: `generator_phase=0.9` → `success: False`
   naming the schema code. This is Beta's reproduction (`:55`), inverted.
4. **G-NEW-1/2:** unchanged. They pass ints and must still pass, which shows integer jobs are unaffected.
5. **A mutant:** restore `int(job['generator_phase'])`; G-REF-2 and the new G-WRK case must catch it.
6. `scripts/prove_unchanged_outside.py --base 0caeaf9 --file sieve_filter.py --allow execute_sieve_job
   GPUSieve.run_hybrid_sieve` and the import proof: unchanged result expected.
7. Focused re-runs of `tests/test_pwc_backup_stage2.py` and `tests/test_pwc_backup_step1.py`. Their jobs carry ints,
   so no change is expected. A full before/after pass only if the owner wants one for an 8-line change.
