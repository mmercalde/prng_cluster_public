# X-CUPY-CACHE-RESET · R1 · REVIEW_REQUEST · empty each rig worker's CuPy kernel cache before start (DRAFT, not landed, not yet sent)

## Prior record

**Searched:** `docs/LEADS.md` L-7; `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md` §8–§9;
`docs/CLAUDE_CODE_PLAN_RIG_ARM_V1_V2.md`; the register (`…/REGISTER_GPU_FAULT_PRIOR_WORK.md`) entries on the
kernel cache: S152 per-worker `CUPY_CACHE_DIR` (`:153`, `:633`) and E30 `22bb145`.

**Labels:**
- per-worker cache dirs: ALREADY RECORDED (S152, `c06e4e4`);
- emptying them before start: **NEW**;
- no ruling changed.

**Not searched:** ser8, the private repository.

```
id / round   X-CUPY-CACHE-RESET / R1        type REVIEW_REQUEST (BEFORE-COMMIT: launch path for runs that count)
base         048fd47                         draft diff  docs/drafts/DRAFT_launcher_cupy_cache_reset.diff (35 lines)
```

**Basis:** 2026-11-16, rrig6600 and rrig6600c, phase 4, single events per cell:

| | cold (compiled) | warm (cache hit) |
|---|---|---|
| run #1 after boot | clean 4/4 | fault 1/1 (Run C) |
| run #2 on the same boot | clean 1/1 (Run E) | fault 3/3 |

**Change:**
- New `RIG_CUPY_CACHE_RESET` (default `0`, validated `0|1`, exit 64 otherwise).
- With `1`, each rig worker's remote command starts `rm -rf /tmp/cupy_cache_gpu$N || exit $RC_REMOTE_MKDIR`
  before the existing `mkdir -p`. That is the worker's own directory only, in its own CT.
- With `0` the remote command is byte-identical: the prefix is empty, no added space. Checked by `od -c` of
  the generated line.
- The mode is echoed at launch. The Zeus local cache (`/tmp/cupy_cache_local_gpu$N`) is untouched.

## Questions

Answer each with `NO OBJECTION` · `OBJECTION` (`LIVE`/`THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.

**Q1. Default byte-identity.** Is the default remote command unchanged? Do the parity gate and the 25/25
sentinel and argv checks see no difference?

**Q2. Scope and safety.**
- Is `rm -rf` limited to `/tmp/cupy_cache_gpu$N` with `$N` always set?
- Is reusing `RC_REMOTE_MKDIR` for its failure acceptable?
- Can a reset race a still-running worker of a previous run? The launcher already refuses live workers.

**Q3. Is "cold every run" the right minimal mitigation** given the 2×2 above, or should it wait for the
remaining confounder? (In warm runs the arange/fill and sieve dispatches land within ~1 ms of each other
after a ~220–237 s idle.) Cost: ~0.3–1.2 s of compile per phase per worker.

**Q4. Gates.** What would you require? Proposed:
- a launcher-block gate for the default (no prefix) and for `1` (prefix with that worker's `$N`);
- a bad value exits 64;
- a mutant that drops the prefix while the variable is `1`.
