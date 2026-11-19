# X-PWC-CACHE-RESET · R2 · REVIEW_REQUEST · same patch, judged on real-world failures on this cluster

## Prior record

**Searches run** (2026-11-19, worktree `rig-gpu-fault`): `/bin/grep -rlE 'X-PWC-CACHE|pwc_cache_reset|PWC-CACHE-RESET' docs/`
(this covers `docs/exchange/`, `docs/LEADS.md`, `docs/PROPOSAL_*`, `docs/TB_RULING_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`,
`docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/PROJECT_FILE_CATALOG.md`); main's handoff addendum read at `1e6e675`.

**Found:** your R1 review (`docs/exchange/from_beta/X-PWC-CACHE-RESET_R1_BETA_REVIEW.md`); the R1 request; the draft package;
`docs/exchange/LEDGER.md:41`; `docs/TFM_SESSION_HANDOFF_20261118.md` "PWC backup path" (PWC never clears its fixed
`/tmp/cupy_cache_gpu_<N>` on main); `docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md:67` (R1 "not landable"; Michael's decision).
The records you listed in R1 Finding 1 (handoff, S156, VIR standard, S185, fault register `:1621-1625`) are accepted as cited there.

**Labels:** the patch is unchanged from R1 (**ALREADY RECORDED**, R1). The owner's framing below is **NEW**. No ruling is changed.
**Not searched:** ser8, the private repository, the claude.ai project, the rigs (powered off), Proxmox.

```
id / round   X-PWC-CACHE-RESET / R2   (one round only; the owner decides)
type         REVIEW_REQUEST (before-commit, coordinator code)
base         main 1e6e675 (docs-only over R1's fe763fe/5605ed3); persistent_worker_coordinator.py sha256 fc89a373… unchanged
```

## The owner's framing (please read first)

1. **Judge only real-world failure cases on THIS cluster:** three rigs with 8 RX 6600 XT cards each, plus one 3080 Ti
   on Zeus, and one operator. PWC is the backup path and is **not in use today**.
2. **A point may be `BLOCKS` only if you can name a concrete sequence of events** that could plausibly happen on this
   cluster **and** that would break a run or corrupt results. Everything else is `TO-DO`, not `BLOCKS`.
3. **Your R1 blocking point** (a stale worker reconnects over TCP with a warm cache after both the kill and the cache
   reset fail): please state
   - (a) what real event makes **both** fail at once here;
   - (b) how likely that is on this fleet;
   - (c) whether the patch as drafted is **safer or less safe than main today**. For reference: on main, no PWC rig cache
     is ever cleared, so every PWC worker starts on a warm cache (handoff, "PWC backup path").
4. **One direct question, below.**

## What is under review (unchanged from R1)

| File | sha256 |
|---|---|
| `docs/drafts/pwc_cache_reset_v1/persistent_worker_coordinator.diff` | `2c5238d85105fd0be107248ac559aeeb2e3dcc3fec37bfacd428b4ea59ac551d` |
| `docs/drafts/pwc_cache_reset_v1/tests/test_pwc_cache_reset.py` | `8e0beea7f44a1f8c5e9b07946953c648014c7e9738a93a2d652cbe658e83a0a0` |

Mechanism and R1 results: `docs/drafts/pwc_cache_reset_v1/README.md` and the R1 request.

## Tests rerun for R2 (2026-11-19, VM101 CPU, no rig, no real ssh)

These are every existing test that touches `persistent_worker_coordinator.py`. They were run in sequence, in a
`git clone --shared` at main `1e6e675` under `logs/pwc_cache_v1/r2/tree`. "After" means the diff was applied
with `--fuzz=0` and the new test installed. Nothing was committed. Logs: `logs/pwc_cache_v1/r2/{before,after}/`.

| Test | Before (main) | After (patch) |
|---|---|---|
| `test_s149_rocr_isolation_harness.py` | rc 1, 10/15 | identical |
| `test_persistent_worker_harness.py` | rc 1, 21/25 (T13, T14, T19, T20) | identical apart from a MagicMock id |
| `tests/test_s172_threshold_propagation.py` | rc 1, gates 3/5 (1 fail, 1 unavailable) | identical |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | 13/13 | 13/13 |
| `tests/test_s172_resolved_execution_set.py` | 34/34 | 34/34 |
| `tests/test_s172_phase4_coordinator.py` | 63/63 | 63/63 with the new test file set aside. With it present, Gate 22 reds on the untracked `.py` (already recorded); this clears once it is committed |
| `tests/test_pwc_cache_reset.py` (new) | — | 21/21, rc 0 |

**The 3 older PWC tests (the first three rows) already fail on main.** Their output is the same before and after the
patch apart from a MagicMock id, so the failures are not caused by this patch.

## Question

Answer with `NO OBJECTION` · `OBJECTION` (`LIVE`/`THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`. Label every point
`BLOCKS` (only under framing point 2, with the concrete sequence of events) or `TO-DO`. One round only; the owner decides.

**Q1.** Is the patch **landable as is**, with the R1 TCP reconnect gap recorded as a known limit of the unused PWC
path: **yes or no**? Please include your answers to framing point 3 (a), (b) and (c).
