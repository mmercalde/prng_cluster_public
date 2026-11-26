# BETA R1 · X-LL6-BUNDLE · summary and proposed responses (2026-11-25)

- **Task:** `~/distributed_prng_analysis/logs/e2e_session/TASK_WF_POSTLAND_AND_LL6_BETA_20261125.txt`, Part B.
- **Exchange** (left uncommitted in the main tree, as the task says):
  - request `docs/exchange/to_beta/X-LL6-BUNDLE_R1_ALPHA_REVIEW_REQUEST_inference_bundle.md`, sha256 `a6182534082f94c3a1c3afa76fd2e214772fd74431d7de5e8f45b2b6783a2954`;
  - review `docs/exchange/from_beta/X-LL6-BUNDLE_R1_BETA_REVIEW.md`, sha256 `a3ad5d1ffa99863b20642953ff071b94020177160a40312c7812d45ea34f8c5f`;
  - ledger row appended by the relay (`M docs/exchange/LEDGER.md`).
- **The relay:**
  - run from the main tree at 19:16, under nohup and waited on in the foreground;
  - verdict **`REVIEW RECEIVED` (exit 0)**, 399 s;
  - log `logs/beta_relay/X-LL6-BUNDLE_R1.log`; my copy of its output: `~/tfm_worktrees/ll6-land/data/ll6_beta/relay_X-LL6-BUNDLE_R1.out`.
- **Before sending,** both of the task's conditions held. They were polled every ~2 minutes from 19:08. At 19:15, `pgrep -af 'beta_relay|e2e_walk_forward_selftest'` was empty and `from_beta/X-REGISTRY-REPAIR_R1_BETA_REVIEW.md` was present, with its ledger row at 19:14.
- **Nothing in the five files or the tests has changed since the review.** No commit anywhere. Nothing was written in the main tree apart from the request file, which the relay protocol requires. The relay wrote the review and the ledger row.

## Prior record

- **Read:** Beta's review in full; the request; `docs/exchange/RELAY_PROTOCOL.md` §2.4 (what to do per position); this worktree's `READY_LAND_LL6_20261125.md`.
- **Labels:**
  - Beta's positions: `NEW` (this thread's first round).
  - The two canonical-path gaps (Q3): `ALREADY RECORDED` (READY note §5, request limits 2-3), as Beta also notes.
- `CHANGES A RULING`: none.
- **Not searched:** ser8, the private repository, the rigs.

## Plain summary

- **Four of Beta's five positions raise no objection.** Q1, Q2 and Q4 are `NO OBJECTION`; Q3 is a `SUGGESTION`. Beta ran the default refactor, the canonical loader and A-NORM itself.
  - Beta's own A-NORM on all 12,156 real block-0 rows with the four archived models: NN 7.46e-9, trees 0.
  - Step 6 scores on those rows equal the archived CPU scores exactly.
- **Q5 is `OBJECTION · LIVE`, about the test, not the shipped code.**
  - The no-write gate passed while LightGBM made `/tmp/matplotlib-…` cache dirs in Beta's run.
  - It excludes all of `data/`, so a sibling write is invisible.
  - The Step 6 CLI gate's log check failed falsely in Beta's run on a LightGBM warning.
  - A mutant on Chapter 13's bundle handoff survives.
  - The wrong-checkpoint mutant is missing.
- **The fix is inside this landing's scope** (the two test files). Per protocol §2.4, a `LIVE` objection holds the commit, and the fix comes back to Beta for another round.
- **Proposed:** accept Q5 and fix the tests; do not contest the label.

## Positions and proposed responses

| Q | Beta | Severity | What Beta found | Proposed response (nothing done yet) |
|---|---|---|---|---|
| Q1 Default behaviour | NO OBJECTION | — | Refactor keeps the checkpoint read, best-effort branches, float32, the `1e-12` clamp and logger; legacy Chapter 13 route preserved. Beta reproduced the scaler, CLI and legacy gates. With `MPLCONFIGDIR` set, all runner log lines matched. WATCHER, GPU and walk-forward fingerprint: UNAVAILABLE to Beta | Record. No change |
| Q2 Canonical loader and A-NORM | NO OBJECTION | — | Same sidecar factory and scaler reader as Step 6. Captured Chapter 13's 20 real NN forwards (eval mode): ≤ 9.38e-9 against Step 6. On 12,156 block-0 rows with the archived models: Step 6 equal to archived scores, checkpoint hashes equal, A-NORM NN 7.46e-9, trees 0. End-to-end Chapter 13 cycle and hit rate: not measured | Record. Add Beta's block-0 parity figures to the READY note as Beta's evidence, labelled as such |
| Q3 NN mode, tier comparison | SUGGESTION | — | Eval mode at load is optional for LL-6. Put the tier fix with LL-7 or LL-8 before tier output is relied on, so one validated vector and scaler feed both paths. Test it with named rows | **Adopt as recorded follow-up**, outside this landing: add to the READY note's limits and to the LL-7/LL-8 entry. No code change in LL-6 |
| Q4 Scaler source | NO OBJECTION | — | Keep Step 6's resolution in LL-6. Beta's fixture gap: 4.55 `resolved`, 0 `as_written`; Alpha's 1.89. A-NORM has a known failure when the written NN path is missing. Hit-rate impact unmeasured | Record. Add to the READY note limits: "A-NORM fails by construction for a relocated NN directory; check parity on the actual inference paths before reading such an attribution" |
| Q5 Tests after landing | **OBJECTION** | **LIVE** — "A CPU replay on VM 101 created a cache directory outside `--out` while the no-write gate passed." | (a) `/tmp/matplotlib-…` dirs made by the LightGBM CLI runs (default cache unwritable in Beta's environment); the gate saw nothing because it compares the tree's git status minus all of `data/`. (b) A sibling-write sentinel outside `--out` but under `data/` was invisible. (c) The CLI gate compared logs against the baseline without a log control, so LightGBM's cache warning failed it falsely; the unmutated gate was then red, and the three CLI mutant kills could not be credited from the count alone (Beta verified them separately). (d) Mutant `bundle=None` at the handoff `chapter_13_orchestrator.py:365` survived. (e) No wrong-checkpoint mutant (W2 had one). (f) The summary records the default baseline constant, not an override | **Accept and fix in the two test files** (in scope; no change to the three shipped files). Then a fresh full run and a request for round R2. Details below |

**Additional findings:**

| # | Beta | Proposed response |
|---|---|---|
| 1 | Prior record omits `OWNER_DIRECTION_learning_loop_team_20261123.txt:2-6`, `TASK_LEARN_WHY_BLOCK0_20261124.txt:2-5`, `READY_WHY_BLOCK0_20261124.md:120-129` (branch, real-model A-NORM, development) and system map §4.5 | Accept. Add all four to the READY note's prior record and to the R2 request |
| 2 | **THEORETICAL**: before activation on real artifacts, the canonical adapter accepts positional vectors by width alone, and the analysis does not bind a ranked record to the bundle's schema or checkpoint (system map §4.5; Ch13 audit B15) | Record as THEORETICAL. It holds nothing. List it in the READY note limits as a precondition for using `canonical` on real artifacts (LL-5/LL-7/LL-8 work). It is not in the positions table, so the relay did not append it to `DEFERRED.md`; the lead decides whether to add it there by hand |

## Q5: the proposed test fixes (not applied)

1. **Cache and temp roots under `--out`.** Before any import or child process, set `MPLCONFIGDIR`, `TMPDIR` and `XDG_CACHE_HOME` to subdirectories of `--out`, for the test and every child (Step 5 training, Step 6 runner, mutant runs). This is the pattern `scripts/diagnostics/e2e_walk_forward.py:388-396` already uses. In my runs no `/tmp/matplotlib-*` appeared, because `~/.cache/matplotlib` exists and is writable, but the test must not depend on that.
2. **No-write gate on the exact boundary:**
   - compare the tree's status with only the selected `--out` subtree excluded, not all of `data/`;
   - also snapshot `/tmp`'s top-level entries owned by this user, and the cache roots, before and after;
   - add a negative control that writes a sibling sentinel under `data/` and outside `--out`, and must turn the gate red.
   - Beta's point stands that a before/after comparison cannot see a temp file that is created and removed during the run. Redirecting the temp and cache roots (fix 1) is the answer to that part.
3. **Step 6 CLI log control.** Compare the two baseline runs' logs too. Treat a line as expected-volatile only if it differs between the two baseline runs, and declare that set. Require the new run to match the baseline outside it. That way an environment warning fails on both sides, or is shown as volatile, instead of failing one side only.
4. **Handoff mutant.** A gate that drives the real handoff: `_load_best_model_if_available` under `canonical`, then the call at `chapter_13_orchestrator.py:355-366`, with the archive cwd under `--out`. The input must be scaled and name-keyed, as in the existing root-cause gate. Add mutant `handoff_bundle_dropped` (`bundle=model_info.get('bundle')` → `bundle=None`), which must turn it red.
5. **Wrong-checkpoint mutant**, carried over from W2: a same-family, same-width alternate NN checkpoint whose bundle must fail A-NORM against Step 6 on the original. For example, train a second NN fixture with another seed, and swap `resolve_checkpoint_path`'s result in a mutant.
6. **Provenance:** write the actual `--baseline-rev` used into the summary, not the default constant.
7. **Mutant credit:** a mutant counts only if its target gates are red **and** the same gates were green in the unmutated run. Both are checked and printed.

After these fixes:
- a fresh full run of the LL-6 test and the same landed suites, one at a time;
- update the READY note with Beta's Q2 and Q4 evidence, Finding 1 and Finding 2;
- the R2 request, `X-LL6-BUNDLE_R2_ALPHA_REPLY.md`, with the changed sha256s.

All of this waits for the lead's go. The task said: nothing changed yet.

## Not done

- No file was changed after the review: the shipped files, the tests and the READY note are as reviewed.
- R2 is not requested.
- The THEORETICAL finding is not added to `DEFERRED.md` by hand.
