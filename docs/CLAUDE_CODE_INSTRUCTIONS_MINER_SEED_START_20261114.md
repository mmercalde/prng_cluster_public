# CLAUDE CODE INSTRUCTIONS — CARRY THE SEED START INTO THE SIEVE, AND REFRESH THE RUN 4 EVIDENCE

## Prior record

**Searches run (public mirror, HEAD `b1cd2fa`, 2026-11-14):**

- `grep -rnI "base_start" docs/exchange docs/TB_RULING_* docs/BACKLOG.md docs/TODO_CONSOLIDATED.md docs/LEADS.md docs/PROPOSAL_*.md` (no hit) and `grep -rlI "base_start" docs`.
- `grep -n "assign_stripes(\|partition_macro_stripes(" miner/*.py window_optimizer_integration_final.py`.
- `grep -n "seed_start\|start_seed\|base_seed" persistent_worker_coordinator.py zmq_sqlite_coordinator.py`.
- `grep -n "GOVERNED_FILES" -A 12 scripts/gate12_parity_gate.py`.
- Read in full: `docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R1_BETA_REVIEW.md`, `_R2_`, `_R3_`;
  `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` Q1.
- Not covered by a fresh search of `docs/PROJECT_FILE_CATALOG.md`, `docs/CHAPTER_*` and
  `docs/*_AUDIT.md` for this subject. **Do that search in your first step and add what you find.**

**Documents that already decide this subject:**

| Document | What it settles | Label |
|---|---|---|
| `docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md` (`915d382`), its miner-finding section | The RANGE-MINER route sweeps from 0 whatever `seed_start` is. Shown by reading and by execution on CPU (stripes `[0,250)…[750,1000)` for `seed_start=2147483648`) | ALREADY RECORDED. This brief fixes it |
| `docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R1_BETA_REVIEW.md` Q13, Q10 | Beta confirms the finding by source. The temporary guard blocks Run 4 on purpose; the `base_start` work and the Route-A refresh are prerequisites | ALREADY RECORDED |
| `window_optimizer.py:675-703` (`assert_miner_seed_start_supported`, G-MINER-START) | The temporary guard. Its comment says this brief replaces it | ALREADY RECORDED |
| Owner, 2026-11-14 | The pre-dispatch refusal also covers PWC and ZMQ, which build their work from a fixed 0 (`persistent_worker_coordinator.py:1276`, `zmq_sqlite_coordinator.py:831`) | ALREADY RECORDED in chat; first written here |
| `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md` (`35f5503`) M-B | The worker's executor was driven over `[2³¹, 2³²)` on one NVIDIA GPU: 0 / 0 / 60 / 23,568 survivors. Not run on ROCm | ALREADY RECORDED |
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md` §6, §8 item 4 | The Route-A evidence gets a scoped refresh before Run 4 | ALREADY RECORDED. Part 2 here |
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154-179` | Run 4's claim boundary | ALREADY RECORDED. Not changed |
| `scripts/gate12_parity_gate.py:174` | `miner/range_miner_coordinator.py` is in the rig parity set | ALREADY RECORDED. Consequence below |

**Items this brief raises:**

| Item | Label |
|---|---|
| `docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md:213-215` says every seed in `[seed_start, …)` is visited. Wrong for a non-zero start | ALREADY RECORDED as contradicted (stage 1 report). The doc correction is listed for Alpha, not done here |
| Editing `miner/range_miner_coordinator.py` changes a rig-parity file. The rigs need the new copy before the parity gate can pass at power-on | NEW consequence. Owner's step at power-on; say so at the top of the report |
| A second call to `partition_macro_stripes` at `miner/range_miner_coordinator.py:5104` | NEW. Not read by Alpha. Establish what it is before anything else |

**Not searched:** VM101's live state, ser8 archives, the private repository, the project
PDFs. Anchors were read at `b1cd2fa`. **Re-read each one before relying on it.**

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** record `git rev-parse HEAD` in your first step (expected `b1cd2fa` or later).
**Review class:** BEFORE-COMMIT. This changes the path of a counted run.
**Report:** `docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md`, in your worktree.

**Why.** Run 4 is declared over `[2³¹, 2³²)`. As built it would sweep `[0, 2³¹)` again and
the record would call the upper half covered. Stage 1 now refuses that run. This brief makes
the run real.

**The yardstick (CLAUDE.md).** TFM is judged by prediction-pool hit rate against `k/1000` on
unseen draws. Nothing here is about state recovery or a "true seed".

## 0. Rules for this session

- **No commit, push, branch or tag.** Michael lands everything.
- **Worktree:** `git worktree add --detach /home/michael/tfm_worktrees/miner-start HEAD`
  (hyphen, not underscore: the landing script refuses underscores). Write only inside it,
  except the review request file (§4).
- **No fleet, no GPU, no SSH to any rig.** CPU only, dispatch intercepted or fake workers.
- **No writes to live state:** the live database, the ledgers, `/home/michael/miner_staging`,
  `/home/michael/distributed_prng_analysis/optuna_studies`, the halt file.
- **May edit:** `miner/range_miner_coordinator.py` (the smallest change that works),
  `window_optimizer_integration_final.py`, `window_optimizer.py`, and tests.
- **Not edited:** `miner/range_miner_worker.py`, `miner/range_miner_protocol.py`,
  `miner/dataset_authority.py`, `prng_registry.py`, `sieve_gpu_worker.py`, the kernels,
  `utils/seed_coverage_ledger.py`, `utils/run_finalizer.py`, `utils/step1_search_ledger.py`,
  `agents/watcher_agent.py`, `persistent_worker_coordinator.py`, `zmq_sqlite_coordinator.py`,
  `gate12_launch.sh`, the 22-array contract.
- **Use your judgement on small things and keep going.** A suite that would write to live
  state: skip or redirect that one case, record it, continue. An unexplained red that was
  red at baseline: record it, continue. Stop only at a stop gate (§6).
- **Another session may be running** in `/home/michael/tfm_worktrees/pwc-backup`. It is
  read-only. Leave that directory alone.
- **First step:** `pgrep -af beta_relay.py`; `git rev-parse HEAD`; `git status --porcelain` in
  the main tree; create the worktree; activate the venv; copy the three git-ignored `daily3`
  dataset files into the worktree read-only (sha256 equal to the main tree's); the catalog,
  chapter and audit search named above. Then the baseline of every suite you will re-run.

## 1. Part 1 — the seed start reaches the sieve

**M first.** Reproduce the stage 1 demonstration on CPU before editing: a request with
`seed_start=2147483648` is assigned stripes from 0. Then say what `:5104` is.

| # | Requirement |
|---|---|
| R1 | The start travels: Step 1's `seed_start` → the `run_trial_miner` call (`window_optimizer_integration_final.py:1718`) → `run_trial_miner` (`miner/range_miner_coordinator.py:12336`) → the serve context → `serve_trial` (`:9389`) → `assign_stripes(..., base_start=…)` (`:10324`), for every phase and stage. Every other place that partitions or re-assigns stripes (`:5104`, retries, lease expiry, resume) uses the same start. No path can fall back to 0 |
| R2 | The interval is checked before dispatch: `0 ≤ seed_start`, `seed_start + total_seeds ≤ 2³²`. Outside it refuses |
| R3 | The swept interval is evidence. The stage 1 evidence reader takes it from committed, contiguous `done` stripe rows (`window_optimizer_integration_final.py`, the miner evidence reader). It must now yield `[seed_start, seed_start + total_seeds)` for a real non-zero run and still refuse a mismatch. Prefer deriving the interval from the stripe rows that already carry `seed_start`. If a ledger schema change is the only way, read the migration note at `miner/range_miner_coordinator.py:1400` first and stop if a live-ledger migration is needed |
| R4 | G-MINER-START (`window_optimizer.py:675-703`) no longer refuses the RANGE-MINER route. The second guard (credit only when swept equals requested) stays exactly as it is |
| R5 | **Owner decision:** the pre-dispatch refusal now covers PWC and ZMQ. A Step 1 run on either route with `seed_start != 0` refuses before any dispatch, naming the reason. The guard lives in `window_optimizer.py` / the integration file; the two coordinator files are not edited |
| R6 | Check `tests/gate_s172_prod_shape.py` (G-PROD-SHAPE) for any assumption that stripes start at 0. Report what you find. Edit it only if it would fail a correct non-zero run, and then ask Beta about that edit |
| R7 | Nothing changes for a run at `seed_start = 0`: the assigned stripes, payloads and evidence for attempt 9's shape are identical before and after. Prove it |

**Tests** (CPU, real coordinator, fake workers; each refusal has a mutant shown red):

- `seed_start = 2³¹`, `total = 2³¹`, stripe size as Run 4 (`67108864`): 32 stripes per
  phase, first at `2147483648`, last ending at `4294967296`, contiguous, no gap or overlap,
  in all four phases.
- A small non-zero start with a short final stripe.
- Retry, lease expiry and resume keep the start.
- R2 at the exact endpoints and one past each.
- R3: a real non-zero run credits; a mutated stripe row (gap, wrong start, missing `done`)
  refuses.
- R5: PWC and ZMQ at non-zero refuse before dispatch; at 0 they behave as before.
- R7: the zero-start control, byte-equal stripe assignments.
- End to end, dispatch intercepted: WATCHER with an open pinned search over `[2³¹, 2³²)`
  builds the Run 4 command, the child passes G-MINER-START, the coordinator assigns
  upper-half stripes, the evidence reader returns `[2³¹, 2³²)`, credit is given, and the
  history then reads COMPLETE over `[0, 2³²)` in a throwaway database.

## 2. Part 2 — refresh the Run 4 evidence

`tests/test_s172_run4_routing_patch.py` (4/20) and `tests/run4_prelaunch_provenance_check.py`
are red because their stubbed databases have no search lookup, so WATCHER fails closed.
Refresh the fixtures so they exercise today's path: an open pinned search over
`[2³¹, 2³²)`, the seven pins unchanged, `--seed-start 2147483648 --max-seeds 2147483648`.
Change fixtures only. Weaken no assertion and keep no obsolete fallback. Report each
assertion that changed and why. If an assertion can only pass by being weakened, leave it
red and report it.

## 3. Report

`docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md`, opening with `## Prior record`. At the top,
for the owner, in plain words: what now happens when Run 4 is launched; that the rigs need
the new `miner/range_miner_coordinator.py` before the parity gate at power-on; what was not
verified (no GPU, no fleet, no ROCm). Then: first-step output; M; each requirement with
`file:line`; tests with counts and mutants; suites before and after; Beta's last position
per question; anything unresolved; the landing list with sha256 in two groups (files the
landing script takes; files under `miner/` and any other hand-landed file).

## 4. Review by Beta, carried by this session

As `docs/exchange/RELAY_PROTOCOL.md` §2. Check `pgrep -af beta_relay.py` first and wait if
one runs. Request file, in the main tree:
`docs/exchange/to_beta/X-MINER-START_R1_ALPHA_REVIEW_REQUEST_seed_start_into_sieve.md`,
opening with `## Prior record`, review class BEFORE-COMMIT, files by absolute worktree path.
At most three rounds. Fix what you agree with inside this brief's scope. Stop when you
disagree with a LIVE objection or its fix is outside scope.

**Questions for Beta (use as written):**

1. Does the start reach every place stripes are partitioned or re-assigned, with no path
   back to 0?
2. Is the swept interval taken from execution evidence, and can a wrong interval be credited?
3. Is a run at `seed_start = 0` unchanged?
4. The PWC and ZMQ refusal: before dispatch, and nothing else changed on those routes?
5. G-PROD-SHAPE: any assumption of a zero start left, and is any edit to it sound?
6. The Run 4 fixtures: was any assertion weakened?
7. What remains unverified until a real fleet run, and is anything in it a reason not to
   launch Run 4 over `[2³¹, 2³²)` once the rigs carry this file?
8. Any governing record this work missed.

## 5. Stop gates

Stop and report when: M does not reproduce; R1 needs an edit to a file listed as not
edited; a live-ledger migration is needed; R7 cannot be shown; Beta holds a LIVE objection
you disagree with or whose fix is outside scope; the third round ends.
