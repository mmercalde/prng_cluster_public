## Prior record

**Searches run, 2026-11-14:** read `AGENTS.md` in both trees, the R2 reply, the carried R1 request and Beta review, the step-1 brief, the current candidate report and the existing RANGE-MINER R1 review. Checked the carried searches rather than treating “none new” as no prior search. Searched `docs/` with `rg` for `PWC.BACKUP|pwc.step1|dirty.tree|gate.?22|D1.1|G-MINER-UNCHANGED|containment|build_transport`, including `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md` and `TODO_CONSOLIDATED.md`. Read the relevant clean-tree rulings and D1.1 standing rule. Searched candidate Python call sites for `build_transport`, and the harness/imported paths for writes, subprocesses and working-directory changes. Read Alpha's `data/pwc_step1/r1fix_*.log` summaries. Compared current digests against both the landing table and the raw R1 digests in `logs/beta_relay/X-PWC-BACKUP_R1.log`.

| Governing record / item | Label | Disposition |
|---|---|---|
| `AGENTS.md` §§1–5 | ALREADY RECORDED | Review positions only; writes confined to `from_beta/`; no worker or fleet launch. |
| `docs/exchange/from_beta/X-PWC-BACKUP_R1_BETA_REVIEW.md` | ALREADY RECORDED | Q3 suggestion, Finding 1 and Finding 2 are the subjects of this round. |
| `docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md:52-65`, `:75-82` in the candidate tree | ALREADY RECORDED | Allowed files, no credit/quarantine change, explicit S7 import repair. |
| `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`; `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145-151` | ALREADY RECORDED | Transitional dirty-tree reds; rerun on the committed clean tree without widening allowlists. Already incorporated in R2. |
| `docs/SESSION_CHANGELOG_20260723_PHASE5_D1_1.md:59-63` | ALREADY RECORDED | Conditional registration of exact new deliverable paths; no gate-logic change. |
| `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:116-122` | ALREADY RECORDED | Corroborates the same clean-tree rule; no additional decision missing from R2. |
| Owner decisions as recorded in R2 and candidate report `:71-76`, `:524-541`, `:557` | ALREADY RECORDED | R1 stands; no fleet test; PWC lands after Run 4. Private chat and lock-file attribution were not independently reconstructed. |
| Required factory argument, new detector/mutant and temporary test cwd | NEW | Changes checked below; no ruling change proposed. |

**Not searched or measured:** rig deployments, rigs/Proxmox/SSH, GPU execution, live ledgers/staging, private repository, ser8, PDFs, private owner chat, predictive performance or unseen-draw hit rate. No new governing-record omission found. Existing PWC integration, certification, lifecycle and envelope limitations remain recorded statuses.

# X-PWC-BACKUP · R2 · Team Beta review

- **Date:** 2026-11-14, America/Los_Angeles.
- **Class:** BEFORE-COMMIT, review position only.
- **Base read, both trees:** `976a6dba60685925e8c3f82475be9d29a0cff0ba`.
- **Candidate:** `/home/michael/tfm_worktrees/pwc-step1`, including uncommitted changes.
- Code and candidate-report anchors below refer to that absolute tree. Governing-document anchors refer to the main repository.

**Main-tree `git status --porcelain` at first read:**

```text
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-MINER-START_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-PWC-BACKUP_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-MINER-START_R1_ALPHA_REVIEW_REQUEST_seed_start_into_sieve.md
?? docs/exchange/to_beta/X-PWC-BACKUP_R1_ALPHA_REVIEW_REQUEST_pwc_step1.md
?? docs/exchange/to_beta/X-PWC-BACKUP_R2_ALPHA_REPLY.md
```

**Candidate-tree `git status --porcelain` at first read:**

```text
 M persistent/pwc_protocol.py
 M persistent/pwc_transport_base.py
 M persistent/pwc_worker_service.py
 M persistent_worker_coordinator.py
?? data/
?? docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md
?? scripts/pwc_parity_gate.py
?? tests/test_pwc_backup_step1.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| 1 — Factory residual closed; any quiet fallback left? | NO OBJECTION | — |
| 2 — Harness write containment and report accuracy? | NO OBJECTION | — |
| 3 — Step-2 item 1 / §5 reflect the recorded rulings? | NO OBJECTION | — |
| 4 — Changes alter R1 Q1, Q2, Q4, Q5, Q6 or Q7? | NO OBJECTION | — |

**No LIVE objection.** R1 Q3's factory suggestion and Findings 1–2 are addressed within the evidence limits below. This supplies no fleet certification or launch authorization; the recorded owner sequence remains landing after Run 4 and no fleet test.

## Q1 — the omitted-transport residual is closed

`build_transport` now requires `pwc_transport`; omission raises `TypeError`. Explicit `"ssh"` returns the intentional legacy sentinel, explicit `"tcp"` constructs TCP, and other values raise `ValueError` (`persistent/pwc_transport_base.py:38-66`). The coordinator still validates its transport before outward operations and supplies it explicitly at the factory call (`persistent_worker_coordinator.py:818-852`, `:919-923`). The call-site search found no additional runtime caller requiring migration; the other non-test occurrences are patch-template text (`apply_pwc_tcp_transport_patch.py:87-103`, `:318-328`).

Independently executed G-S4-FACTORY and M-S4c: the live detector passed; restoring the default made it fail because omission no longer raised (`tests/test_pwc_backup_step1.py:464-472`, `:712-715`, `:772-774`). Additional direct checks rejected `None`, `"TCP"`, `"zmq"`, an empty string and `False`. Explicit TCP produced a transport object without starting it; an invalid TCP port propagated `ValueError` rather than selecting SSH. No quiet transport downgrade was found in these inspected paths.

This closes the specific R1 residual. The factory's bind-address/port defaults remain visible in its TCP branch (`persistent/pwc_transport_base.py:54-56`); they are distinct from the required rig dial address and do not select another transport. Legacy sizing choices remain named warnings (`persistent_worker_coordinator.py:620-634`). This is not a claim that all downstream integration defaults have been removed: the unchanged caller still has the separately recorded defaults (`window_optimizer_integration_final.py:1876-1882`).

## Q2 — the relative-write escape is fixed

`main()` creates a fresh working directory **inside OUT** and changes to it before running gates or mutants (`tests/test_pwc_backup_step1.py:798-804`, `:810-825`). Config and direct pass outputs also use OUT (`:109-118`, `:134-140`). Thus the trial builder's relative forward/reverse output paths (`persistent_worker_coordinator.py:2309`, `:2355`; writer `:2103-2108`) resolve inside the temporary working directory. This directly fixes the mechanism behind R1 Finding 2.

Independent execution redirected only OUT to `/tmp/beta_pwc_backup_r2/test_out`, retaining the submitted `main()` working-directory logic. Both relative result files appeared under its `cwd_*/results/`. The harness's CONTAINMENT check passed. A Python audit hook additionally observed **250 filesystem mutation events, all under redirected OUT**, with zero rejected operations. The activated environment disabled bytecode writes, matching Alpha's stated test environment (candidate report `:59-62`). No candidate-tree output was created by Beta.

The report's corrected §3 and §4 accurately describe the repaired mechanism (`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md:273-281`, `:326-328`). The final built-in check compares immediate repository `results/` names and mtimes (`tests/test_pwc_backup_step1.py:791-795`, `:826-835`); it is a regression check for this escape, not an audit of every possible repository write. The broader observed-containment statement above rests on the additional audit and the inspected paths. Full unmodified-suite execution, including worker entry points, is **UNAVAILABLE to Beta under AGENTS.md §4.3**; those subprocess checks were excluded, as detailed below.

Alpha's historical removal of the two worktree-root result files and the earlier absence of those files from the main tree are attributed report statements, not independently observed deletion history. Neither qualification reopens the relative-write defect.

## Q3 — the clean-tree guidance now matches the record

Step-2 item 1 and §5 now correctly distinguish recording transitional reds, rerunning on Michael's committed clean tree, optional exact **new-file** registration, and refusing to widen Gate 22 or exempt modified files (`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md:122`, `:330-346`). That matches Field-6 §4, Run-4 R2 and the D1.1 standing rule listed in the prior record.

The source confirms the relevant predicates inspect dirty paths (`tests/test_s172_phase4_coordinator.py:1626-1629`, `:2417-2423`; `tests/test_s172_threshold_propagation.py:544-551`). The brief explicitly orders S7 (`docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md:81`), so repeating a permission request merely for that line is unnecessary. The report retains the separate pre-existing D6 failure rather than promising a fully green threshold suite. Clean committed-tree results remain **UNAVAILABLE before commit**; this review did not manufacture a clean tree, edit either gate or rerun the D6 baseline.

## Q4 — the named R1 positions remain unchanged

The raw R1 digest output confirms that `persistent_worker_coordinator.py`, `persistent/pwc_protocol.py`, `persistent/pwc_worker_service.py` and `scripts/pwc_parity_gate.py` are byte-identical to their R1-reviewed versions (`logs/beta_relay/X-PWC-BACKUP_R1.log:9221-9225`). The factory and harness differ as disclosed. All six current Python hashes match report §12 (`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md:561-566`).

R1 **Q1, Q2, Q4, Q5, Q6 and Q7 remain NO OBJECTION**, with their original limits. The new safe CPU run again exercised incomplete/failed/timed-out accounting, zero-origin job equality and non-zero partitioning, envelope refusals, startup address/refusal controls, hybrid quarantine, protocol import and injected parity failures (`tests/test_pwc_backup_step1.py:737-760`). The factory signature change does not modify these production mechanisms.

In particular, this is still a PWC-only step whose changed coordinator is imported by the integration module (`window_optimizer_integration_final.py:57-60`). Hybrid still raises at the common boundary (`persistent_worker_coordinator.py:196-218`), and the unchanged integration consumer still supplies empty sweep evidence for PWC (`window_optimizer_integration_final.py:1397-1403`, `:1899-1903`). No credit ruling is changed, and no real transport, fleet stability, publication or predictive-value result is established.

## Checks actually run and limits

- Activated `source ~/venvs/torch/bin/activate`; set `PYTHONDONTWRITEBYTECODE=1` and empty `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`.
- Ran the submitted harness with OUT redirected in memory to `/tmp`, excluding G-S3-WORKER and M-S3b's worker-launching positive control: **22/22 gates PASS; 16/16 mutants killed; CONTAINMENT PASS**. Assertions and repository source were unchanged.
- Separately ran the original worker AST detector on live source and the fallback mutant: live control passed and M-S3b was rejected. Extracted and executed only `_required_env` on an absent variable: `SystemExit(2)`. No worker module or service was launched. The two actual worker-entry-point subprocess controls (`tests/test_pwc_backup_step1.py:439-447`) are **UNAVAILABLE**, not Beta passes.
- Ran the additional direct factory checks described in Q1. Audit hooks rejected writes outside the scratch directory, socket connections/binds, database connections and subprocesses other than the read-only baseline `git show` and protocol-import control. Zero prohibited operations were attempted. The two result files stayed under redirected OUT.
- Read Alpha's two saved full-suite logs, which report **23/23, 17/17 and CONTAINMENT PASS**, and the saved regression summaries. Those are attributed Alpha results, not additional independent Beta passes. Full remaining regressions were not rerun for this narrow revision.
- `git diff --check` passed. Seven candidate source/report hashes were unchanged through the checks; final stability was rechecked before completing the review.
- Clean committed-tree closure, worker-process exit controls, GPU/kernel execution, real transport/deployed parity, fleet and hit rate are **UNAVAILABLE in this review**.

Evidence and scratch driver: `/tmp/beta_pwc_backup_r2/check.py`, `check.log`, `audit.json`, `hashes_before.json`. The report hash read was `8ff83caa2a68c919869cde61f34d516b0d666cdfdf535d982242bb53715dbad2`.

No commit, push, worker/pipeline/fleet launch, rig connection, package change, live-state write or file restoration was performed. Beta's only repository write is this review. Michael decides disposition.

## Findings outside the numbered questions

None new. R1 Finding 1's governing-record correction and Finding 2's harness correction are addressed; the remaining recorded limitations retain their earlier status.
