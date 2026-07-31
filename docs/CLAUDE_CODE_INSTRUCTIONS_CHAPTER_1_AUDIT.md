# CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_AUDIT.md — REV1

**Documentation audit: `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` vs. live source.**

**This is an AUDIT ONLY. Do not correct the chapter. Do not change any code.** The
correction pass is a separate, separately-authorized deliverable that will be written
*from* your findings after Michael and Team Alpha review them. Producing a report that
lets someone else safely rewrite the chapter is the deliverable.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. You do
NOT commit, push, or run WATCHER. STOP at the gate.

---

## 0. Why this exists

The chapter documents are the project's durable reference, and they have **never been
systematically verified against the rewritten pipeline.** That gap has already caused
material harm:

- The stale "~62 features" figure propagated from `feature_importance.py:95-119` into
  multiple documents and was repeated as fact for months. The real value is 91 extracted /
  89 trained.
- In one session Team Alpha, Team Beta and Claude Code *independently* recommended removing
  `skip_min`/`skip_max` from variable-skip search — a cornerstone — because none of them had
  read the document defining it. **That definition is in this chapter**
  (`skip_min` = "Minimum skip for variable PRNGs", `skip_max` = "Maximum skip for variable
  PRNGs"). Michael stopped it.

So the failure mode is not "the docs are untidy." It is: **a stale or absent chapter causes
correct components to be broken by people acting in good faith.**

## 1. The falsifiable question

> For every substantive claim in `docs/CHAPTER_1_WINDOW_OPTIMIZER.md`, is it **accurate**
> against live source at HEAD, and if not, what is the true state?

## 2. Known starting signals (verify, do not assume)

- The chapter header declares **Version 3.1** and cites **`~868 + ~595` lines** for
  `window_optimizer.py` + `window_optimizer_integration_final.py`. Compare to actual line
  counts. A large divergence quantifies expected drift; report both numbers.
- **`docs/CHAPTER_1_PATCH_S114.md` exists as a separate document.** Determine whether its
  content was ever folded into the chapter, or whether the chapter and the patch now
  disagree. An unmerged patch is itself a finding.
- The chapter's §4.3 is titled "Threshold Philosophy" — check it against the
  **whitepaper §7** rationale (loose thresholds are required to preserve a learnable
  manifold) and against the just-repaired propagation path (`8a55a68`).

## 3. Required classification

Classify **every** substantive claim into exactly one of:

| class | meaning |
|---|---|
| **ACCURATE** | matches live source; cite `file:line` |
| **STALE** | was true, no longer is; give the current truth + `file:line` |
| **SUPERSEDED** | replaced by a different mechanism; name the replacement |
| **CONTRADICTED-BY-CODE** | the doc states intent the code does not implement — **distinguish carefully whether the doc or the code is wrong** |
| **UNVERIFIABLE** | cannot be checked from available surfaces; say why (VIR-5) |

**The CONTRADICTED-BY-CODE class is the most important one and the easiest to get wrong.**
`skip_min`/`skip_max` are the worked example: the chapter documents them for variable PRNGs,
the hybrid kernels do not accept them. **The chapter is right and the code is defective.**
Do not assume the code is the authority. When a doc and the code disagree, report the
disagreement and — where you can establish it — which one reflects design intent, citing
the whitepaper, git history, or other documents.

## 4. Areas requiring specific attention

Work the whole chapter, but these carry known risk:

1. **§3.1 `WindowConfig` / §3.2 `SearchBounds`** — every field: does it exist, does it carry
   the documented meaning, does its value reach the code that claims to consume it? This is
   where the skip-bound and offset dead dimensions live.
2. **§4 Search Bounds Configuration** — the chapter claims a "Single Source of Truth"
   (§4.1). Verify that claim. Note `distributed_config.json` holds **bare-metal** rig
   addresses **deliberately** (`CLAUDE.md` §3: *not a bug, must not be corrected*) — do not
   report that as an error.
3. **§4.3 Threshold Philosophy** — against whitepaper §7 and the repaired path.
4. **§6.2 BayesianOptimization / §8 Bayesian flow** — the Optuna search space, what is
   actually sampled, and (critically) **whether each sampled parameter reaches execution.**
   Two dead dimensions are already known: hybrid `skip_min`/`skip_max`
   (`prng_registry.py:1027, :805, :885, :1159`; `range_miner_worker.py:776`, `:871`,
   `_hybrid_prefix:177-193`) and forward-hybrid `offset` (sampled
   `window_optimizer_bayesian.py:423`). **Report any others you find** — this defect class
   has appeared four times.
5. **§7.2 `test_configuration()`** — recently changed by the threshold repair (`8a55a68`);
   verify the chapter against current behavior including `resolve_directional_threshold()`.
6. **§8.3 "Test Both Modes (V2.0)"** — constant + variable. Note Team Beta has ruled
   combined-session sequential sieving **non-certifying and prohibited by default**, and
   production re-optimization is **per-session**. Report whether the chapter reflects that.
7. **`offset` semantics** — the chapter, `parameter_registry.json:38-43`, and the loader
   reportedly give three different interpretations (head-relative array index vs. "time
   offset from current draw"). Establish what the code does and what each document claims.

## 5. Out of scope

- **Do not edit the chapter or any other document.**
- Do not change code, tests, or config.
- Do not re-audit the threshold propagation path (done: `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`)
  or the dataset consumer contract (done: `DAILY3_CONSUMER_CONTRACT_v1.md`) — **cite them**
  where the chapter overlaps.
- Do not fix the skip-bound dead dimension. That is the next correctness deliverable and has
  its own brief.

## 6. Verification-integrity controls (VIR-1…6)

- **execution proof** — every ACCURATE/STALE verdict carries a `file:line` anchor read this
  session.
- **clean control** — state explicitly which chapter sections you verified and found correct.
  A report listing only defects gives no evidence the rest was checked.
- **fault-injection control** — n/a for a read-only audit; say so rather than omitting it.
- **completion sentinel** — terminate with an explicit
  `PASS | FAIL | UNAVAILABLE | INCOMPLETE` and a per-section coverage table. A section not
  reached is `INCOMPLETE`, never silently absent.
- **unavailable-observer behavior** — anything you cannot verify is `UNAVAILABLE`, not
  assumed correct.
- **audit claim scope (VIR-6)** — declare searched surfaces and unavailable ones. If you
  check only VM 101 and not the rigs' deployed copies, say so. **The repository is not the
  system.**

## 7. Deliverable

`docs/CHAPTER_1_AUDIT_v1.md`, containing:

1. **Header reality check** — declared version and line counts vs. actual.
2. **`CHAPTER_1_PATCH_S114.md` disposition** — merged, unmerged, or conflicting.
3. **Per-section classification table** — section → claim → class → `file:line` → true state
   if not accurate.
4. **Dead-dimension inventory** — every sampled-but-unreached parameter found, with the hop
   where it dies.
5. **Doc-vs-code conflicts** — each with your assessment of which reflects design intent and
   the evidence for that assessment.
6. **A prioritized correction list** — what a future rewrite must change, ordered by
   consequence, so the correction pass can be scoped from it.
7. **Coverage + completion sentinel.**

Then STOP for Team Alpha review. **Do not commit.**
