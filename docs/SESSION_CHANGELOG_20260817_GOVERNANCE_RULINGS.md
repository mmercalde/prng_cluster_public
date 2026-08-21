# SESSION CHANGELOG — 2026-08-17 — GOVERNANCE RULINGS

**Session type:** governance / sequencing (no code changes)
**Predecessor:** `SESSION_CHANGELOG_20260817_GATE12_ATTEMPT9.md` (the passing run itself)
**Repo state at open:** `7b10d2d` (Gate-12 passed, skill v26) · **at close:** `73633e7`
**Naming per:** `docs/TB_RULING_CHANGELOG_NUMBERING.md` (date + topic canonical until
SER8-backlog reconciliation)

---

## 1. What this session was

The governance session immediately following the Gate-12 Attempt-9 pass. Two Beta rulings
were received, recorded, and executed; the Gate-12 sequencing items they mandated were
completed; and one material finding was made — the window-anchor design artifact presupposed
by Beta's first ruling does not exist — which produced the second ruling.

## 2. Beta ruling 1 — Gate-12 Attempt-9 acceptance

Committed verbatim: `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` (`d391a5c`).

- **Gate-12 Attempt 9 PASSED; coverage/cursor certification ACCEPTED.**
- **MP-1 drain-starvation defect CLOSED** (R-1..R-4). Do not reopen R-1.
- **Fields 1 and 2: MISSED as written, remedy not refuted.** Field 1's `<100 s` pump
  prediction failed on its own precondition (pump-call population moved 4.6×); per-call
  collapse 1.463→0.210 s is corroboration, not a rewritten pass. Field 2's `<60 s` staging
  prediction missed at 159.9 s; the `staging/msg` ratio ruled non-binding; −76.5% absolute
  and 70.3%→21.7% of serve-loop wall, with 25/25 serviced / queue 0 / zero lease expiries,
  closes the causal question.
- **Field 6: UNOBSERVED — instrumentation-output defect.** Both `_bp` falsifier fields were
  computed but never persisted to the `[S172-BP] summary` line. Bounded observability-only
  repair mandated (own gate, no logic changes, no perf work); **no Gate-12 rerun**; first
  production observation comes from the next Step-1 run.
- Publication symlinks: no blocker; NPZs stay out of git; `.s172_accumulator/generations/`
  is durable data plane needing its own backup policy.
- Sequencing: anchor `e9ca800` + preserve bundle → window-anchor merge → field-6 repair →
  Phase 7.

## 3. Sequencing item 1 executed — anchor + preserve

- Annotated tag **`gate12-passed-attempt9` → `e9ca800`**, pushed to both remotes; verified
  from an independent clone that the tag resolves to the launch HEAD on the public remote.
- Forensic bundle: `logs/gate12_20260817_181819*` + launcher log tarred on VM101 →
  `gate12_attempt9_forensic_bundle_20260817.tar.gz`, sha256
  `583fbab3f4f7772f5405f302dbea596e8303a71420a0b2445149025470743fa2`, copied to ser8
  `~/Downloads/` (forensic-archive convention). Hash recorded durably in BACKLOG §19.

## 4. Sequencing item 3 prepared — field-6 repair brief

`docs/CLAUDE_CODE_INSTRUCTIONS_FIELD6_OBSERVABILITY_REPAIR.md` (`206ec4a`). Written from
source reads, not the ruling alone. Key findings folded in:

- The defect is **narrower than "never persisted"**: `staging_backpressure_metrics()`
  already exports both fields via `dict(self._bp)` (`:7226`); the omission is solely the
  `[S172-BP] summary` format string (`log_staging_backpressure_summary`) — the artifact a
  production run actually persists.
- **Silent-failure trap identified:** the mandated `None`/UNOBSERVED sentinel would make the
  existing `int()` casts in the update block (`:7978-7985`) raise TypeError into a blanket
  `except: pass` — permanently fake-UNOBSERVED. Scope B rewrites the update None-aware;
  mutant M3 exists to prove the gate catches the regression.
- Constraints carried: no new `def` in the coordinator (AST scope-proof gate), gate extends
  the committed staging-backpressure suite (Gate-22), grep-stable line appended-only,
  UNOBSERVED literal, three gate arms (population variance / UNOBSERVED pin / dict↔line
  coherence), mutation evidence M1-M3.
- **Rider:** the `:7741` `_pump_deferred` docstring debt attaches by its own rule (first
  commit that legitimately touches the definition); recommended wording taken verbatim from
  the R-1..R-4 report §3.1; flagged as debt-rule, not scope creep.

Also in `206ec4a`: **BACKLOG §19** — accumulator backup/recovery policy item (Beta-ruled
real, non-blocking), both dispositions recorded, bundle sha256 pinned. Diff verified as pure
append (32 insertions, 0 deletions).

## 5. Finding — the window-anchor design artifact does not exist

Sequencing item 2 ("perform the window-anchor production merge") presupposed a design.
Searched: both remotes, all refs (unshallowed, incl. GitHub-only
`claude/project-status-review-r222oc`), full history · VM101 branches/worktrees/stash +
date-keyed name-agnostic find since 2026-08-10 (owner-run) · ser8 `~/Downloads` (owner-run)
· prior chat sessions. **Negative on all surfaces.** What exists is the problem
characterization: Chapter 2 F-4 (CONFIRMED, not repaired), `AUDIT_STEP1_OFFSET_REACH.md`,
skill §2.21, and attack-plan D.1 ("described only, not implemented … a proposal to Beta").
Declared unavailable: pre-repository ser8 archives.

Ruling request committed: `docs/TB_RULING_REQUEST_WINDOW_ANCHOR_SEQUENCING.md` (`3b5e577`)
— four questions: hidden-artifact check, scope (a)/(b)/(c), kernel-ABI constraint,
re-sequencing.

## 6. Beta ruling 2 — window-anchor / generator-phase sequencing

Committed verbatim: `docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md` (`73633e7`).
**APPROVED WITH TWO CORRECTIONS.**

- Beta's "merge" wording was Beta's own sequencing error; **no hidden design exists;
  proposal phase authorized**: `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md`,
  sequence design → review → implement → acceptance → merge, against
  `gate12-passed-attempt9`.
- Scope: **(a) separation + (b) hybrid semantics mandatory; (c) skip-bound semantics OUT.**
- Correction 1: Alpha's "forward hybrids receive no offset" too broad — **per-variant
  capability matrix required** (lcg32/pcg32 hybrids carry phase; java/minstd/xorshift32/
  xorshift128 hybrids do not; all covered reverse hybrids carry trailing `int32(offset)`).
- Correction 2 (binding semantic contract): `window_anchor` = which observed records form
  the residue window; `generator_phase` = generator advances before first comparison;
  **never reconstructed from one another; never emulated** on no-phase variants.
- **Frozen kernel ABI BINDING for v1.** Any need for independent phase on the four no-phase
  forward hybrids = separate kernel-ABI v2 dependency with its own certification cycle.
- **`generator_phase = 0` for v1 and D.1** — not an Optuna dimension; `[0,100]` neither
  inherited nor raised; anchor domain derived from data:
  `0 ≤ window_anchor ≤ N_filtered − window_size`.
- Step-3 consumer law `offset = train_history_len` is a **consumer continuation law** —
  untouched, renamed `continuation_phase` in design narrative, changes need their own
  ruling.
- **Re-sequencing APPROVED:** field-6 now → Phase-7 soak after field-6 lands
  (**non-certifying for anchor semantics** — observability/autonomy evidence only, must not
  be cited as window-anchor acceptance evidence) → proposal in parallel → design approval →
  implement → post-change semantic/parity acceptance → merge → D.1 differential experiment.

## 7. State at close

| track | state |
|---|---|
| Field-6 repair | brief committed, **head of executable sequence**, not yet executed |
| Phase-7 soak | gated behind field-6 landing; classified non-certifying for anchor semantics |
| Window-anchor proposal | authorized, to be drafted (fresh session per long-context rule) |
| WATCHER failure-authority defect | still open, untouched this session |
| Certified baseline | `gate12-passed-attempt9` = `e9ca800`, both remotes, bundle archived |

## 8. Commits this session

| commit | content |
|---|---|
| `d391a5c` | TB ruling: Gate-12 acceptance (verbatim + dispositions table) |
| `206ec4a` | Field-6 repair brief + BACKLOG §19 |
| `3b5e577` | TB ruling request: window-anchor design absence |
| `73633e7` | TB ruling: window-anchor sequencing (verbatim + dispositions table) |
| tag | `gate12-passed-attempt9` → `e9ca800`, both remotes |

All dual-pushed; each verified on the public remote from an independent clone.
