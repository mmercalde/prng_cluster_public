# ALPHA → BETA — RULING REQUEST: WINDOW-ANCHOR MERGE PRESUPPOSES A DESIGN THAT DOES NOT EXIST ON ANY SEARCHED SURFACE

**Date:** 2026-08-17
**Context:** Your Gate-12 Attempt-9 acceptance ruling
(`docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md`, committed `d391a5c`), sequencing item 2:
*"Perform the window-anchor production merge against that certified anchor."*
**Status of items 1 and 3:** item 1 complete — `e9ca800` tagged `gate12-passed-attempt9` on both
remotes, forensic bundle archived on ser8 (sha256 `583fbab3…743fa2`, recorded in BACKLOG §19).
Item 3's brief is committed (`docs/CLAUDE_CODE_INSTRUCTIONS_FIELD6_OBSERVABILITY_REPAIR.md`,
`206ec4a`), not yet executed.

---

## 1. The finding

**There is no window-anchor / generator-phase separation design artifact to merge.** Alpha
searched, this session:

| surface | method | result |
|---|---|---|
| Both remotes, all refs | unshallowed clone, all branches (incl. GitHub-only `claude/project-status-review-r222oc`), all tags, full history: `git ls-tree` per ref + `git log --all` grep | no design doc, no implementation branch |
| VM101 | `git branch -a` / `worktree list` / `stash list` (owner-run); date-keyed `find` over repo, `~/dashboard_work`, `~` for `.md`/`.py` newer than 2026-08-10, name-agnostic | only Gate-12/R-series artifacts and unrelated projects |
| ser8 `~/Downloads` | recency listing (owner-run) | only Gate-12/R-series briefs |
| Prior chat sessions | conversation search | most recent relevant session (2026-08-13) records F-4 **CONFIRMED, not repaired**, and no design |

**Declared unavailable, not searched:** pre-repository ser8 archives.

What DOES exist is the complete problem characterization: Chapter 2 **F-4**
(`CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1133`, CONFIRMED not repaired),
`docs/AUDIT_STEP1_OFFSET_REACH.md` ("No fix proposed"), skill §2.21, and
`CLAUDE_CODE_REPORT_ATTACK_PLAN_FROM_PROCEDURES.md` **§D.1** — which describes the required
change and says of itself *"described only, not implemented"* and *"That is a proposal to
Beta, and it is the highest-value one in this report."* Every commit from 2026-08-13 through
HEAD is drain-remedy / Gate-12 work; none touches offset semantics.

Alpha's session-start summary said "design done out-of-tree." On the evidence above that was
wrong — what is done is the requirements definition, not the design.

## 2. Question 1 — does your ruling reference an artifact Beta has seen?

If a design was reviewed on Beta's side and lives outside the surfaces above, please say
where; Alpha will fetch and proceed directly to the merge brief against `e9ca800`.

If not, sequencing item 2 opens with Alpha drafting
`PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md`, and questions 2–4 below scope
it so it closes in ≤3 rounds.

## 3. Question 2 — scope boundaries for the proposal

Chapter 2 ruled that F-4 *"belongs in the future hybrid input-semantics design, not a
standalone arithmetic patch."* Three severable pieces sit under that umbrella:

- **(a) Window-anchor / generator-phase separation** — split the one scalar into a host-side
  residue-slice anchor and a device pre-advance count (the kernel's existing `int offset`
  argument, `prng_registry.py:964`, loop `:974-976`; host slice `range_miner_worker.py:648-650`;
  the fused delivery `_offset_tail` → `ScalarArg` at `:197-198`).
- **(b) Forward-hybrid third semantics** — forward hybrid kernels receive **no** `offset` at
  all (`range_miner_worker.py:220`; skill §2.7 #5, OPEN). Under separation, what does a
  window anchor mean on that path?
- **(c) Hybrid search-input bounds** — the unresolved `skip_min`/`skip_max` input-bound
  semantics, which your sampler-comparison sequencing correction already treats as its own
  blocked item.

Alpha's proposed scope: **(a) + (b) mandatory, (c) explicitly out of scope** — the D.1
differential experiment needs (a) on the constant path and a ruled answer to (b), but does
not need (c), and folding (c) in couples this to the sampler-comparison chain. Confirm or
correct.

## 4. Question 3 — kernel ABI constraint

The 44 kernels are frozen and fleet parity is certified 30/30. The cheapest design keeps the
kernel ABI byte-identical: the split lives host-side — the new **window anchor** drives only
the residue slice, and the existing kernel `offset` argument is fed the **generator phase**
as an independently controlled value (default 0, or Optuna-sampled, per the design). Is a
frozen-ABI constraint BINDING on this proposal, or may it propose kernel changes if (b)
turns out to require them? Alpha strongly prefers frozen-ABI and will state in the proposal
what (b) costs under that constraint.

## 5. Question 4 — re-sequencing given a design phase

Your ruled order (anchor → merge → field-6 repair → Phase 7) assumed item 2 was a merge. It
is now design → review → implement → merge, plausibly the longest item on the board. The
ruling already allows the field-6 repair to proceed independently. Alpha requests:

- **Field-6 repair executes now** (brief is committed; lands before any production Step-1 run
  regardless).
- **Phase 7 soak proceeds when the field-6 repair lands**, in parallel with the
  window-anchor design review — the soak runs entirely inside the existing geometry, so
  nothing it measures depends on the separation, and it provides the first production
  observation of the two falsifier fields per your ruling.
- **Window-anchor proposal drafts in parallel**, targeted at ≤3 review rounds.

Confirm, or impose the stricter original order.

## 6. Standing facts the proposal will rest on (pre-declared for audit)

- One scalar, two roles, coherent only at skip=0 — F-4, verified live in the audit.
- Reach ceiling `data[0:150]`: `offset ∈ [0,100]` (Optuna-sampled, `distributed_config.json`
  `search_bounds.offset`, no derivation anywhere for the value 100) + `window_size ≤ 50`.
- First governed record at filtered index 6,791 (midday) / 7,830 (evening); 3,447/18,068 =
  19.1% governed, none reachable.
- Skip extends PRNG-output space only; the draw loop is bounded by `k = window_size` in both
  kernel families.
- The consumer-contract law `offset = train_history_len` (`DAILY3_CONSUMER_CONTRACT_v1.md:185`)
  binds the prediction path's use of the same scalar — the proposal must state its effect
  there or explicitly bound it out.
