# SESSION CHANGELOG — 2026-08-28 — RUN-4 ROUTE-A ROUTING PATCH

**SR-2 changelog. No S-number.**
**Session span:** the Route-A implementation cycle, from Beta's brief review through commit and
the post-commit runway.
**Commits produced:** `3e2537b` (patch + acceptance artifacts), `e75a6eb` (housekeeping).
**Certified production digest:** `agents/watcher_agent.py` sha256
`d14bcb3b3395877017e33e15fe00fb31b775945bdb1cf3d6bce8bd556a208e5f`.
**Base at session start:** `69ca910`. **Tree at session end:** clean.

---

## 0. WHAT THIS SESSION CLOSED, AND WHAT IT DID NOT

**CLOSED.** The S172 Run-4 Route-A routing patch is designed, reviewed, implemented, mutation-
tested, approved by Team Beta and committed. WATCHER can now route an explicit seven-key operator
warm-start pin to Step 1 through the real CLI path, and cannot route one from anywhere else.

**NOT CLOSED, and explicitly not claimed.** The patch certifies **routing and containment only**.
It does not make Run 4 correct, does not certify population equivalence, and resolves neither L-1
nor the coordinator ingress byte bound. Run 4 remains separately Michael-authorized. The
2026-08-22 mid-run freeze remains **UNDETERMINED**. See §7.

---

## 1. GOVERNANCE CYCLE

| stage | artifact | outcome |
|---|---|---|
| Design ruling (prior session) | `TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md` | architecture set: literal seven-key allowlist, no manifest defaults, all-seven-or-none, eighth key stripped |
| Brief review | `TB_RULING_RUN4_ROUTING_PATCH_BRIEF_REVIEW.md` | **architecture APPROVED**, four §3 corrections RATIFIED, **2 blockers**, revision required |
| Brief rev 1 → 2 | `S172_RUN4_ROUTING_PATCH_BRIEF_ROUTE_A.md` | four bounded edits incorporated |
| Patch review R1 | `TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R1.md` | **DO NOT COMMIT** `0398c0d1…`; **2 blockers** |
| Brief rev 2 → 3 | same file | R1 corrections + MOVE justification + bool-superset flag |
| Patch review R2 | `TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md` | **APPROVED FOR COMMIT** at `d14bcb3b…` |

**Three review rounds, inside the ≤3 target.** No round required a design re-ruling; every
correction was bounded and executable.

### 1.1 Brief-review blockers (both closed)

- **B1 — invocation-local pin ownership.** Authority belongs to one pipeline invocation, not to
  the `WatcherAgent`, module state, `retry_params` or daemon state. Implemented as a keyword-only
  parameter threaded from `run_pipeline`; proven by **G-INVOCATION-ISOLATION** (two `run_pipeline`
  invocations on one agent: 7/7 then 0/7) and mutant **M2a**.
- **B2 — `G-UNPINNED-IDENTICAL` oracle.** The pinned-executable design violated **EXEC-PIN-1**
  (pinned `run_step` would consume the post-patch `_step1_declared_params` and false-green). Fixed
  by the preferred route: a **pre-edit clean control captured from untouched `69ca910` before the
  first edit** — base commit, fixture inputs, exact 47-token argv, artifact hash, completion
  sentinel. Provenance verified independently: the recorded target digest equals
  `git show 69ca910:agents/watcher_agent.py | sha256sum`, so the capture demonstrably ran on an
  unmodified tree.

Also from that review: **G-EXACT** redefined as value **and type** equality (never `is`), and
**M2** split into **M2a** (lifetime authority) / **M2b** (default contamination).

### 1.2 Patch-review R1 blockers (both closed)

- **B1 — operator origin.** Capturing from generic `params` at `run_pipeline` entry proved a
  bundle *existed*, not *who supplied it* — and `chapter_13_triggers.py:616` is a live
  programmatic caller that passes `params`. Fixed with a dedicated keyword-only
  `_operator_pin_params` channel, default `None`, populated **only** by the real
  `--run-pipeline --params` seam. Fail-loud (`Step1PinAuthorityError`) when the seven appear in
  ordinary params. New gate **G-ORIGIN**, new mutant **M4**.
  - **Seam decision: MOVE, not duplicate — RATIFIED by Beta.** Both paths were traced before
    choosing. MOVE makes the fail-loud check unconditional (no authorized-invocation exemption,
    so no branch weakens it) and stops `_build_retry_params` (`:2234`, which copies
    `original_params` wholesale) from re-carrying the seven. DUPLICATE would trip fail-loud on the
    authorized run itself and needs an authority-keyed exemption, re-creating the coupling B1
    removes. Justification is written at source in `split_operator_pin_params.__doc__` and in
    brief §4.2.
- **B2 — presence vs usable value.** `params[k] is not None` had two holes: `''` counted as
  supplied (seven empty strings logged "pin accepted, 7 keys" while the builder routed zero), and
  an explicit `None` silently collapsed to "no pin" instead of failing as malformed. Fixed:
  `present = SEVEN ∩ params.keys()`; empty → no pin; otherwise all seven present **and** none
  carrying a value the builder treats as absent. New gate **G-VALUE-USABLE**, 30 negative cases.
  - **`True`/`False` are rejected as well — RATIFIED as not unauthorized expansion.** Derived from
    the builder's own semantics at `agents/watcher_agent.py:2009-2020`: `False` omits the flag
    (`:2013`); `True` emits a valueless numeric option (`:2011`), and since all seven are declared
    `type=int`/`type=float` at `window_optimizer.py:1514-1526`, argparse either consumes the next
    token or aborts the step — corrupted either way. Flagged as a superset of the ruling in brief
    §4.3 rather than slipped in.
- **Non-blocking, done:** bundle built over `sorted(STEP1_EXPLICIT_PIN_KEYS)` so `step1_pin_argv`
  is stable across runs. Visible in the dry-run argv (§5).

---

## 2. WHAT THE PATCH DOES

One file, `agents/watcher_agent.py`, **+374/−13**.

- **WALL 1** (`run_step`, the real first wall — `allowed_params = set(default_params.keys())`, and
  the manifest carries **zero** warm keys, which is why the pre-patch chain was dead): the seven
  join `final_params` **from the invocation-local bundle only**. `default_params` untouched.
- **WALL 2**: `_INTERNAL_ONLY_PARAMS` narrows to the seven **only** under an operator pin.
  `warm_start_session` — the eighth internal-only key, absent from `window_optimizer.py` — stays
  stripped unconditionally on every path.
- **Authority channel**: `_operator_pin_params` (keyword-only, default `None`) →
  `capture_step1_pin_bundle` → invocation-local frozen bundle → threaded to each Step-1
  `run_step`. Discarded on return. No instance attribute, no daemon state.
- **Provenance, pinned path only, absent (never `None`) unpinned**: `step1_pin_source` proves
  **authority**; `step1_pin_argv` records **what that authority requested**. Emitted to the
  `EXEC CMD` log record **and** stamped into the step's structured result at all five
  dispatch-outcome returns. No decision logic consumes either.
- **Declaration parity**: `_step1_declared_params` mirrors the seven so the two notions of
  "declared" cannot drift. Verified inert for fleet resolution by **G-PARITY-INERT**.
- **Still closed, as prohibited**: manifest `default_params`, optimizer/integration changes,
  direct threshold wiring, `search_strategy`, `seed_count`, search bounds, six-of-seven downstream
  repair, L-1, ingress byte bound.

---

## 3. ACCEPTANCE EVIDENCE

`tests/test_s172_run4_routing_patch.py` — **20/20 PASS** (14 gates, 6 mutants), grown from 12+5
at R1.

**Gates:** G-ALLOWLIST-EXACT · G-UNPINNED-IDENTICAL · G-CHAIN · G-EXACT · G-PARTIAL-CLOSED ·
G-EIGHTH · G-NO-SYNTH · G-NO-LLM · G-PROVENANCE · G-INVOCATION-ISOLATION · G-ONE-TRIAL ·
G-PARITY-INERT · **G-ORIGIN** · **G-VALUE-USABLE**

**Mutants — all APPLIED / EXECUTED / DETECTED:**

```
M1  -unconditional-strip          red [g_chain,g_exact]              green [g_unpinned_identical]
M1b -wall2-only-unconditional     red [g_chain,g_exact]              green [g_unpinned_identical,
                                                                            g_allowlist_exact,
                                                                            g_partial_closed,
                                                                            g_provenance]
M2a -lifetime-authority           red [g_invocation_isolation]       green [g_chain,g_unpinned_identical]
M2b -default-contamination        red [g_unpinned_identical,
                                       g_provenance]                 green [g_chain]
M3  -derived-allowlist            red [g_allowlist_exact]            green [g_chain,g_exact]
M4  -origin-from-ordinary-params  red [g_origin]                     green [g_chain,g_unpinned_identical,
                                                                            g_value_usable]
```

**Regressions:** Brief-I `26/26` · Brief-I mutants `DETECTED 14/15, INVALID 1, SURVIVED 0` ·
WATCHER LLM integration `13/13` · frozen `69ca910` unpinned control **list-equal at 47 tokens**.

**Three methodological points worth carrying forward:**

1. **M1b is the load-bearing mutant.** M1 empties the allowlist, a constant with *two* consumers,
   so its blast radius exceeds its label. M1b kills WALL 2's narrowing **in source** and nothing
   else — G-ALLOWLIST-EXACT and G-PARTIAL-CLOSED stay green, which is the evidence the reds are
   WALL-2 reds. **G-PROVENANCE also stays green under M1b**: WALL 1 still fires, so the pin is
   accepted and logged **while nothing routes**. Beta's words: nobody may treat the marker alone
   as proof the seven reached the optimizer. The source mutation is proven by a **three-point
   digest** (before / mutated / restored), because a matching before-and-after digest alone is
   equally consistent with the mutant never having been applied.
2. **Vacuity rule.** `_run_gate_under_mutant` credits only a gate's own terminal verdict; a gate
   that *raises* under a mutant is neither detection nor still-green and the mutant terminates
   `INCOMPLETE`. Introduced after finding the harness had been crediting any exception as
   detection — the defect the brief's own §7 forbids. **No detection in this session rests on an
   exception.**
3. **Short-circuit vacuity.** G-PROVENANCE short-circuits on its log arm, so the structured-result
   arm would never execute under M2b. Probed separately with the log arm bypassed: on a clean tree
   the unpinned result carries neither key; under M2b it carries both, with a 61-token
   contaminated argv on a dispatch supplying no warm keys. The arm reds on its own.

---

## 4. COMMITS

**`3e2537b` — S172 RUN-4 Route-A: explicit operator warm-start pin routing (TB R2 APPROVED).**
16 files, +4555/−13: the patch, the suite, the STEP-0 control + fixtures + capture script, brief
revision 3, both results records, and four TB rulings.

**`e75a6eb` — Housekeeping.** `CLAUDE.md` lead-handling contract; skill **v27.1** (§6 boot-selector
recovery rule); `LEADS.md`; the crash-forensic correction addendum; the netconsole runbook; the
Beta note; the Run-4 pre-launch check. **Closed the SR-3 three-copy drift** — worktree and
installed `~/.claude/skills/tfm-project-facts/SKILL.md` were already byte-identical at v27.1
(`6e8e715c…`); the **committed** copy was the stale one at v27. ser8 (copy 3) was **UNAVAILABLE**,
not verified — unreachable from VM101.

Both dual-pushed by Michael. **Claude committed nothing and launched nothing.**

---

## 5. POST-COMMIT RUNWAY (TB R2 ruling steps 1–4)

**Step 1 — committed digest ✅**
`git show HEAD:agents/watcher_agent.py | sha256sum` → `d14bcb3b3395877017e33e15fe00fb31b775945bdb1cf3d6bce8bd556a208e5f`, matching the approved value.

**Step 2 — clean-tree / Gate-22 ✅ without widening anything.**
The check is `gate22_coexistence()`, `tests/test_s172_phase4_coordinator.py:1621`, registered at
`:4220`. It reads **full `git status --porcelain`** (not tracked-only), slices `ln[3:]` so
modified *and* untracked both count, filters `.endswith(".py")`, and asserts
`changed_py <= allowed` at `:2417`. With the tree clean, `changed_py = set()` → passes. The R2
commit made every new `.py` tracked-and-clean, so **none of them needed an allowlist entry** —
confirmed by inspection. *(Mid-session the new pre-launch script itself reddened this gate as a
stray untracked `.py`; committing it in `e75a6eb` cleared it, which is the correct fix rather than
widening the allowlist.)*

**Step 3 — Run-4 infrastructure hardening ⚠ PARTIAL.** See §7.

**Step 4 — pre-launch provenance dry-run ✅ 10/10.**
`tests/run4_prelaunch_provenance_check.py` drives the **real** CLI seam — production argparse JSON
→ production `split_operator_pin_params` → production `run_pipeline(_operator_pin_params=…)` →
production `run_step` — intercepting only `_run_step_streaming`. No GPU, no fleet, no optimizer,
no step execution.

```
SEAM-MOVED · SEAM-AUTHORITY · THREADED · ARGV-BUILT (61 tokens) · GEOMETRY-COMPLETE
GEOMETRY-EXACT (window=12 offset=25 session_idx=1 fwd=0.71 rev=0.47 skip_min=6 skip_max=99)
EIGHTH-KEY-ABSENT · ONE-TRIAL · PROVENANCE-MARKER · PROVENANCE-ARGV
                                        10/10 PASS   RUN4_PRELAUNCH_PROVENANCE_OK
```

Built Step-1 command, verbatim (warm-start flags in sorted order — the audit-stability improvement
Beta accepted, visible in production output):

```
python3 window_optimizer.py --lottery-file daily3.json --strategy bayesian --max-seeds 2147483648
--prng-type java_lcg --output optimal_window_config.json --test-both-modes --trials 1
--trse-context trse_context.json --enable-pruning --n-parallel 1 --worker-pool-size 25
--seed-cap-nvidia 5000000 --seed-cap-amd 2000000 --seed-start 0 --pwc-transport tcp
--min-workers 24 --use-range-miner --miner-stripe-size 67108864 --miner-substripes 8
--staging-dir /home/michael/miner_staging --staging-workers 4 --staging-queue-depth 2
--staging-capacity-timeout 600.0 --staging-high-water-bytes 17179869184
--warm-start-fwd-thresh 0.71 --warm-start-offset 25 --warm-start-rev-thresh 0.47
--warm-start-session-idx 1 --warm-start-skip-max 99 --warm-start-skip-min 6 --warm-start-window 12
```

---

## 6. NETCONSOLE FINDING — L-3

Checking sender arm-state for step 3, the live probe returned **UNAVAILABLE**: all nine rig
endpoints (`.120/.154/.162` bare-metal, `.122/.156/.164` CT, `.121/.155/.163` hosts) answered
"No route to host". Only `.128` (VM101's own Proxmox host) responded, so VM101's networking was
fine — the rig fleet is powered off. **Nothing was rebooted or modified.** Reported as unobserved,
never as "not armed" (VIR-5).

The archived capture then contradicted a committed document:
`docs/RIG_CRASH_FORENSIC_20260822.md:24` records `netconsole = EMPTY`, but
`logs/netconsole_all_rigs.log` holds **11 packets dated 2026-08-22** from all three Proxmox hosts —
an `NC-TEST3`/`NCPROOF` arm test at 19:25-19:26 and `systemd-shutdown` lines at 19:52 (`.155`) and
20:33 (`.121`/`.163`).

**Operator explanation (Michael): all 11 are post-incident cleanup.** He always shuts the rigs
down after a crash and did so that evening. The `watchdog did not stop!` line belongs to an
orderly shutdown path, not a fault.

**What that settles, precisely.** The doc's disjunction was *"no event"* vs *"not active"*. The
`NCPROOF` packets **close the "not active" branch** — senders were armed and delivering on all
three hosts — leaving *"no event during the freeze itself"* as the correct and only reading. **It
does not explain the freeze.** An armed-but-silent netconsole shows no kernel message reached the
wire; that is not evidence of a healthy host.

**Residual question — resolved, and it was the second option.** Whether the frozen rigs accepted
the 20:33 shutdown directly, or whether these are a boot-check-then-shutdown cycle: kernel
monotonic uptime stamps give implied boot times **18:52:41 / 18:58:18 / 18:58:55**, all *after*
the run log's last write at **18:42:04**, and uptime runs continuous from the 19:25 arming to each
shutdown (wall-vs-uptime drift <0.07 s on all three, so no second reboot in that window). These
packets come from post-incident boot sessions. **The freeze is bounded to 18:42-18:52 and left no
netconsole trace** — a bound, not a hypothesis.

**Recorded as:** `LEADS.md` **L-3** (filed **OPEN**, not closed — the register's fixed vocabulary
is `OPEN · CLOSED-BY-EXPERIMENT · DEFERRED-BY-OPERATOR`, only a *failed follow-up experiment*
closes a lead, and two follow-ups are unrun); a dated **CORRECTION ADDENDUM appended** to the
forensic doc with the original 401 lines untouched; `RUNBOOK_NETCONSOLE_REARM.md`; and an
informational note for the next Beta package (no ruling requested).

---

## 7. CARRIED FORWARD

| item | status |
|---|---|
| **L-3 — 2026-08-22 mid-run freeze** | **OPEN. UNDETERMINED, bounded to 18:42-18:52.** No hypothesis offered. Needs the fleet up and the root-free fault surfaces; host kernel ring is UNAVAILABLE from inside unprivileged LXC. |
| **NCPROOF gate on power-on day** | Re-arm netconsole and confirm **three** `NCPROOF` lines, one per host, **before any run starts**. An unarmed sender during a future freeze recreates exactly the ambiguity L-3 just resolved. |
| **Runbook sender commands `[RECONSTRUCTED]`** | `RUNBOOK_NETCONSOLE_REARM.md` §2 is reconstructed from the observed `NCPROOF` packets — they prove the shape that worked, not the command that produced it. There is no install script or `modprobe` line anywhere in the tree. **Correct in place on first live execution** and drop the marker. Reboot persistence is unconfigured and not assumed. |
| **GPU power caps · off-host power telemetry** | **Unverifiable fleet-down.** The other two R2 step-3 items; not attempted. |
| **Physical circuit check** | **OWED.** Forensic doc H2 — *shared-circuit voltage sag under the synchronised 24-GPU phase-4 resume* — remains **OPEN, co-leading**, not demoted. Its stated follow-ups are per-rig instrumented supply logging across a phase boundary and **a circuit map for the three rigs**. Physical work; nothing in this session touched it. |
| **Run 4** | **Michael-authorized separately.** Not scheduled here. |
| Privileged seams `_operator_pin_params` / `_pin_bundle` | Standing constraint adopted by Beta; **no new production caller without governance review**. `BACKLOG` §22. Detectors: G-ORIGIN, M4. |
| Stale root `watcher_agent.py` | `BACKLOG` §23. 72 KB, 2026-04-24, zero R2 seams, `run_step(step, params)` at `:1547`. Wrong-file-edit hazard. **Deletion not proposed** — that is Beta's ruling to make. |
| Beta's recorded-but-unrepaired list | Six-of-seven optimizer check (WATCHER-MANIFEST-ROUTING-1) · `_ws_source` mislabel (**WATCHER provenance remains authoritative**) · L-1 float32/float64 seam · ingress byte bound · `step_runner`'s separate dispatch surface (**not** part of the certified Run-4 route). All unchanged; none blocked the commit. |
| Housekeeping note | Unusable-value rejection shares `Step1PinBundleError` with partial-pin failures. Acceptable while the durable error text stays specific; **do not** expand the change to rename classification fields. |

---

## 8. HARD-RULE COMPLIANCE

- **Never commit or push from an agent sandbox.** Michael committed `3e2537b` and `e75a6eb` and
  dual-pushed both. Claude prepared explicit `git add` lines and drafts only, and used `-a` never.
- **Never launch the pipeline autonomously.** No pipeline, no fleet run, no GPU work. The
  provenance check intercepts before dispatch by construction.
- **Fleet stays down.** Preflight, execution-set resolution and dispatch stubbed with the
  deterministic fixtures recorded in the STEP-0 artifact's `stub_boundary`. The only network
  activity was a read-only ping sweep and read-only SSH attempts for §6; nothing was rebooted or
  modified.
- **No hardcoding.** Every value from a live check or a committed source file; provenance key
  names exported as module constants so the gate binds to production names.
- **Read live code before patching. Verify before fixing. Fix forward.** Both R1 blockers were
  traced in live source before a line changed; no backup was restored.
- **Anchor every claim.** All `file:line` anchors, digests and artifacts recorded in the two
  results files.
- **Lead handling.** L-3 filed before any conclusion was written; not closed on an operator
  explanation.

---

## 9. ARTIFACT INDEX

```
agents/watcher_agent.py                              d14bcb3b…   +374/-13, certified
tests/test_s172_run4_routing_patch.py                            20/20, 14 gates + 6 mutants
tests/capture_run4_clean_control.py                              STEP-0 capture (Blocker 2)
tests/fixtures/run4_routing_clean_control.py                     FROZEN control fixture
tests/fixtures/run4_pin_harness.py                               pin + structured-result harness
tests/fixtures/run4_clean_control_69ca910.txt                    STEP-0 oracle, 47-token argv
tests/run4_prelaunch_provenance_check.py                         R2 step 4, 10/10
docs/S172_RUN4_ROUTING_PATCH_BRIEF_ROUTE_A.md                    brief revision 3
docs/S172_RUN4_ROUTING_SUITE_RESULTS.txt                         R1 record (rejected 0398c0d1…)
docs/S172_RUN4_ROUTING_SUITE_RESULTS_R2.txt                      R2/R2a record, 576 lines
docs/TB_RULING_RUN4_ROUTING_PATCH_BRIEF_REVIEW.md                brief review
docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R1.md                   R1, DO NOT COMMIT
docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md          R2, APPROVED
docs/LEADS.md                                                    L-3 (OPEN)
docs/RIG_CRASH_FORENSIC_20260822.md                              + CORRECTION ADDENDUM
docs/RUNBOOK_NETCONSOLE_REARM.md                                 stub, sender [RECONSTRUCTED]
docs/NOTE_FOR_BETA_20260828_NETCONSOLE_EVIDENCE.md               informational
docs/BACKLOG.md                                                  §22 privileged seams, §23 stale root
```

**Fallback parity:** `code=current @ e75a6eb`, `env=needs capture` — pytest 9.1.1 was installed
into `~/venvs/torch` this session and is **not** yet in a committed requirements artifact. Per the
box contract §4/§5 that dependency change belongs in the reproducible env artifact; recorded here
rather than hand-patched anywhere.

**END OF SESSION.**
