# SESSION CHANGELOG — 2026-08-14 — S182 — D6 INTEGRATION REPAIR (code-parity gate + launcher wait set)

**Host:** VM101 `zeus-ubuntu` (192.168.3.177), user `michael`, `~/venvs/torch`.
**Base:** HEAD `69ff2228bb19913183e08aaa735b85aa4a20516c` — unchanged at START and END.
Working tree at session start was **clean** (`git status --porcelain` empty).
**Brief:** `~/dashboard_work/CCODE_BRIEF_D6_INTEGRATION_REPAIR_v1_0.md` v1.0 — Team Beta RRR on the
D6 dry-run forensic, 2026-08-14: **narrow D6 integration repair AUTHORIZED**, fleet-deployment /
prelaunch integration layer only.
**Full report:** `~/dashboard_work/D6_INTEGRATION_REPAIR.md`.
**The attempt-6 coordinator/worker mechanics remain CERTIFIED and FROZEN and were not reopened.**

**NOTHING COMMITTED · NOTHING PUSHED · NOTHING DEPLOYED TO THE RIGS · D6 NOT RERUN ·
ATTEMPT 6 REMAINS HELD.**

---

## 1. A — `scripts/gate12_parity_gate.py` (NEW, 806 lines)

A fail-closed rig **source-parity** gate alongside the GPU truth gate, running **before worker
dispatch** in both the D6 dry-run procedure and real Gate-12 launches. The pre-launch battery proved
the cards; nothing proved the code, and on 2026-08-14 the D6 dry run dispatched into three rigs
carrying a `miner/range_miner_worker.py` last deployed 2026-08-02 — 24 of 25 workers died at
argparse.

- **Expected values are the full 64-hex SHA256 of the canonical clean local tree, derived at run
  time.** The forensic's 12-character display prefixes are not in the file as comparison values, and
  an AST arm enforces that no string literal outside a docstring carries a hex run ≥ 12 (one
  allowlist entry, by exact value, for the parser's validation alphabet).
- **`--verify-clean` (default on)** refuses if any governed file is dirty locally, so "canonical
  clean tree" is a checked property. An unrunnable `git` is UNAVAILABLE, never a silent clean.
- **Acceptance authority is content identity, never Git identity.** `local HEAD` prints once tagged
  `[CONTEXT ONLY]`; `evaluate()` provably does not reference it.
- **One ssh per rig** hashes every governed path. BEGIN/END sentinels make truncation detectable;
  each rig reports its own `hostname`, so three machines cannot be one machine answering thrice.
- **The governed set is PINNED (10 files)** and cross-checked against an AST-derived project-local
  import closure of `miner/range_miner_worker.py`. A worker-side project import added without parity
  coverage REFUSES until the pin is extended. Stdlib and site-packages are never hashed.
- **MISSING is a MISMATCH, not an UNAVAILABLE** — the probe ran and observed the absence.
- **§C evidence bundle** (`--evidence-json`): per rig per file, hostname · canonical path · expected
  full sha256 · observed full sha256 · size · MATCH|MISMATCH|UNAVAILABLE · collection timestamp,
  plus the governed set, the derived closure and the full expectation map.

### 1.1 A NEW MEASURED FINDING — the forensic's five-file closure was incomplete

`miner/__init__.py:19` imports the coordinator at **module scope**, so the worker's **statically
reachable project-local import / deployment closure** is **ten** project files, not five. Run
read-only against the live fleet, the gate finds two facts the forensic did not report:

| file | VM101 `69ff222` | all three rigs |
|---|---|---|
| `miner/range_miner_coordinator.py` | `5cf41f8332efa89d…` (563,487 B) | `d6cc26bfa09f2c00…` (255,156 B) — **MISMATCH, not previously measured** |
| `execution_set.py` | `d21614701c31a7b4…` | **ABSENT — not previously measured** |

Live verdict: **`18 MATCH · 12 MISMATCH · 0 UNAVAILABLE`, REFUSED.**

**Not claimed:** that either fact changed any past result. These are bytes measured **today**; the
§5.5 residual is untouched. The worker calls exactly one name out of the coordinator
(`DEFAULT_WORKER_ADMISSION_TIMEOUT`, `range_miner_worker.py:1265`) and Alpha did not audit whether
its value differs between vintages. `execution_set.py` is statically reachable but **not executed on
today's normal worker path** — it is reached only through deferred imports the worker never calls,
which is why the workers start without it. The gate governs it anyway.
**RULED AT R1:** Beta **REJECTED** a call-graph exception and **APPROVED** the ten-file static
closure; `execution_set.py` stays governed and **must be deployed** — see R1.1 below.

## 2. B — `scripts/launch_fleet_manual.sh` (wait-set correction)

```
launch_fleet_manual.sh completion
    MEANS:         all worker DISPATCH operations have been dispositioned
    DOES NOT MEAN: all launched worker PROCESSES have exited
```

The argument-less `wait` at `:228` waited for the local worker too — a `nohup … &` background job of
the same shell. With `RUN_NONCE` set that worker parks at the release barrier, and release is written
by a later step of the caller the launcher has not returned to. Measured 2026-08-14:
`SESSION_RELEASE_ABORTED waited_s=595.299`.

- Two PID sets: `REMOTE_DISPATCH_PIDS` (the wait set) and `LOCAL_WORKER_PIDS` (**excluded**).
- Each remote PID is waited **individually** so its status is read: a failing ssh is a dispatch that
  did not land and the launcher refuses. The old loop counted iterations.
- The local worker is asserted **ALIVE** before return — a dead local worker is the silent
  single-worker loss (its log already carries its sentinel, so the sentinel gate would pass on a dead
  process; cohort freezes at 25, admits 24).
- **Release was NOT moved earlier.** The launcher creates no token; sentinel verification still
  precedes release.

## 3. `gate12_launch.sh` — §0.6

The parity gate is invoked after the GPU gate and **before the clean slate**, so a refusal leaves the
box exactly as it found it. `${PIPESTATUS[0]}` is used, not the pipeline status. Its §C bundle goes
to `logs/gate12_${STAMP}_source_digests.json` (gitignored via `.gitignore:62`) and is echoed into
`$EVID`.

## 4. `tests/test_s172_d6_integration_repair.py` (NEW) — 65/65 green

```
COMPLETION SENTINEL: PASS — S172 D6 integration repair, the parity gate and the
launcher wait-set battery are green
```

Part A (32) parity gate: clean control + eight fault-injection arms (digest mismatch, missing file,
ssh 255, truncation, malformed output, wrong hostname, missing directory, unknown status), closure
derivation with positive and negative controls, expectation provenance, git-identity exclusion, the
canonical-tree precondition on a real throwaway git repo, and end-to-end `main()`.
Part B (26) launcher: RED authenticity first, three structural arms plus self-protection, **the
required four-part gate** (launcher returns · local worker alive and parked · no REGISTER · no
release token), the RED arm, dispatch and liveness disposition, release-not-moved, six frozen
surfaces.
Part D (4) mutants, each proven APPLIED, EXECUTED and DETECTED. Part C (3) harness integrity.

**RED arm pinning:** anchor `69ff222`, **full** sha256
`793c97ea5904315c92b56973b5a9ba321b72c530723459e008a9bcf32e39afc4`, defect surface (bare `wait` at
line 228) verified present before the arm is trusted; a drifted anchor reports **UNAVAILABLE**, never
a pass. The arm asserts only that the wait cycle is launch-blocking — **the predicted 900-second
endpoint is deliberately not asserted**, per Beta.

No real rig is contacted by the suite and no real fleet is launched: Part A replaces the ssh
transport with a real shim on PATH against fixture trees (the gate runs unmodified); Part B copies
the launcher's live bytes (digest equality asserted) beside stub config, a stub worker and a real
loopback listener.

## 5. Record-only, and the procedure

`~/dashboard_work/D6_DRYRUN_PROCEDURE.md`: the `pgrep -c … || echo 0` **double-zero construct is
removed** (STEP 7) — `pgrep -c` prints `0` *and* exits 1, so the `|| echo 0` printed a second zero,
the attempt-1 `0/8` pathology; `rc ≤ 1` is now a real count and anything else is **UNAVAILABLE**. New
**STEP 1.5** runs the parity gate before dispatch. **STEP 2's expectation corrected**: the launcher
now returns while the fleet stays parked.

## 6. Scope proof — nothing frozen moved

`~/dashboard_work/D6_INTEGRATION_REPAIR_SCOPE_PROOF.txt`, per-definition AST digests against `69ff222`:

```
miner/range_miner_coordinator.py: 248 -> 248   UNCHANGED 248 | CHANGED 0 | ADDED 0 | REMOVED 0
miner/range_miner_worker.py:       73 ->  73   UNCHANGED  73 | CHANGED 0 | ADDED 0 | REMOVED 0
§4.3 bounded-admission block (AST subtree): IDENTICAL 6326ebb4f31561a8
All 35 named no-touch definitions: IDENTICAL
All ten fleet-deployed (governed) files: whole-file byte-IDENTICAL to 69ff222
NO-TOUCH VERDICT: PASS
```

Deployment, when Beta authorizes it, therefore deploys `69ff222`'s bytes — not this session's.

## 7. Regression battery at final state

Run **sequentially** (concurrent S172 runs flake Part B's free-space arm). Logs:
`~/dashboard_work/d6_repair_logs/`. Full table in `D6_INTEGRATION_REPAIR.md` §6.

```
d6_integration_repair 65/65 · attempt6_remediation 78/78 · phase4_coordinator 62/63
f1_lease_origin 18/18 · f1_f2_active_lease 16/16 · defect_a_transport_recovery 29/29
admission_liveness 16/16 · resolved_execution_set 34/34 · elapsed_roundtrip 6/6
staging_backpressure 50/50 · staging_partb 24/24 · admission_binding 11/20 (PRE-EXISTING)
phase3_worker 17/17 · phase2_protocol 6/6 · phase1_scaffolding 6/6
gate12_gpu_gate 8/9 (PRE-EXISTING) · gate12_cleantree_admission 30/31 (PRE-EXISTING)
```

Every tally is **IDENTICAL** to the attempt-6 R2 certification run for the thirteen suites in that
table. `phase4_coordinator`'s one red is **Gate 22** on the two new untracked `.py` files — expected
during development, not a regression, and **not** a reason to widen Gate 22; it self-clears on a
clean committed tree.

### 7.1 Two pre-existing reds found, measured, and NOT fixed

`test_gate12_gpu_gate.py` (arm `P2-REFUSAL-PRECEDES-SAMPLER`) and
`test_gate12_cleantree_admission.py` (arm `W-ADMISSION-FIRST`) are red. **Proven not chargeable to
this repair** by the differential-worktree method: a pristine `git worktree` at `69ff222` produces
the **identical** arms and the **identical** 8/9 and 30/31 — the differential is empty.

Cause, reported as an observation: both arms assert a line-offset ordering inside `gate12_launch.sh`
in which the **fleet launch comes after the coordinator**, and the attempt-6 remediation (§8.4.3,
committed at `69ff222`) deliberately inverted that order — the fleet now starts first and parks at
the release barrier. Neither suite appears in the attempt-6 R2 regression table, which is how two
committed ordering gates came to encode a superseded launch order unnoticed. **Alpha did not repair
them**: correcting a Beta-authored ordering assertion is outside "a rig code-parity gate and the
launcher wait-set correction." Flagged for Beta's direction.

## 8. Files changed

```
 M gate12_launch.sh
 M scripts/launch_fleet_manual.sh
?? scripts/gate12_parity_gate.py
?? tests/test_s172_d6_integration_repair.py
?? docs/SESSION_CHANGELOG_20260814_S182.md
```

Outside the repository (deliverables, not tracked here):
`~/dashboard_work/D6_INTEGRATION_REPAIR.md`, `~/dashboard_work/D6_INTEGRATION_REPAIR_SCOPE_PROOF.txt`,
`~/dashboard_work/D6_DRYRUN_PROCEDURE.md` (modified), `~/dashboard_work/d6_repair_logs/`.

## 9. What was NOT done

No deployment to the rigs (every rig contact was `sha256sum`/`hostname` over ssh, read-only); no
commit, no push, no D6 rerun, no attempt-6 launch; no coordinator or worker change; no ruling on the
`execution_set.py` pin question; no assertion of the predicted 900-second endpoint; no claim that
§5.5 is closed. **D6 remains undischarged and attempt 6 remains HELD.**

---

# R1 — STALE ORDERING ASSERTIONS (TEST-ONLY), same session

**Brief:** `~/dashboard_work/CCODE_BRIEF_D6_R1_STALE_ORDERING_v1_0.md` v1.0 — Team Beta RRR on the
D6 integration repair, 2026-08-14: **core repair ACCEPTED; narrow TEST-ONLY R1 required before
commit/deployment.** Same day, same uncommitted package, so it is logged here rather than split into
a changelog nobody could join to §§1-9.

**NO PRODUCTION BEHAVIOUR CHANGED.** Nothing committed, pushed, deployed; D6 not rerun; attempt 6
still HELD.

## R1.1 Beta's acceptances, recorded so they are not revisited

The parity gate and the launcher wait-set repair both pass review. **The ten-file static deployment
closure is APPROVED and must NOT be narrowed** — Beta REJECTED a call-graph exception for
`execution_set.py`, because *"does this execution happen to reach this file?"* depends on arguments,
branch paths, future code, failure handling and deferred imports, and is far harder to prove than
static project-local reachability. **`execution_set.py` stays governed and must be deployed; so must
the canonical rig copy of `miner/range_miner_coordinator.py`.** This settles §1.1's open question in
the opposite direction from the narrowing Alpha offered, and the gate needed **no change** to comply.
Also accepted: the extra SSH-status check and the local-liveness assertion (approved, not struck),
the 65/65 battery, the pinned historical bare-`wait` RED, and the scope proof.

## R1.2 The two corrected arms — 9/9 and 31/31

Both were pre-existing at pristine `69ff222` (§7.1). Both expressed a **fan as a chain**, which
smuggled in `coordinator < fleet` — the *pre*-attempt-6 order — purely as a side effect of the
writing. Beta's diagnosis, now carried verbatim in both docstrings: *"the mistake here was allowing
older tests to claim more ordering territory than the property they actually governed."*

`test_gate12_gpu_gate.py` · `P2-REFUSAL-PRECEDES-SAMPLER` → **9/9**

```
OWNS      the GPU gate precedes, INDIVIDUALLY, each of parity gate ·
          clean-slate mutation · fleet dispatch · sampler creation ·
          coordinator creation; its refusal exits before the FIRST of them
NOT OWNED the relative order of fleet, sampler and coordinator
```

`test_gate12_cleantree_admission.py` · `W-ADMISSION-FIRST` → **31/31**

```
OWNS      admission < GPU gate < parity gate < clean slate < config rotation
                    < pre-dispatch clean-tree assertion;
          pre-dispatch assertion < each of fleet / sampler / coordinator;
          admission refusal exit < clean slate
NOT OWNED the relative order of fleet, sampler and coordinator (asserted as a
          FAN out of the pre-dispatch assertion, never compared to each other)
```

A needle that no longer matches is now RED rather than silently ordered by the not-found sentinel.
**Both arm names are unchanged** — `P2-REFUSAL-PRECEDES-SAMPLER` now names less than it owns, and the
name is left alone because R1 is scoped to the assertion and a rename would orphan two committed
documents that cite it; each docstring says so, and it is flagged for Beta.

## R1.3 The corrected arms are falsifiable — proven

Three mutations of `gate12_launch.sh`, applied inside a throwaway `git worktree` at `69ff222`
(nothing in the repository touched). Log:
`~/dashboard_work/d6_repair_logs/R1_ordering_arms_falsifiability.log`.

```
baseline  unmutated                                     both PASS
M-A       parity-gate invocation DELETED                both FAIL   (needle guard)
M-B       sampler created BEFORE the pre-dispatch wall  W FAIL, P2 PASS
M-C       PRE-attempt-6 order restored (coord < fleet)  both PASS   <- the point
```

**M-C is the one that matters:** under the stale shape it reddened both arms; under the corrected
shape both stay green, because neither owns that relationship any more. The correction removed a
false constraint without removing the true one.

## R1.4 Record-only wording correction

*"Executing closure"* → **"statically reachable project-local import / deployment closure"**, in
comments and docstrings only, in `scripts/gate12_parity_gate.py` (module docstring ×2, one comment,
plus a new paragraph recording Beta's ruling and naming `execution_set.py` as the live example of
*statically reachable but not executed on today's normal worker path*) and
`tests/test_s172_d6_integration_repair.py` (docstring, one comment).

**Proven to change no executable content** (scoped to these comment-and-docstring edits; the
separately authorized message string is R1.4a): the pre-R1 text was reconstructed by reversing each
edit and both versions reduced to `ast.unparse` with docstrings stripped —
`8e6dcff8a8916118…` **==** `8e6dcff8a8916118…`, bytes differing as expected.

**One instance was flagged rather than changed — and Beta then authorized it.**
`scripts/gate12_parity_gate.py:729` is a *printed refusal message*, not a comment or docstring, so it
fell outside Beta's comments/docstrings exception and Alpha raised it instead of acting on it. Beta
subsequently authorized it as terminology-only; **it has been applied** — see R1.4a.

## R1.4a The flagged refusal message — AUTHORIZED BY BETA AND APPLIED

Beta authorized the R1.4 flag as **terminology-only, output text, no behavioural change**, requiring
only a syntax check and the 65/65 D6 integration suite. Applied:

```
BEFORE   * {f} is imported into the worker's executing closure and is NOT governed
AFTER    * {f} is in the worker's statically reachable project-local import /
           deployment closure and is NOT governed
```

The grammar was adapted as Beta directed — *"is imported into … and"* → *"is in … and"*, because the
new noun phrase names a closure the file **is in**. Only that message changed; the refusal block's
header line above it does not carry the phrase and was outside the authorization.

**Scope, measured:** per-definition AST digests 18 → 18, **exactly one definition moved (`main`)**,
zero added, zero removed. The module's docstring-stripped AST digest *does* move here
(`8e6dcff8a8916118…` → `936104cc6d99542f…`) — the correct signature of an output-text change inside
executable code rather than in a comment, and why it needed its own authorization rather than riding
along with R1.4. `ast.parse` clean; `test_s172_d6_integration_repair.py` **65/65**; porcelain
unchanged.

## R1.5 Recorded for the D6 rerun (not run)

`~/dashboard_work/D6_DRYRUN_PROCEDURE.md` STEP 4 now carries Beta's evidence requirement — **a
sentinel record cannot substitute for process liveness; a process can emit its sentinel and
subsequently die.** Before cleanup or release the parked-fleet run must prove, together:
`remote 24/24 alive+parked · local 1/1 alive+parked · sentinels 25/25 current nonce · REGISTER 0 ·
release token 0`.

## R1.6 Return evidence

```
test_gate12_gpu_gate.py              9/9
test_gate12_cleantree_admission.py  31/31
test_s172_d6_integration_repair.py  65/65
test_s172_attempt6_remediation.py   78/78

production diff since the last submission: NONE
scope proof: coordinator 248 -> 248 / 0 changed · worker 73 -> 73 / 0 changed
             ten governed files byte-identical to 69ff222 · NO-TOUCH VERDICT: PASS
```

R1 file surface — all four are test or documentation:

```
 M tests/test_gate12_gpu_gate.py               corrected arm
 M tests/test_gate12_cleantree_admission.py    corrected arm
?? tests/test_s172_d6_integration_repair.py    wording only
?? scripts/gate12_parity_gate.py               wording, plus the one Beta-authorized
                                               refusal-message string (R1.4a)
```

`git status --porcelain` at R1 START was the five entries of §8; at R1 END it is those five plus
`M tests/test_gate12_gpu_gate.py` and `M tests/test_gate12_cleantree_admission.py`.

**Gate 22** stays at its known development-state RED while the new `.py` files are untracked; its
allowlist was **not** widened. It self-clears once Michael commits the certified package.
