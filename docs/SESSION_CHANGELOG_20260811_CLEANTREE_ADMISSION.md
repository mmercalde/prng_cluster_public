# SESSION CHANGELOG — 2026-08-11 — Gate-12 clean-tree admission repair

**Brief:** `~/dashboard_work/CCODE_BRIEF_CLEANTREE_ADMISSION_REPAIR_v1_1.md` (v1.1, Beta-approved
2026-08-11, with the BINDING SEQUENCING AMENDMENT).
**R1 brief:** `~/dashboard_work/CCODE_BRIEF_R1_CLEANTREE_SUITE_PROVENANCE_v1_0.md` — Beta
RETURN FOR NARROW R1, same day: **architecture ACCEPTED, certification HELD** on two suite
provenance defects. Both were in the test file; see "R1 correction" below. Suite **25/25 → 31/31**.
**Report:** `~/dashboard_work/CLEANTREE_ADMISSION_REPAIR.md` — full design, evidence and anchors
(amended for R1, not rewritten).
**Host:** VM101, `~/distributed_prng_analysis`, HEAD `3254a306…`, branch `main`.
**Nothing committed, nothing pushed, attempt 4 NOT launched, frozen attempt-3 bundle NOT modified**
(`sha256sum -c` → 13/13 OK after all work).

---

## What was wrong

Gate-12 attempt 3 (`distributed_config_t1_d606edbe`, 2026-08-10) completed four stages, 128/128
stripes over the full `[0, 2^31)` domain and satisfied the saturation verdict, then was refused at
publication:

```
utils.run_finalizer.RunParameterError: repository_tree_clean is False … commit 3254a306…
```

Two defects, both in the launch harness, neither in D3.5:

1. **Admission was too permissive.** `gate12_launch.sh:54` ran
   `echo "--- TREE STATE ---"; git status --porcelain` inside the evidence block — it **printed**
   the tree state and never **tested** it. Verified from the frozen bundle: the launch-time
   `--- TREE STATE ---` output at 17:28:31 is byte-identical to the failure-time porcelain. The
   harness printed the reason the run was going to fail, two hours before it failed, and dispatched
   the fleet anyway.
2. **The harness dirtied the tree by its own hand, after admission and before dispatch.** The clean
   slate renamed `optimal_window_config.json` (ignored, `.gitignore:115`) to
   `optimal_window_config.json.pregate12_${STAMP}`, a name **no** ignore rule matches — measured
   with `git check-ignore -v`, not assumed. Testing the predicate once at admission would therefore
   not have been sufficient.

**New fact, not in the brief:** the rename did **not** fire in attempt 3 (no
`optimal_window_config.json` existed at launch — it had been rotated away on 2026-08-07), but
attempt 3 *wrote* one at 19:48, so the `[ -f ]` guard is satisfied now. The pre-repair harness would
have manufactured the residue on the **next** launch. Attempt 3 escaped Defect 2 by accident.

## What changed

| file | status | change |
|---|---|---|
| `scripts/gate12_cleantree_gate.py` | **NEW** | the gate |
| `gate12_launch.sh` | **MODIFIED** +100/−3 | §0 evidence note, §0.4 admission gate, rotation destination, §1.9 pre-dispatch assertion |
| `tests/test_gate12_cleantree_admission.py` | **NEW** | C1–C5, C5A, two RED arms, eight mutants; **R1-corrected** (pinned anchor + derived fixture) |
| `docs/SESSION_CHANGELOG_20260811_CLEANTREE_ADMISSION.md` | **NEW** | this file |

**ONE PREDICATE, SHARED NOT COPIED.** The gate imports
`window_optimizer_integration_final._repository_state` — the function whose second return value
becomes the finalizer's `repository_tree_clean` argument (`…final.py:2972` → `:2992`) — as an
object. A second `git status --porcelain` implementation would be a second predicate, and a second
predicate that can disagree with the first is this defect recurring in a new costume.
`decide(clean)` is unary by construction, so the human-readable entry listing cannot influence the
verdict; `W-ONE-DECISION-PATH` proves by AST that the listing's only caller is `render_refusal`,
which runs after the verdict.

Sequence now: `§0.4 admission (clean) → clean slate → rotation into logs/ → §1.9 pre-dispatch
(still clean) → sampler → coordinator → fleet → D3.5`. Measured ordering
`114 < 145 < 154 < 174 < 192 < 211 < 224 < 248`, refusal `exit 1` at 118.

The rotation destination is now `logs/gate12_${STAMP}_pregate12_optimal_window_config.json`.
`logs/` is ignored as a **whole directory** (`.gitignore:62`), so the rollback copy survives beside
the run's other artifacts and **no filename exception was added anywhere** — the prohibition is
satisfied by construction, not by restraint.

Both gate invocations read `${PIPESTATUS[0]}`, never `$?`: `cmd | tee` exits with tee's status,
which is 0 essentially always, and `if ! cmd | tee` would print REFUSED and launch anyway. Same
self-caught class as the GPU gate's fix in `4643a11`; `M2` executes the rejected form and shows it
launching.

## Evidence

`tests/test_gate12_cleantree_admission.py` — **31/31 green, exit 0** (25/25 before the R1
correction added six provenance gates). No rig, GPU, coordinator or
fleet is contacted: C1 runs the **real, unmodified** `gate12_launch.sh` inside a sandboxed `HOME`
behind recording shims for `pkill`/`ssh`/`setsid`/`nohup`/`ss`, and refuses to proceed unless
`command -v` confirms the shims won the PATH race.

- **C1** attempt-3 fixture → rc 1, all three entries named, refusal cites `run_finalizer.py:1589`;
  the shim witness file is **empty** — zero dispatch, zero GPU work.
- **C1-RED** the **pinned** pre-repair source on the identical fixture **printed all three entries
  and still reached the clean slate** (8 shim calls). The gate would have caught attempt 3.
- **C2** clean → PASS. **C3** modified tracked production file → refuse. **C4** modified tracked
  test/governance file → refuse.
- **C5** for five fixtures, the launch verdict is fed to the real `finalize_run` (reaching
  `run_finalizer.py:1589`): `clean=ADMIT/ACCEPT; attempt3=REFUSE/REJECT;
  modified-production=REFUSE/REJECT; modified-governance=REFUSE/REJECT;
  ignored-config-present=ADMIT/ACCEPT`. **No ADMIT/REJECT row.**
- **C5A** real extracted rotation region, clean repo + pre-existing ignored config:
  `initial=CLEAN, post-preparation=CLEAN, pre-dispatch rc=0, coordinator dispatched=NO`.
  **C5A-RED** the committed rename executed verbatim: **CLEAN → DIRTY**, caught by §1.9 by name.
- **VIR-5** a non-git root makes the producer raise → reported `UNAVAILABLE` and refused, never
  rendered as clean.
- **Mutants** M1 verdict inverted (applied, executed, rc 0 on the attempt-3 state; unmutated gate
  rc 1 on the same fixture) · M2 decorative `if ! … | tee` · M3 rotation destination reverted
  (CLEAN → DIRTY) · M4 extraction non-vacuity.

**Regression battery** (sequential, venv active): D3.5 finalizer **60/60** · gate-12 GPU gate
**9/9** · concurrency sampler **49/49** · phase-4 coordinator **62/63**. The single phase-4 failure
is Gate 22's standing untracked-`.py` sensitivity naming exactly the two new files — expected, not a
regression, **allowlist NOT widened**, self-clears on the clean committed tree. Fifth occurrence.

A real `git worktree add --detach HEAD` checkout returned `PASS` at rc 0 — a clean-control on this
repository's actual HEAD, not a fixture.

`W-NO-WEAKENING` asserts sha256 identity to HEAD for `utils/run_finalizer.py`,
`window_optimizer_integration_final.py` and `.gitignore`, and that no `*.stale_*` / `*.db-shm` /
`*.db-wal` / `*.pregate12*` exception exists. **D3.5 was not weakened in any respect.**

## R1 correction — adversarial-input provenance (Beta return, same day)

Beta accepted the architecture and returned two defects, **both in
`tests/test_gate12_cleantree_admission.py`**. `gate12_launch.sh` and
`scripts/gate12_cleantree_gate.py` needed no change and got none — the launch script's diff is
still `100 insertions(+), 3 deletions(-)`.

**R1 — the RED arms read `HEAD`, so they die on commit.** `git show HEAD:gate12_launch.sh` is the
pre-repair script only while HEAD is `3254a306…`. Now pinned to `PRE_REPAIR_COMMIT` and
integrity-checked: `_launch_source_at()` refuses to return an object that does not resolve **or**
that no longer carries **both** old defect surfaces — the untested `--- TREE STATE ---` porcelain
print and the `mv … optimal_window_config.json.pregate12_` rename. Every consumer reports
**UNAVAILABLE** (VIR-3, recorded as not-green) instead of passing.

**The probes run over executable lines only, and that is load-bearing.** The repaired script quotes
both surfaces verbatim in its own header comments explaining what it fixed, so a raw-text probe
would match the repaired script and be blind to exactly the drift it exists to detect. `M6` measures
**2/2 surfaces missing** in the repaired script.

**R2 — C1 claimed frozen-bundle derivation and hard-coded the fixture.** `ATTEMPT3_ENTRIES` as
three string literals is gone. `attempt3_entries()` reads the bundle's `git_status_porcelain.txt`,
**verifies it against the digest the bundle's own `SHA256SUMS.txt` records for that path**, parses
the porcelain, rejects renames / absolute paths / `..`, and asserts cardinality 3 and status `??`.
No fallback to constants. `M7` drives four tampers against **copies** — absent, truncated,
digest-mismatch, not-untracked — all REFUSED. The bundle is opened read-only, two files, and `M8`
re-verifies it **inside the suite**: **13/13 OK**.

**R3 — demonstrated against an actual post-repair HEAD, by simulation.** A `--no-hardlinks` clone
in scratch (separate object DB — nothing written to the repo's `.git`) received the three files and
a commit, giving HEAD `3938f7a4…` with an empty porcelain. The corrected suite run from that
checkout: **31/31 green**, `R1-ANCHOR-INTEGRITY` reporting *"HEAD has moved… the pin is what keeps
the RED arms alive"*, both RED arms still crediting. Acceptance met: **green before and after the
repair commit.** Main repo HEAD unmoved at `3254a306…`.

**Counter-demonstration, measured.** A mutant restoring the original `HEAD`-relative helper, run in
the same post-repair clone: **27/31**, failing `C1-RED-OLD-ADMITS` (witness `0 shim calls` — the
repaired script correctly refused, so the arm had nothing to observe), `C1-RED-NO-TREE-GUARD`,
`C5A-RED-OLD-DIRTIES` and `M6`.

**One refinement to Beta's wording, as measurement not disagreement:** the arms would have gone
**loudly RED**, not silently vacuous. The hazard stands and is arguably sharper — a permanent
adversarial suite that reds on a *legitimate* commit invites the next reader to delete or weaken
its RED arms to restore green, and that is how it goes vacuous. The pin removes the pressure: after
the commit the suite is still 31/31.

New gates: `R1-ANCHOR-INTEGRITY`, `R2-FIXTURE-DERIVED`, `M5-ANCHOR-MISSING-KILLED`,
`M6-ANCHOR-DRIFT-KILLED`, `M7-EVIDENCE-TAMPER-KILLED`, `M8-BUNDLE-INTACT`.

**Re-run vs unchanged, stated plainly per Beta:** only the changed suite was re-run (31/31), plus
the new R3 clone run (31/31). D3.5 60/60, GPU gate 9/9 and sampler 49/49 are **not re-credited** —
unchanged inputs. Phase-4 was not re-run because Gate 22's input is the *set* of changed/untracked
`.py` paths, which is the same two files; editing an already-flagged file cannot change that set.

**Alpha observation 1 — premise MEASURED FALSE, recorded.** Importing
`window_optimizer_integration_final` does **not** monkeypatch `MultiGPUCoordinator.optimize_window`
and does **not** print `✅ Window optimizer integrated into MultiGPUCoordinator`. Both statements
are the last two lines of the **function** `add_window_optimizer_to_coordinator()`
(`window_optimizer_integration_final.py:1937`, the module's final top-level `def`; the file is
3,135 lines, so `:3134-3135` sit inside its body) and run only when it is **called**, which the
gate never does. `python3 -c "import window_optimizer_integration_final"` writes **0 bytes to
stdout and 0 to stderr**, rc 0, and the gate's output carries no such line on either stream. No
emoji reaches `$EVID`; there is nothing to suppress. The residual half is correct and is recorded:
the module-scope import means an import-chain failure kills the gate by traceback rather than by
its designed refusal — fail-closed still holds via non-zero exit and `${PIPESTATUS[0]}`, but by
exit status rather than by design. Left as-is; reimplementing the producer is prohibited.

**Alpha observation 2 — accepted.** §9 step 3 of the report now copies the three residue files to a
**sibling** directory with a digest manifest **before** deletion — never into
`/home/michael/gate12_attempt3_20260810_200824/`, which is hash-verified. `optimal_window_config.json.stale_1786149572`
is the only surviving copy of the 2026-05-11 Step-1 config.

## Generated-config provenance (required investigation)

- **`optimal_window_config.json`** — Step 1's primary deliverable (`instructions.txt:4368-4371`,
  read live). Ignored at `.gitignore:115`, which is an **explicit re-ignore** overriding the
  `!*_config.json` negation at `:43`. Correctly outside certified Git-visible state; unchanged.
- **`optimal_window_config.json.pregate12_*`** — produced only by `gate12_launch.sh`. **Zero
  instances have ever existed**; the guard never fired. Destination now inside `logs/`.
- **`optimal_window_config.json.stale_*`** — **no in-repo producer exists and none ever has.**
  `git log --all -S'.stale_' -- '*.py' '*.sh'` returns nothing (deleted paths included); a live
  full-text sweep of the working tree, untracked and gitignored files included, finds `.stale_` only
  in `docs/*.md` prose and `.gitignore:120`. It is a **hand rotation**: file mtime and content both
  `2026-05-11 19:24:23 −0700`, while the suffix `1786149572` decodes to **2026-08-07 17:39:32 −0700**
  — the suffix is the rotation timestamp; `mv` preserved the content mtime.
  **Correction:** `docs/CLAUDE_CODE_REPORT_S172_STAGING_CAPACITY_AMENDMENT.md:23` calls it *"a
  2026-05-11 runtime rotation"*; 2026-05-11 is the **content** date, the rotation was 2026-08-07.
  Sibling `zmq_job_queue.db.stale_23846e2c` (2026-03-31) is the same shape, and `.gitignore:120`
  `zmq_job_queue.db.stale_*` was added for it by `ba833a4` — an ignore rule written for one hand
  rotation, the class never generalised. That is why the identical action four months later was
  Git-visible and killed attempt 3.

**No residue was deleted.** The three untracked entries are pre-existing operator residue, not
harness output, so disposing of them is outside this repair. Adding `.gitignore` exceptions for them
is explicitly prohibited by the brief. Measured and reassuring: the root `miner_ledger.db` is the
**stale** ledger (live one is `/home/michael/miner_staging/miner_ledger.db`), and its `-shm`/`-wal`
mtimes (2026-08-10 10:40 / 2026-08-09 07:48) predate attempt 3's 17:28 launch and were unchanged by
it — nothing in the run path opens the root ledger, so removing them will not resurrect them
mid-run.

## Findings reported, not acted on

- **WATCHER scored the failed attempt-3 run as PASS** — `Parse Method: file_exists`,
  `Confidence 1.00`, `Action PROCEED`, after the `RunParameterError`. Out of scope per the brief,
  but it **does** share an authority seam with this repair: WATCHER's Step-1 verdict is
  `file_exists` over `optimal_window_config.json`, the same generated config traced above, written
  by the optimizer *before* D3.5 runs, so its presence cannot distinguish "published" from "refused
  at publication". `--end-step 1` held (log shows the benign `Triggering Step 2`, not
  `STEP 2: Scorer Meta-Optimizer (run #N)`). Nothing was changed in WATCHER.
- **Run-created residue is not covered.** The invariant is admission → preparation → dispatch.
  Attempt 3 gives positive evidence the run creates nothing Git-visible (launch-time and
  failure-time porcelain identical), but that is an observation, not a gate. Closing it would need a
  third evaluation before `_finalize_run_d3_5`, i.e. a production change — outside this
  authorisation.

## Next

Michael reviews → Beta certifies the narrow amendment → **preserve then** remove the three residue
entries (copy to a sibling directory with digests first — never into the hash-verified bundle) → stage
explicitly (never `git add -a`) → commit → dual-push → phase-4 self-clears to 63/63 →
porcelain-empty proof → existing prelaunch battery → only then Gate-12 attempt 4. The seven-part
completion authority of skill §2.33 is unchanged by this repair.
