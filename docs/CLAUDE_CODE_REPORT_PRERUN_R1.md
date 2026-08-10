# CLAUDE CODE REPORT — PRE-RERUN ITEMS, REVISION 1 (SAMPLER + ANCHOR)

**Host:** VM101 (`192.168.3.177`), repo `~/distributed_prng_analysis`, HEAD **`c4e0037`**, venv
`~/venvs/torch`. **Authority:** Team Beta ruling *"PRE-RERUN ITEMS REVIEW"* (2026-08-09), as
transcribed in `docs/CLAUDE_CODE_INSTRUCTIONS_PRERUN_R1.md`.

**Constraints honoured:** no commit, no push, no launch, no fleet, no port 5700 bind, Gate 12 HELD.
The coordinator, miner, ledger, seed-domain/coverage surface and every certified suite are
untouched. The GPU probe repair in `preflight_check.py` was not modified, not re-verified and not
reached by sampler work.

## BASE VERIFICATION (before any edit)

```
git log --oneline -1                     c4e0037
tests/test_preflight_gpu_probe.py        12/12 checks green
tests/test_gate12_concurrency_sampler.py 14/14 checks green
```

## FINAL STATE (after the last change — this is the run the evidence below describes)

```
tests/test_preflight_gpu_probe.py        12/12 checks green   (unchanged in substance)
tests/test_gate12_concurrency_sampler.py 29/29 checks green   (14 retained + 15 new)
port 5700                                UNBOUND
/home/michael/miner_staging/miner_ledger.db   mtime 2026-08-09 12:47:17.388348534  (unchanged)
./miner_ledger.db                             mtime 2026-08-04 18:09:56.049862973  (unchanged)
repo HEAD                                c4e0037 · reflog head unchanged · 0 stashes
```

---

## 1. THE SNAPSHOT MECHANISM, AND WHY IT IS SUFFICIENT UNDER WAL

**Mechanism:** an explicit `BEGIN DEFERRED` … `COMMIT` around both reads of one sample, expressed
as a context manager `read_snapshot(conn)` and applied in `sample_run`. The read-only connection is
additionally set to `isolation_level = None`.

```python
@contextlib.contextmanager
def read_snapshot(conn):
    conn.execute("BEGIN DEFERRED")
    try:
        yield conn
    finally:
        conn.execute("COMMIT")
```

**Why sufficient under WAL.** `BEGIN DEFERRED` opens no transaction immediately; the first read
statement takes a snapshot of the WAL as of that moment — the reader records the wal-index end mark
— and every subsequent read *in the same transaction* is served from that same mark regardless of
what any writer commits meanwhile. WAL readers never block writers and are never invalidated by
them, so the coordinator continues at full speed and the sample is still one instant. Two
autocommit reads are two independent read transactions and take two marks; that is the defect.

**Three points that make it the smallest *provably* sufficient change rather than the smallest
change:**

- **`DEFERRED` specifically.** `BEGIN IMMEDIATE` would attempt a write lock, which a `mode=ro`
  connection cannot take. `DEFERRED` is the only form available here, and it is the correct one.
- **`isolation_level = None` is load-bearing, not tidying.** Under the default (`""`), the sqlite3
  module emits implicit `BEGIN`s ahead of DML only — SELECTs run in autocommit — so an explicit
  `BEGIN` races the module's own transaction bookkeeping. `None` means the module never emits
  `BEGIN`/`COMMIT` of its own and ours is the only transaction there is.
- **The `COMMIT` is as important as the `BEGIN`.** A read transaction that is opened and never
  released would serve *every later sample* from the first snapshot the sampler ever took — the
  evidence file would freeze at the first instant and still look plausible. Gate **A3** exists for
  precisely that, and its mutant is a `read_snapshot` that BEGINs without COMMITting.

**Scope note (stated rather than assumed):** `discover_run_id` is the latch, not part of a sample,
and is deliberately left outside the snapshot. Atomicity is a property required *of one sample*;
latching a run one microsecond before sampling it changes nothing about whether that sample
describes one instant.

## 2. THE ESTAB UNAVAILABLE PATH, AND PROOF THE VERDICT IS UNAFFECTED

`estab_count` is replaced by `estab_observation(port)`, returning
`{"estab": Optional[int], "estab_status": "OK"|"UNAVAILABLE", "estab_reason": str|None}`.

| condition | before | now |
|---|---|---|
| `ss` not on PATH | `-1` (a number, rendered into the TSV as one) | `None` / UNAVAILABLE / `ss_not_found` |
| `ss` exits non-zero | **`0`** — the defect Beta names | `None` / UNAVAILABLE / `ss_exit_N:<stderr>` |
| `ss` times out | `-1` | `None` / UNAVAILABLE / `timeout` |
| output unparseable | a fabricated line count | `None` / UNAVAILABLE / `unparseable_ss_output:<line>` |
| `ss` succeeds, nothing established | `0` | **`0` / OK** — unchanged, and gated as the converse |

**Precision about the defect, since it matters for what was actually broken.** The literal `0`
arose on the **non-zero-exit** path: `subprocess.run` without `check=True` does not raise, so a
failed `ss` with empty stdout became `len([]) == 0`. The missing-binary path returned `-1`, which is
not `0` but is still a *number written into the evidence file* — an analyst reading the TSV sees a
count either way. Both are now UNAVAILABLE. Beta's characterisation is correct in substance; this
paragraph records the two distinct code paths behind it.

Rendering: `render_estab()` emits the literal `UNAVAILABLE` in the TSV, never `0`, `-1` or `None`,
and a new `estab_reason` column carries why. The summary gains an ESTAB block that reports
max/min over *observed* samples and counts unavailable samples with their reasons, under a heading
stating ESTAB is not a term in either criterion.

**Proof the verdict is unaffected — structural, then behavioural.**

- *Structural:* `evaluate()` has no ESTAB input and cannot acquire one. ESTAB aggregation lives in a
  separate `summarize_estab()` consumed only by the renderer. Gate **G3** (retained, unmodified)
  asserts `"estab" not in v` on the verdict object.
- *Behavioural:* gate **B7** runs the same fixture three times, identical but for ESTAB — absent,
  `0`, and an absurd `400` — and asserts the verdict objects compare **equal** and both rendered
  verdict lines are identical across all three. Its mutant (an `evaluate` that lets ESTAB into the
  occupancy term) reds it.

## 3. THE TWO SEPARATE VERDICTS, WITH THE EXACT PREDICATE FOR EACH

Let `W` be the qualifying window: the longest run of consecutive samples each satisfying
`compute_active ≥ threshold ∧ queued_pending ≥ 1`, and `|W| ≥ min_window_samples`.

**VERDICT 1 — SUSTAINED SIMULTANEITY** *(existing, unchanged)*

```
satisfied  ⟺  ∃W
```

**VERDICT 2 — TURNOVER UNDER FULL OCCUPANCY** *(new)*

```
pending_delta  = W[0].queued_pending − W[-1].queued_pending
transitions    = Σ over consecutive (a,b) in W of  max(0, (b.done + b.staging) − (a.done + a.staging))
done_delta     = W[-1].done − W[0].done

turnover_satisfied  ⟺  ∃W  ∧  ( pending_delta > 0  ∨  transitions > 0 )
```

Both terms are reported numerically in the summary — `pending_delta`, `transitions`, `done_delta`,
and pending at window start/end — satisfied or not.

Three properties worth stating explicitly:

- **"while occupancy remained at the threshold" is satisfied by construction**, not by a separate
  test: every sample in `W` is at or above the threshold by the definition of `W`, so anything
  counted inside `W` necessarily happened while the fleet was full.
- **`transitions` clamps per step and sums `done + staging`**, so a stripe moving `staging → done`
  is not counted a second time. It measures stripes *leaving* the claimed/pending pool.
- **The `∨` is load-bearing, not defensive.** Across a stage boundary the backlog is replenished, so
  `pending` can be flat while stripes are genuinely being consumed. Gate **C4** is that case, and
  its mutant is a `_turnover` that keeps only the drain term.

**They are not collapsed.** The summary prints two labelled lines; there is no combined boolean
anywhere in the rendered output. The exit status keeps them apart too:

```
0 = both satisfied · 2 = simultaneity failed · 3 = simultaneity satisfied, turnover failed
```

Gate **C5** asserts each label appears exactly once, that the two rendered values can differ, and
that the three exit codes are distinct; its mutant is a `_turnover` that mirrors verdict 1.

**End-to-end confirmation** against a synthetic 25-claimed / 7-pending ledger held static — the
exact shape Beta names as the trap:

```
VERDICT 1 — SUSTAINED SIMULTANEITY        : SATISFIED
VERDICT 2 — TURNOVER UNDER FULL OCCUPANCY : NOT SATISFIED
pending at window start / end             : 7 -> 7
  pending delta (positive == drained)     : 0
stripes transitioned into done/staging    : 0
EXIT=3
```

## 4. THE RE-ANCHORED MUTATION CHECK — AND A CORRECTION TO §6's PREMISE

**The change, and only this change:** `test_preflight_gpu_probe.py` now reads the pre-fix probe out
of a pinned commit instead of `HEAD`, matching the convention the repo already uses
(`tests/test_s172_staging_backpressure.py:1550-1560`):

```python
_PRE_FIX_REV = "c4e003743893f489b85310aa8a2d36505185a2ec"  # probe repair's parent
...
["git", "-C", str(REPO), "show", f"{_PRE_FIX_REV}:preflight_check.py"]
```

**How post-commit correctness was demonstrated without committing.** A throwaway **`--depth 1`
clone** of the repo was made into the session scratchpad (`git clone --depth 1 file:///…`, 9.6 MB —
depth 1 suffices because `c4e0037` *is* HEAD, so it is the clone's only commit). The two modified
files were copied in and committed **inside the clone**, making its HEAD the mutated source — the
exact post-commit condition. The suite was then run there:

```
clone HEAD after the throwaway commit : 5bea59d "THROWAWAY: simulate the probe repair landing"
tests/test_preflight_gpu_probe.py     : 12/12 checks green
M1A-MUTANT-AUTHENTIC                  : `|| echo 0` construct located in c4e0037:preflight_check.py
```

The clone was deleted afterwards. **The repository was not touched:** HEAD is still `c4e0037`, the
reflog head is still the original commit, there are no stashes and no new refs.

### ⚠ DISAGREEMENT — REPORTED, NOT WORKED AROUND

**Beta §6 states that once committed, "`HEAD` becomes the mutated source and the check inverts."
Measured in the post-commit clone, it does not invert. It goes VACUOUS — which is worse.**

Restoring the `HEAD` anchor inside the post-commit clone and re-running gives **12/12, M1A still
green.** The reason is that the certified repair *documents the construct it replaced*:
`preflight_check.py:62` contains, in a comment, the legacy shell string **verbatim**. So
`LEGACY_SHELL in git show HEAD:preflight_check.py` keeps matching after the commit — against the
repair's own commentary rather than against any executable probe. Measured:

| revision | string present anywhere | present outside comments |
|---|---|---|
| post-commit HEAD (the repaired file) | **True** | **False** ← no executable pre-fix probe |
| `c4e0037` (the pinned baseline) | True | **True** |

Consequences, stated plainly:

- **The re-anchoring is still required and is still correct** — arguably more so. A gate that
  silently stops testing anything is a worse outcome than one that reds loudly, and the `HEAD`
  anchor produced the former, not the latter. §6's *conclusion* stands; its *predicted failure
  mode* does not hold for this file.
- **`G-MATRIX-DIFF-a` is not quite the same class after all.** That gate counted call sites, so a
  moved baseline changed a number and red. This one is a substring test, and its baseline moving
  changes *what the substring matches* while the assertion stays true.
- **A residual weakness I am NOT fixing, because the instruction is "change only the anchor,
  nothing else":** M1A would also pass if the pinned baseline mentioned the construct only in a
  comment. Pinning the hash makes that moot today (at `c4e0037` the match is genuinely the
  executable probe, proven above), but the gate does not *itself* enforce it. If Beta wants that
  closed, the bounded change is one added assertion — the match must survive comment-stripping —
  and I will make it on instruction. It is out of scope here.

## 5. RED-FIRST / MUTATION EVIDENCE, PER NEW ARM

Every new arm was run against the defect it exists to catch. Mutants are module-attribute swaps
undone in a `finally`; no production file is modified by the harness.

| arm | mutant installed | result |
|---|---|---|
| A1-SNAPSHOT-IS-ONE-INSTANT | `read_snapshot` → no-op (two autocommit reads) | **RED** |
| A3-SNAPSHOT-RELEASED | `read_snapshot` BEGINs, never COMMITs | **RED** — `OperationalError: cannot start a transaction within a transaction` on the next sample |
| A2-MUTANT-STRADDLES-INSTANTS | interleaving trigger broken (vacuity probe) | **RED** |
| B1-SS-MISSING-UNAVAIL | pre-R1 `estab_count` | **RED** (`-1`) |
| B2-SS-NONZERO-EXIT-UNAVAIL | pre-R1 `estab_count` | **RED** (`0` — Beta's defect, reproduced) |
| B3-SS-UNPARSEABLE-UNAVAIL | pre-R1 `estab_count` | **RED** |
| B4-SS-TIMEOUT-UNAVAIL | pre-R1 `estab_count` | **RED** (`-1`) |
| B5-OBSERVED-ZERO-IS-ZERO | pre-R1 `estab_count` | **green — correctly.** The legacy code gets a genuine zero right; this is the converse control and must not red |
| B6-TSV-RENDERS-UNAVAILABLE | `render_estab` → `str(None)` | **RED** |
| B7-VERDICT-UNAFFECTED-BY-ESTAB | `evaluate` lets ESTAB into occupancy | **RED** |
| C1-STATIC-QUEUE-NO-TURNOVER | `_turnover` always satisfied | **RED** |
| C3-UNION-ONLY-NEITHER | `_turnover` always satisfied | **RED** |
| C2-DRAINING-QUEUE-BOTH | `_turnover` never satisfied | **RED** |
| C4-TRANSITIONS-ALONE-SUFFICE | `_turnover` keeps only the drain term | **RED** (and C2 stays green under the same mutant — the terms are independent) |
| C5-VERDICTS-NOT-COLLAPSED | `_turnover` mirrors verdict 1 | **RED** |

**Gate A1/A2 method (VIR-2).** A wrapper connection fires **one real writer commit** — a separate
connection, a real `UPDATE`, a real `COMMIT` — in the gap between the sample's two reads, against a
**WAL** fixture (required: in rollback-journal mode the writer would simply block on the reader's
shared lock and the interleaving under test could not occur). Fixture: 25 compute-active workers,
one claim each as F1 enforces, plus 7 queued; the interleaved write completes three.

*Self-consistency predicate:* under F1 there is exactly one compute-active claim per serial worker,
so `compute_active` (read 1) and `claimed_rows` (read 2) are the same quantity measured twice. A
sample that straddles a transition disagrees with itself.

```
snapshot  : active=25 claimed=25 done=0 pending=7   (32 rows conserved, self-consistent)
mutant    : active=25 (before) vs claimed=22 done=3 (after)  — two instants in one sample
next sample after COMMIT : active=22 done=3        — the snapshot really was released
```

The wrapper asserts `fired is True` in every arm, so if production SQL is ever reshaped past the
trigger the arm reds rather than passing vacuously.

**One disclosure on mutant authenticity.** For the GPU probe, M1A anchors the legacy transcription
to a commit. **The sampler cannot do that: `scripts/gate12_concurrency_sampler.py` is UNTRACKED**
(`?? scripts/gate12_concurrency_sampler.py`), so there is no revision to `git show` a pre-R1
baseline out of. The legacy `estab_count` used in the B-arm mutant is transcribed from the working
tree as it stood before this session's edit, and reproduces exactly the behaviour Beta describes
("returns `0` when `ss` is unavailable or fails"). This is stated rather than glossed: it is a
weaker anchor than a hash, and it becomes a hash the moment the file is committed.

## 6. THE GPU PROBE'S CERTIFIED LOGIC IS BYTE-UNCHANGED APART FROM THE ANCHOR

**`preflight_check.py` was not opened for editing at any point this session.**

- mtime `2026-08-09 19:57:33` — predates this session (first command `2026-08-10 07:0x`), whereas
  every file this session did change carries an `2026-08-10 07:xx` mtime.
- `sha256 cfbde94c71b66d07a613b4ef49dbc38088efdb4005d28899e5846c2f2c346730`
- `git diff c4e0037 -- preflight_check.py` = `236 insertions(+), 54 deletions(-)` — the certified
  repair, and nothing else; it contains no anchor logic (`grep -c "HEAD:"` → 0).
- The anchor lives entirely in the **test** file. Within that file the diff is: one added
  `_PRE_FIX_REV` constant with its rationale comment, the `git show` argument, and two doc/detail
  strings that named `HEAD`. The three-outcome logic, all twelve gates, the fixtures, the SSH shim
  and the gating arms are untouched — **12/12, identical gate names and identical outcomes** before
  and after.

`gate12_launch.sh` was likewise not modified (mtime `2026-08-09 20:07:32`).

## 7. FILES CHANGED

| file | status | change |
|---|---|---|
| `scripts/gate12_concurrency_sampler.py` | untracked, modified | `read_snapshot` + `isolation_level=None`; `sample_run` wrapped; `estab_count` → `estab_observation` + `render_estab` + `format_tsv_row` + `summarize_estab`; `_turnover` and the second verdict; two-verdict summary + ESTAB block; exit codes 0/2/3; docstring |
| `tests/test_gate12_concurrency_sampler.py` | untracked, modified | 14 arms retained (G6's verdict-line assertion updated for the two-verdict layout); **15 new arms** A1–A3, B1–B7, C1–C5; `build_ledger(wal=)`, `samples_from` accepts done/staging, `verdict_line`, `_InterleavingConn`, `fake_ss` |
| `tests/test_preflight_gpu_probe.py` | untracked, modified | **anchor only** — `_PRE_FIX_REV` pinned to `c4e0037`, plus its rationale comment and two strings that named `HEAD` |

**Not changed:** `preflight_check.py` · `gate12_launch.sh` · every coordinator, miner, ledger,
seed-domain and certified-suite file.

### `gate12_launch.sh` — DELIVERED UNCHANGED, because the interface did not change

Beta's instruction was to update the launcher *if the sampler's interface changes*. It does not.
Verified behaviourally rather than by reading: the sampler was invoked with the launcher's **exact**
flag set (`--out --summary --interval 2 --threshold 25 --min-window-samples 2 --port 5700
--max-seconds 7200`) against a nonexistent ledger, and it died at the **ledger guard**, not at
argparse — every flag was accepted. No new required argument exists (`--out`, `--summary` remain the
only two, and the launcher passes both). The manifest-derived default ledger path still resolves to
`/home/michael/miner_staging/miner_ledger.db`.

The only observable change to the launcher's contract is the sampler's **exit status**, which the
launcher never inspects (it backgrounds the sampler and reads the TSV and verdict files). Editing
the script solely to re-document that would be an unrequested change to a Beta-reviewed artifact, so
it is reported here instead. **Nothing was run from it.**

### Standing-item note

No new untracked `.py` file was created — the three modified files were already untracked from the
previous round — so the phase-4 Gate 22 untracked-`.py` sensitivity is unchanged by this work.

## VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** every figure quoted above is from a run in this session on VM101 under
  `~/venvs/torch`; both suites print a terminal `N/N checks green` sentinel.
- **clean control:** G6/C2 (both criteria genuinely met → SATISFIED); A1 (atomic sample under a real
  interleaved write); B5 (a genuine zero stays 0).
- **fault-injection control:** the 15-row mutant table in §5; every new arm reds under its defect.
- **completion sentinel:** `12/12` and `29/29`; the mutation harness prints `ALL MUTANTS BEHAVED AS
  REQUIRED`.
- **unavailable-observer behaviour:** ESTAB terminates OK / UNAVAILABLE and is rendered as such; a
  ledger read error is printed and the sample is kept out of the verdict.
- **audit claim scope:** the sampler, its suite, and the GPU-probe suite's anchor line. **No claim
  is made about the certified probe logic beyond "not touched"** — it was deliberately not
  re-verified.
- **searched surfaces:** live working tree on VM101; `git show` at `c4e0037` and at a throwaway
  post-commit HEAD; `git status/log/reflog/for-each-ref`; `docs/` — `CLAUDE_CODE_INSTRUCTIONS_PRERUN_R1.md`,
  `CLAUDE_CODE_REPORT_PRERUN_PROBE_AND_SAMPLER.md`, `TB_SUBMISSION_PRERUN_ITEMS_AND_RERUN_REQUEST.md`,
  `TB_NOTE_R1_INFLIGHT_AND_ACCESS_PATTERN.md`; `tests/` for the existing pinned-baseline convention;
  live `ss` and `sqlite3` behaviour on the box.
- **unavailable surfaces:** no committed baseline exists for the sampler (untracked — §5 disclosure).
  The fleet, the coordinator and the ledger were **deliberately** not exercised: no launch, no fleet,
  no port bind.
- **governance trail searched:** `TB_*` and `CLAUDE_CODE_*` documents listed above.
- **chapters searched:** none — this item touches no pipeline stage.
