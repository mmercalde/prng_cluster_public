# CLAUDE CODE INSTRUCTIONS — PRE-RERUN R3 (P2 · S3 · S4 · E1, NOTHING ELSE)

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD **`49ff9b4`** (accepted R2 lineage —
do NOT revert). `source ~/venvs/torch/bin/activate` before every test.

**Authority:** Team Beta ruling *"PRE-RERUN R2 / GATE-12 AUTHORIZATION REVIEW"* (2026-08-10).
Beta: *"This should be extremely narrow. Only four things remain."* Close all four; touch
nothing else.

**THE OPERATIVE BETA CHECKLIST — copied verbatim per Beta §11, and this is the list you close
against, not this brief's own framing:**

```
P1  CLOSED
P2  OPEN

S2  CLOSED
S3  PARTIAL — turnover logic good; combined authority line absent
S4  OPEN

E1  OPEN
```

*(R2's traceability failure was Alpha verifying "closed" against Alpha's own brief instead of
this checklist. If anything in this brief conflicts with the Beta ruling text, the ruling wins —
report the conflict, do not resolve it silently.)*

**Hard constraints:** no commit, no push, **no launch, no fleet, no port 5700 bind, no real-rig
SSH**; Gate 12 HELD. **No changes authorized to:** F1/F2 · coordinator · miner · ledger · lease
policy · retry matrix · seed domain · coverage authority · dataset authority · publication ·
execution-set production logic · generic `PreflightChecker` advisory policy.

**Base verification before any edit:** HEAD `49ff9b4` · probe suite **12/12** · sampler suite
**38/38**.

---

## P2 — Gate-12-only GPU fail-close in `gate12_launch.sh`

**Falsifiable question:** does the Gate-12 harness refuse to launch unless all three rigs
truthfully report `status == OK` and `gpu_count == 8`?

Beta ordered a **Gate-12 harness rule only** — NOT a change to generic preflight advisory
policy. Current launch flow (`gate12_launch.sh`) is evidence block → clean slate → sampler
(`:84`) → coordinator → fleet (`:121`) with **no GPU gate anywhere**.

**Required:** before the sampler/coordinator sequence begins, run the **already-certified
truthful probe** — reuse `_build_gpu_probe_script` / `_parse_gpu_probe`
(`preflight_check.py:103` / `:133`; `check_gpu_health` is the existing caller at `:457`) —
against **192.168.3.122 · 192.168.3.156 · 192.168.3.164**, and proceed **only** on
`OK, 8/8 × 3`. Do not re-implement the probe; a second probe with its own parsing is a second
place for the `|| echo 0` class of defect to live.

**Enumerate the full input space and state the behaviour for each (self-check #14) — a rule
validated only against the case that motivated it is untested:**

| probe result on ANY rig | launch |
|---|---|
| `OK, count == 8` on all three | **proceed** |
| `UNAVAILABLE` (no binary / non-zero exit / ssh failure / timeout) | **REFUSE** |
| `ERROR` (unparseable) | **REFUSE** |
| `OK, count != 8` — including the genuine observed `0` | **REFUSE** |

The refusal must abort **before** the sampler starts and before any coordinator process is
created, print which rig and which outcome, and write that to the evidence block. UNAVAILABLE
must be reported as UNAVAILABLE — never as a count.

**Gates (fixture-driven, ssh-shim technique already in `test_preflight_gpu_probe.py`; no real
rig is contacted):** `8/8 × 3 → allowed` · `one UNAVAILABLE → refused` · `one ERROR → refused` ·
`one count mismatch → refused`. **Mutant:** the gate bypassed (or its result ignored) must red
the refusal arms.

## S3 — the authoritative combined saturation verdict line

**Falsifiable question:** does the summary artifact itself carry the Gate-12 authority, rather
than leaving it implicit in an exit code the launch script never consumes?

**No algorithmic change** — Beta confirms `exit_code()` already encodes the semantics. Add to
`render_summary` (`scripts/gate12_concurrency_sampler.py:643`; current output has only the two
sub-verdict lines at `:705-706`) an explicit **top-level authority line**:

```
GATE-12 SATURATION VERDICT : SATISFIED
```

iff `v["satisfied"] and v["turnover_satisfied"]`, else `NOT SATISFIED`. **Keep both diagnostic
sub-verdicts underneath it, unchanged.**

**Gates:** `simultaneity=yes, turnover=no → OVERALL NOT SATISFIED` · `yes/yes → OVERALL
SATISFIED` (drive through real `evaluate` + `render_summary`, not a hand-built dict). **Extend
the F1 self-describing arm's required-element list to include the authority line** — otherwise a
mutant deleting the line passes the existing 17-element check. **Mutant:** authority line
computed with `or` instead of `and` must red.

## S4 — durable simultaneous worker identities in the TSV

**Falsifiable question:** after the process exits, can the evidence file prove *which* 25
workers were simultaneously active — auditable against the frozen execution cohort?

`sample_run` already computes `active_workers: Set[str]` (`:277`, returned `:288`) but
`TSV_COLUMNS` has no identity column and `format_tsv_row` (`:394-400`) serializes only the
numeric `LEDGER_FIELDS` — the set dies in-process.

**Required:** a new column `active_workers_json` persisted in **every** sample:

- **observed sample:** a **sorted** JSON array of the worker IDs. **Invariant, asserted where
  the row is rendered:** `len(parsed array) == compute_active`.
- **UNOBSERVED sample:** the `UNOBSERVED` marker — **never `[]`**. An empty array is a claim of
  an observed zero-worker instant. ⚠ **Trap, read it in source first:** `unobserved_row:311`
  already seeds `active_workers: set()`, so a renderer that serializes the set unconditionally
  emits `[]` on exactly the samples where it must not. **Key the rendering off `obs_status`, not
  off the set's contents.**
- **the genuine no-run-yet observed zero** (`:899-901`) legitimately renders `[]` — that is an
  observation of nothing, and the distinction between `[]` and the marker is the point.

Sortedness makes the field deterministic; state the encoding (JSON, no whitespace dependence) at
the column definition. Summary/`evaluate` semantics are **unchanged** — this is persistence, not
a new criterion term.

**Gates:** observed sample with 25 workers → 25 sorted IDs persisted, invariant holds ·
UNOBSERVED sample → marker, **not** `[]` · observed-zero → `[]`. **Mutant:** unconditional
serialization of the set (the `[]`-on-unobserved defect) must red.

## E1 — the stale certified execution-set test

**Falsifiable question:** does `test_s172_resolved_execution_set.py` pass in full against the
certified execution-set correction (localhost GPU count 2 → 1)?

The stale expectation is `tests/test_s172_resolved_execution_set.py:667`:
`assert len(workers) == 2 + 8` inside `g_consumer_legacy_test_connectivity`. **Test-only fix,
already authorized.** Beta: prefer **deriving** the expectation from the authoritative
fixture/config over a fresh magic `1 + 8`. The derivation is in the test's own hands: the arm
builds `s = _cli_set(declared_nodes=["localhost", "rrig6600"])` at `:658`, and
`s.gpu_count()` (`execution_set.py:220-221`, `sum(n.gpu_count for n in self.nodes)`) is the
authoritative count for exactly those nodes. **Read the arm in full before editing** — if the
worker list is asserted anywhere else in the same arm or the return string bakes in a count,
make those consistent with the same derivation. **No production execution-set change.**

**Deliverable:** the full suite **GREEN**, run end-to-end, with the count in the pass output
shown to be derived (temporarily perturbing the fixture count must move the expectation — show
that once, then restore).

---

## VERIFICATION (Beta §13 minimum — all six suites, full runs, terminal sentinels)

```
tests/test_preflight_gpu_probe.py            (12/12, untouched by R3 unless a P2 gate lands here)
tests/test_gate12_concurrency_sampler.py     (38/38 + the S3/S4 arms)
tests/test_s172_resolved_execution_set.py    (GREEN — E1)
tests/test_seed_domain_cursor_amendment.py
tests/test_s172_phase4_coordinator.py        (63/63)
tests/test_s172_f1_f2_active_lease.py        (16/16)
```

plus the explicit P2 launch-gate arms above. Red-first / mutation evidence per new arm. Long
suites: `python3 -u <suite> | tee /tmp/<name>.log` — **never pipe to `tail`**. Confirm port 5700
unbound and both ledger mtimes unchanged at start and end.

## REPORT — `docs/CLAUDE_CODE_REPORT_PRERUN_R3.md`

1. **The §10 checklist verbatim at the top, with your closing evidence per row** — closure is
   claimed against Beta's list, nothing else.
2. P2: the gate's placement in the launch flow, the full input-space table with the observed
   behaviour per row, and the fixture technique.
3. S3: the authority line, a **verbatim summary sample** showing it above both sub-verdicts, and
   the extended F1 element list.
4. S4: the column definition, the three rendering cases with TSV excerpts, the invariant's
   enforcement site.
5. E1: the derivation used and the full-suite green output.
6. Red-first / mutation evidence per new arm; the R1+R2 mutation tables re-run unchanged.
7. Byte-unchanged confirmation for everything outside the four items — **`sha256` for
   `preflight_check.py` and every file on the no-touch list that borders this work.**
8. Files changed, from `git status` — the `git add` list is built from this section, never from
   recall. **Any disagreement reported, not worked around.**
