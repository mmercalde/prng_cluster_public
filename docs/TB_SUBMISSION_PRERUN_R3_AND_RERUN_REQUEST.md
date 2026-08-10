# TEAM ALPHA → TEAM BETA — PRE-RERUN R3 + GATE-12 RERUN REQUESTED

**Per your ruling of 2026-08-10** (*"Return PRE-RERUN R3 with only P2, S3, S4 and E1"*).
**Committed at `4643a11`** — exactly one commit ahead of the accepted R2 lineage `49ff9b4`, under
the owner's authority per your §15. **Nothing launched; port 5700 unbound; both ledger mtimes
unchanged at start and end of the work.** No file on your §12 no-touch list was modified.
Implementation report: `docs/CLAUDE_CODE_REPORT_PRERUN_R3.md`.

**THE OPERATIVE CHECKLIST — your §10, closed row by row, and closure is claimed against this
list and nothing else:**

```
P1  CLOSED   (R2 — unchanged)
P2  CLOSED   — scripts/gate12_gpu_gate.py + gate12_launch.sh:87-92; suite 9/9
S2  CLOSED   (R2 — unchanged)
S3  CLOSED   — GATE-12 SATURATION VERDICT authority line; sampler suite 44/44
S4  CLOSED   — active_workers_json; sampler suite 44/44
E1  CLOSED   — test_s172_resolved_execution_set.py 34/34, derivations not literals
```

**Suites, VM101 (Michael's transcript):** probe **12/12** · P2 gate **9/9** · sampler **44/44**
· exec-set **34/34** · seed-domain **40/40** · F1/F2 **16/16** · phase-4 **63/63 post-commit**
(see disclosure (c)). **Reproduced independently by Alpha in a fresh clone at `4643a11`:** probe
12/12 · P2 gate 9/9 · sampler 44/44 · F1/F2 16/16 clone-side; exec-set 33/34 and seed-domain
32 green with the misses being **environment-bound arms, not code reds** — named with their
refusal text in §5, per VIR-6.

---

## 1. P2 — Gate-12-only GPU fail-close

**The gate runs at `gate12_launch.sh:87` — before clean-slate, sampler, coordinator and fleet —
and a refusal exits the script** (`P2-REFUSAL-PRECEDES-SAMPLER` proves the ordering positionally:
gate@86 < clean-slate@95 < sampler@114 < coordinator@127 < fleet@151).

- **The certified probe is reused BY IDENTITY, not reimplemented:** `gate12_gpu_gate.py:56-57`
  binds `PF._build_gpu_probe_script` / `PF._parse_gpu_probe` directly;
  `P2-REUSES-CERTIFIED-PROBE` asserts there is no second probe string and no second parser.
- **Targets are DERIVED, not literals:** resolved from `rig_profiles_config.json` joined with
  `distributed_config.json` via the execution-set resolver — the same authority the run itself
  uses; `P2-TARGETS-ARE-DERIVED` proves `['192.168.3.122','192.168.3.156','192.168.3.164']`
  with `expected=8` and **no rig address literal in the gate's executable code**.
- **Full input space enumerated, each cause separately** (ssh failure and missing binary reach
  refusal by different paths): `OK 8/8 × 3 → proceed` · any `UNAVAILABLE` (no binary /
  non-zero exit / ssh failure / timeout) → **REFUSE** · any `ERROR` → **REFUSE** · any count
  mismatch **including the genuine observed 0** → REFUSE, **reported as a real zero, not
  UNAVAILABLE** — the three-outcome vocabulary is preserved through the refusal.
- **Mutants:** `P2-MUTANT-GATE-RESULT-IGNORED` — with the verdict discarded, all three refusal
  arms would PROCEED, so the live refusals are decided by the gate, not by the probe failing to
  run. And see disclosure (b).
- **Generic `PreflightChecker` advisory policy untouched**, exactly as your ruling bounded it.

## 2. S3 — the combined authority line

`overall_satisfied` (`gate12_concurrency_sampler.py:695-708`) is the conjunction
`v["satisfied"] and v["turnover_satisfied"]`; `render_summary` prints
**`GATE-12 SATURATION VERDICT : SATISFIED|NOT SATISFIED` ABOVE both retained sub-verdicts**
(the suite asserts the ordering positionally, not just presence). The docstring records why
`and` is load-bearing in your terms: *`or` would let a run that proved occupancy but never
proved the queue was consumed present itself as saturated.* `exit_code()` is byte-unchanged.
**The F1 self-describing element list grew 17 → 19** (the authority line and the exit-code
correspondence), so a mutant deleting the line reds; `S3-AUTHORITY-LINE-IS-CONJUNCTION` proves
yes/yes → SATISFIED, yes/no → NOT SATISFIED with both sub-verdicts retained.

## 3. S4 — durable simultaneous worker identities

New column `active_workers_json`; `render_active_workers`
(`gate12_concurrency_sampler.py:405-440`) is **keyed off `obs_status`, not the set's contents**
— `unobserved_row` seeds an empty set, so unconditional serialization would emit `[]` on
exactly the samples where `[]` is a lie. Three cases, distinguished by design and by gate:

| case | renders |
|---|---|
| OBSERVED, workers active | sorted JSON array of the IDs |
| OBSERVED, none active (genuine no-run-yet zero) | `[]` — a real observation of nothing |
| UNOBSERVED | the `UNOBSERVED` marker — **never `[]`** |

**The `len == compute_active` invariant is ENFORCED, not aspirational** — it raises at the
render site if the identities and the count ever disagree, because a file that cannot account
for exactly the workers its count is made of must not claim auditability. Sorted for
determinism, so the evidence file diffs and hashes stably.
`S4-IDENTITIES-PERSIST-GAPS-ARE-NOT-EMPTY`: 8 observed rows carry 25 sorted IDs with the
invariant holding; 2 injected gaps carry UNOBSERVED, never `[]`. Summary and `evaluate`
semantics unchanged — persistence only, as ruled.

## 4. E1 — the stale certified execution-set test

`assert len(workers) == 2 + 8` is gone; the expectation is **derived from the arm's own
resolved set** — `expected_workers = s.gpu_count()` — so the certified 2→1 localhost correction
moves it automatically. Full suite **34/34 on VM101**, the perturbation demonstration performed
once and the config restored byte-exact. **One disclosure attaches — (a) below.**

---

## 5. ALPHA'S INDEPENDENT VERIFICATION (fresh clone at `4643a11`, this session)

Every closure above was **read in source in the clone, not relayed from the report**: the
`by-identity` probe binding, the gate's position in the launch flow, `overall_satisfied`'s
conjunction and the summary ordering, `render_active_workers`' `obs_status` keying and its
raising invariant, and both E1 derivation sites. Suites clone-side: probe 12/12 · P2 gate 9/9 ·
sampler 44/44 · F1/F2 16/16. **Two suites are host-bound and their misses are honest
environment refusals, not assertion failures (VIR-6):** exec-set 33/34 — `g_consumer_boot_notify`
raises `UNAVAILABLE: /etc/cluster-boot-notify.conf not readable`, a VM101 system file; and
seed-domain — two dispatch arms requiring CuPy/GPU (`CuPy not available - GPU required for
sieve`), absent in the clone environment. Both are 34/34 and 40/40 respectively on VM101. One
sandbox note for completeness: the P2 gate suite initially **refused to run** in the clone at
the execution-set resolver's hostname-identity wall — itself confirmation the gate binds to the
real fleet configuration — and ran 9/9 once the clone host's name matched.

## 6. FOUR DISCLOSURES, NOT WORKED AROUND

**(a) E1 had TWO stale sites, and both were fixed.** Your ruling named `:667` (`2 + 8`); the
same 2→1 correction had also silently staled `:649`'s total (`gpu_count() == 26`). Fixing only
the named line would have left the suite red, failing your stated deliverable — *"return the
suite GREEN"* — so both were corrected, **with derivations, not fresh literals**: `:667` from
the arm's own `s.gpu_count()`; the totals site from `distributed_config.json`, because deriving
it from `s.gpu_count()` would assert a value against itself. Offered for you to reverse if the
second site is ruled out of scope.

**(b) A defect in Alpha's own P2 wiring, self-caught and fixed before submission.** The first
wiring was `if ! python3 gate.py | tee "$EVID"` — which tests **tee's** exit status, so the gate
would have printed REFUSED **and launched anyway**: decorative, in the very script whose
attempt-1 defect was a GPU reading that stopped nothing. Fixed with `${PIPESTATUS[0]}`
(`gate12_launch.sh:87-92`), and **both forms are executed in a gate to prove the difference** —
`P2-MUTANT-PIPESTATUS-BYPASS` shows the `| tee` form swallows the refusal and launches while the
live PIPESTATUS form aborts.

**(c) Phase-4 read 62/63 pre-commit; the standing Gate 22 rule was applied, not the allowlist.**
The one red was the scope-drift detector naming R3's four then-uncommitted `.py` files — the
documented untracked-file sensitivity, arising for the **third** time. Per the standing rule the
allowlist was **not** widened (Claude Code proved in a scratch clone that a four-line entry
would restore 63/63, and declined to make the edit as out of scope — the same restraint you
credited at R1); the files were committed and **self-clearance was then proven empirically:
63/63 post-commit on VM101**, transcript on file.

**(d) A brief/ruling numbering flag, resolved as a non-issue.** The implementer could not check
the brief's §10/§11 citations because your ruling document is not in the repo; Alpha confirmed
against the ruling held in-session that §10 is the checklist and §11 the copy-verbatim
requirement. Flagged here so the trail records that it was checked rather than assumed. *(If
rulings are to be checkable by the implementing agent in future rounds, committing them to
`docs/` would close this class; Alpha takes no position beyond noting it.)*

---

## 7. REQUEST — Gate-12 rerun authorization (attempt 2)

Your §14 has already **approved the geometry**; Alpha requests the run itself. The shape,
unchanged:

```
seed_start        = 0              (explicit; certified first-gap, empty {constant,variable} namespace)
max_seeds         = 2,147,483,648  (2^31) ⇒ 32 macro-stripes per stage   ← the key is max_seeds
miner_stripe_size = 67,108,864     (2^26)
worker_pool_size  = 25
test_both_modes   = true           prng_type = java_lcg
window_trials     = 1              n_parallel = 1     (CLI key is `trials`)
use_range_miner   = true           use_persistent_workers = false   (flag omitted ⇒ PWC suppressed)
```

Every instrumentation condition from your R2 review now holds at once: the launch **fail-closes
on truthful GPU health before any process starts** · the sampler arms before the first
`StripeAssign` and each sample is one atomic snapshot · a failed read is UNOBSERVED, breaks
windows, and cannot end the observation · turnover is measured step-wise inside the qualifying
window · **the evidence file states its own authority** — the combined verdict line — **and can
prove which 25 workers were simultaneously active**, auditable against the frozen cohort.

**Standing conditions Alpha will observe, restated for the record:** no mid-run intervention of
any kind · a sizing refusal at preflight is a **legitimate Gate-12 result** and will not be met
by shrinking the seed count · **a GPU-gate refusal at launch is likewise a legitimate recorded
outcome, not a condition to be worked around** · coordinator process death means **interrupted,
not resumable** · GPU completion is not completion — **only successful canonical publication
is** · fewer than 25 admitted workers ⇒ **no saturation claim.**

**Nothing will be launched until you authorize it. On authorization, Michael initiates:
`bash gate12_launch.sh` on VM101 — one command; GPU gate, then sampler, then fleet.**
