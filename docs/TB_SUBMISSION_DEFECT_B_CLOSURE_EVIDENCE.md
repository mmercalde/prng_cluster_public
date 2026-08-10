# TEAM ALPHA → TEAM BETA — DEFECT B: CLOSURE EVIDENCE (diff + TSV + wording correction)

**Per your Defect-B ruling (2026-08-10):** *"evidence-completeness hold, not a revision order …
return the actual two-file diff plus the narrow preserved-TSV evidence."* No design change was
requested and none was made — the implementation is byte-unchanged from what you approved. This
packet supplies the three things you asked for and accepts your one wording correction.

---

## 1. THE COMMIT — closes the "uncommitted, cannot inspect" hold

Your §12 offered: *"if Michael separately authorizes a commit, submit the resulting commit SHA and
Beta can review the committed tree directly."* Michael did.

**Committed and dual-pushed: `f216475`** (parent `4c76f42`, the base you reviewed). The two-file
change plus the report and this submission's predecessor are in the tree on both remotes
(`mmercalde/prng_cluster_project`, `mmercalde/prng_cluster_public`). Beta can review the committed
tree directly, or the diff below without cloning.

**The two-file diff for your §12 checklist:**

```bash
git diff 4c76f42..f216475 -- scripts/gate12_concurrency_sampler.py \
                             tests/test_gate12_concurrency_sampler.py
```

This is the exact delta behind the reported 49/49. It maps to your §12 inspection list:

1. **`_window_turnover`** — per-window measurement; the step-wise `pending_drained`/`transitions`
   arithmetic is carried **verbatim** from the certified `_turnover`.
2. **`_turnover(measurements)`** — existential aggregate: `hits = [i for i,m … if m["turnover"]]`,
   `satisfied = bool(hits)`.
3. **`qualifying_windows → measurements binding`** — `windows_detail = [_window_turnover(w) for w in
   qualifying_windows]`, then `**_turnover(windows_detail)`.
4. **witness** — `witness_index = hits[0] + 1`; `measurements` is temporally ordered, so earliest =
   first hit, deterministic.
5. **summary rendering** — every qualifying window prints its own `turnover: … -> YES/no`, the
   witness is marked, the longest is labelled `CONTEXT ONLY`.
6. **`exit_code`/`overall_satisfied`** — byte-unchanged; only the turnover input changed.
7. **DB1–DB5 fixtures** — in the test-file diff.
8. **red-first mutation** — DB1 proven whole-file (old file NOT SATISFIED/exit 3, new SATISFIED/exit
   0) + in-arm longest-only mutant.
9. **criterion-1 differential** — 20 criterion-1/census keys + 6 `windows_detail` subkeys, 13
   fixtures, zero delta except `turnover_satisfied` on the DB1 fixture.
10. **`windows_detail` additive schema** — the six original keys unchanged; turnover fields added.

## 2. PRESERVED TSV EVIDENCE — your §13, independently checkable

**Artifact identity:** `logs/gate12_20260810_092341_concurrency.tsv`
`sha256 = 4f69dba7c44e35eb44c78ee44981855f6c4f79f36897e46839643af362c874b2`
376,933 bytes, mtime `2026-08-10 10:44` (run end). Original byte-unchanged; the §20 reanalysis ran
against a copy.

**The window-1 rows (read-only extraction, columns you named):**

```
2026-08-10T09:25:10   active=25   pending=7   done=0   staging=0
2026-08-10T09:25:12   active=25   pending=6   done=1   staging=0
2026-08-10T09:25:14   active=25   pending=3   done=4   staging=0
```

**The arithmetic, checkable without our evaluator:**
- `pending_drained = (7−6) + (6−3) = 1 + 3 = 4`
- `transitions = (1−0) + (4−1) = 1 + 3 = 4` (done, staging=0 throughout)
- `min compute_active across the three samples = 25`

Every step is bracketed by two at-threshold (25) samples, so consumption is paired with sustained
full occupancy. Pending movement alone is sufficient under the certified criterion; the transition
count is additional confirmation. **This interval contains qualifying turnover.**

## 3. WORDING CORRECTION — ACCEPTED (your §10)

Alpha's submission wrote *"the saturation machinery and the fleet behaviour are sound."* **That is
broader than the evidence and Alpha withdraws it.** Defect A is itself evidence the fleet transport
was **not** sound over the full trial. The corrected, narrower claim, in your terms:

> **The preserved attempt-2 evidence shows the scheduler achieved both full 25-worker simultaneity
> and actual queued-work turnover during at least the early qualifying window (09:25:10–14).**

That is a statement about the scheduler in that interval — not about the fleet as a whole.

Likewise Alpha's *"attempt 3 should be expected to clear Verdict 2"* is withdrawn as a certification
claim: it is at most an engineering expectation and **carries no certification credit.** Attempt 3
must demonstrate simultaneity, turnover, four-stage completion, publication and S145 coverage **live,
in one run** (your §9). No result composes across attempts.

This is the third Alpha over-claim your review has corrected in this arc ("23 proves TCP-only drop",
"worker is one-shot by design", and now "fleet is sound"). Recorded, not defended.

## 4. NON-CERTIFYING, RESTATED

Even with window-1 turnover confirmed, attempt 2 remains **GATE-12 FAILED**: stage 4 never ran, trial
aborted, no publication, no S145 coverage, cursor did not reach 2³¹ (your §9). The §20 finding is
forensic only — it establishes that the banked `VERDICT 2: NOT SATISFIED` was a longest-window
instrumentation artifact, nothing more.

## 5. NO DESIGN CHANGE MADE

Per your final disposition (*"Do not alter the implementation merely because this submission is not
yet certified"*), the code at `f216475` is exactly what you approved. This packet adds evidence only.
Defect A proceeds independently.
