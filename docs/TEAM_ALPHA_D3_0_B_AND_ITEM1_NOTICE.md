# TEAM ALPHA → TEAM BETA — one ruling request and one disclosure, from PHASE6_PREREQS REV4

**`docs/PHASE6_PREREQS.md` REV4 is a status correction and is NOT submitted for approval.** Five of
seven statuses were wrong; they are now measured
(`docs/S172_PHASE_7_PREREQ_REPORT.md`). This notice carries the two items inside it that Beta
should not have to find by reading a 392-line checklist.

---

## 1. RULING REQUESTED — D3.0-B was never completed, and Phase 6 certified anyway

**REV3 stated, in a TB-approved document:** *"D3.0-B (legacy writer corrections) must also complete
before Phase 6 certification."*

**No commit completing it exists.** `git log --all --grep` over the label returns only the review
that raised it. **And the defect it targets is live at HEAD**, verified this session:

```python
# convert_survivors_to_binary.py:184
encode_prng_type(s.get('prng_type', s.get('prng_base', 'java_lcg')))
```

A record carrying **neither** `prng_type` **nor** `prng_base` still silently becomes `'java_lcg'`
rather than failing closed — the exact residual silent default D3.0-B exists to purge
(`docs/TEAM_ALPHA_REVIEW_S172_PHASE5_D3_0.md` §5.4).

**Phase 6 certified at `d98298c` regardless.**

**What Alpha does NOT know, and will not assume:** whether Beta waived D3.0-B, superseded it, or it
was simply never raised at certification. **The repository does not say.** It is plausible that Wall
A/B never exercised the legacy writer, which would make the omission harmless *in fact* while
leaving a stated prerequisite unmet *on paper* — but that is a guess, and Alpha is not presenting it
as a finding.

**Ruling requested:** dispose of D3.0-B — **waived · superseded · or open and requiring completion.**

**Alpha is not proposing to fix it.** Two reasons: it touches the legacy writer, which is outside
every current work item; and if it genuinely should have blocked certification, that is Beta's
call to make explicit rather than Alpha's to quietly close.

*Alpha raises it because a checklist that silently loses a prerequisite is worse than one that never
carried it — the next reader inherits a document that looks complete.*

---

## 2. DISCLOSURE — the owner has waived item 1; REV4 edits a TB-approved gating list

**Not a ruling request.** Michael, as owner, has mandated that the Phase-7 soak runs at **25 GPUs**
(24 AMD + 1 NVIDIA, live inventory verified exactly). The second 3080Ti remains assigned to VM100.

**REV3's Phase-7 gating line required items 1–7. REV4 now reads:** items 2, 3, 5, 6, 7 required ·
**item 1 explicitly waived by the owner** · item 4 required.

**Alpha is disclosing this because it modifies a TB-approved document's gating list**, and Beta
should not discover it inside a status correction.

**The technical consequence Alpha did act on**, since it is not a permission question: under the
Resolved Execution Set ruling a partial set must be **explicit and frozen, never inferred from who
answers.** Measured:

```
set_id                     = adcc2ae5714c98b0f232c62c1aa33ef43d9cd16eeb66c4f480a0b779d61af138
requested_admission_count  = 25
admission_count            = 25      admission_clamped() = False
```

**25 by construction, not by clamp** — as required.

**One qualification Alpha flags rather than hides.** The frozen set carries **26 worker identities**,
because `distributed_config.json` declares `localhost.gpu_count: 2` from the two-card configuration.
So `min(25, 26) = 25` with no clamp recorded. **This is not the failure `eff6616` closed** — the
threshold is 25, 25 real workers exist, the 180 s window is meetable, and no production path spawns
workers by iterating `gpu_count` (they launch explicitly with `--gpu-id N`). **The cost is
auditability:** provenance logs 26 identities admitting 25, which *reads* like a 26-set that came up
short.

Alpha has recommended setting `localhost.gpu_count: 1` to match measured hardware, with three
caveats recorded in REV4: it changes `set_id`; it must be committed **before** launch because item
5's clean-tree wall rejects a dirty tree at finalization; and it is a **distinct field from that
file's bare-metal addresses, which stay untouched** (CLAUDE.md §3). **Owner decision, not yet
applied.**

---

## 3. Where the soak stands

**Two blockers remain, both operational, neither requiring a ruling:**

- **Item 4** — VM101 is still on DHCP (`inet 192.168.3.177/24 … dynamic`). A lease move mid-soak
  disconnects every worker from the coordinator. Router-side reservation in progress.
- **Item 6** — **code parity fails.** All three rigs carry `miner/range_miner_coordinator.py` at
  `ee0db06` and `miner/dataset_authority.py` at `8600e75`, with `utils/checkpoint_d6_2.py` absent —
  and **both stale modules sit inside the worker's executing import closure**
  (`miner/__init__.py:19`, confirmed via `sys.modules` on each rig). The rig copies pre-date
  `18a2419`, **the D6.2 repair this soak exists to exercise.** Redeploy and digest re-verification
  are under way.

Items 2, 3, 5 and 7 are closed on live measurement.

## 4. VIR declaration

**Audit scope:** repo at `3561cda` plus live hosts (VM101 and three CT100 workers), measured
2026-08-02. **Searched surfaces:** tracked repo including full git history for the D3.0-B label; the
live rig filesystems by sha256; `sys.modules` in a real interpreter on each rig.
**Unavailable surfaces:** whether Wall A/B exercised the legacy writer — **that is why §1's
disposition question is Beta's and not Alpha's answer**; the second 3080Ti (assigned to VM100).
