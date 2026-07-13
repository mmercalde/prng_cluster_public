# Documentation Reconciliation — Config of Record + Correction Checklist
**Version:** 1.0.0
**Date:** 2026-07-13
**Author:** Team Alpha (Claude)
**Status:** DRAFT — for operator review; no existing files edited by this document
**Scope:** Front-door docs (`README.md`, `PROJECT_MAP.md`, `CURRENT_STATUS.txt`,
`instructions*.txt`, `Cluster_operating_manual.txt`) and older `docs/` chapter /
proposal versions that contradict the live configuration.
**Supersedes (partially):** `docs/DOCUMENTATION_AUDIT_20260131.md` — that audit
identified several of the items below in Jan 2026 but the fixes were only
partially applied (see §4).

---

## 0. Purpose

Multiple documents describe the cluster, pipeline, and PRNG registry in ways
that no longer match the code and config. This document does two things:

1. Establishes a single **Config of Record** (§1) sourced from machine-readable
   ground truth, so there is one place to point at.
2. Provides an **exact file:line correction checklist** (§3) to bring the
   drifted docs into alignment.

This document changes no existing files. It is a plan, not a patch.

---

## 1. Config of Record (authoritative)

Sourced from `distributed_config.json`, `config_manifests/feature_registry.json`,
`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (SPEC FROZEN), and the
S172/S174 session changelogs.

| Item | Authoritative value | Source |
|------|---------------------|--------|
| Nodes | 4: Zeus/localhost + 3 AMD rigs | `distributed_config.json` |
| Zeus GPUs | 2× RTX 3080 Ti (CUDA) | `distributed_config.json` |
| AMD rigs | `.120`, `.154`, `.162` — **8× RX 6600 each** (ROCm) | `distributed_config.json` |
| Total GPUs | **26** (2 + 8 + 8 + 8) | `distributed_config.json` |
| User / paths | `michael` / `/home/michael/distributed_prng_analysis` | `distributed_config.json` |
| PRNG registry | **44** (11 base × 4 variants) | `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §0 |
| Feature vector | **64 total** = 50 per-seed + 14 global; **62 training** (excl. score, confidence) | `feature_registry.json` |
| Window bounds | **min 6 / max 50 / default 12** (S172 raised min 2→6) | `distributed_config.json` `search_bounds.window_size` |
| Step 0 | **TRSE** (`trse_step0.py` v1.15.0) — Temporal Regime Segmentation; implemented, wired into WATCHER, consumed by Step 1 (passive, Rule A) | `trse_step0.py`, `agents/watcher_agent.py:387-410`, `window_optimizer_bayesian.py:487-520` |
| Step 1 backend | PWC (persistent workers, TCP); S172 RANGE-MINER opt-in **not yet implemented** (Phase 1 stub) | `SESSION_CHANGELOG_20260507_S174.md`, `miner/range_miner_coordinator.py` |
| Infra direction | Proxmox containerization of rigs (TB-approved; LXC-vs-VM pending `rrig6600c` trial) | `PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md` |
| Latest session | S174 (ready-gate hard fix) + S172 infra pivot | git HEAD `2cf23cf` (2026-07-11) |

### 1.1 Caveats on the authoritative values
- `feature_registry.json` is itself dated 2025-12-26 (v1.0.0, Session 17) and has
  not been re-verified against post-S170 code. The **64/62** figure is the
  machine-readable contract and should be treated as authoritative until the
  registry is re-audited; if a re-audit changes it, update the registry first,
  then this table.
- "Step 0" is **not** genuinely ambiguous in the code (this was a misread in an
  earlier pass). The label had two historical meanings: (i) PRNG fingerprinting,
  archived Session 17 (Dec 2025), and (ii) **TRSE**, which took over the slot in
  S119–S122 (Mar 2026) and is the current, code-authoritative Step 0. Docs that
  still describe (i) as "Step 0" are simply stale. See §2 items E and H.

---

## 2. Misalignment inventory

Severity: 🔴 high (can mislead operations/agents) · 🟠 stale baseline · 🟡 hygiene.

| # | Sev | Issue | Wrong value(s) | Correct value |
|---|-----|-------|----------------|---------------|
| A | 🔴 | GPU topology | "2 rigs × 12× RX 6600", "~285 TFLOPS" | 3 rigs × 8× RX 6600 = 26 GPUs |
| B | 🔴 | PRNG registry count | "46 PRNGs" | 44 (11×4) |
| C | 🔴 | Feature count | "91" / "89 training" / "48" | 64 total / 62 training |
| D | 🔴 | Window bounds | "min 2 / max 500" | min 6 / max 50 / default 12 |
| E | 🟠 | Step 0 identity — stale docs describe archived fingerprinting | MAP/STATUS: "Step 0 = PRNG fingerprinting, ARCHIVED"; S121 plan: "wiring NOT yet done" | Step 0 = TRSE, implemented + wired + consumed by Step 1 (code-authoritative) |
| F | 🟠 | "FULLY OPERATIONAL 285.69 TFLOPS" whole-doc baselines | bare-metal, pre-pivot | needs S172/S174 + Proxmox context |
| G | 🟠 | Front-door docs date-frozen | README S135 (Mar); STATUS/MAP Session 17 (Dec 2025) | S174 / S172-infra (Jul 2026) |
| H | 🔴 | **BUG** — dangling WATCHER Step 0 manifest | `watcher_agent.py:399` sets `STEP_MANIFESTS[0]="trse.json"` but `agent_manifests/trse.json` **does not exist** | create the manifest (S121 plan §2.B) or remove the reference |
| I | 🟡 | Duplicate/near-dup files | `CURRENT_STATUS.txt` vs `CURRENT_Status.txt`; multi-version guides/proposals | one source of truth each |
| J | 🟡 | Hostname convention | `rig-6600` vs `rrig6600` vs raw IP | standardize |
| K | 🟡 | Repo/remote push guidance | README assumes private `origin` + `public` | N/A in public mirror |

---

## 3. Correction checklist (exact file:line)

> Line numbers are as of HEAD `2cf23cf`. Verify before editing; some files may
> shift. Each edit is a value correction only — no restructuring implied.

### A. GPU topology (→ 3 rigs × 8, drop "285 TFLOPS" or recompute)
- [ ] `PROJECT_MAP.md:179-182` — table lists only rig-6600 + rig-6600b at "12×"; add rig-6600c, change all to "8×", fix "Total: 26 GPUs, ~285 TFLOPS" (26 is right, TFLOPS figure is unverified)
- [ ] `agent_contexts/step1_window_optimizer.md:8` — "12× RX 6600" → "8× RX 6600" (×2 rigs) → 3 rigs
- [ ] `docs/SOAK_TEST_HANDOFF_PROMPT.md:19-20` — "12×" → "8×"; `:23` — recheck "~285 TFLOPS"
- [ ] `instructions.txt`, `instructions_10-16.txt`, `docs/instructions.txt`, `Cluster_operating_manual.txt` — every "285.69 TFLOPS" and per-rig "12×" (see item F; may be simpler to mark these deprecated wholesale)

### B. PRNG count (46 → 44)
- [ ] `PROJECT_MAP.md:239` — "46 PRNG algorithm definitions" → "44"
- [ ] `agent_contexts/step1_window_optimizer.md:10` — "46 PRNG algorithms" → "44"
- [ ] `docs/SOAK_TEST_HANDOFF_PROMPT.md:10` — "(46 PRNG algorithms)" → "44"
- [ ] `instructions.txt:3084,3196,3419,3436` — "46 PRNG variants/PRNGs" → "44"
- [ ] (historical, optional) `llm_test_results/*` — leave as-is; they are captured LLM outputs, not living docs

### C. Feature count (→ 64 total / 62 training)
- [ ] `README.md:15` — "91-feature extraction per survivor" → "64-feature (62 training)"
- [ ] `docs/COMPLETE_OPERATING_GUIDE_v2_0.md:223` — "91 features per survivor (89 training)" → "64 (62 training)"
- [ ] `agent_contexts/step2_5_scorer_meta.md:59` — "48 per-seed features" → "50 per-seed (64 total)"
- [ ] `docs/IMPLEMENTATION_CHECKLIST.md:129` — "48 features" → "62 training features"
- [ ] Confirm remaining "50 per-seed" mentions are correct (they are, as a sub-count) before touching

### D. Window bounds (→ min 6 / max 50 / default 12)
- [ ] `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:274,300` — `"window_size": {"min": 2, "max": 500}` → `{"min": 6, "max": 50, "default": 12}`
- [ ] `docs/PROPOSAL_Unified_Agent_Context_Framework_v3_2_0.md:644,665` — same
- [ ] `docs/PROPOSAL_Unified_Agent_Context_Framework_v3_2_3.md:673,694` — same
- [ ] (older proposal versions `_v3_2_7/9` may also contain it — grep before editing)

### E. Step 0 identity — no decision required (code-authoritative)
**Resolved:** Step 0 = **TRSE** (`trse_step0.py` v1.15.0). It is implemented,
wired into WATCHER (`agents/watcher_agent.py:387-410`: `STEP_SCRIPTS[0]`,
`STEP_NAMES[0]`, `STEP_MANIFESTS[0]`), and actively consumed by Step 1
(`window_optimizer_bayesian.py:487-520`, Rule A narrows the window ceiling to
≤32 when `regime_type=short_persistence` ∧ `regime_type_confidence≥0.70` ∧
`regime_stable`; Rules B/C log only). Step 1 also writes `confirmed_windows`
back to `trse_context.json` (`window_optimizer.py:769-798`). The
"archived PRNG fingerprinting" is a **superseded earlier meaning** of the Step 0
label (Session 17, Dec 2025), predating TRSE.

`README.md` is **already correct** on Step 0 — no edit needed. Fix the stale docs:
- [ ] `PROJECT_MAP.md:14,22` — reframe "Step 0 PRNG Fingerprinting — ARCHIVED" as historical; add current Step 0 = TRSE row
- [ ] `CURRENT_STATUS.txt` — same (Session-17 archived-fingerprinting narrative is historical, not current)
- [ ] `docs/TRSE_INTEGRATION_PLAN_S121.md:3` — status line "wiring into pipeline NOT yet done" is **stale**; items §2.A (WATCHER) and §2.C (Step 1 passive read + Rule A) have since landed. Update to "wired" and note §2.B (manifest) is the one outstanding item — see item H.

### H. BUG — dangling WATCHER Step 0 manifest (🔴 real, not doc-only)
`agents/watcher_agent.py:399` sets `STEP_MANIFESTS[0] = "trse.json"`, but
`agent_manifests/trse.json` **does not exist** (dir contains only the six
Step 1–6 manifests; `find . -name 'trse*.json'` → none). S121 plan §2.B specified
this manifest (`agent_name`, `pipeline_step:0`, `primary_output:trse_context.json`,
`evaluation_type:file_exists`, `disable_llm_parsing:true`, `retry_policy:none`) but
it was never committed.
- [ ] Verify how WATCHER handles a missing manifest on a `--start-step 0` run (does it fail, or fall through to defaults?)
- [ ] Either **create** `agent_manifests/trse.json` per S121 §2.B, or **remove** the dangling `STEP_MANIFESTS[0]` entry if Step 0 is intended to run without a manifest
- Note: `trse_context.json` itself is not gitignored and not committed — it is a runtime artifact regenerated on Zeus each run; that is expected, not a gap.

### F/G. Stale baselines and dates
- [ ] `README.md:2` — "Updated: S135 (2026-03-10)" → current session
- [ ] `CURRENT_STATUS.txt:1` / `PROJECT_MAP.md:2` — "Session 17 (Dec 2025)" → current
- [ ] `instructions.txt`, `instructions_10-16.txt`, `docs/instructions.txt`, `Cluster_operating_manual.txt` — add a deprecation banner at top pointing to the Config of Record and the S172/S174 changelogs, OR retire them. These predate the Proxmox pivot and S174 stability work and read as "current" when they are not.

### I/J/K. Hygiene
- [ ] Resolve `CURRENT_STATUS.txt` vs `CURRENT_Status.txt` (two files, different case/content) → one canonical file
- [ ] Consolidate multi-version guides/proposals (`COMPLETE_OPERATING_GUIDE_v1_1/v2_0/.docx`; `PROPOSAL_Unified_Agent_Context_Framework_v3_2_{0,3,7,9,10}`) — keep the latest, mark older as superseded
- [ ] Standardize rig naming (`rig-6600` vs `rrig6600`) across README + changelogs (pick one; note IPs remain the config key)
- [ ] `README.md:69-71` — private `origin` + `public` remote and `git push origin main && git push public main` do not apply in the public mirror; either scope the instruction as "private repo only" or drop it here

---

## 4. Why the Jan-2026 audit didn't fully land

`docs/DOCUMENTATION_AUDIT_20260131.md` already recorded items A (12×→8×, line 47,
77, 80, 144) and related fixes, and `doc_update_report_20260131_*.txt` logged
edits. Yet `PROJECT_MAP.md`, `agent_contexts/step1_window_optimizer.md`, and
`docs/SOAK_TEST_HANDOFF_PROMPT.md` still carry the "12×" / "46 PRNG" values.
The audit's file sweep missed these targets (or they were regenerated after).
Recommendation: after applying §3, add a post-edit grep gate to CI or the
session-start hook:

```bash
# should return ZERO matches after reconciliation
grep -rniE '12[× x] ?RX ?6600|46 ?PRNG|285\.69 ?TFLOPS|"min": 2, "max": 500' \
  --include='*.md' --include='*.txt' . \
  | grep -vE 'DOCUMENTATION_(AUDIT|RECONCILIATION)|doc_update_report|llm_test_results'
```

---

## 5. Suggested order of operations

1. Fix the **H bug** first (missing `agent_manifests/trse.json`) — it is a real
   wiring gap, not doc drift; decide create-vs-remove and confirm WATCHER behavior.
2. Apply mechanical value fixes A–D (safe, unambiguous — no decision needed).
3. Apply E doc-cleanup (mark archived-fingerprinting as historical; README needs no change).
4. Add deprecation banners to the F-tier legacy docs (fast, high signal).
5. Refresh dates + narratives G, then hygiene I–K.
6. Add the §4 grep gate so drift can't silently return.

*This document is a draft plan. It edits nothing. The only genuine open question
is the "285.69 TFLOPS" figure (unverified — see item A/F); Step 0 is resolved
(item E) and item H is a code bug to fix, not a decision.*
