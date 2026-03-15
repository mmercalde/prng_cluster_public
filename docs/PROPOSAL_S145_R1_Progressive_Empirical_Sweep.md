# PROPOSAL S145-R1 — Progressive Empirical Sweep of Seed IDs 0→2³² for CA/java_lcg
**Cross-Session Survivor Accumulation with Persistent Optuna Continuity**

**Date:** 2026-03-15 | **Author:** Team Alpha | **Approved by:** Team Beta (conditional)
**Supersedes:** PROPOSAL_S145_Complete_Seed_Space_Sweep.md (rejected)

---

## TB Approval Status

| Component | Status |
|---|---|
| Cross-session survivor accumulation | ✅ Approved |
| Merge by best per-seed `score` | ✅ Approved |
| Manifest field corrections (live fields) | ✅ Approved |
| Timeout patching | ✅ Approved |
| Optional pruning | ✅ Approved |
| WATCHER fresh-study invariant patch | ✅ Required before resume works |
| `.gitignore` accumulator exception | ✅ Approved |
| "Complete 32-bit sweep" claim for java_lcg | ❌ Rejected |
| "Practically sufficient coverage" conclusion | ❌ Rejected — deferred to post-sweep |
| "Step 1 retired permanently" | ❌ Rejected |
| "Same Optuna study resumes under auto-advance" (unpatched) | ❌ Blocked until WATCHER patched |

---

## 1. Executive Summary

This proposal implements a large-scale empirical frontier expansion across seed IDs
0 → 4,294,967,296 for the CA Daily 3 / java_lcg PRNG. It is executed across 2–4
sequential sessions using existing cluster infrastructure.

**What this is:**
A progressive empirical sweep targeting the seed range where CA survivor evidence
is concentrated. Each session adds permanently to an accumulated survivor pool.
Optuna learning carries forward across sessions via study resume.

**What this is not:**
- Not a mathematically exhaustive sweep of java_lcg (state space is 2^48)
- Not proof that the CA ADM seeds only in 0→2^32
- Not grounds to retire Step 1 permanently

**Value delivered regardless of exhaustiveness:**
- More survivors → better ML training data → better predictions
- Accumulated permanent survivor population — nothing is ever discarded
- Optuna TPE learns progressively across sessions — converges faster over time
- Foundation for post-sweep yield analysis to determine practical coverage

---

## 2. What S145 (Original) Got Wrong — Corrections Applied

### 2.1 The 2^32 Collapse Claim — Rejected

The original proposal claimed java_lcg's effective search space collapses to 2^32
because output extraction discards the lower 16 bits (`state >> 16`). This is wrong.

The Java LCG state transition is:
```
state_{n+1} = (25214903917 × state_n + 11) mod 2^48
```

The multiplication propagates the lower 16 bits of `state_n` into the upper bits of
`state_{n+1}` via carry. Two seeds differing only in their lower 16 bits produce
identical output at step 0, but diverge after exactly one step. For any window_size
≥ 2 with any skip ≥ 1, the full 48-bit state matters completely. The mathematical
search space for java_lcg is 2^48 = 281 trillion seeds.

### 2.2 Revised Framing — Empirical Motivation

The sweep target 0 → 2^32 is justified empirically, not mathematically:

- Bidirectional survivors found consistently in 0 → 50M range across all real-data runs
- Winning configs use window_size 2–8 — short persistence consistent with per-session
  reseeding from a constrained initialization source
- Skip range 5–56 matches documented ADM operational procedures exactly
- S143 PA experiment found massive survivors immediately under similar conditions

**Working hypothesis:** The CA ADM seeds from a constrained source (timestamp,
counter, or small-integer initialization) whose practical range falls within 0 → 2^32.
This is a hypothesis requiring post-sweep validation — not a pre-sweep conclusion.

### 2.3 `m` Correction

Original proposal misstated `m = 0xFFFFFFFFFFFFFFFF` (64-bit).
Correct value: `m = 0xFFFFFFFFFFFFULL` (48-bit mask = 2^48 - 1).

### 2.4 Manifest Field Corrections

Original proposal targeted `parameter_bounds.seed_count` (line 123).
Live WATCHER launch path uses `default_params.max_seeds`. Both fields exist
separately. All patches now target the confirmed live fields.

### 2.5 Merge Policy Correction

Original proposal used `trial_score` as merge key.
Correct field per live code and TB ruling: per-seed `score`.

---

## 3. Architecture

### 3.1 Two Independent Tracking Systems

**Seed Tracker:** `exhaustive_progress` table in `prng_analysis.db`
Tracks `seed_range_start` → `seed_range_end` per run. Fires once at WATCHER
launch. Advances `seed_start` automatically via `MAX(seed_range_end)`.

**Optuna Study:** `optuna_studies/window_opt_XXXX.db`
Tracks TPE trial history and learned parameter landscape. Persists across
sessions when operator explicitly sets `study_name` in `default_params`.

### 3.2 WATCHER Fresh-Study Invariant — Patched

The live invariant (lines 1407–1408) unconditionally resets `resume_study=False`
and clears `study_name` whenever `seed_start` advances. This defeated cross-session
Optuna continuity regardless of `default_params` settings.

**S145-R1 patch:** Conditionalizes the invariant on `study_name` presence:

```python
# [S145-R1] Conditionalized — preserves Optuna continuity when study_name
# explicitly set by operator. Default behavior (no study_name) unchanged.
_explicit_study = final_params.get('study_name', '')
if not _explicit_study:
    final_params['resume_study'] = False   # INVARIANT: new range = fresh study
    final_params['study_name'] = ''
    logger.info(f"[COVERAGE] ... forcing fresh study (no explicit study_name)")
else:
    final_params['resume_study'] = True
    logger.info(f"[COVERAGE] ... preserving Optuna continuity "
                f"(study_name='{_explicit_study}' explicitly set) [S145-R1]")
```

Backward compatible: default behavior (no study_name set) is identical to pre-S145.

### 3.3 Session Flow — 4-Session Example

| Session | seed_start | seed_end | Optuna | Action required |
|---|---|---|---|---|
| Run 1 | 0 | 1,073,741,824 | Fresh study: `window_opt_AAAA.db` | Note study name from log |
| Run 2 | auto | 2,147,483,648 | Resume `window_opt_AAAA` — TPE continues | Set `study_name` in manifest |
| Run 3 | auto | 3,221,225,472 | Resume `window_opt_AAAA` — TPE converges | Set `study_name` in manifest |
| Run 4 | auto | 4,294,967,296 | Resume `window_opt_AAAA` — TPE fully informed | Set `study_name` in manifest |

### 3.4 Survivor Accumulation

Each run appends to `bidirectional_survivors_all.json`. Merge policy: best
per-seed `score` wins on conflict. The NPZ fed to Steps 2–6 is rebuilt from
the full accumulated set after every session. Per-run
`bidirectional_survivors.json` still written identically — no existing
contract broken.

---

## 4. Timing

Throughput basis: **2,082,140 seeds/sec** — verified S130 soak test.

| Config | Seeds/session | Time/trial | 50 trials | 100 trials |
|---|---|---|---|---|
| 2 sessions | 2,147,483,648 | 34.4 min | 28.6 hrs | 57.3 hrs |
| 4 sessions (recommended) | 1,073,741,824 | 17.2 min | 14.3 hrs | 28.6 hrs |

**Recommendation:** 4 sessions × 50 trials = ~14 hours per session.
Timeout set to 900 minutes provides ~50% buffer above 50-trial estimate.

---

## 5. Required Code Changes

Five files, seven surgical changes. Patch script: `apply_s145r1_progressive_sweep.py`

### 5.1 `window_optimizer_integration_final.py` — Accumulator

Replace existing NPZ conversion call (~line 1297) with accumulator block.

```python
# [S145-R1] SURVIVOR ACCUMULATOR
_accum_path = 'bidirectional_survivors_all.json'
_prior_survivors = json.load(open(_accum_path)) if os.path.exists(_accum_path) else []
_merged = {s['seed']: s for s in _prior_survivors}
for s in bidirectional_deduped:
    if s['seed'] not in _merged or \
       float(s.get('score', 0)) > float(_merged[s['seed']].get('score', 0)):
        _merged[s['seed']] = s
_merged_list = sorted(_merged.values(), key=lambda x: x['seed'])
json.dump(_merged_list, open(_accum_path, 'w'))
print(f"[S145-R1][ACCUMULATOR] {len(_merged_list):,} total survivors")
# NPZ conversion now uses accumulator
subprocess_run(["python3", "convert_survivors_to_binary.py", _accum_path], check=True)
```

### 5.2 `agent_manifests/window_optimizer.json` — Four Values

| Field | Location | From | To |
|---|---|---|---|
| `timeout_minutes` | `actions[0]` line 44 | `240` | `900` |
| `max_seeds` | `default_params` line 200 | `10000000` | `1073741824` |
| `window_trials` | `default_params` line 205 | `100` | `50` |
| `enable_pruning` | `default_params` line 208 | `false` | `true` |

### 5.3 `agents/watcher_agent.py` — Two Changes

**Change A** (~line 1407): Conditionalize fresh-study invariant (see Section 3.2)

**Change B** (line 2796): Step 1 timeout override
```python
# From:
step_timeout_overrides={0: 1, 1: 480, 5: 360}
# To:
step_timeout_overrides={0: 1, 1: 900, 5: 360}  # [S145-R1]
```

### 5.4 `.gitignore` — One Line

After line 44 (`!schema_*.json`):
```
!bidirectional_survivors_all.json   # [S145-R1] persistent cross-run survivor accumulator
```

---

## 6. Operational Procedure

### 6.1 Pre-Launch

```bash
# Apply patch script
cd ~/distributed_prng_analysis
python3 apply_s145r1_progressive_sweep.py

# Smoke test (100k seeds, 2 trials — verify accumulator fires)
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1

# Verify in log:
# [S145-R1][ACCUMULATOR] N total survivors
# ✅ Converted bidirectional_survivors_all.json to bidirectional_survivors_binary.npz
```

### 6.2 Run 1

```bash
# default_params.study_name = "" (fresh study)
nohup bash -c 'PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 0 --end-step 1 \
    > logs/sweep_r1.log 2>&1' &

# After completion — note study name from log:
grep "Optuna study:" logs/sweep_r1.log
# → optuna_studies/window_opt_XXXXXXXXXX.db

# Mandatory commit:
git add -f bidirectional_survivors_binary.npz bidirectional_survivors_all.json
git add apply_s145r1_progressive_sweep.py
git commit -m "feat(s145-r1): run 1 complete — seeds 0→1,073,741,824"
git push origin main && git push public main
```

### 6.3 Runs 2, 3, 4

Before each run, set `study_name` in `default_params`:
```json
"study_name": "window_opt_XXXXXXXXXX"
```

Launch identically to Run 1 (change log filename). Seed start advances
automatically. Optuna resumes from prior TPE state.

After each run: commit NPZ and accumulator to both remotes.

### 6.4 Post-Sweep Analysis (Required Before Any Sufficiency Claim)

See Section 7.

---

## 7. Evidence Needed Before Any Practical Sufficiency Claim

Per TB ruling, "0 → 2^32 is practically sufficient coverage for the CA ADM"
is a hypothesis requiring post-sweep evidence. The following analysis is
required before making any sufficiency claim:

**7.1 Yield Decay Analysis**
Plot `bidirectional_survivors / seeds_searched` per session. If yield drops
significantly toward zero in sessions 3–4, that supports the hypothesis that
the signal is concentrated in lower seed IDs. If yield remains roughly constant
across all four sessions, the signal is distributed and the hypothesis is weaker.

**7.2 Seed Distribution Analysis**
Plot the seed values of all accumulated survivors. If they cluster in specific
ranges (timestamp-consistent bands, powers of 2 ± offset), that is evidence
of a constrained initialization source. Uniform distribution across 0 → 2^32
would argue against practical sufficiency.

**7.3 Survivor Quality vs Seed Range**
Compare average per-seed `score` across sessions. If quality degrades in higher
seed ranges, the lower range is more signal-rich. If quality is consistent, the
full space contributes equally.

**7.4 Sufficiency Threshold**
A practical sufficiency claim requires: yield in sessions 3–4 is < 5% of
yield in session 1, AND survivor quality in sessions 3–4 is not materially
better than in sessions 1–2. Until both conditions are met, Step 1 remains
active and the sweep is characterized as progressive frontier expansion only.

---

## 8. Backward Compatibility

| Concern | Analysis | Verdict |
|---|---|---|
| `bidirectional_survivors.json` | Still written identically per run | ✅ Safe |
| `bidirectional_survivors_binary.npz` | Same path, same schema — accumulator feeds it | ✅ Safe |
| Steps 2–6 contract | All read NPZ only — unchanged | ✅ Safe |
| WATCHER freshness gate | Checks NPZ existence — unchanged | ✅ Safe |
| Short runs (50M seeds) | `max_seeds` overridable — unchanged behavior | ✅ Safe |
| Fresh-study default | No `study_name` set = fresh study as before | ✅ Safe |
| Step 1 normal operation | All changes opt-in via `study_name` and `max_seeds` | ✅ Safe |
| Remote workers | No disk writes on rigs — unaffected | ✅ Safe |
| Chapter 13 / autonomous loop | Reads `survivors_with_scores.json` — unaffected | ✅ Safe |

---

## 9. Summary

| File | Change | TB Status |
|---|---|---|
| `window_optimizer_integration_final.py` | Survivor accumulator, per-seed score merge | ✅ Required |
| `agent_manifests/window_optimizer.json` | 4 field corrections against live values | ✅ Required |
| `agents/watcher_agent.py` | Fresh-study invariant conditionalized | ✅ Required |
| `agents/watcher_agent.py` | Step 1 timeout 480 → 900 | ✅ Required |
| `.gitignore` | Accumulator JSON exception | ✅ Required |

**Patch script:** `apply_s145r1_progressive_sweep.py` — backups, anchored
replacements, line-count verification.

**This proposal does not claim:**
- Mathematical exhaustiveness for java_lcg
- Practical sufficiency for CA ADM (deferred to post-sweep analysis)
- Permanent retirement of Step 1

**This proposal delivers:**
- Permanent, accumulative survivor pool growing with every run
- Progressive Optuna learning across sessions
- Infrastructure for post-sweep yield analysis
- Additive value whether the sweep covers 1 session or 40

---

*Team Alpha — S145-R1 — 2026-03-15*
*TB Approval: Conditional — approved with mandatory edits applied*
