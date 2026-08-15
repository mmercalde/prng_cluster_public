# FALLBACK-PARITY REVIEW — PASS 1 OF 2 (VM101 baseline), 2026-08-15

**Verdict: pass 1 RUN. Fallback parity remains UNRESOLVED and is NOT converted to PASS.**

Beta ruled the S183/S184 fallback-parity line **not measured, not credited, and not to be silently
converted to PASS.** This document does not convert it. It runs the half of the review that is
runnable today, records the baseline pass 2 will diff against, and leaves the overall verdict open.

**Host:** VM101 `zeus-ubuntu-vm` (192.168.3.177), user `michael`, venv `~/venvs/torch`.
**Trigger:** phase boundary — D6 dry run #3 PASS (S184), immediately before attempt 6.
**Authority:** `CLAUDE.md` §5 (fallback-parity review, read-and-report ONLY) and §4 (env capture).

---

## 1. What the governance actually requires — checked, not assumed

### 1.1 Beta's §18 pre-launch conditions do NOT include fallback parity

The §18 battery is: launch-tree HEAD pin · phase-4 suite green · GPU truth gate 8/8 on each of the
three remote rigs · frozen eligible cohort 25 · `worker_pool_size = 25`. **No fallback-parity item
appears in it**, and none appears in `PHASE6_PREREQS.md` REV5's seven checklist items either
(searched: `docs/PHASE6_PREREQS.md`, `-i fallback` → one hit, line 352, about GPU allocation on a
bare-metal boot, not about parity).

**Therefore fallback parity is NOT an attempt-6 launch blocker.** It is a `CLAUDE.md` §5 obligation
triggered *"at each phase boundary / TB milestone"*, and D6 #3 is such a boundary. Not a blocker and
not optional are both true at once; conflating them is what produced two sessions of
`not re-measured`.

### 1.2 `CLAUDE.md` §5 — the two-pass structure, and why `.127` being down does not excuse pass 1

§5 is explicitly **two-pass, *because* Zeus runs one OS at a time** and 101 and `.127` are never up
simultaneously:

```
1. On 101 (up now):  record git rev-parse HEAD, pip freeze, run the current phase harness.
2. Separately, after booting .127: git fetch + commits-behind, diff its pip freeze
   against 101's, run the same harness after a pull.
3. Produce a parity report: code / env diffs / harness pass-fail, dated.
```

**Pass 1 names no `.127` operation at all.** Every one of its three actions is performed on 101, and
the "up now" parenthetical is the procedure telling you to run it precisely while the other target is
down. `.127` being down is the *designed condition* for pass 1, not an obstacle to it.

**So the non-applicability argument fails on pass 1 and succeeds only on pass 2.** Formally:

| pass | applicable today? | why |
|---|---|---|
| **1** (101 baseline) | **YES — RUN, and it is recorded below** | needs only 101, which is up |
| **2** (`.127` comparison) | **NO — NON-APPLICABLE while `.127` is down** | requires booting `.127`, which requires taking Zeus out of Proxmox and VM101 down with it |

**Boot-state evidence, measured 2026-08-15 from VM101:**

```
192.168.3.127  DOWN (no ICMP reply)     <- bare-metal Zeus, the frozen fallback
192.168.3.128  UP                       <- pzeus Proxmox host, under which VM101 runs
```

This is the expected boot state, not a defect (`CLAUDE.md` §3: every machine is a boot-selector).
Pass 2 is deferred **on an availability ground, stated formally**, not skipped.

---

## 2. PASS 1 — the measured baseline

### 2.1 Code layer

```
local HEAD    e8e755681ef73fdeaaa602470e0bf0b2623744fd   (branch main)
origin/main   e8e755681ef73fdeaaa602470e0bf0b2623744fd   private  mmercalde/prng_cluster_project
public/main   e8e755681ef73fdeaaa602470e0bf0b2623744fd   mirror   mmercalde/prng_cluster_public
working tree  git status --porcelain EMPTY
```

Read via `git ls-remote` — no `fetch`, no ref mutation, no tree change. **Both remotes are in sync
with local HEAD**, which is the property the code layer needs: `.127` is one `git pull` from current
no matter how far 101 pivots, and there is nothing unpushed for it to be unable to reach.

`e8e7556` landed after the D6 #3 evidence was recorded and is **docs-only** — `git diff --name-only
3218718..HEAD` yields one `.md` and no `.py`. **The dry-run evidence taken at `3218718` therefore
still describes the current production code**, and the §18 launch-tree pin should be read against
production-code identity, not against a HEAD that a documentation commit moves.

**Code-layer status: `code=[current]`.** Pass 2 will report `behind N` from `.127`'s own `git fetch`.

### 2.2 Environment layer

```
Python        3.10.12
venv          /home/michael/venvs/torch
interpreter   /home/michael/venvs/torch/bin/python3
packages      263
freeze sha256 b259c57fdd6d5a830d88a756de448203fc84dcf9e784b9822ababc21d6f4e473
```

Load-bearing versions for this phase:

```
torch==2.5.1+cu121      cupy-cuda12x==13.5.1    optuna==4.4.0     numpy==1.22.0
scipy==1.11.4           scikit-learn==1.7.1     pandas==1.5.3
xgboost==3.0.4          lightgbm==4.6.0
torchvision==0.20.1+cu121                       torchaudio==2.5.1+cu121
```

**The full freeze is committed at `docs/FALLBACK_PARITY_PASS1_20260815_pipfreeze.txt`, and that
placement is load-bearing.** Pass 2 runs on `.127`, which can see **only committed files**. The
working copy under `logs/fallback_parity/` is gitignored (`.gitignore:62`), so a baseline left only
there would be **invisible to the machine that has to diff against it** — the review would have
looked complete on 101 and been unrunnable on `.127`. Verified with `git check-ignore` both ways.

**Env-layer status: `env=[baseline captured on 101; parity UNKNOWN until pass 2]`.** No claim is made
about `.127`'s environment. Nothing was compared, because there is nothing up to compare against.

### 2.3 Harness

```
tests/test_s172_phase4_coordinator.py    63/63 checks green    rc=0
```

This is the suite Beta's §18 names, run on 101 at `e8e7556` in `~/venvs/torch`. It is the harness
pass 2 must re-run on `.127` after a pull, so the two results are comparable by construction.

Side conditions verified after the run: **port 5700 never bound** (the suite passes
`miner_port=5700` as a captured config value with `_serve=_capture`, so the serve path is stubbed and
nothing listens), no worker/pipeline process left behind, porcelain still clean.

**Harness status: `harness=[PASS 63/63 on 101]`.**

---

## 3. FINDING — the committed environment artifact `CLAUDE.md` §4 requires does not exist

Reported, not fixed. `CLAUDE.md` §4 states VM101 is *"evolving (it's the dev box) but ENV-CAPTURED.
Every dependency change gets committed to a reproducible artifact (`requirements`/setup), so 'frozen'
and 'current' are both one command away — never a hand-rebuilt box."*

**Measured 2026-08-15: no such tracked artifact exists.**

```
git ls-files | grep -iE 'requirements|setup\.py|setup\.cfg|pyproject|environment\.ya?ml|constraints'
    -> only docs/FLEET_STATE_REQUIREMENTS_v1.md (a substring match, not an env artifact)
find . -maxdepth 2 -iname 'requirements*' -o -iname 'pyproject*'
    -> no results, tracked or untracked
git check-ignore -v requirements.txt
    -> not matched by .gitignore, so absence is genuine and not a visibility artifact
```

**Why this matters more than the missing pass 2.** §5 is explicit that the code layer is
self-healing via git while the environment layer *"rots silently if 101's dep changes go uncaptured —
this is the layer that actually breaks a fallback."* The uncaptured layer is exactly the one with no
artifact. Today's freeze is a **point-in-time observation**, not the reproducible artifact §4 asks
for: re-provisioning from it is not a defined operation, and nothing detects the next dep change.

**No remediation is performed here, deliberately.** §5: *"Remediation is NOT part of the review. Do
not `pip install` / pull-to-patch `.127` to 'fix' drift — that is an unreproducible hand-modification
(the frozen lesson's anti-pattern). If the review finds a missing dep, capture it in the committed
`requirements`/setup artifact and re-provision from that."* Creating that artifact is a scoped change
with its own review, and it is **not** an attempt-6 blocker. Recommended as a separate item.

---

## 4. Pass 2 — deferred, with the exact procedure

To be run **after booting Zeus to its bare-metal `.127` target**, which takes VM101 down. Not to be
scheduled against attempt 6.

```bash
# on .127, after boot
cd /home/michael/distributed_prng_analysis
git fetch origin && git rev-list --count HEAD..origin/main     # commits behind -> code=[behind N]
git pull
source <the .127 venv>
pip freeze > /tmp/pass2_env.txt
diff <(sed -n '/^--- pip freeze ---$/,$p' docs/FALLBACK_PARITY_PASS1_20260815_pipfreeze.txt \
        | tail -n +2) /tmp/pass2_env.txt                        # env diffs
python3 -u tests/test_s172_phase4_coordinator.py                # same harness, expect 63/63
```

Then complete §5 step 3: a dated parity report carrying code ✅/behind-N, the env diff, and harness
pass/fail. **Until that runs, fallback parity has no verdict.**

---

## 5. STATUS LINE

```
fallback parity: code=[current — HEAD==origin/main==public/main==e8e7556, porcelain empty],
                 env=[101 baseline captured, 263 pkgs, freeze sha256 b259c57f…e473;
                      PARITY UNKNOWN — .127 not compared],
                 harness=[phase-4 63/63 PASS on 101],
                 pass 2=[NON-APPLICABLE 2026-08-15 — .127 DOWN by boot state (.128 Proxmox UP);
                         DEFERRED, not waived, not credited]
```

**This is not a PASS and must not be cited as one.** Pass 1 is a baseline, not a verdict: a review
that has compared nothing cannot have found parity. The one substantive finding — the missing §4
environment artifact — is a report, not a repair.
