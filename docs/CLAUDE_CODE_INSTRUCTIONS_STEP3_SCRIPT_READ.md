# CLAUDE_CODE_INSTRUCTIONS_STEP3_SCRIPT_READ.md — REV1

**Read `run_step3_full_scoring.sh` and report what it does. READ-ONLY. No fix authorised.**

**Base:** HEAD `0c0e603` or later. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`,
**run from `/home/michael/distributed_prng_analysis`.**

**Deliverable:** your final message only. **Create NO file.** *(The tree is clean and the Phase-7
soak launches tomorrow — do not dirty it for a one-file read.)*

---

## 0. Why this specific file

**`STEP_SCRIPTS[3] = "run_step3_full_scoring.sh"`**, while `full_scoring.json`'s actions name
`generate_full_scoring_jobs.py`, `full_scoring_worker.py` and `aggregate_scoring_results.py` —
**none of which is the shell script.** Recorded at `PROJECT_FILE_CATALOG.md` §5.1 item 2 and
`PIPELINE_BEHAVIOUR_MODEL.md` §18 obs 3, both as a **structural fact, not diagnosed.**

**Its Step-2 sibling has the identical shape, and that one is a soak hazard.**
`run_scorer_meta_optimizer.sh`:

- **`:87`** invokes `convert_survivors_to_binary.py` — **TB-prohibited while D3.0-B is open**;
- **`:97`** `mv "$TMP_NPZ" bidirectional_survivors_binary.npz` — **a D3.5 finalizer-owned
  compatibility symlink.** Replacing it with a regular file makes `run_finalizer.py:1406` raise
  `PublicationError` — **at publication, hours into a run**;
- **`:120`** then `scp`s the regular file to the rigs.

That is why the Phase-7 soak is confined to `--start-step 1 --end-step 1`.

**Nobody has opened the Step-3 script.** This brief opens it.

**No outcome is expected. "It is clean" is a real and useful result.**

## 1. What to establish

Read `run_step3_full_scoring.sh` in full, then answer each with a `file:line`:

1. **Does it invoke `convert_survivors_to_binary.py`, or any legacy writer?** *(TB prohibits
   invoking the converter while D3.0-B is open — a call site is a finding even if unreached.)*
2. **Does it `mv`, `cp`, `>`, `tee`, `ln` or otherwise write to any path that is a D3.5
   finalizer-owned alias?** The owned names are `bidirectional_survivors_all.npz` and
   `bidirectional_survivors_binary.npz` at repo root. **Check both, and check for indirection
   through a variable** — the Step-2 script writes `$SURVIVORS`, defined ~30 lines earlier.
3. **Does it write anything else into the repo root** that would dirty the tree and trip item 5's
   clean-tree preflight at finalization?
4. **Does it `scp`, `ssh` or otherwise touch the rigs or CT100 workers?**
5. **What does it actually invoke** — and does that match `full_scoring.json`'s three named
   actions, or is it a separate path?
6. **Is it reachable in a run confined to `--start-step 1 --end-step 1`?** Trace it: `STEP_SCRIPTS`
   → the loop guard at `agents/watcher_agent.py:2365` → `run_step`.
7. **Does `chapter_13_triggers.py`'s independent `STEP_SCRIPTS` map (`:646-653`) reach it?** That
   map is **not bounded by `--end-step`**.

## 2. Method

**Read the file. Do not infer from its name or from the Step-2 script.** They share a shape; that
is a reason to check, not a reason to assume the same content.

**If it does something the Step-2 script does not, say so.** If it is materially safer, say that
too — plainly, without hedging.

## 3. What NOT to do

- **Do not fix, patch or edit anything.** Not the script, not the manifests, not `STEP_SCRIPTS`.
- **Do not execute the script**, or any part of it, or anything it calls.
- **Do not invoke `convert_survivors_to_binary.py`** — TB prohibits it while D3.0-B is open.
- Do not run the pipeline or WATCHER. Do not touch `miner/`. Do not contact a rig.
- **Do not create a file. Do not commit or push.**

## 4. Report — final message only

Numbered answers to §1.1–§1.7, each with `file:line`. Then **one plain sentence**:

> **"Does this script pose the same hazard as its Step-2 sibling — yes, no, or partly?"**

Then anything you noticed that §1 did not ask about, labelled as an observation.

**Then STOP.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** state the file's line count and quote the lines your answers rest on.
  **Answers without quoted lines are unverifiable.**
- **clean control:** `NOT_APPLICABLE` — a read, not a detector. **Write `NOT_APPLICABLE`, never
  `PASS`.**
- **fault-injection control:** `NOT_APPLICABLE` — same reason.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **`PASS` means the seven
  questions were answered from the file — NOT that the script is safe.** State that distinction.
- **audit claim scope:** repo-scoped at the stated HEAD. **Static read only — nothing was executed,
  so runtime behaviour is inferred from source and must be labelled as such.**
- **searched surfaces:** the script, `agent_manifests/full_scoring.json`, `agents/watcher_agent.py`,
  `chapter_13_triggers.py`, and anything the script sources or calls.
- **unavailable surfaces:** anything the script invokes that you did not open — **name it rather
  than assuming it is harmless.**
