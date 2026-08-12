#!/usr/bin/env python3
"""GATE-12 CLEAN-TREE ADMISSION GATE — a harness rule, not a D3.5 policy change.

WHAT THIS IS
------------
Gate-12 attempt 3 (`distributed_config_t1_d606edbe`) ran four stages, 128/128
stripes over the full `[0, 2^31)` domain, satisfied the saturation verdict, and
was then REFUSED at publication:

    utils.run_finalizer.RunParameterError: repository_tree_clean is False

Two hours of fleet compute were dispatched from a repository state that
publication was predetermined to reject. Three untracked entries —
`miner_ledger.db-shm`, `miner_ledger.db-wal`,
`optimal_window_config.json.stale_1786149572` — were PRINTED into the launch
evidence block by `gate12_launch.sh` (`--- TREE STATE ---`) and never TESTED.

D3.5 is correct and is NOT weakened. `repository_tree_clean` is a certified
prerequisite (`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md:450`, suite
gate **F37** at `tests/test_s172_phase5_d3_5_finalizer.py:1379`). The defect is
upstream, in admission. This module moves the same refusal to the front of the
run, where it costs zero GPU-seconds.

WHY IT REUSES THE CERTIFIED PRODUCER
------------------------------------
`_repository_state` is imported from `window_optimizer_integration_final`, not
reimplemented. That function is the ONE producer of the boolean the finalizer
receives: `window_optimizer_integration_final.py:2972` computes it and `:2992`
passes it as `repository_tree_clean=` into `_finalize_run_d3_5`. A second
`git status --porcelain` implementation would be a SECOND PREDICATE, and a
second predicate that can disagree with the first is this defect recurring in a
new costume. There is one predicate; this module only decides what to DO with
it, earlier.

THE RULE — fail-close on anything that is not a demonstrably clean tree:

    _repository_state()[1] is True     -> PROCEED
    _repository_state()[1] is False    -> REFUSE, naming the offending entries
    _repository_state() raises         -> REFUSE  (UNAVAILABLE, never "clean")

The third row is VIR-5: a predicate that could not be evaluated is not a
predicate that passed. `_repository_state` runs git with `check=True`, so a git
failure surfaces as an exception rather than as an empty — and therefore
"clean" — porcelain string.

THE ENTRY LISTING IS DIAGNOSTIC, NOT THE PREDICATE
--------------------------------------------------
Beta requires the refusal to name the offending entries. `_repository_state`
returns only a boolean, so the listing comes from a separate, clearly-labelled
`git status --porcelain` read whose ONLY consumer is the human-readable message.
It never decides anything: if that read fails, or races to an empty result, the
refusal still stands on the producer's boolean. `decide()` takes the boolean and
nothing else, which is what makes that structural rather than a promise.

TWO PHASES, ONE PREDICATE
-------------------------
`--phase admission`    the pre-launch gate: before clean slate, before the
                       sampler is armed, before any coordinator process exists.
`--phase pre-dispatch` the last assertion before sampler/coordinator creation.
                       Testing once at admission is insufficient: attempt 3's
                       own harness could dirty the tree AFTER admission by
                       renaming `optimal_window_config.json` to a name no Git
                       ignore rule covers. Both phases evaluate the identical
                       predicate; only the wording differs.

Exit status: 0 = proceed, 1 = REFUSE. The caller aborts on non-zero.
"""

import argparse
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import window_optimizer_integration_final as WOI                    # noqa: E402

# The certified producer surface, imported rather than copied. Named here so the
# dependency is explicit and a rename cannot silently fall back to a local copy.
_repository_state = WOI._repository_state

EXIT_PROCEED = 0
EXIT_REFUSE = 1

# The enforcement site this gate is a forward projection of. Quoted in the
# refusal so the operator sees WHY the launch stopped, not merely THAT it did.
_D3_5_ENFORCEMENT = "utils/run_finalizer.py:1589"
_D3_5_PRODUCER = "window_optimizer_integration_final.py:2972 -> :2992"

_PHASES = {
    "admission": (
        "CLEAN-TREE ADMISSION GATE",
        "This runs BEFORE the clean slate, BEFORE the sampler is armed and "
        "BEFORE any\ncoordinator process is created, so a refusal leaves the "
        "box exactly as it found\nit and costs ZERO GPU-seconds.",
    ),
    "pre-dispatch": (
        "CLEAN-TREE PRE-DISPATCH ASSERTION",
        "This runs AFTER launch preparation and immediately BEFORE "
        "sampler/coordinator\ncreation. Admission proved the tree was clean; "
        "this proves launch preparation\nPRESERVED it. No operation between "
        "admission and fleet dispatch may create a\nGit-visible dirty state.",
    ),
}


def evaluate(repo_root):
    """Return (clean, commit, error). Fail-closed: an exception is NOT clean.

    `clean` is None when the predicate could not be evaluated at all. That is a
    distinct outcome from False and is rendered as UNAVAILABLE — "the check
    could not run" and "the check ran and found a dirty tree" are different
    facts, and both refuse.
    """
    try:
        commit, clean = _repository_state(repo_root=repo_root)
    except Exception as e:                                          # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"
    if not isinstance(clean, bool):
        # The finalizer's own argument contract rejects a non-bool
        # (`run_finalizer.py:1584`). Refuse here for the same reason.
        return None, commit, f"producer returned non-bool {clean!r}"
    return clean, commit, None


def decide(clean):
    """The whole decision, and it reads exactly one input.

    Deliberately takes the boolean alone — not the repo root, not the porcelain
    listing. A future edit that wants the entry list to influence the verdict
    would have to change this signature, which is the point.
    """
    return EXIT_PROCEED if clean is True else EXIT_REFUSE


def diagnostic_entries(repo_root):
    """Human-readable listing of WHAT makes the tree dirty. NOT the predicate.

    Returns (entries, error). Its only consumer is the refusal message. A
    failure here degrades the message, never the verdict.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=60)
    except Exception as e:                                          # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    if proc.returncode != 0:
        return None, f"git exited {proc.returncode}: {proc.stderr.strip()}"
    return [ln for ln in proc.stdout.splitlines() if ln.strip()], None


def render_refusal(repo_root, clean, commit, error, phase):
    """The refusal text. Names the entries; states that publication rejects."""
    out = []
    if clean is False:
        out.append(f"GATE-12 CLEAN-TREE : REFUSED — "
                   f"repository_tree_clean is False at {commit}")
        entries, derr = diagnostic_entries(repo_root)
        out.append("")
        if entries:
            out.append(f"Offending entries ({len(entries)}) — "
                       f"`git status --porcelain`, diagnostic listing only:")
            for ln in entries:
                out.append(f"  * {ln}")
        elif entries == []:
            # The listing raced the predicate. The predicate still governs.
            out.append("Offending entries: the diagnostic listing came back "
                       "EMPTY while the authoritative")
            out.append("predicate reported dirty. The predicate governs; this "
                       "gate still refuses.")
        else:
            out.append(f"Offending entries: UNAVAILABLE ({derr}). The "
                       f"authoritative predicate reported")
            out.append("dirty; this gate still refuses.")
    else:
        out.append("GATE-12 CLEAN-TREE : REFUSED — the clean-tree predicate "
                   "could not be evaluated")
        out.append(f"                     UNAVAILABLE: {error}")
        out.append("")
        out.append("An unobservable tree is not a clean tree (VIR-5).")

    out.append("")
    out.append(f"PUBLICATION WOULD REJECT THIS STATE. D3.5 raises "
               f"RunParameterError at")
    out.append(f"{_D3_5_ENFORCEMENT} when repository_tree_clean is False — the "
               f"same boolean, from the")
    out.append(f"same producer ({_D3_5_PRODUCER}). Attempt 3 reached that "
               f"refusal AFTER four")
    out.append("stages and 128/128 stripes of fleet compute. This gate reaches "
               "it before any.")
    out.append("")
    if phase == "admission":
        out.append("ABORTING BEFORE THE CLEAN SLATE, BEFORE THE SAMPLER IS "
                   "ARMED AND BEFORE ANY")
        out.append("COORDINATOR PROCESS IS CREATED. Zero GPU-seconds consumed; "
                   "retry once the tree")
        out.append("is committed or the residue is removed.")
    else:
        out.append("ABORTING BEFORE SAMPLER/COORDINATOR CREATION. Admission "
                   "passed, so launch")
        out.append("PREPARATION dirtied the tree — that is a harness defect, "
                   "not an operator one.")
        out.append("Report it; do not retry until the preparation step is "
                   "fixed.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", default=_REPO_ROOT,
                    help="repository to evaluate (default: this checkout)")
    ap.add_argument("--phase", default="admission", choices=sorted(_PHASES),
                    help="which of the two evaluations this is")
    args = ap.parse_args(argv)

    title, placement = _PHASES[args.phase]
    print("=" * 70, flush=True)
    print(f"GATE-12 {title} — publication's predicate, evaluated early",
          flush=True)
    print("=" * 70, flush=True)
    print(f"repo root         : {args.repo_root}", flush=True)
    print(f"predicate         : window_optimizer_integration_final."
          f"_repository_state", flush=True)
    print(f"                    (the D3.5 producer itself; not reimplemented "
          f"here)", flush=True)
    print(f"enforced later at : {_D3_5_ENFORCEMENT} (F37)", flush=True)
    print(placement, flush=True)
    print("", flush=True)

    clean, commit, error = evaluate(args.repo_root)
    rc = decide(clean)

    if rc == EXIT_PROCEED:
        print(f"HEAD              : {commit}", flush=True)
        print(f"git status --porcelain : empty", flush=True)
        print("", flush=True)
        print(f"GATE-12 CLEAN-TREE : PASS — repository_tree_clean is True. "
              f"Launch may proceed.", flush=True)
        return rc

    print(render_refusal(args.repo_root, clean, commit, error, args.phase),
          flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
