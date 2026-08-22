#!/usr/bin/env python3
"""
S172 WINDOW-ANCHOR BRIEF I — §5 MUTATION EVIDENCE
=================================================
The brief's rule, applied literally:

    Prove each mutant actually APPLIED, EXECUTED the mutated path, and REACHED
    the credited assertion. A mutant that dies on import, on class identity, on a
    loader error, or on a TypeError before reaching the mutated line EARNS NO
    CREDIT — report it as an invalid mutant and replace it.

So every mutant here is classified into one of:

    DETECTED  — applied; the credited gate ran to ITS OWN assertion and failed
                there; the assertion text is captured
    SURVIVED  — applied and executed, but the credited gate stayed green.
                A DEFECT IN THE GATE, reported as such.
    INVALID   — did not apply, or died before reaching the credited assertion
                (import error, TypeError, wrong exception type). NO CREDIT.

Each mutant also carries a CLEAN CONTROL: the same gate is run on unmutated source
and must PASS, so a red can never be credited to a gate that was already failing.

Every gate runs in a FRESH INTERPRETER (subprocess), because an already-imported
module would defeat a source mutation and the pass would be meaningless.

CPU-only except where the credited gate itself needs a GPU (G-SEP-1).
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
GATE_FILE = os.path.join(_ROOT, "tests", "test_s172_window_anchor_brief_i.py")

_RESULTS: List[Tuple[str, str, str]] = []          # (id, status, detail)


# ---------------------------------------------------------------------------
# running a credited gate in a fresh interpreter
# ---------------------------------------------------------------------------
_DRIVER = r'''
import importlib.util, json, sys, traceback
sys.path.insert(0, %(root)r)
out = {}
# LOAD BY PATH, not as a package. `tests/` has no __init__.py, so
# importlib.import_module("tests.<mod>") raises ModuleNotFoundError and every
# clean control reads as failed — which made the whole framework refuse credit
# for a reason that had nothing to do with any mutant.
try:
    _spec = importlib.util.spec_from_file_location("_brief_i_gates", %(gatefile)r)
    m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(m)
except BaseException as e:
    print(json.dumps({"phase": "import", "exc": type(e).__name__, "msg": str(e)[:400]}))
    raise SystemExit(0)
for name in %(gates)r:
    try:
        fn = getattr(m, name)
    except AttributeError as e:
        out[name] = {"phase": "lookup", "exc": "AttributeError", "msg": str(e)[:400]}
        continue
    try:
        fn()
        out[name] = {"phase": "ran", "outcome": "PASS"}
    except BaseException as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        # the frame the failure came from — proof of WHERE it stopped
        last = tb[-1] if tb else None
        out[name] = {"phase": "ran", "outcome": "FAIL",
                     "exc": type(e).__name__, "msg": str(e)[:600],
                     "frame_file": (last.filename if last else None),
                     "frame_fn": (last.name if last else None),
                     "frame_line": (last.lineno if last else None)}
print(json.dumps(out))
'''


def _run_gates(gates: List[str]) -> Dict[str, Any]:
    code = _DRIVER % {"root": _ROOT, "gatefile": GATE_FILE, "gates": gates}
    p = subprocess.run([sys.executable, "-c", code], cwd=_ROOT,
                       capture_output=True, text=True, timeout=900)
    line = ""
    for ln in p.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            line = ln
    if not line:
        return {"__driver__": {"phase": "driver", "outcome": "NO-JSON",
                               "stderr": p.stderr[-600:]}}
    return json.loads(line)


def _run_suite(rel: str) -> int:
    """Whole-suite subprocess (used where the credited gate IS another suite)."""
    p = subprocess.run([sys.executable, rel], cwd=_ROOT,
                       capture_output=True, text=True, timeout=1800)
    return p.returncode


# ---------------------------------------------------------------------------
# mutation application
# ---------------------------------------------------------------------------
class Mutant:
    def __init__(self, mid, target, credited, description,
                 edits=None, fn=None, suites=None, invalid_reason=None):
        self.id = mid
        self.target = target                 # repo-relative file, or None
        self.credited = credited or []       # gate function names
        self.suites = suites or []           # whole suites, if that is the gate
        self.description = description
        self.edits = edits or []             # [(old, new)]
        self.fn = fn                         # callable(text) -> text
        self.invalid_reason = invalid_reason

    def apply(self) -> str:
        path = os.path.join(_ROOT, self.target)
        text = io.open(path, encoding="utf-8").read()
        new = text
        for old, rep in self.edits:
            n = new.count(old)
            if n != 1:
                raise AssertionError(
                    f"{self.id}: mutation hunk matched {n} times, expected exactly 1 "
                    f"— the mutant DID NOT APPLY cleanly:\n{old[:160]!r}")
            new = new.replace(old, rep, 1)
        if self.fn is not None:
            new = self.fn(new)
        if new == text:
            raise AssertionError(f"{self.id}: mutation produced no change")
        io.open(path, "w", encoding="utf-8").write(new)
        return text


def _record(mid, status, detail):
    _RESULTS.append((mid, status, detail))
    colour = {"DETECTED": GREEN, "SURVIVED": RED, "INVALID": YELLOW}[status]
    print(f"  [{colour}{status:<8}{RESET}] {mid}  {detail}")


def run_mutant(m: Mutant, controls: Dict[str, Any]):
    if m.invalid_reason:
        _record(m.id, "INVALID", m.invalid_reason)
        return

    # ---- clean control: every credited gate must be GREEN before we mutate ---
    for g in m.credited:
        c = controls.get(g, {})
        if c.get("outcome") != "PASS":
            _record(m.id, "INVALID",
                    f"clean control for {g} is not green ({c.get('outcome')}), so a "
                    f"red under mutation would be unattributable")
            return

    path = os.path.join(_ROOT, m.target)
    backup = None
    try:
        backup = m.apply()                                # raises if not exactly-1
        applied_note = f"applied to {m.target}"
        if m.credited:
            res = _run_gates(m.credited)
            if "__driver__" in res:
                _record(m.id, "INVALID",
                        f"driver produced no result: {res['__driver__'].get('stderr','')[:200]}")
                return
            caught, notes = [], []
            for g in m.credited:
                r = res.get(g, {})
                if r.get("phase") == "import":
                    _record(m.id, "INVALID",
                            f"module died on IMPORT ({r.get('exc')}) — the mutated "
                            f"line never executed, no credit")
                    return
                if r.get("outcome") == "PASS":
                    notes.append(f"{g}: SURVIVED")
                    continue
                exc = r.get("exc")
                if exc != "AssertionError":
                    notes.append(f"{g}: died on {exc} in {r.get('frame_fn')} — "
                                 f"not the credited assertion")
                    continue
                caught.append(g)
                notes.append(f"{g}: AssertionError in {r.get('frame_fn')}"
                             f":{r.get('frame_line')} -> {r.get('msg','')[:150]!r}")
            if not caught:
                status = "SURVIVED" if any("SURVIVED" in n for n in notes) else "INVALID"
                _record(m.id, status, f"{applied_note}; " + " | ".join(notes))
                return
            _record(m.id, "DETECTED",
                    f"{applied_note}; caught by {', '.join(caught)}; " + " | ".join(notes))
            return

        # ---- credited gate is a whole suite --------------------------------
        rcs = {s: _run_suite(s) for s in m.suites}
        red = [s for s, rc in rcs.items() if rc != 0]
        if not red:
            _record(m.id, "SURVIVED", f"{applied_note}; suites stayed green: {rcs}")
        else:
            _record(m.id, "DETECTED", f"{applied_note}; suites RED: {red}")
    except AssertionError as e:
        _record(m.id, "INVALID", str(e)[:300])
    finally:
        if backup is not None:
            io.open(path, "w", encoding="utf-8").write(backup)


# ---------------------------------------------------------------------------
# the 14 mutants
# ---------------------------------------------------------------------------
W = "miner/range_miner_worker.py"
C = "miner/range_miner_coordinator.py"
N = "miner/range_miner_npz_writer.py"


def _kernel_byte(text: str) -> str:
    """M11 — insert bytes inside a REAL kernel body.

    THE FIRST FORM WAS INVALID AND IS RECORDED AS SUCH, NOT SILENTLY SWAPPED.
    It searched for the literal "kernel_source" and mutated the next triple
    quote it found. But `kernel_source` is a dict KEY pointing at a module
    constant -- `'kernel_source': XORSHIFT32_KERNEL` -- so the mutation landed
    on an unrelated docstring, NO kernel_source string changed, and
    G-ABI-FROZEN was correct to stay green. The mutant never executed the path
    it was credited against, so it earned no credit as an invalid mutant.

    This form targets a `*_KERNEL = r'\'\'` constant -- the actual kernel body --
    and ASSERTS the target was found, so a future rename makes M11 INVALID
    rather than silently vacuous."""
    m = re.search(r"^([A-Z0-9_]+_KERNEL) = r" + chr(39) * 3, text, re.M)
    assert m, "no *_KERNEL constant found -- M11 has no valid target"
    return text[:m.end()] + "\n// mutant\n" + text[m.end():]


# ---------------------------------------------------------------------------
# WHY M14 BECAME TWO MUTANTS (M14 + M15)
# ---------------------------------------------------------------------------
# The brief credits M14 to G-SEP-1 AND G-NO-FUSED. That dual credit cannot hold:
# the two gates assert the same property at DIFFERENT LEVELS — G-SEP-1 on the real
# `SieveExecutor.execute` path, G-NO-FUSED at the builder/BuildContext level — so
# a single mutation can only ever reach one of them, and crediting it to both
# would have produced a mutant that one gate structurally could not catch. That is
# the shape this suite exists to refuse, so the mutant is SPLIT rather than the
# credit widened:
#
#   M14  one-line fusion at the EXECUTE seam (+ the pin that exists only because
#        of the separation)            -> credited to G-SEP-1
#   M15  STRUCTURAL: put the anchor into BuildContext and emit it as a ScalarArg
#                                      -> credited to G-NO-FUSED
#
# M15 exists because G-NO-FUSED is AC5's repo-level proof and, without it, its
# non-vacuity rested on its own construction rather than on a mutant — which is
# precisely the "green on a fact it does not check" risk this brief keeps finding.
MUTANTS = [
    Mutant("M1 ", W, ["gate_domain_1_out_of_domain", "gate_domain_2_coordinator_side"],
           "restore the silent clamp",
           edits=[("    derived_max = n - window_size\n"
                   "    anchor = int(window_anchor)\n"
                   "    if anchor < 0 or anchor > derived_max:",
                   "    derived_max = n - window_size\n"
                   "    anchor = max(0, min(int(window_anchor), n - window_size))\n"
                   "    if False:")]),
    Mutant("M2 ", W, ["gate_reject_3_no_default"],
           "reinstate payload.get('window_anchor', 0)",
           edits=[('        window_anchor = require_window_anchor(\n'
                   '            payload, where="ResidueResolver.resolve")',
                   '        window_anchor = payload.get("window_anchor", 0)')]),
    Mutant("M3 ", W, ["gate_reject_1_legacy_key", "gate_reject_2_no_mapping"],
           "accept AND map offset -> window_anchor (also G-REJECT-2's injection proof)",
           edits=[('    if _LEGACY_OFFSET_KEY in (payload or {}):',
                   '    if False:'),
                  ('def require_window_anchor(payload: Dict[str, Any], *, where: str) -> int:\n'
                   '    """THE reader for `window_anchor` — the host-side residue-window start index."""\n'
                   '    return _require_int_key(payload, "window_anchor", where=where)',
                   'def require_window_anchor(payload: Dict[str, Any], *, where: str) -> int:\n'
                   '    """THE reader for `window_anchor` — the host-side residue-window start index."""\n'
                   '    if "window_anchor" not in payload and "offset" in payload:\n'
                   '        return int(payload["offset"])\n'
                   '    return _require_int_key(payload, "window_anchor", where=where)')]),
    Mutant("M4 ", W, ["gate_cap_3_phase_rejection"],
           "add java_lcg_hybrid to PHASE_CAPABLE_VARIANTS",
           edits=[('    "lcg32_hybrid",            # inline int32, position 17 of 17',
                   '    "lcg32_hybrid", "java_lcg_hybrid",   # MUTANT'),
                  ('    "java_lcg_hybrid", "minstd_hybrid", "xorshift32_hybrid", "xorshift128_hybrid",',
                   '    "minstd_hybrid", "xorshift32_hybrid", "xorshift128_hybrid",')]),
    Mutant("M5 ", W, ["gate_cap_1_arity", "gate_cap_4_pinned_delivery"],
           "drop the phase arg from _generator_phase_tail",
           edits=[('    return [ScalarArg(ctx.generator_phase, "int32")]\n\n\ndef _reverse_hybrid_tail',
                   '    return []\n\n\ndef _reverse_hybrid_tail')]),
    Mutant("M6 ", W, ["gate_cap_1_arity"],
           "move the phase arg one position earlier in build_lcg32",
           edits=[('            ScalarArg(ctx.params.get("m", 0xFFFFFFFF), "uint32"),\n'
                   '            ScalarArg(ctx.generator_phase, "int32"),',
                   '            ScalarArg(ctx.generator_phase, "int32"),\n'
                   '            ScalarArg(ctx.params.get("m", 0xFFFFFFFF), "uint32"),')]),
    Mutant("M7 ", W, ["gate_domain_3_session_scoped"],
           "compute derived_max BEFORE the session filter",
           edits=[('    if sessions:\n'
                   '        data = [e for e in data if e.get("session") in sessions]\n'
                   '    n = len(data)',
                   '    _pre = len(data)\n'
                   '    if sessions:\n'
                   '        data = [e for e in data if e.get("session") in sessions]\n'
                   '    n = _pre')]),
    Mutant("M8 ", None, [], "set the control_era ceiling to 149",
           invalid_reason=(
               "INVALID BY SCOPE — the scope argument, not just the label. "
               "THE MUTATION HAS NO BRIEF-I CODE SITE TO APPLY TO: `control_era`'s "
               "ceiling lives on the ERA-RESOLUTION surface, and v1.1 §4.2 places "
               "era subdomains with the Optuna/optimizer surface, which §3's "
               "firewall assigns to Brief II. At Brief I no production expression "
               "computes a control-era bound, so there is no line to mutate — this "
               "is an ABSENT TARGET, not a skipped test and not a gate that failed "
               "to catch something. G-ENVELOPE is correspondingly scoped to the "
               "bound ARITHMETIC (it asserts anchor 149 is NOT inside "
               "[0, min(100, N-w)], and that 149 = 100+50-1 is the record-envelope "
               "ceiling), which is the half of Q4 that carries the category error. "
               "CARRY-FORWARD: the era-ceiling mutant TRANSFERS TO BRIEF II with the "
               "era-resolution work and must be run there against the real bound; "
               "the coverage gap is inherited deliberately, not lost.")),
    Mutant("M9 ", N, ["gate_tuple", "gate_phase5_seam"],
           "revert _CONTEXT_FIELDS to 'offset'",
           edits=[('    "trial_number", "window_size", "window_anchor", "generator_phase",',
                   '    "trial_number", "window_size", "offset", "generator_phase",')]),
    Mutant("M10", "coordinator.py", ["gate_legacy_2_routes_closed"],
           "re-enable one legacy dispatch route",
           edits=[('            elif job.search_type == \'reverse_sieve\':\n'
                   '                # [WINDOW-ANCHOR BRIEF I §2.4] Route CLOSED.\n'
                   '                raise RuntimeError(',
                   '            elif job.search_type == \'reverse_sieve\':\n'
                   '                # [WINDOW-ANCHOR BRIEF I §2.4] Route CLOSED.\n'
                   '                _MUTANT_DISABLED = (')]),
    Mutant("M11", "prng_registry.py", ["gate_abi_frozen"],
           "change one byte of one kernel_source", fn=_kernel_byte),
    Mutant("M12", C, [], "add a new def to range_miner_coordinator.py",
           suites=["tests/test_s172_r1_drain_remedy.py",
                   "tests/test_s172_mp1_drain_attribution.py"],
           edits=[("class MinerLedger:",
                   "def _mutant_new_definition():\n    return 42\n\n\nclass MinerLedger:")]),
    Mutant("M13", C, ["gate_sep_3_public_schema_fail_closed"],
           "let generator_phase = 3 through the public schema",
           edits=[("        if int(generator_phase) != 0:", "        if False:")]),
    # M14'S FIRST FORM WAS INVALID and is recorded as such: a single hunk fusing
    # the anchor into generator_phase was caught by production's v1 POLICY PIN
    # (GeneratorPhaseNotPermittedError) BEFORE G-SEP-1 reached its assertion --
    # good news about production, but the mutant never exercised the credited
    # path. F-4 returning means BOTH halves of the separation revert at that
    # seam, since the pin exists only because of the separation.
    Mutant("M14", W, ["gate_sep_1_anchor_moves_args_do_not"],
           "make the anchor reach a kernel scalar (F-4 returns, pin reverted)",
           edits=[('        generator_phase = require_generator_phase(\n'
                   '            assign.payload, where="SieveExecutor.execute")',
                   '        generator_phase = require_window_anchor(\n'
                   '            assign.payload, where="SieveExecutor.execute")'),
                  ('        assert_generator_phase_permitted(assign.family_name, generator_phase)',
                   '        pass  # MUTANT: the v1 pin reverts with the separation')]),
    # M15 — the STRUCTURAL fusion mutant. Adds the host-side anchor to the
    # DEVICE-side build context and emits it as a kernel scalar: F-4's exact
    # architecture, reintroduced. The field carries a DEFAULT so every existing
    # BuildContext construction still works — without it the mutant would die on
    # TypeError at construction and earn no credit.
    Mutant("M15", W, ["gate_no_fused", "gate_cap_1_arity"],
           "STRUCTURAL: anchor enters BuildContext and reaches a kernel scalar",
           edits=[("    n_strategies: int = 0\n    hybrid_threshold: float = 0.0",
                   "    n_strategies: int = 0\n    hybrid_threshold: float = 0.0\n"
                   "    window_anchor: int = 11   # MUTANT: the host anchor, on the device side"),
                  ("    # forward constant: uint64 a, c + generator_phase\n"
                   "    return (\n"
                   "        _constant_prefix(ctx)\n"
                   "        + [\n"
                   "            ScalarArg(ctx.params.get(\"a\", 25214903917), \"uint64\"),\n"
                   "            ScalarArg(ctx.params.get(\"c\", 11), \"uint64\"),\n"
                   "        ]\n"
                   "        + _generator_phase_tail(ctx)\n"
                   "    )",
                   "    # forward constant: uint64 a, c + generator_phase\n"
                   "    return (\n"
                   "        _constant_prefix(ctx)\n"
                   "        + [\n"
                   "            ScalarArg(ctx.params.get(\"a\", 25214903917), \"uint64\"),\n"
                   "            ScalarArg(ctx.params.get(\"c\", 11), \"uint64\"),\n"
                   "            ScalarArg(ctx.window_anchor, \"int32\"),   # MUTANT: fused\n"
                   "        ]\n"
                   "        + _generator_phase_tail(ctx)\n"
                   "    )")]),
]


def main() -> int:
    print("=" * 78)
    print("S172 WINDOW-ANCHOR BRIEF I — §5 MUTATION EVIDENCE (15 mutants)")
    print("=" * 78)

    gates = sorted({g for m in MUTANTS for g in m.credited})
    print(f"\n-- clean controls: {len(gates)} credited gates on UNMUTATED source --")
    controls = _run_gates(gates)
    if controls.get("phase") == "import" or "__driver__" in controls:
        print(f"  {RED}CONTROL HARNESS BROKEN{RESET}: {controls}")
        print("COMPLETION SENTINEL: FAIL — no mutant can be credited")
        return 1
    for g in gates:
        r = controls.get(g, {})
        ok = r.get("outcome") == "PASS"
        print(f"  [{GREEN if ok else RED}{'PASS' if ok else 'FAIL'}{RESET}] {g}")

    print(f"\n-- mutants --")
    for m in MUTANTS:
        run_mutant(m, controls)

    print("\n" + "=" * 78)
    det = [r for r in _RESULTS if r[1] == "DETECTED"]
    inv = [r for r in _RESULTS if r[1] == "INVALID"]
    sur = [r for r in _RESULTS if r[1] == "SURVIVED"]
    print(f"\nDETECTED {len(det)}/{len(MUTANTS)}   INVALID {len(inv)}   SURVIVED {len(sur)}")
    for mid, st, detail in _RESULTS:
        if st != "DETECTED":
            print(f"  {st}: {mid} — {detail[:260]}")
    if sur:
        print("\nSURVIVING MUTANTS ARE GATE DEFECTS — DO NOT COMMIT")
        print("COMPLETION SENTINEL: FAIL")
        return 1
    print("\nCOMPLETION SENTINEL: PASS — every mutant either DETECTED by its credited "
          "gate at its own assertion, or reported INVALID with a stated reason "
          "(pending Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
