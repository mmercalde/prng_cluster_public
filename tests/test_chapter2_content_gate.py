#!/usr/bin/env python3
"""
test_chapter2_content_gate.py — content-and-source-agreement gate for Chapter 2.

Authority : docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_AND_2_CLOSURE.md (REV1) §5,
            which required an executable gate wherever a chapter edit touches one,
            and the 2026-08-02 closure finding that Chapter 2 had none.
Guards    : docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md
Sibling   : tests/test_chapter1_p0_corrections.py — same shape, same sentinel
            vocabulary, same mutant discipline.

    G-SKIP-RATIONALE      §5.1 physical model + §5.1.1 near-removal record +
                          the standing WIRE-IN rule. THE MOST IMPORTANT GATE.
    G-LANE-COUNT          §6's three-lane explanation is present, and the count
                          the chapter publishes MATCHES a live recount of
                          prng_registry.py by the §6.2.1 method.
    G-HYBRID-DEFECT       §5.4's defect callout is present AND still true of
                          live source (the prefix builders are re-derived).
    G-SOURCE-ANCHORS      every explicit `file.py:line` anchor resolves, and the
                          §1.1 / §7.2 facts corrected at closure are re-derived
                          from source rather than trusted.
    G-CLOSURE-INTEGRITY   §14 closure statement, its sentinel, the F-1…F-8
                          finding table and the VIR-6 declaration are intact.
    G-CHAPTER-SUBSTANCE   the chapter is not a fragment. THE 248e48c GATE.


WHY THIS GATE EXISTS — the failure mode it defends against
----------------------------------------------------------
Chapter 2 was destroyed by `248e48c`, a commit titled "chore: move CHAPTER docs",
which copied a 34-line fragment over a 743-line chapter. Nobody read either file.
The loss was not noticed for months, and in the interim Team Alpha, Team Beta and
Claude Code INDEPENDENTLY recommended deleting `skip_min`/`skip_max` — a
cornerstone of the design — because the document explaining why skip exists no
longer existed to be read (§5.1.1). Michael stopped the removal.

A gate is a message to whoever does not open the file. This one is therefore
scoped to CONTENT PRESENCE and SOURCE AGREEMENT, deliberately NOT to prose style:
every assertion below is either "this load-bearing statement is still here" or
"this number still matches the source it claims to describe". Rewording a
paragraph must not red this gate. Deleting the paragraph must.

VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)
-----------------------------------------
execution proof
    No gate trusts the chapter about source. G-LANE-COUNT RE-DERIVES the lane
    count from live `prng_registry.py` and compares. G-HYBRID-DEFECT RE-DERIVES
    the prefix-builder asymmetry by importing the live module and inspecting the
    returned arg lists. G-SOURCE-ANCHORS resolves every citation against the
    files on disk.
clean control
    Every gate has a negative arm on the unmutated tree: the count agrees, the
    anchors resolve, the required sections are present at their real sizes.
    G-LANE-COUNT additionally proves its own counter DISCRIMINATES, by running
    it against a source in which a lane test has been removed.
fault-injection control
    Six mutants, one per gate, each a temp-file copy of the chapter (or of a
    source file) that MUST turn its gate red. Never written into the repo.
detector independence
    The detectors are a live recount, a live module import, and filesystem
    resolution. None shares an expression with the prose it checks.
completion sentinel
    PASS | FAIL | UNAVAILABLE | INCOMPLETE, printed at the end. Only PASS accepts.
unavailable-observer behavior
    A surface that cannot be reached reports UNAVAILABLE and is NOT green.
audit claim scope
    Repo-scoped, VM 101 working tree. NO GPU, NO RIG, NO FLEET, NO PIPELINE, and
    no subprocess into the CLI. This is deliberate: four arms of the Chapter 1
    gate require a reachable fleet, so a red there can mean "rigs are off"
    (recorded in Chapter 1 §17.2). This gate has no such dependency — a red here
    always means the chapter or its source agreement actually changed.
"""

import ast
import os
import re
import sys
import tempfile
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CHAPTER = os.path.join(_ROOT, "docs", "CHAPTER_2_BIDIRECTIONAL_SIEVE.md")
REGISTRY = os.path.join(_ROOT, "prng_registry.py")
WORKER = os.path.join(_ROOT, "miner", "range_miner_worker.py")

RESULTS = []          # (name, "PASS" | "FAIL" | "UNAVAILABLE", detail)
_GREEN = "\033[92m"
_RED = "\033[91m"
_YEL = "\033[93m"
_OFF = "\033[0m"


class _Unavailable(Exception):
    """The surface this gate needs could not be reached (VIR-5)."""


def _check(name, fn):
    try:
        fn()
    except _Unavailable as e:
        RESULTS.append((name, "UNAVAILABLE", str(e)))
        print(f"  [{_YEL}UNAVAILABLE{_OFF}] {name}: {e}", flush=True)
    except Exception:
        RESULTS.append((name, "FAIL", traceback.format_exc()))
        print(f"  [{_RED}FAIL{_OFF}] {name}", flush=True)
    else:
        RESULTS.append((name, "PASS", ""))
        print(f"  [{_GREEN}PASS{_OFF}] {name}", flush=True)


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _temp_chapter(text, tag):
    """Write a mutated chapter to a TEMP path, never into the repo."""
    d = tempfile.mkdtemp(prefix=f"ch2_gate_{tag}_")
    p = os.path.join(d, "CHAPTER_2_BIDIRECTIONAL_SIEVE.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def _mutate(text, old, new, what):
    """Replace `old` once; the anchor MUST exist or the harness is stale."""
    if old not in text:
        raise AssertionError(
            f"mutation anchor not found ({what}) — this harness is stale "
            f"relative to the chapter it guards: {old[:90]!r}")
    return text.replace(old, new, 1)


# ===========================================================================
# the §6.2.1 counting method — ONE implementation, used by the gate and by the
# mutant. The chapter publishes this same method as runnable code.
# ===========================================================================

def count_lane_tests(source_text):
    """
    Structural count: a mod-1000 comparison whose following two lines carry the
    mod-8 and mod-125 conjuncts. Indifferent to casts, spacing and index naming
    — it counts the conjunction, not its formatting.
    """
    lines = source_text.split("\n")
    return [i + 1 for i, l in enumerate(lines)
            if "% 1000" in l
            and "% 8" in "\n".join(lines[i:i + 3])
            and "% 125" in "\n".join(lines[i:i + 3])]


def count_kernels(source_text):
    return [i + 1 for i, l in enumerate(source_text.split("\n"))
            if 'extern "C" __global__' in l]


def kernel_owner(source_text, line_no):
    """Name of the kernel whose body contains `line_no`."""
    lines = source_text.split("\n")
    owner = None
    for start in count_kernels(source_text):
        if start <= line_no:
            m = re.search(r"void\s+(\w+)", " ".join(lines[start - 1:start + 3]))
            owner = m.group(1) if m else "?"
        else:
            break
    return owner


# ===========================================================================
# G-SKIP-RATIONALE — the most important gate in this file
# ===========================================================================

# The standing rule. This sentence is the whole reason §5.1 was written: three
# parties reasoned their way to deleting a cornerstone because it was absent.
WIRE_IN_RULE = ("The correct action on finding skip bounds unwired\n"
                "is to wire them in, not to remove them")

NEAR_REMOVAL_RECORD = ("Team Alpha, Team Beta and Claude Code independently "
                       "recommended deleting")


def gate_skip_rationale(chapter=None):
    text = _read(chapter or CHAPTER)

    # --- §5.1 must exist as a section, not as a passing mention -------------
    assert "### 5.1 Why skip exists — the physical model" in text, \
        "§5.1 heading is gone — the physical model section was removed or renamed"
    assert "#### 5.1.1 Why this paragraph is in this chapter" in text, \
        "§5.1.1 heading is gone — the record of WHY §5.1 exists was removed"

    # --- the standing rule --------------------------------------------------
    assert WIRE_IN_RULE in text, (
        "the WIRE-IN rule is absent or altered. This is the single most "
        "load-bearing sentence in the chapter: it is what stops a future reader "
        "re-deriving the already-rejected conclusion that skip_min/skip_max "
        "should be deleted (§5.1.1)")

    # --- the near-removal record itself ------------------------------------
    assert NEAR_REMOVAL_RECORD in text, \
        "the record of the three-party near-removal was removed from §5.1.1"
    assert "Michael stopped the removal." in text, \
        "the record of who stopped the removal was dropped"
    assert "the document explaining why skip exists did not exist to be read" in text, \
        "§5.1.1's root-cause statement was dropped"

    # --- the physical model: the three procedural facts skip rests on -------
    for required, why in (
        ("California State Lottery Daily & SuperLotto Plus Draw Procedures",
         "the source document skip's justification rests on"),
        ("2021-06-09", "the effective date that pins WHICH revision is meant"),
        ("One automatic pre-test session runs before an automatic Daily draw",
         "the corrected pre-test claim (the count was wrong once already)"),
        ("Draw equipment is selected per session",
         "per-session equipment selection"),
        ("Fantasy 5", "the co-drawn evening games that sit between Daily 3 values"),
    ):
        assert required in text, f"§5.1 lost {why}: {required!r}"

    # --- the epistemic qualification. Removing this would silently upgrade
    #     "candidate gaps" into "proven state advances" — a Beta ruling.
    assert "candidate gaps* supporting skip as a detector" in text or \
           "candidate gaps** supporting skip as a detector" in text, \
        ("§5.1's epistemic qualification is gone — without it the chapter reads "
         "as claiming the omitted outputs are PROVEN state advances, which Beta "
         "explicitly ruled it must not claim")
    assert "not proven state advances" in text, \
        "§5.1's 'not proven state advances' qualification was removed"

    # --- the citation honesty marker ---------------------------------------
    assert "citation `UNAVAILABLE`" in text or "**citation `UNAVAILABLE`**" in text, \
        ("§5.1 no longer marks its central citation UNAVAILABLE — the PDF is "
         "still not in the repo, and dropping the marker would present unverified "
         "ruling text as source-verified")

    # --- the correction that was itself an Alpha-introduced error ----------
    assert "two pre-test draws" in text, \
        ("the record of the corrected two-pre-test error was dropped; it must "
         "stay so the wrong claim is not re-introduced from an older document")

    # CLEAN CONTROL — §5.1 is a real section with substance, not a stub.
    start = text.index("### 5.1 Why skip exists")
    end = text.index("### 5.2 Constant skip mode")
    body = text[start:end]
    assert len(body.split("\n")) > 40, \
        f"§5.1 collapsed to {len(body.splitlines())} lines — it is a stub, not the section"


def mutant_wire_in_rule_removed():
    """FAULT INJECTION — delete the WIRE-IN rule; the gate must go RED."""
    text = _read(CHAPTER)
    mutated = _mutate(text, WIRE_IN_RULE, "(rule removed by mutant)", "WIRE_IN_RULE")
    p = _temp_chapter(mutated, "wirein")
    try:
        gate_skip_rationale(chapter=p)
    except AssertionError as e:
        assert "WIRE-IN rule is absent" in str(e), \
            f"gate reddened for the wrong reason: {e}"
    else:
        raise AssertionError(
            "gate stayed green with the WIRE-IN rule deleted — it is vacuous, "
            "and the exact deletion that nearly happened in production would pass")


# ===========================================================================
# G-LANE-COUNT — content presence AND live source agreement
# ===========================================================================

LANE_BLOCK_BEGIN = "<!-- BEGIN LANE TEST COUNT"
LANE_BLOCK_END = "<!-- END LANE TEST COUNT"


def _lane_block(text):
    a = text.find(LANE_BLOCK_BEGIN)
    b = text.find(LANE_BLOCK_END)
    if a < 0 or b < 0:
        raise AssertionError(
            "the machine-readable lane-count block (§6.2.1) is missing — the "
            "chapter's number can no longer be checked against source")
    return text[a:b]


def _parse_lane_block(block):
    out = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.strip().partition(":")
        k = k.strip()
        if k and re.fullmatch(r"[a-z_]+", k):
            out[k] = v.strip()
    return out


def gate_lane_count(chapter=None, registry=None):
    text = _read(chapter or CHAPTER)
    src = _read(registry or REGISTRY)

    # --- §6 must still explain the construction ----------------------------
    assert "## 6. Three-Lane CRT Architecture" in text, "§6 heading is gone"
    for required, why in (
        ("1000 = 8 × 125", "the CRT construction itself"),
        ("gcd(8, 125) = 1", "the coprimality that makes the decomposition valid"),
        ("Chinese Remainder Theorem", "the theorem named"),
    ):
        assert required in text, f"§6 lost {why}: {required!r}"

    # the redundancy finding (F-1) — the correction that matters most in §6
    assert "CRT-redundant" in text or "exactly equivalent" in text, \
        ("§6.3's redundancy finding is gone — without it the chapter re-asserts "
         "the refuted triple-validation power claim (F-1)")

    # --- the count, re-derived from LIVE source ----------------------------
    declared = _parse_lane_block(_lane_block(text))
    for key in ("lane_test_count", "total_kernels", "single_lane_exception",
                "method", "source_file"):
        assert key in declared, f"lane-count block lacks {key}"

    live_hits = count_lane_tests(src)
    live_kernels = count_kernels(src)

    assert int(declared["lane_test_count"]) == len(live_hits), (
        f"CHAPTER IS STALE: §6.2.1 declares {declared['lane_test_count']} lane "
        f"tests, live {os.path.basename(REGISTRY)} has {len(live_hits)}. "
        f"Re-run the published method and correct the prose — do NOT edit the "
        f"block to match")
    assert int(declared["total_kernels"]) == len(live_kernels), (
        f"CHAPTER IS STALE: §6.2.1 declares {declared['total_kernels']} kernels, "
        f"live registry has {len(live_kernels)}")

    # --- the single-lane exception is real and correctly named -------------
    owners = {kernel_owner(src, h) for h in live_hits}
    all_kernels = set()
    for start in live_kernels:
        m = re.search(r"void\s+(\w+)",
                      " ".join(src.split("\n")[start - 1:start + 3]))
        all_kernels.add(m.group(1) if m else "?")
    missing = all_kernels - owners
    assert len(missing) == 1, (
        f"§6.2.1 claims exactly one kernel lacks the three-lane test; live "
        f"source has {len(missing)}: {sorted(missing)}")
    assert declared["single_lane_exception"] in missing, (
        f"§6.2.1 names {declared['single_lane_exception']!r} as the single-lane "
        f"exception, but live source says it is {sorted(missing)[0]!r}")

    # --- the verbatim three-lane block still matches source ----------------
    # Compared on WHITESPACE-NORMALISED text, because the chapter pads the block
    # for column alignment (`%    8`) while the source does not (`% 8`). The
    # tokens either side of the padding must still match exactly — this is not a
    # loose match, it is the same comparison with layout removed.
    norm_src = re.sub(r"\s+", " ", src)
    norm_text = re.sub(r"\s+", " ", text)
    for lane in ("(output % 1000) == (unsigned int)(residues[i] % 1000)",
                 "(output % 8) == (unsigned int)(residues[i] % 8)",
                 "(output % 125) == (unsigned int)(residues[i] % 125)"):
        assert lane in norm_src, \
            f"the lane §6.2 quotes verbatim is no longer in the registry: {lane!r}"
        assert lane in norm_text, \
            f"§6.2's verbatim block lost a lane: {lane!r}"

    # --- the withdrawn number must not creep back as a CLAIM ---------------
    # The chapter legitimately QUOTES "39 occurrences" in order to withdraw it,
    # in §6.2.1, in the VIR-6 amendment and in the version history. So a bare
    # substring test would forbid the historical record itself. What must never
    # reappear is 39 asserted as the count: every mention has to sit inside
    # withdrawal language.
    WITHDRAWAL_VOCAB = ("withdraw", "previously", "superseded", "amended",
                        "settled", "not reproducible", "unreproducible")
    doc_lines = text.split("\n")
    for i, line in enumerate(doc_lines):
        if "39 occurrence" not in line and "'39" not in line and '"39' not in line:
            continue
        window = " ".join(doc_lines[max(0, i - 2):i + 3]).lower()
        assert any(v in window for v in WITHDRAWAL_VOCAB), (
            f"line {i + 1} states '39' as the lane-test count without marking it "
            f"withdrawn — the unreproducible number is back as a claim:\n  {line.strip()}")

    # CLEAN CONTROL — the counter DISCRIMINATES. Remove one lane test from a
    # copy of the source and the count must drop by exactly one.
    lines = src.split("\n")
    first = live_hits[0]
    gutted = "\n".join(lines[:first - 1] + ["// lane test removed by control"]
                       + lines[first + 2:])
    assert len(count_lane_tests(gutted)) == len(live_hits) - 1, (
        "clean control: the counter did not notice a removed lane test, so its "
        "agreement with the chapter proves nothing")


def mutant_lane_count_stale():
    """FAULT INJECTION — hand-edit the declared count; the gate must go RED."""
    text = _read(CHAPTER)
    block = _lane_block(text)
    declared = _parse_lane_block(block)
    real = int(declared["lane_test_count"])
    mutated_block = block.replace(f"lane_test_count:        {real}",
                                  f"lane_test_count:        {real + 4}", 1)
    assert mutated_block != block, "lane_test_count mutation anchor not found"
    p = _temp_chapter(text.replace(block, mutated_block, 1), "lanecount")
    try:
        gate_lane_count(chapter=p)
    except AssertionError as e:
        assert "CHAPTER IS STALE" in str(e), \
            f"gate reddened for the wrong reason: {e}"
    else:
        raise AssertionError(
            "gate stayed green against a hand-edited count — it is vacuous, and "
            "the unreproducible-number failure it exists to prevent would recur")


# ===========================================================================
# G-HYBRID-DEFECT — the callout, and whether it is STILL TRUE of source
# ===========================================================================

DEFECT_HEADING = ("### 5.4 The defect: hybrid kernels do not execute the "
                  "requested skip semantics")
DEFECT_CONSEQUENCE = ("**Consequence, stated plainly: hybrid optimization "
                      "results are non-certifying.**")


def gate_hybrid_defect(chapter=None):
    text = _read(chapter or CHAPTER)

    assert DEFECT_HEADING in text, \
        "§5.4's defect heading is gone — the hybrid skip defect was un-documented"
    assert DEFECT_CONSEQUENCE in text, (
        "§5.4's consequence statement is absent. Without it the chapter records "
        "a defect without recording that hybrid results CANNOT BE CERTIFIED")
    assert "remains **OPEN**" in text or "remains OPEN" in text, \
        "§5.4 no longer marks the defect OPEN — it is described, not repaired"
    assert "0 of 22 hybrid kernels declare" in text, \
        "§5.4 lost the audited hybrid-kernel count"

    # --- EXECUTION PROOF: is the defect still true of live source? ---------
    # The chapter asserts an asymmetry between two argument-prefix builders.
    # Re-derive it from the live file by AST, NOT by importing the module:
    # importing range_miner_worker pulls in sieve_gpu_worker, whose import
    # replaces sys.stdout and discards the importer's buffered output. An AST
    # parse is also a stronger detector than substring matching — it reads the
    # actual returned list, so a skip bound mentioned in a comment or a docstring
    # cannot satisfy it.
    if not os.path.exists(WORKER):
        raise _Unavailable(f"{WORKER} missing — cannot re-derive the defect")
    try:
        tree = ast.parse(_read(WORKER))
    except SyntaxError as e:                     # pragma: no cover
        raise _Unavailable(f"cannot parse range_miner_worker: {e}")

    def _returned_args(fn_name):
        """The arg-constructor calls in `fn_name`'s returned list."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.List):
                        return [ast.unparse(e) for e in sub.value.elts]
                raise _Unavailable(f"{fn_name} does not return a list literal")
        raise _Unavailable(f"range_miner_worker has no {fn_name} — structure changed")

    const_args = _returned_args("_constant_prefix")
    hyb_args = _returned_args("_hybrid_prefix")

    const_has_skip = any("skip_min" in a or "skip_max" in a for a in const_args)
    hyb_has_skip = any("skip_min" in a or "skip_max" in a for a in hyb_args)

    assert const_has_skip, (
        "_constant_prefix no longer emits the skip bounds — §5.4's asymmetry "
        "claim is stale and the chapter must be re-verified")
    assert not hyb_has_skip, (
        "SOURCE CHANGED: _hybrid_prefix now emits a skip bound. If the defect "
        "was repaired, §5.4 is stale and must be updated — this gate is telling "
        "you the chapter no longer describes the code")

    # the chapter states a 13-element hybrid prefix; count the real elements
    assert len(hyb_args) == 13, (
        f"§5.4 states _hybrid_prefix returns a 13-element prefix; live source "
        f"builds {len(hyb_args)}. The chapter is stale")
    assert "13" in text and "_hybrid_prefix" in text, \
        "§5.4 lost its reference to the 13-element hybrid prefix"

    # CLEAN CONTROL — the two builders really do differ, so the assertion above
    # is discriminating rather than trivially true of both.
    assert const_has_skip != hyb_has_skip, \
        "clean control: both prefixes agree on skip_min, so §5.4 has no asymmetry"


def mutant_hybrid_defect_removed():
    """FAULT INJECTION — drop the non-certifying consequence; gate must go RED."""
    text = _read(CHAPTER)
    mutated = _mutate(text, DEFECT_CONSEQUENCE, "(consequence removed by mutant)",
                      "DEFECT_CONSEQUENCE")
    p = _temp_chapter(mutated, "hybdefect")
    try:
        gate_hybrid_defect(chapter=p)
    except AssertionError as e:
        assert "consequence statement is absent" in str(e), \
            f"gate reddened for the wrong reason: {e}"
    else:
        raise AssertionError(
            "gate stayed green with the non-certifying consequence deleted — a "
            "reader could then treat hybrid results as certifiable")


# ===========================================================================
# G-SOURCE-ANCHORS — every citation resolves; closure-corrected facts re-derived
# ===========================================================================

_SEARCH_PREFIXES = ("", "tests/phase6", "miner")


def _resolve(rel):
    for pref in _SEARCH_PREFIXES:
        p = os.path.join(_ROOT, pref, rel)
        if os.path.exists(p):
            return p
    return None


def gate_source_anchors(chapter=None):
    text = _read(chapter or CHAPTER)

    # --- every explicit file.py:line anchor resolves and is in range --------
    bad, checked = [], 0
    for m in re.finditer(r"`([^`]*)`", text):
        seg = re.sub(r"sha256:[0-9a-f]+", "", m.group(1))
        for mm in re.finditer(r"([\w/]+\.py):(\d+)(?:-(\d+))?", seg):
            rel, a, b = mm.group(1), int(mm.group(2)), mm.group(3)
            b = int(b) if b else a
            path = _resolve(rel)
            if path is None:
                bad.append(f"{rel}:{a}-{b} — FILE NOT FOUND")
                continue
            checked += 1
            n = len(_read(path).split("\n"))
            if b > n or a < 1:
                bad.append(f"{rel}:{a}-{b} — file has {n} lines")
    assert not bad, ("chapter anchors do not resolve against live source:\n  "
                     + "\n  ".join(bad))
    assert checked >= 40, \
        f"only {checked} anchors checked — the chapter lost most of its citations"

    # --- §1.1 facts corrected at closure, RE-DERIVED from source -----------
    src = _read(REGISTRY)
    lines = src.split("\n")

    # a/c hardcoded in exactly TWO kernel bodies, both reverse kernels
    hard = [i + 1 for i, l in enumerate(lines) if "25214903917ULL" in l]
    owners = sorted({kernel_owner(src, h) for h in hard})
    assert len(owners) == 2, (
        f"§1.1 states a/c are hardcoded in TWO kernel bodies; live source says "
        f"{len(owners)}: {owners}. The chapter is stale")
    assert all("reverse" in o for o in owners), (
        f"§1.1 states both hardcoding kernels are REVERSE kernels; live: {owners}")
    assert "two**\n  kernel bodies" in text or "**two**" in text, \
        "§1.1 lost the corrected two-kernel statement"
    assert "hardcoded in the three" not in text, \
        "the withdrawn 'three non-parameterised kernels' claim is back in §1.1"

    # the default_params anchor must point at a real default_params block
    m = re.search(r"`:(\d+)-(\d+)`\*\* \(and `java_lcg_hybrid`", text)
    if m is None:
        m = re.search(r"\*\*`:(\d+)-(\d+)`\*\*", text)
    assert m, "§1.1's corrected default_params anchor is gone"
    a, b = int(m.group(1)), int(m.group(2))
    window = "\n".join(lines[a - 2:b])
    assert "default_params" in window and "25214903917" in window, (
        f"§1.1's default_params anchor :{a}-{b} no longer points at the "
        f"default_params block — it points at:\n{window[:200]}")
    assert ":1004" not in text or "closing" in text, \
        "the withdrawn :1004 default-params anchor is back without its correction"

    # --- §7.2.1's two SEPARATED consumers exist ----------------------------
    #
    # REPOINTED, not relaxed. These two assertions previously required the FUSED
    # consumers (`def _offset_tail`, `min(int(offset), n - window_size)`) to be
    # present in worker source — a tripwire on the F-4 defect. Window-Anchor
    # Brief I repaired F-4, so the tripwire fired exactly as designed and its own
    # message asked for the finding to be re-verified. It was: the fused pair is
    # gone and the SEPARATED pair is what must now be asserted.
    #
    # Self-exclusion: the probe reads WORKER, never this file, which contains both
    # search strings in its own source. Asserted rather than assumed, because a
    # probe that reads its own file is green on a fact it does not check.
    wsrc = _read(WORKER)
    assert os.path.abspath(os.path.join(_ROOT, WORKER)) != os.path.abspath(__file__), \
        "the source probe is pointed at this gate file, not at the worker"

    # device side: the phase reaches the kernel through the frozen slot
    assert "def _generator_phase_tail" in wsrc, (
        "§7.2.1's device-side consumer (_generator_phase_tail) is gone from source "
        "— the generator phase no longer has a named delivery surface")
    # host side: the anchor is VALIDATED against a derived domain, never clamped
    assert "if anchor < 0 or anchor > derived_max:" in wsrc, (
        "§7.2.1's host-side anchor validation is gone from source — if the clamp "
        "returned, F-4 returned with it")
    assert "min(int(offset), n - window_size)" not in wsrc, (
        "the FUSED clamp is back in worker source — §7.2.1 describes a separation "
        "that no longer exists")

    # :578 — RETAINED. F-4 must stay in the chapter; the finding is re-disposed,
    # never deleted. (Known weak: keyword presence over a 1,463-line document.
    # Recorded as follow-up debt, deliberately NOT repaired here — see the Brief I
    # report; this gate is not being redesigned.)
    assert "F-4" in text and "offset" in text, "§7 lost the F-4 offset finding"


def mutant_anchor_broken():
    """FAULT INJECTION — point an anchor past EOF; the gate must go RED."""
    text = _read(CHAPTER)
    mutated = _mutate(text, "`prng_registry.py:984-986`",
                      "`prng_registry.py:984-99986`", "anchor")
    p = _temp_chapter(mutated, "anchor")
    try:
        gate_source_anchors(chapter=p)
    except AssertionError as e:
        assert "do not resolve" in str(e), f"gate reddened for the wrong reason: {e}"
    else:
        raise AssertionError(
            "gate stayed green against an anchor pointing past EOF — the anchor "
            "drift this chapter has already suffered twice would go unnoticed")


# ===========================================================================
# G-CLOSURE-INTEGRITY — the closure statement and finding table
# ===========================================================================

def gate_closure_integrity(chapter=None):
    text = _read(chapter or CHAPTER)

    assert "## 14. Closure statement" in text, "the §14 closure statement is gone"
    assert "CHAPTER 2 CLOSURE:  PASS" in text, \
        "the closure sentinel is absent or no longer PASS"
    assert "verified-and-bounded, not finished" in text, (
        "the closure statement lost its qualification — 'closed' would then read "
        "as 'finished', which it is not")

    # the four things a closure statement must carry (brief §3)
    for required in ("Verified against", "What is verified",
                     "What remains open", "is NOT"):
        assert required in text, f"closure statement lacks a required part: {required!r}"

    # --- the finding table: F-1…F-8 all still recorded ---------------------
    for f in range(1, 9):
        assert f"**F-{f}**" in text, \
            f"finding F-{f} was dropped from the §12 audit table"

    # dispositions that are rulings, not opinions — these must not be softened
    assert "NOT the repair" in text, \
        "F-4's disposition (settles C-2 as an observed inconsistency, NOT the repair) was altered"
    assert "Do not rename the hosts" in text, (
        "F-5's standing hazard note is gone — renaming the rig hosts would "
        "activate obsolete overrides")

    # --- VIR-6 declaration ------------------------------------------------
    assert "audit claim scope (VIR-6)" in text, "the VIR-6 declaration is gone"
    assert "Unavailable surfaces" in text, \
        "the declared unavailable-surface list is gone — surfaces would read as clean"
    assert "the *bodies* of the 40 non-java_lcg kernels" in text, \
        "the residual unavailability (40 unread kernel bodies) was dropped"

    # the §12.1 inherited items must not silently vanish
    assert "### 12.1 Open items this chapter inherits but does not own" in text, \
        "§12.1's inherited-open-items section is gone"
    assert "java_lcg_cpu" in text and "no fix authorized" in text, \
        "the java_lcg_cpu skip-mismatch item (TB: no fix authorized) was dropped"


def mutant_closure_sentinel_removed():
    """FAULT INJECTION — remove the closure sentinel; the gate must go RED."""
    text = _read(CHAPTER)
    mutated = _mutate(text, "CHAPTER 2 CLOSURE:  PASS", "(sentinel removed)",
                      "closure sentinel")
    p = _temp_chapter(mutated, "closure")
    try:
        gate_closure_integrity(chapter=p)
    except AssertionError as e:
        assert "closure sentinel is absent" in str(e), \
            f"gate reddened for the wrong reason: {e}"
    else:
        raise AssertionError(
            "gate stayed green with the closure sentinel deleted — it is vacuous")


# ===========================================================================
# G-CHAPTER-SUBSTANCE — THE 248e48c GATE
# ===========================================================================
#
# 248e48c ("chore: move CHAPTER docs") copied a 34-line fragment over a 743-line
# chapter. Every gate above checks a specific claim; this one checks that the
# chapter still EXISTS at scale. It is the cheapest and bluntest of the six, and
# it is the one that would have caught the incident that motivated all of them.

MIN_LINES = 1000          # chapter is ~1470; 743 was the pre-destruction size
MIN_BYTES = 60_000

REQUIRED_SECTIONS = [
    "## 1. Mathematical Foundation",
    "## 2. Forward Sieve",
    "## 3. Reverse Sieve",
    "## 4. Bidirectional Intersection",
    "## 5. Skip/Gap Handling",
    "## 6. Three-Lane CRT Architecture",
    "## 7. `offset` — one name, incompatible meanings",
    "## 8. The engine today: RANGE-MINER",
    "## 11. What is certified, and what is not",
    "## 12. Audit findings from this pass",
    "## 13. Verification declaration (VIR-1…6)",
    "## 14. Closure statement",
]


def gate_chapter_substance(chapter=None):
    path = chapter or CHAPTER
    assert os.path.exists(path), f"{path} does not exist"
    text = _read(path)
    n_lines = len(text.split("\n"))
    n_bytes = len(text.encode("utf-8"))

    assert n_lines >= MIN_LINES, (
        f"CHAPTER TRUNCATED: {n_lines} lines, expected >= {MIN_LINES}. This is "
        f"the 248e48c failure mode — a fragment copied over the chapter. Do not "
        f"lower this threshold to make the gate pass; restore the content")
    assert n_bytes >= MIN_BYTES, \
        f"CHAPTER TRUNCATED: {n_bytes} bytes, expected >= {MIN_BYTES}"

    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, (
        f"CHAPTER TRUNCATED: {len(missing)} required section(s) absent: {missing}")

    # a fragment can also be produced by keeping headings and gutting bodies
    for section, floor in (("## 5. Skip/Gap Handling", 100),
                           ("## 6. Three-Lane CRT Architecture", 100)):
        i = text.index(section)
        nxt = text.find("\n## ", i + 1)
        body = text[i:nxt if nxt > 0 else len(text)]
        assert len(body.split("\n")) >= floor, (
            f"{section} is {len(body.splitlines())} lines — the headings survived "
            f"but the body was gutted")

    # CLEAN CONTROL — the thresholds discriminate: a 34-line fragment (the real
    # 248e48c payload size) must not satisfy them.
    assert MIN_LINES > 34 and MIN_BYTES > 34 * 80, \
        "clean control: the thresholds would accept the historical fragment"


def mutant_chapter_truncated():
    """FAULT INJECTION — reproduce 248e48c: a 34-line fragment. Gate must go RED."""
    text = _read(CHAPTER)
    fragment = "\n".join(text.split("\n")[:34])
    p = _temp_chapter(fragment, "trunc")
    try:
        gate_chapter_substance(chapter=p)
    except AssertionError as e:
        assert "CHAPTER TRUNCATED" in str(e), \
            f"gate reddened for the wrong reason: {e}"
    else:
        raise AssertionError(
            "gate stayed green against a 34-line fragment — it would NOT have "
            "caught 248e48c, which is the incident it exists for")


# ===========================================================================

GATES = [
    ("G-SKIP-RATIONALE", gate_skip_rationale),
    ("G-LANE-COUNT", gate_lane_count),
    ("G-HYBRID-DEFECT", gate_hybrid_defect),
    ("G-SOURCE-ANCHORS", gate_source_anchors),
    ("G-CLOSURE-INTEGRITY", gate_closure_integrity),
    ("G-CHAPTER-SUBSTANCE", gate_chapter_substance),
]

MUTANTS = [
    ("M1 delete the WIRE-IN rule", mutant_wire_in_rule_removed),
    ("M2 hand-edit the declared lane count", mutant_lane_count_stale),
    ("M3 remove the non-certifying consequence", mutant_hybrid_defect_removed),
    ("M4 point an anchor past EOF", mutant_anchor_broken),
    ("M5 remove the closure sentinel", mutant_closure_sentinel_removed),
    ("M6 truncate the chapter to a 34-line fragment (248e48c)",
     mutant_chapter_truncated),
]


def main():
    print("=" * 70, flush=True)
    print("CHAPTER 2 CONTENT GATE — presence + live source agreement", flush=True)
    print("=" * 70, flush=True)

    print("\n--- GATES (clean tree) ---", flush=True)
    for name, fn in GATES:
        _check(name, fn)

    print("\n--- MUTANTS (fault-injection controls) ---", flush=True)
    for name, fn in MUTANTS:
        _check(name, fn)

    total = len(RESULTS)
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = [r for r in RESULTS if r[1] == "FAIL"]
    unavail = [r for r in RESULTS if r[1] == "UNAVAILABLE"]

    print("\n" + "=" * 70, flush=True)
    print(f"{passed}/{total} checks green "
          f"({len(GATES)} gates + {len(MUTANTS)} mutants)", flush=True)

    if failed:
        print("\nFAILURES (DO NOT COMMIT):", flush=True)
        for name, _, detail in failed:
            print(f"\n--- {name} ---\n{detail}", flush=True)

    if unavail:
        print("\nUNAVAILABLE (not counted as green — VIR-5):", flush=True)
        for name, _, detail in unavail:
            print(f"  {name}: {detail}", flush=True)

    if failed:
        sentinel = "FAIL"
    elif unavail:
        sentinel = "INCOMPLETE"
    elif passed == total:
        sentinel = "PASS"
    else:
        sentinel = "INCOMPLETE"

    print(f"\nSENTINEL : {sentinel}", flush=True)
    return 0 if sentinel == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
