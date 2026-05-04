#!/usr/bin/env python3
"""
S170-K512: repair lingering 512-sized local skip buffers in prng_registry.py.
"""

from __future__ import annotations
import ast
import re
import sys
from pathlib import Path
from typing import Tuple

MARKER = "S170-K512"
TARGET_DEFAULT = "prng_registry.py"
MAX_WIN = "2048"

SUSPICIOUS_ARRAY_NAMES = (
    "best_skip_seq",
    "current_skip_seq",
    "temp_skip_seq",
    "skip_seq",
)

def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def subn(pattern: str, repl: str, src: str, label: str) -> Tuple[str, int]:
    new, n = re.subn(pattern, repl, src, flags=re.MULTILINE)
    if n:
        print(f"[S170-K512] {label}: {n}")
    return new, n

def audit_or_die(src: str) -> None:
    failures = []

    for name in SUSPICIOUS_ARRAY_NAMES:
        pat = rf"\b(?:unsigned\s+int|int)\s+{name}\s*\[\s*512\s*\]"
        for m in re.finditer(pat, src):
            line_no = src.count("\n", 0, m.start()) + 1
            failures.append(f"line {line_no}: remaining {name}[512]")

    for m in re.finditer(r"pos\s*\*\s*512\s*\+\s*i", src):
        line_no = src.count("\n", 0, m.start()) + 1
        failures.append(f"line {line_no}: remaining pos * 512 + i stride")

    suspicious_512_line = re.compile(
        r"(draw_idx|\bi\b|seq_size|skip_seq|best_skip_seq|current_skip_seq|temp_skip_seq).*\b512\b|"
        r"\b512\b.*(draw_idx|\bi\b|seq_size|skip_seq|best_skip_seq|current_skip_seq|temp_skip_seq)"
    )
    for i, line in enumerate(src.splitlines(), 1):
        if suspicious_512_line.search(line):
            if "was" in line or "Previous limit" in line or "hardcoded" in line or "Version" in line:
                continue
            if "window_sizes" in line:
                continue
            failures.append(f"line {i}: suspicious remaining 512 context: {line.strip()[:160]}")

    if failures:
        print("[S170-K512] post-patch audit failures:", file=sys.stderr)
        for f in failures[:80]:
            print(f"  - {f}", file=sys.stderr)
        die("dangerous 512 patterns remain; restore backup and inspect manually", 20)

def patch_text(src: str) -> Tuple[str, int]:
    total = 0
    out = src

    array_names = "|".join(map(re.escape, SUSPICIOUS_ARRAY_NAMES))
    pat = rf"\b(unsigned\s+int|int)\s+({array_names})\s*\[\s*512\s*\]\s*;"
    repl = rf"\1 \2[{MAX_WIN}];  // [{MARKER}] was [512]"
    out, n = subn(pat, repl, out, "local skip arrays [512] -> [2048]")
    total += n

    out, n = subn(r"pos\s*\*\s*512\s*\+\s*i", "pos * k + i", out, "legacy skip_sequences stride")
    total += n

    narrow_repls = [
        (r"draw_idx\s*<\s*512", f"draw_idx < {MAX_WIN}", "draw_idx < 512"),
        (r"i\s*<\s*k\s*&&\s*i\s*<\s*512", f"i < k && i < {MAX_WIN}", "i < k && i < 512"),
        (r"i\s*<\s*512\s*&&\s*i\s*<\s*k", f"i < {MAX_WIN} && i < k", "i < 512 && i < k"),
        (r"if\s*\(\s*draw_idx\s*<\s*512\s*\)", f"if (draw_idx < {MAX_WIN})", "if draw_idx < 512"),
        (r"if\s*\(\s*i\s*<\s*512\s*\)", f"if (i < {MAX_WIN})", "if i < 512"),
    ]
    for pat, repl, label in narrow_repls:
        out, n = subn(pat, repl, out, label)
        total += n

    seq_patterns = [
        (r"seq_size\s*=\s*\(\s*k\s*<\s*512\s*\)\s*\?\s*k\s*:\s*512", f"seq_size = (k < {MAX_WIN}) ? k : {MAX_WIN}"),
        (r"seq_size\s*=\s*\(\s*512\s*<\s*k\s*\)\s*\?\s*512\s*:\s*k", f"seq_size = ({MAX_WIN} < k) ? {MAX_WIN} : k"),
    ]
    for pat, repl in seq_patterns:
        out, n = subn(pat, repl, out, "seq_size 512 cap")
        total += n

    return out, total

def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else TARGET_DEFAULT)
    if not target.exists():
        die(f"target not found: {target}")

    src = target.read_text()
    before = src
    patched, edits = patch_text(src)

    if patched == before:
        print("[S170-K512] no text changes needed; running audit")
    else:
        try:
            ast.parse(patched)
        except SyntaxError as e:
            die(f"patched file fails AST parse before write: {e}", 30)

        audit_or_die(patched)
        target.write_text(patched)
        print(f"[S170-K512] patched {target} ({edits} edits)")

    final = target.read_text()
    try:
        ast.parse(final)
    except SyntaxError as e:
        die(f"final file fails AST parse: {e}", 31)

    audit_or_die(final)
    print("[S170-K512] AST: OK")
    print("[S170-K512] audit: OK — no dangerous 512 skip-buffer patterns remain")

if __name__ == "__main__":
    main()
