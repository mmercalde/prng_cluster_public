#!/usr/bin/env python3
import re, sys, pathlib

p = pathlib.Path("prng_registry.py")
src = p.read_text()

# 1) Locate the mt19937_cpu_simple function block
m = re.search(r"^def\s+mt19937_cpu_simple\([^\)]*\):\n", src, re.M)
if not m:
    print("Could not find mt19937_cpu_simple(). Aborting.")
    sys.exit(1)

start = m.start()

# Heuristic end of block: next top-level 'def ' or end-of-file
m_end = re.search(r"^def\s+\w+\(", src[m.end():], re.M)
end = (m.end() + m_end.start()) if m_end else len(src)

block = src[start:end]

# 2) Fix the nested extractor signature IF it’s wrong
#    Accept any current param list and collapse to empty.
block_fixed = re.sub(
    r"(^\s*def\s+mt19937_extract\s*\()[^\)]*(\)\s*:)", 
    r"\1\2", 
    block, 
    flags=re.M
)

# 3) Fix all call sites INSIDE this function from
#    mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A)
#    to mt19937_extract()
block_fixed = re.sub(
    r"mt19937_extract\s*\(\s*mt\s*,\s*mti\s*,\s*N\s*,\s*M\s*,\s*UPPER_MASK\s*,\s*LOWER_MASK\s*,\s*MATRIX_A\s*\)",
    "mt19937_extract()",
    block_fixed
)

# 4) Write back only if changed
if block_fixed != block:
    p.write_text(src[:start] + block_fixed + src[end:])
    print("✅ Repaired mt19937_cpu_simple(): signature & call sites updated.")
else:
    print("No changes were necessary.")
