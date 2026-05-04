#!/usr/bin/env python3
"""
S171-COORD-DISPATCH-THROTTLE

Minimal diagnostic patch.

Adds coordinator-side sleep immediately before TCP/PWC job submission.

Env var:
  PRNG_PWC_COORD_DISPATCH_THROTTLE_MS=0   # default off

Purpose:
  Test whether Zeus-side dispatch burst pressure contributes to pool8 ROCm/SMU crashes.

Default behavior unchanged.
"""

from pathlib import Path
import ast
import sys

P = Path("persistent_worker_coordinator.py")
MARKER = "S171-COORD-DISPATCH-THROTTLE"

def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

src = P.read_text()

if MARKER in src:
    print("[S171-COORD-DISPATCH-THROTTLE] already patched")
    sys.exit(0)

# Add helper after import time
old = "import time\n"
new = """import time

# [S171-COORD-DISPATCH-THROTTLE]
def _s171_coord_dispatch_throttle():
    \"\"\"Optional coordinator-side dispatch pacing. Default OFF.\"\"\"
    try:
        ms = int(os.environ.get("PRNG_PWC_COORD_DISPATCH_THROTTLE_MS", "0"))
        if ms > 0:
            time.sleep(ms / 1000.0)
    except Exception:
        pass

"""
if src.count(old) != 1:
    die(f"import anchor count={src.count(old)}, expected 1")
src = src.replace(old, new, 1)

# Patch likely TCP submit call.
# This is intentionally narrow: throttle immediately before submit_job().
anchors = [
    "self._tcp_transport.submit_job(",
    "_tcp_transport.submit_job(",
]

patched = False
for anchor in anchors:
    idx = src.find(anchor)
    if idx != -1:
        # Find beginning of the line containing submit_job
        line_start = src.rfind("\n", 0, idx) + 1
        indent = src[line_start:idx]
        insert = indent + "# [S171-COORD-DISPATCH-THROTTLE] env-gated dispatch pacing\n" + indent + "_s171_coord_dispatch_throttle()\n"
        src = src[:line_start] + insert + src[line_start:]
        patched = True
        print(f"[S171-COORD-DISPATCH-THROTTLE] patched before anchor: {anchor}")
        break

if not patched:
    die("Could not find TCP submit_job anchor. Need live dispatch snippet.")

ast.parse(src)
P.write_text(src)

print("[S171-COORD-DISPATCH-THROTTLE] AST OK")
print("[S171-COORD-DISPATCH-THROTTLE] DONE")
