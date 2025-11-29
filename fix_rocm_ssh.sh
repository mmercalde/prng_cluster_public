#!/usr/bin/env bash
set -euo pipefail

# === USER SETTINGS (edit if your env differs) ==========================
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
RIGS=(${RIGS:-"192.168.3.120" "192.168.3.154"})
RIG_USER="${RIG_USER:-michael}"
AMD_PY="/home/michael/rocm_env/bin/python"
LOCALHOST="localhost"
# ======================================================================

cd "$PROJECT_DIR"

echo "==> Working directory: $PWD"

# Sanity: required files present
req=(coordinator.py distributed_worker.py enhanced_gpu_model_id.py distributed_config.json)
for f in "${req[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f in $PWD"; exit 1; }
done

# Backup dir
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup_$TS"
mkdir -p "$BK"
cp -v coordinator.py distributed_worker.py enhanced_gpu_model_id.py distributed_config.json "$BK"/

echo "==> Backups saved in $BK/"

# ---------- 1) PATCH coordinator.py -----------------------------------
python3 - "$PWD/coordinator.py" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()

# Ensure cmd_body uses the explicit interpreter
src = re.sub(
    r'(cmd_body\s*=\s*)f?"?python\s+-u\s+\{?worker\}?',
    r'\1f"{py} -u {worker}"',
    src,
    flags=re.M
)

# AMD branch: replace the big bash -lc + exports block with minimal run
pat_amd = re.compile(
    r'if self\._is_rocm\(node\):\n(.*?)else:\n',
    flags=re.S
)

def repl_amd(m):
    return (
        "if self._is_rocm(node):\n"
        "            # Minimal, robust AMD path: interpreter from config; env lives in worker files\n"
        "            return (\n"
        "                f\"cd '{node.script_path}' && \"\n"
        "                f\"cat > {payload_filename} <<'JSON'\\n{j}\\nJSON\\n\"\n"
        "                f\"if command -v timeout >/dev/null 2>&1; then \"\n"
        "                f\"timeout -k 10 {tmo} {cmd_body}; \"\n"
        "                f\"else {cmd_body}; fi ; \"\n"
        "                f\"[ -f {result_file} ] && cat {result_file} ; \"\n"
        "                f\"rm -f {payload_filename} {result_file} || true\"\n"
        "            )\n"
        "        else:\n"
    )

src2 = pat_amd.sub(repl_amd, src)
if src2 == src:
    print("WARN: AMD block not replaced (structure mismatch); continuing", file=sys.stderr)
src = src2

# CUDA timeout wrapper: ensure duration appears right after -k
src = re.sub(
    r'timeout -k 10\s*\{env_prefix\}\{tmo\}\s*\{py\}\s*-u',
    r'timeout -k 10 {tmo} {env_prefix}{py} -u',
    src
)

# Also fix other possible variant that inlines full cmd_body later
src = re.sub(
    r'timeout -k 10\s*\{env_prefix\}\{tmo\}\s*\{cmd_body\}',
    r'timeout -k 10 {tmo} {env_prefix}{cmd_body}',
    src
)

# And ensure the CUDA wrapper uses {cmd_body} consistently
src = re.sub(
    r'(timeout_wrapper\s*=\s*\(\n\s*?f"if command -v timeout .*? then\s*"\n\s*?\+\s*)'
    r'f?"timeout -k 10 .*?\\n"',
    r'\1f"timeout -k 10 {tmo} {env_prefix}{cmd_body}\\n"',
    src,
    flags=re.S
)

p.write_text(src)
print("Patched coordinator.py")
PY

# ---------- 2) Add ROCm prelude to enhanced_gpu_model_id.py ------------
python3 - "$PWD/enhanced_gpu_model_id.py" <<'PY'
import sys, pathlib, re
p = pathlib.Path(sys.argv[1])
s = p.read_text()

if "HSA_OVERRIDE_GFX_VERSION" in s and "CUPY_CACHE_DIR" in s and "HIP_PATH" in s:
    print("enhanced_gpu_model_id.py: prelude already present (skipping)")
else:
    # Insert right after shebang / header
    lines = s.splitlines(True)
    insert_at = 0
    # keep shebang if present
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    prelude = (
        "# ===== ROCm/NVIDIA environment prelude (MUST be first, before importing cupy) =====\n"
        "import os as _os, socket as _socket\n"
        "_HOST = _socket.gethostname()\n"
        "if _HOST in (\"rig-6600\", \"rig-6600b\"):\n"
        "    _os.environ.setdefault(\"HSA_OVERRIDE_GFX_VERSION\", \"10.3.0\")\n"
        "    _os.environ.setdefault(\"HSA_ENABLE_SDMA\", \"0\")\n"
        "    _os.environ.setdefault(\"ROCM_PATH\", \"/opt/rocm\")\n"
        "    _os.environ.setdefault(\"HIP_PATH\", \"/opt/rocm/hip\")\n"
        "    _os.environ.setdefault(\"CUPY_CACHE_DIR\", _os.path.expanduser(\"~/distributed_prng_analysis/.cache/cupy\"))\n"
        "\n"
    )
    # place before first cupy import
    s = "".join(lines[:insert_at]) + prelude + "".join(lines[insert_at:])
    p.write_text(s)
    print("Patched enhanced_gpu_model_id.py (inserted ROCm prelude)")
PY

# ---------- 3) Add ROCm prelude to distributed_worker.py ---------------
python3 - "$PWD/distributed_worker.py" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()
if "HSA_OVERRIDE_GFX_VERSION" in s and "CUPY_CACHE_DIR" in s and "HIP_PATH" in s:
    print("distributed_worker.py: prelude already present (skipping)")
else:
    inject = (
        "\n# ===== ROCm/NVIDIA environment prelude (MUST be before any cupy import anywhere) =====\n"
        "_HOST = socket.gethostname()\n"
        "if _HOST in (\"rig-6600\", \"rig-6600b\"):\n"
        "    os.environ.setdefault(\"HSA_OVERRIDE_GFX_VERSION\", \"10.3.0\")\n"
        "    os.environ.setdefault(\"HSA_ENABLE_SDMA\", \"0\")\n"
        "    os.environ.setdefault(\"ROCM_PATH\", \"/opt/rocm\")\n"
        "    os.environ.setdefault(\"HIP_PATH\", \"/opt/rocm/hip\")\n"
        "    os.environ.setdefault(\"CUPY_CACHE_DIR\", os.path.expanduser(\"~/distributed_prng_analysis/.cache/cupy\"))\n"
        "\n"
    )
    # After the initial imports block (right after typing import)
    anchor = "from typing import Dict, Any"
    idx = s.find(anchor)
    if idx != -1:
        idx += len(anchor)
        s = s[:idx] + inject + s[idx:]
    else:
        s = inject + s
    p.write_text(s)
    print("Patched distributed_worker.py (inserted ROCm prelude)")
PY

# ---------- 4) Warn if passwords are present in config -----------------
if grep -q '"password"' distributed_config.json; then
  echo "⚠️  WARNING: 'password' found in distributed_config.json."
  echo "   Consider switching to SSH keys and removing that field."
fi

# ---------- 5) Sync worker files to AMD rigs ---------------------------
echo "==> Syncing worker files to rigs..."
for IP in "${RIGS[@]}"; do
  scp -q distributed_worker.py enhanced_gpu_model_id.py "${RIG_USER}@${IP}:~/distributed_prng_analysis/"
  echo "   -> synced to ${IP}"
done

# ---------- 6) Verify file hashes locally vs remote --------------------
echo "==> Verifying hashes..."
sha256sum distributed_worker.py enhanced_gpu_model_id.py > .hash_local.txt
for IP in "${RIGS[@]}"; do
  ssh -o BatchMode=yes "${RIG_USER}@${IP}" "sha256sum ~/distributed_prng_analysis/distributed_worker.py ~/distributed_prng_analysis/enhanced_gpu_model_id.py" \
    | sed "s/^/${IP} /" >> .hash_remote.txt || true
done
echo "Local hashes:"
cat .hash_local.txt
echo "Remote hashes:"
cat .hash_remote.txt || true
echo "NOTE: visually compare – they should match for each file."

# ---------- 7) Remote smoke test (no coordinator) ----------------------
echo "==> Running remote smoke test..."
cat > /tmp/_remote_job.json <<'JSON'
{"job_id":"debug_remote","prng_type":"xorshift","mapping_type":"mod","seeds":[1,2,3,4],"samples":1000,"grid_size":4,"lmax":8}
JSON

for IP in "${RIGS[@]}"; do
  scp -q /tmp/_remote_job.json "${RIG_USER}@${IP}:~/distributed_prng_analysis/_remote_job.json"
  echo "---- ${IP} ----"
  ssh -o BatchMode=yes "${RIG_USER}@${IP}" "cd ~/distributed_prng_analysis && ${AMD_PY} -u distributed_worker.py _remote_job.json --gpu-id 0" \
    || { echo "Smoke test FAILED on ${IP}"; exit 1; }
done

echo "==> All done! If the smoke test printed success JSON from each rig, you're good."
