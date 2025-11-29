#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR:-$PWD}"

RIGS=("192.168.3.120" "192.168.3.154")
RUSER="${RUSER:-michael}"
RDIR="/home/michael/distributed_prng_analysis"

# collect local python files (including rig-specific variants)
CANDIDATES=()
for f in distributed_worker*.py enhanced_gpu_model_id*.py; do
  [[ -f "$f" ]] && CANDIDATES+=("$f")
done

read -r -d '' PRELUDE <<'PY'
# ROCm environment setup - MUST BE FIRST
import os, socket
HOST = socket.gethostname()
if HOST in ("rig-6600", "rig-6600b"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm/hip")
# Optional cache dir to avoid permissions surprises
os.environ.setdefault("CUPY_CACHE_DIR", os.path.expanduser("~/distributed_prng_analysis/.cache/cupy"))
PY

fix_one () {
  local f="$1"
  echo "• normalizing $f"
  cp -v "$f" "${f}.bak.$(date +%Y%m%d_%H%M%S)"
  python3 - "$f" <<<"$PRELUDE" <<'PY'
import sys, re, pathlib
path = pathlib.Path(sys.argv[1])
s = path.read_text()

shebang = "#!/usr/bin/env python3\n"
if s.startswith("#!/"):
    nl = s.find("\n")
    shebang = s[:nl+1]
    s = s[nl+1:]

s = re.sub(r'\n?#\s*ROCm environment setup - MUST BE FIRST.*?(?=\n[^#]|\Z)', '\n', s, flags=re.S)
s = re.sub(r'#\s*===== ROCm/NVIDIA environment prelude .*?\n', '', s)

prelude = sys.stdin.read().rstrip() + "\n\n"
s = shebang + prelude + s.lstrip()
s = re.sub(r'\n{3,}', '\n\n', s, count=1)

path.write_text(s)
print("fixed")
PY
}

echo "==> normalizing local files"
for f in "${CANDIDATES[@]}"; do fix_one "$f"; done

echo "==> syncing to rigs"
for ip in "${RIGS[@]}"; do
  for f in "${CANDIDATES[@]}"; do
    scp -q "$f" "${RUSER}@${ip}:${RDIR}/" && echo "   -> ${ip}:${f}"
  done
done

echo "==> remote sanity"
for ip in "${RIGS[@]}"; do
  echo "---- ${ip} ----"
  ssh -o BatchMode=yes "${RUSER}@${ip}" "head -n 15 ${RDIR}/enhanced_gpu_model_id.py | nl"
  ssh -o BatchMode=yes "${RUSER}@${ip}" 'source ~/rocm_env/bin/activate && python -c "import cupy as cp; print(\"OK devices:\", cp.cuda.runtime.getDeviceCount())"'
done

echo "✅ done. Next:"
echo "   python3 coordinator.py daily3.json -c distributed_config.json --test-only"
