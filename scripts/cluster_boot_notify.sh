#!/usr/bin/env bash
set -u
source /etc/cluster-boot-notify.conf
HOST="$(hostname)"
IP="$(hostname -I | awk '{print $1}')"
TS="$(date -Is)"
EXPECTED_GPUS="${EXPECTED_GPUS:-2}"
# Count NVIDIA GPUs
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [[ "$GPU_COUNT" -eq "$EXPECTED_GPUS" ]]; then
    GPU_STATUS="🟢 GPUs: ${GPU_COUNT}/${EXPECTED_GPUS} OK"
else
    GPU_STATUS="🔴 GPUs: ${GPU_COUNT}/${EXPECTED_GPUS} MISSING"
fi

# ── [RESOLVED EXECUTION SET] consumer line ──────────────────────────────────
# Boot notify fires at BOOT, where no run exists and therefore no run-scoped set
# exists. What it can consume — and what makes it stop being a seventh
# independent opinion about the fleet — is the same DECLARED fleet definition the
# resolver reads: rig_profiles_config.json, joined against distributed_config.json.
# It reports how this host is declared there, next to what it actually sees.
#
# Deliberately preserved, because both are correct and neither is a defect:
# this script BLOCKS NOTHING and it is TELEGRAM-ONLY. The block below cannot
# change the GPU verdict above, cannot fail the script, and cannot alter the
# unconditional `exit 0` at the end — every step is guarded and every failure
# leaves SET_LINE empty. EXPECTED_GPUS still comes from the per-host conf; this
# only says what the fleet definition declares, so a divergence becomes visible
# instead of staying two numbers nobody ever compared.
SET_LINE=""
TFM_ROOT="${TFM_ROOT:-/home/michael/distributed_prng_analysis}"
if command -v python3 >/dev/null 2>&1 && [ -f "${TFM_ROOT}/rig_profiles_config.json" ]; then
    SET_LINE="$(TFM_ROOT="$TFM_ROOT" HOST="$HOST" python3 - <<'PY' 2>/dev/null || true
import json, os
try:
    root = os.environ["TFM_ROOT"]
    host = os.environ["HOST"]
    with open(os.path.join(root, "rig_profiles_config.json")) as f:
        pmap = json.load(f)
    with open(os.path.join(root, "distributed_config.json")) as f:
        cfg = {n.get("hostname"): n for n in json.load(f).get("nodes", [])}
    for n in pmap.get("nodes", []):
        if n.get("worker_hostname") != host:
            continue
        declared = cfg.get(n.get("config_hostname"), {}).get("gpu_count")
        eps = n.get("endpoints", {}) or {}
        print("📋 Fleet: node %s declared %s GPU(s); endpoints %s" % (
            n.get("node_id"),
            declared if declared is not None else "?",
            ", ".join("%s=%s" % (k, v) for k, v in sorted(eps.items()))))
        break
except Exception:
    pass
PY
)"
fi

MSG="🟢 ONLINE
Host: ${HOST}
IP: ${IP}
Time: ${TS}
${GPU_STATUS}${SET_LINE:+
${SET_LINE}}"
curl -sS --connect-timeout 5 --max-time 8 \
  -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
  -d "chat_id=${CHAT_ID}" \
  --data-urlencode "text=${MSG}" \
  >/dev/null
exit 0
