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
MSG="🟢 ONLINE
Host: ${HOST}
IP: ${IP}
Time: ${TS}
${GPU_STATUS}"
curl -sS --connect-timeout 5 --max-time 8 \
  -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
  -d "chat_id=${CHAT_ID}" \
  --data-urlencode "text=${MSG}" \
  >/dev/null
exit 0
