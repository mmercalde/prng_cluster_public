#!/usr/bin/env bash
# update_boot_notify_v2.sh
# v2: restore 🟢/🔴 status lights (original message format) + AMD-only GPU count.
# Rewrites /usr/local/bin/cluster_boot_notify.sh ONLY (conf + unit untouched).
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "ERROR: run as root"; exit 1; }
[[ -r /etc/cluster-boot-notify.conf ]] || { echo "ERROR: conf missing — run installer first"; exit 1; }

cat > /usr/local/bin/cluster_boot_notify.sh <<'EOS'
#!/usr/bin/env bash
set -u
source /etc/cluster-boot-notify.conf
HOST="$(hostname)"
IP="$(hostname -I | awk '{print $1}')"
TS="$(date -Is)"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"

# Count AMD GPUs only (PCI vendor 0x1002); excludes Biostar Intel iGPU render node.
GPU_COUNT=0
for v in /sys/class/drm/renderD*/device/vendor; do
    [[ -r "$v" ]] && [[ "$(cat "$v")" == "0x1002" ]] && GPU_COUNT=$((GPU_COUNT+1))
done
if [[ "$GPU_COUNT" -eq 0 ]]; then
    GPU_COUNT="$(lspci -nn 2>/dev/null | grep -Eic 'VGA.*(AMD|ATI)|Display.*(AMD|ATI)')"
fi

if [[ "$GPU_COUNT" -eq "$EXPECTED_GPUS" ]]; then
    GPU_STATUS="🟢 GPUs: ${GPU_COUNT}/${EXPECTED_GPUS} OK"
else
    GPU_STATUS="🔴 GPUs: ${GPU_COUNT}/${EXPECTED_GPUS} MISSING"
fi

if [[ -f /run/systemd/container || -f /.dockerenv ]] || grep -qa container= /proc/1/environ 2>/dev/null; then
    ROLE="CT"
else
    ROLE="HOST"
fi

MSG="🟢 ONLINE (${ROLE})
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
EOS
chmod 755 /usr/local/bin/cluster_boot_notify.sh
echo "[*] Patched to v2 — firing test..."
bash /usr/local/bin/cluster_boot_notify.sh
echo "[*] Done on $(hostname)."
