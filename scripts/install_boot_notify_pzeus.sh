#!/usr/bin/env bash
# install_boot_notify_pzeus.sh
# Telegram boot-notification installer — pzeus Proxmox host (192.168.3.128)
# GPU count via lspci (NVIDIA VGA/3D devices): works even though passthrough
# cards are vfio-bound with no host driver (nvidia-smi/renderD unusable here).
# Expected: 3 = 2x RTX 3080 Ti (passthrough) + 1x GTX 1660 Ti.
#
# Usage (as root on pzeus):
#   BOT_TOKEN=xxx CHAT_ID=yyy ./install_boot_notify_pzeus.sh
set -euo pipefail

CONF=/etc/cluster-boot-notify.conf
SCRIPT=/usr/local/bin/cluster_boot_notify.sh
UNIT=/etc/systemd/system/cluster-boot-notify.service

[[ $EUID -eq 0 ]] || { echo "ERROR: run as root"; exit 1; }

if [[ -z "${BOT_TOKEN:-}" || -z "${CHAT_ID:-}" ]] && [[ -r "$CONF" ]]; then
    # shellcheck disable=SC1090
    source "$CONF"
fi
if [[ -z "${BOT_TOKEN:-}" ]]; then read -rp "BOT_TOKEN: " BOT_TOKEN; fi
if [[ -z "${CHAT_ID:-}"  ]]; then read -rp "CHAT_ID: "  CHAT_ID;  fi
[[ -n "$BOT_TOKEN" && -n "$CHAT_ID" ]] || { echo "ERROR: BOT_TOKEN and CHAT_ID required"; exit 1; }

EXPECTED_GPUS="${EXPECTED_GPUS:-3}"

cat > "$CONF" <<EOF
BOT_TOKEN=${BOT_TOKEN}
CHAT_ID=${CHAT_ID}
EXPECTED_GPUS=${EXPECTED_GPUS}
EOF
chown root:root "$CONF"; chmod 600 "$CONF"
echo "[*] Wrote $CONF (root:root 600)"

cat > "$SCRIPT" <<'EOS'
#!/usr/bin/env bash
set -u
source /etc/cluster-boot-notify.conf
HOST="$(hostname)"
IP="$(hostname -I | awk '{print $1}')"
TS="$(date -Is)"
EXPECTED_GPUS="${EXPECTED_GPUS:-3}"

# NVIDIA GPU count via lspci — driver-independent, so vfio-bound passthrough
# cards are still counted. (nvidia-smi and /sys/class/drm are unavailable for
# cards without a host driver.)
GPU_COUNT="$(lspci -nn 2>/dev/null | grep -Eic '(VGA|3D).*NVIDIA')"

if [[ "$GPU_COUNT" -eq "$EXPECTED_GPUS" ]]; then
    GPU_STATUS="🟢 GPUs: ${GPU_COUNT}/${EXPECTED_GPUS} OK"
else
    GPU_STATUS="🔴 GPUs: ${GPU_COUNT}/${EXPECTED_GPUS} MISSING"
fi

MSG="🟢 ONLINE (HOST)
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
chmod 755 "$SCRIPT"
echo "[*] Wrote $SCRIPT (755)"

cat > "$UNIT" <<'EOU'
[Unit]
Description=Cluster boot Telegram notification
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/cluster_boot_notify.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOU
systemctl daemon-reload
systemctl enable cluster-boot-notify.service >/dev/null 2>&1
echo "[*] enabled: $(systemctl is-enabled cluster-boot-notify.service)"
echo "[*] Firing manual test..."
bash "$SCRIPT"
echo "[*] Done — check Telegram for ONLINE (HOST) from $(hostname)."
