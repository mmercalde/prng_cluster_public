#!/usr/bin/env bash
# install_boot_notify_amd.sh
# Telegram boot-notification installer — AMD / Proxmox variant
# Targets: Proxmox rig hosts (pve6600b/.155, pve6600c/.163) and CT100 workers (.156, .164)
# Mirrors the bare-metal rig pattern documented in docs/TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md
#
# Usage (run as root on the target):
#   BOT_TOKEN=xxx CHAT_ID=yyy ./install_boot_notify_amd.sh
#   ./install_boot_notify_amd.sh              # prompts interactively
#   ./install_boot_notify_amd.sh --test-only  # re-fire existing install
#
# Idempotent: safe to re-run; overwrites script/unit, preserves conf unless new creds given.

set -euo pipefail

CONF=/etc/cluster-boot-notify.conf
SCRIPT=/usr/local/bin/cluster_boot_notify.sh
UNIT=/etc/systemd/system/cluster-boot-notify.service

if [[ "${1:-}" == "--test-only" ]]; then
    echo "[*] Manual test fire..."
    bash "$SCRIPT"
    echo "[*] Sent (check Telegram)."
    exit 0
fi

[[ $EUID -eq 0 ]] || { echo "ERROR: run as root"; exit 1; }

# ---------- credentials ----------
if [[ -z "${BOT_TOKEN:-}" || -z "${CHAT_ID:-}" ]]; then
    if [[ -r "$CONF" ]]; then
        echo "[*] Existing $CONF found — reusing credentials."
        # shellcheck disable=SC1090
        source "$CONF"
    fi
fi
if [[ -z "${BOT_TOKEN:-}" ]]; then read -rp "BOT_TOKEN: " BOT_TOKEN; fi
if [[ -z "${CHAT_ID:-}"  ]]; then read -rp "CHAT_ID: "  CHAT_ID;  fi
[[ -n "$BOT_TOKEN" && -n "$CHAT_ID" ]] || { echo "ERROR: BOT_TOKEN and CHAT_ID required"; exit 1; }

EXPECTED_GPUS="${EXPECTED_GPUS:-8}"

# ---------- conf ----------
cat > "$CONF" <<EOF
BOT_TOKEN=${BOT_TOKEN}
CHAT_ID=${CHAT_ID}
EXPECTED_GPUS=${EXPECTED_GPUS}
EOF
# S159G lesson: root:michael 640 where michael exists (CT100); root:root 600 on Proxmox hosts
if id michael &>/dev/null; then
    chown root:michael "$CONF"; chmod 640 "$CONF"
else
    chown root:root "$CONF"; chmod 600 "$CONF"
fi
echo "[*] Wrote $CONF ($(stat -c '%U:%G %a' "$CONF"))"

# ---------- notify script (AMD variant) ----------
cat > "$SCRIPT" <<'EOS'
#!/usr/bin/env bash
set -u
source /etc/cluster-boot-notify.conf
HOST="$(hostname)"
IP="$(hostname -I | awk '{print $1}')"
TS="$(date -Is)"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"

# AMD GPU count: renderD nodes (works on Proxmox host with amdgpu loaded
# AND inside LXC with /dev/dri bind-mounted); lspci fallback for host.
GPU_COUNT="$(ls /dev/dri/renderD* 2>/dev/null | wc -l)"
if [[ "$GPU_COUNT" -eq 0 ]]; then
    GPU_COUNT="$(lspci -nn 2>/dev/null | grep -Eic 'VGA.*(AMD|ATI)|Display.*(AMD|ATI)')"
fi

if [[ "$GPU_COUNT" -eq "$EXPECTED_GPUS" ]]; then
    GPU_STATUS="GPUs: ${GPU_COUNT}/${EXPECTED_GPUS} OK"
else
    GPU_STATUS="GPUs: ${GPU_COUNT}/${EXPECTED_GPUS} MISSING"
fi

# Label host vs container so the two notifies per rig are distinguishable
if [[ -f /run/systemd/container || -f /.dockerenv ]] || grep -qa container= /proc/1/environ 2>/dev/null; then
    ROLE="CT"
else
    ROLE="HOST"
fi

MSG="ONLINE (${ROLE})
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

# ---------- systemd unit ----------
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
echo "[*] Enabled cluster-boot-notify.service"

# ---------- verify + test ----------
echo "[*] enabled: $(systemctl is-enabled cluster-boot-notify.service)"
echo "[*] Firing manual test..."
bash "$SCRIPT"
echo "[*] Done — check Telegram for ONLINE message from $(hostname)."
