#!/usr/bin/env bash
# =============================================================================
# cluster_notify.sh -- Runtime Telegram notification script
# =============================================================================
# Session 77: WATCHER Telegram Integration
#
# Sibling to cluster_boot_notify.sh (boot-time only).
# This script handles runtime notifications from WATCHER.
#
# Shares credentials with boot notifier via /etc/cluster-boot-notify.conf
# Does NOT modify or replace the existing boot notification service.
#
# Usage:
#   /usr/local/bin/cluster_notify.sh "Your message here"
#
# Properties:
#   - Non-blocking (best-effort)
#   - Silent on failure (exit 0 always)
#   - Reuses existing Telegram bot credentials
#   - Zeus-only (worker rigs do not need this)
#
# Install:
#   sudo install -m 0755 cluster_notify.sh /usr/local/bin/cluster_notify.sh
# =============================================================================

set -u

CONF="/etc/cluster-boot-notify.conf"

# Silent exit if config missing
if [ ! -r "$CONF" ]; then
    exit 0
fi

source "$CONF"

# Silent exit if vars not set
if [ -z "${BOT_TOKEN:-}" ] || [ -z "${CHAT_ID:-}" ]; then
    exit 0
fi

MSG="${1:-}"
if [ -z "$MSG" ]; then
    exit 0
fi

HOST="$(hostname)"
TS="$(date '+%Y-%m-%d %H:%M:%S')"

FINAL_MSG="[WATCHER]
Host: ${HOST}
Time: ${TS}

${MSG}"

curl -sS --connect-timeout 5 --max-time 8 \
    -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -d "chat_id=${CHAT_ID}" \
    --data-urlencode "text=${FINAL_MSG}" \
    >/dev/null 2>&1

exit 0
