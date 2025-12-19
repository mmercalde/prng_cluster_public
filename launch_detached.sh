#!/bin/bash
# Usage: launch_detached.sh <command> <log_file>
# Runs command fully detached and returns PID immediately

nohup bash -c "$1" </dev/null >"$2" 2>&1 &
echo $!
