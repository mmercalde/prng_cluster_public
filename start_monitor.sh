#!/bin/bash
# Auto-start progress monitor with proper display

MONITOR_SCRIPT="$HOME/distributed_prng_analysis/progress_monitor.py"

# Check if already running
if pgrep -f "progress_monitor.py" > /dev/null; then
    echo "📊 Progress monitor already running"
    # If there's a tmux session, attach to it
    if tmux has-session -t prng_monitor 2>/dev/null; then
        tmux attach -t prng_monitor
    fi
    exit 0
fi

# If in tmux, split pane
if [ -n "$TMUX" ]; then
    tmux split-window -h -p 35 "python3 $MONITOR_SCRIPT"
    echo "📊 Progress monitor started in split pane"
else
    # Not in tmux - create session and attach
    tmux kill-session -t prng_monitor 2>/dev/null
    tmux new-session -s prng_monitor "python3 $MONITOR_SCRIPT"
fi
