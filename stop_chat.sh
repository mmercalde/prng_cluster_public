#!/bin/bash
# Stop LLM servers

echo "Stopping LLM servers..."
pkill -f 'llama-server.*port 808[01]'
sleep 1

# Verify
if pgrep -f 'llama-server.*port 808[01]' > /dev/null; then
    echo "❌ Some servers still running, force killing..."
    pkill -9 -f 'llama-server.*port 808[01]'
else
    echo "✅ All LLM servers stopped"
fi
