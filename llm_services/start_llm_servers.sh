#!/bin/bash
# ============================================================================
# LLM Server Startup Script v2.0.0
# Starts DeepSeek-R1-14B as primary WATCHER agent
# Backup (Claude Opus) is via Claude Code CLI - no server needed
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PROJECT_DIR}/models"
LLAMA_CPP="${HOME}/llama.cpp"
LOG_DIR="${PROJECT_DIR}/logs/llm"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  Distributed PRNG Analysis - LLM Server v2.0"
echo "=============================================="
echo ""
echo "Architecture: DeepSeek-R1-14B (primary) + Claude Opus (backup)"
echo ""
echo "Model Directory: $MODEL_DIR"
echo "Log Directory: $LOG_DIR"
echo ""

# Check primary model exists
PRIMARY_MODEL="${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"

if [[ ! -f "$PRIMARY_MODEL" ]]; then
    echo "❌ ERROR: Primary model not found: $PRIMARY_MODEL"
    echo ""
    echo "   Download with:"
    echo "   huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF \\"
    echo "       DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf --local-dir $MODEL_DIR"
    exit 1
fi

echo "✅ Primary model found: $(basename $PRIMARY_MODEL)"
echo "   Size: $(ls -lh "$PRIMARY_MODEL" | awk '{print $5}')"
echo ""

# Kill existing servers
echo "Stopping any existing LLM servers..."
pkill -f "llama-server.*port 8080" 2>/dev/null || true
sleep 2

# Start Primary on both GPUs (Vulkan auto-splits)
echo "Starting Primary (DeepSeek-R1-14B) on port 8080..."
echo "   Backend: Vulkan (auto GPU split)"
echo ""

nohup "${LLAMA_CPP}/llama-server" \
    --model "$PRIMARY_MODEL" \
    --port 8080 \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --threads 12 \
    --batch-size 2048 \
    --host 0.0.0.0 \
    > "${LOG_DIR}/primary.log" 2>&1 &

PRIMARY_PID=$!
echo "   PID: $PRIMARY_PID"

# Wait for server to initialize
echo ""
echo "Waiting for server to initialize (20 seconds)..."
sleep 20

# Health check
echo ""
echo "Performing health check..."

check_health() {
    local name=$1
    local port=$2
    if curl -s --max-time 5 "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "   ✅ $name (port $port): HEALTHY"
        return 0
    else
        echo "   ❌ $name (port $port): NOT RESPONDING"
        return 1
    fi
}

HEALTHY=true
check_health "Primary (DeepSeek-R1-14B)" 8080 || HEALTHY=false

# Check Claude Code availability
echo ""
echo "Checking backup availability..."
if command -v claude &> /dev/null; then
    echo "   ✅ Claude Code CLI: AVAILABLE"
else
    echo "   ⚠️  Claude Code CLI: NOT INSTALLED"
    echo "      Install with: npm install -g @anthropic-ai/claude-code && claude login"
fi

echo ""
if $HEALTHY; then
    echo "=============================================="
    echo "  ✅ LLM Server started successfully!"
    echo "=============================================="
    echo ""
    echo "Primary Endpoint:"
    echo "  http://localhost:8080/completion"
    echo ""
    echo "Backup (Claude Opus 4.5):"
    echo "  Via Claude Code CLI (no server needed)"
    echo ""
    echo "Benchmarks:"
    echo "  Primary:  51 tok/s (prompt: 1472 tok/s)"
    echo "  Backup:   38 tok/s (via Claude Code)"
    echo ""
    echo "Logs:"
    echo "  ${LOG_DIR}/primary.log"
    echo ""
    echo "To stop server:"
    echo "  pkill -f 'llama-server.*port 8080'"
else
    echo "=============================================="
    echo "  ⚠️  Server failed to start!"
    echo "=============================================="
    echo ""
    echo "Check logs: ${LOG_DIR}/primary.log"
    exit 1
fi
