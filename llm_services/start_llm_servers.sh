#!/bin/bash
# ============================================================
# LLM Server Startup Script
# Version: 2.1.0 (Session 57 — context window 8K → 32K)
# Date: 2026-02-01
# ============================================================
#
# Starts DeepSeek-R1-14B on llama.cpp server (port 8080)
# Partitioned across BOTH RTX 3080 Ti GPUs (n_gpu_layers=99)
#
# Context: 32768 tokens (was 8192)
# KV cache estimate: ~2.6GB per GPU
# VRAM headroom: ~5.15GB per GPU
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/llm_server_config.json"

# Model path — adjust if model lives elsewhere
MODEL_DIR="$HOME/distributed_prng_analysis/models"
MODEL_FILE="DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# Server settings
PORT=8080
CTX_SIZE=32768
N_GPU_LAYERS=99
THREADS=4

# ---- Pre-flight checks ----

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Expected: $MODEL_FILE"
    exit 1
fi

echo "✅ Primary model found: $MODEL_FILE"

# Check if server already running
if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "⚠️  Server already running on port $PORT"
    echo "   Kill with: pkill -f 'llama-server'"
    exit 0
fi

# ---- Start primary (DeepSeek-R1-14B) ----

echo "Starting DeepSeek-R1-14B on port $PORT..."
echo "  ctx_size=$CTX_SIZE, kv_cache_est≈2.6GB/GPU, n_gpu_layers=$N_GPU_LAYERS"

nohup /home/michael/llama.cpp/build/bin/llama-server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --ctx-size $CTX_SIZE \
    --n-gpu-layers $N_GPU_LAYERS \
    --threads $THREADS \
    --log-disable \
    > /tmp/llm_server_primary.log 2>&1 &

PRIMARY_PID=$!
echo "  PID: $PRIMARY_PID"

# ---- Wait for health ----

echo -n "Waiting for health check..."
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        echo " ready! (${i}s)"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo " TIMEOUT after ${MAX_WAIT}s"
        echo "❌ Server failed to start. Check /tmp/llm_server_primary.log"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# ---- Verify backup ----

if command -v claude &> /dev/null; then
    echo "✅ Claude Code CLI: AVAILABLE"
else
    echo "⚠️  Claude Code CLI: NOT FOUND (backup unavailable)"
fi

# ---- Summary ----

echo ""
echo "============================================================"
echo "LLM Infrastructure Ready"
echo "  Primary: DeepSeek-R1-14B (port $PORT)"
echo "  Context: $CTX_SIZE tokens"
echo "  GPU layers: $N_GPU_LAYERS (dual 3080 Ti partition)"
echo "  Startup: ${i}s"
echo "============================================================"
echo ""
echo "LLM ctx_size=$CTX_SIZE, kv_cache_est≈2.6GB/GPU"
echo ""
echo "Stop with: pkill -f 'llama-server'"
