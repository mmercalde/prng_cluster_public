#!/bin/bash
# ============================================================================
# Dual-LLM Server Startup Script
# Distributed PRNG Analysis System - Schema v1.0.4
# 
# Starts two specialized LLM servers:
#   - Orchestrator (Qwen2.5-Coder-14B) on GPU0:8080
#   - Math Specialist (Qwen2.5-Math-7B) on GPU1:8081
#
# Usage:
#   ./start_llm_servers.sh [start|stop|restart|status]
#
# Requirements:
#   - llama.cpp built with CUDA support
#   - Model files in ./models/ directory
#   - Two RTX 3080 Ti GPUs (or equivalent with 12GB+ VRAM each)
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PROJECT_DIR}/models"
LOG_DIR="${PROJECT_DIR}/logs/llm"
PID_DIR="${PROJECT_DIR}/run"

# Adjust this path to your llama.cpp installation
LLAMA_CPP="${LLAMA_CPP_PATH:-${HOME}/llama.cpp}"

# Model filenames
ORCHESTRATOR_MODEL="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"
MATH_MODEL="Qwen2.5-Math-7B-Instruct-Q5_K_M.gguf"

# Server ports
ORCHESTRATOR_PORT=8080
MATH_PORT=8081

# Create directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "=============================================="
    echo "  Distributed PRNG Analysis - LLM Servers"
    echo "  Schema Version: 1.0.4"
    echo "=============================================="
    echo ""
}

check_llama_cpp() {
    if [[ ! -f "${LLAMA_CPP}/llama-server" ]]; then
        echo "❌ ERROR: llama-server not found at ${LLAMA_CPP}/llama-server"
        echo ""
        echo "Please install llama.cpp:"
        echo "  cd ~"
        echo "  git clone https://github.com/ggerganov/llama.cpp.git"
        echo "  cd llama.cpp"
        echo "  LLAMA_CUDA=1 make -j\$(nproc)"
        echo ""
        echo "Or set LLAMA_CPP_PATH environment variable to your installation."
        exit 1
    fi
}

check_models() {
    local missing=false
    
    if [[ ! -f "${MODEL_DIR}/${ORCHESTRATOR_MODEL}" ]]; then
        echo "❌ Orchestrator model not found: ${MODEL_DIR}/${ORCHESTRATOR_MODEL}"
        echo "   Download: huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF qwen2.5-coder-14b-instruct-q4_k_m.gguf --local-dir ${MODEL_DIR}"
        missing=true
    fi
    
    if [[ ! -f "${MODEL_DIR}/${MATH_MODEL}" ]]; then
        echo "❌ Math model not found: ${MODEL_DIR}/${MATH_MODEL}"
        echo "   Download: huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct-GGUF qwen2.5-math-7b-instruct-q5_k_m.gguf --local-dir ${MODEL_DIR}"
        missing=true
    fi
    
    if $missing; then
        echo ""
        echo "Please download the required models and try again."
        exit 1
    fi
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi not found. CUDA drivers may not be installed."
        exit 1
    fi
    
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [[ $gpu_count -lt 2 ]]; then
        echo "⚠️  Warning: Only $gpu_count GPU(s) detected. Dual-LLM requires 2 GPUs."
        echo "   Orchestrator and Math servers will compete for GPU resources."
    fi
    
    echo "GPUs detected: $gpu_count"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
}

is_running() {
    local port=$1
    local pid_file="${PID_DIR}/llm_${port}.pid"
    
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

get_pid() {
    local port=$1
    local pid_file="${PID_DIR}/llm_${port}.pid"
    
    if [[ -f "$pid_file" ]]; then
        cat "$pid_file"
    fi
}

start_server() {
    local name=$1
    local model=$2
    local port=$3
    local gpu_id=$4
    local ctx_size=$5
    
    local pid_file="${PID_DIR}/llm_${port}.pid"
    local log_file="${LOG_DIR}/${name}.log"
    
    if is_running $port; then
        echo "   ⚠️  $name already running on port $port (PID: $(get_pid $port))"
        return 0
    fi
    
    echo "   Starting $name on GPU${gpu_id}:${port}..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup "${LLAMA_CPP}/llama-server" \
        --model "${MODEL_DIR}/${model}" \
        --port $port \
        --ctx-size $ctx_size \
        --n-gpu-layers 99 \
        --threads 8 \
        --batch-size 512 \
        --host 0.0.0.0 \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo $pid > "$pid_file"
    echo "   PID: $pid"
}

stop_server() {
    local name=$1
    local port=$2
    
    local pid_file="${PID_DIR}/llm_${port}.pid"
    
    if is_running $port; then
        local pid=$(get_pid $port)
        echo "   Stopping $name (PID: $pid)..."
        kill $pid 2>/dev/null || true
        rm -f "$pid_file"
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 $pid 2>/dev/null; then
                echo "   ✅ $name stopped"
                return 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        kill -9 $pid 2>/dev/null || true
        echo "   ✅ $name force stopped"
    else
        echo "   ℹ️  $name not running"
    fi
}

health_check() {
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

wait_for_startup() {
    local max_wait=$1
    echo ""
    echo "Waiting for servers to initialize (up to ${max_wait}s)..."
    
    for i in $(seq 1 $max_wait); do
        local orch_ok=false
        local math_ok=false
        
        curl -s --max-time 2 "http://localhost:${ORCHESTRATOR_PORT}/health" > /dev/null 2>&1 && orch_ok=true
        curl -s --max-time 2 "http://localhost:${MATH_PORT}/health" > /dev/null 2>&1 && math_ok=true
        
        if $orch_ok && $math_ok; then
            echo "   Both servers ready after ${i}s"
            return 0
        fi
        
        sleep 1
    done
    
    echo "   ⚠️  Timeout waiting for servers"
    return 1
}

show_status() {
    echo "Server Status:"
    echo ""
    
    if is_running $ORCHESTRATOR_PORT; then
        echo "   Orchestrator (Qwen2.5-Coder-14B):"
        echo "      Status: RUNNING (PID: $(get_pid $ORCHESTRATOR_PORT))"
        echo "      Port: $ORCHESTRATOR_PORT"
        echo "      GPU: 0"
        health_check "Health" $ORCHESTRATOR_PORT 2>/dev/null || true
    else
        echo "   Orchestrator: STOPPED"
    fi
    
    echo ""
    
    if is_running $MATH_PORT; then
        echo "   Math Specialist (Qwen2.5-Math-7B):"
        echo "      Status: RUNNING (PID: $(get_pid $MATH_PORT))"
        echo "      Port: $MATH_PORT"
        echo "      GPU: 1"
        health_check "Health" $MATH_PORT 2>/dev/null || true
    else
        echo "   Math Specialist: STOPPED"
    fi
    
    echo ""
    echo "Logs: ${LOG_DIR}/"
    echo "PIDs: ${PID_DIR}/"
}

# ============================================================================
# Main Commands
# ============================================================================

do_start() {
    print_header
    
    echo "Checking prerequisites..."
    check_llama_cpp
    check_models
    check_gpu
    echo ""
    
    echo "Starting LLM servers..."
    start_server "orchestrator" "$ORCHESTRATOR_MODEL" $ORCHESTRATOR_PORT 0 16384
    start_server "math" "$MATH_MODEL" $MATH_PORT 1 8192
    
    wait_for_startup 60
    
    echo ""
    echo "Health checks:"
    local all_healthy=true
    health_check "Orchestrator" $ORCHESTRATOR_PORT || all_healthy=false
    health_check "Math Specialist" $MATH_PORT || all_healthy=false
    
    echo ""
    if $all_healthy; then
        echo "=============================================="
        echo "  ✅ All LLM servers started successfully!"
        echo "=============================================="
        echo ""
        echo "Endpoints:"
        echo "  Orchestrator: http://localhost:${ORCHESTRATOR_PORT}/completion"
        echo "  Math:         http://localhost:${MATH_PORT}/completion"
        echo ""
        echo "Test with:"
        echo "  python -m llm_services.llm_router --health"
        echo "  python -m llm_services.llm_router --query 'Calculate 2^32 mod 1000'"
    else
        echo "=============================================="
        echo "  ⚠️  Some servers failed to start!"
        echo "=============================================="
        echo "Check logs:"
        echo "  tail -f ${LOG_DIR}/orchestrator.log"
        echo "  tail -f ${LOG_DIR}/math.log"
        exit 1
    fi
}

do_stop() {
    print_header
    
    echo "Stopping LLM servers..."
    stop_server "orchestrator" $ORCHESTRATOR_PORT
    stop_server "math" $MATH_PORT
    
    # Also kill any orphaned processes
    pkill -f "llama-server.*port ${ORCHESTRATOR_PORT}" 2>/dev/null || true
    pkill -f "llama-server.*port ${MATH_PORT}" 2>/dev/null || true
    
    echo ""
    echo "✅ All LLM servers stopped"
}

do_restart() {
    do_stop
    sleep 3
    do_start
}

do_status() {
    print_header
    show_status
}

# ============================================================================
# Entry Point
# ============================================================================

case "${1:-start}" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_restart
        ;;
    status)
        do_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
