#!/bin/bash
# Start LLM servers and launch chat

cd ~/distributed_prng_analysis

echo "Starting LLM servers..."
bash llm_services/start_llm_servers.sh

echo ""
echo "Waiting for servers to stabilize..."
sleep 3
echo "Launching chat..."
python3 llm_chat.py
