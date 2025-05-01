#!/bin/bash

set -xe

# Model to run.
MODEL_NAME=Qwen/Qwen3-0.6B

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Prefill instance.
CUDA_VISIBLE_DEVICES=2 NIXL_ROLE="SENDER" vllm serve $MODEL_NAME \
    --port 8100 \
    --enforce-eager \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

# Decode instance.
CUDA_VISIBLE_DEVICES=7 NIXL_ROLE="RECVER" vllm serve $MODEL_NAME \
    --port 8200 \
    --enforce-eager \
    --disable-log-requests \
    --num_gpu_blocks_override 1000 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# Proxy server.
python toy_proxy_server.py --port 8192

# Run lm eval.
# python3 -m pytest -s -x test_accuracy.py
