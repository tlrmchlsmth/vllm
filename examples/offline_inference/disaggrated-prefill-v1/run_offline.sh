#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <role>"
    exit 1
fi

# role could be "producer" or "consumer"

if [[ $1 == "producer" ]]; then
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        LMCACHE_CONFIG_FILE=lmcache_prefill_config.yaml \
        LMCACHE_USE_EXPERIMENTAL=True \
        CUDA_VISIBLE_DEVICES=0 \
        python3 prefill_example-lmcache.py

elif [[ $1 == "consumer" ]]; then
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        LMCACHE_CONFIG_FILE=lmcache_decode_config.yaml \
        LMCACHE_USE_EXPERIMENTAL=True \
        CUDA_VISIBLE_DEVICES=1 \
        python3 decode_example-lmcache.py

else
    echo "Invalid role: $1"
    echo "Should be 'producer' or 'consumer'"
    exit 1
fi
