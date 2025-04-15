#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <role>"
    exit 1
fi

# role could be "producer" or "consumer"

if [[ $1 == "producer" ]]; then
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=lmcache_prefill_config.yaml \
        LMCACHE_USE_EXPERIMENTAL=True \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=0 \
        vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --port 9101 \
        --disable-log-requests \
        --max-num-seqs 50 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {}}'

        #LMCACHE_LOG_LEVEL=DEBUG \
        #--enforce-eager \
        #--gpu-memory-utilization 0.6 \
        #--max-model-len 8192 \
        #--max-num-batched-tokens 8192 \
elif [[ $1 == "consumer" ]]; then
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=lmcache_decode_config.yaml \
        LMCACHE_USE_EXPERIMENTAL=True \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=1 \
        vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --port 9201 \
        --disable-log-requests \
        --max-num-seqs 50 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {}}'

        #LMCACHE_LOG_LEVEL=DEBUG \
        #--enforce-eager \
        #--gpu-memory-utilization 0.6 \
        #--max-model-len 8192 \
        #--max-num-batched-tokens 8192 \
else
    echo "Invalid role: $1"
    exit 1
fi
