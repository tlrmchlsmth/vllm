notes:
    UCX_RNDV_THRESH=0    # Force rendezvous protocol for all messages
    UCX_MEMTYPE_CACHE=n  # Disable memory type caching
    UCX_TLS=rc,ud,dc,cuda_copy,cuda_ipc,gdr_copy  # Prioritize RDMA transports
    UCX_ZCOPY_THRESH=0   # Force zero-copy for all sizes

prefill:
    UCX_LOG_LEVEL=debug \
    NIXL_ROLE="SENDER" \
    CUDA_VISIBLE_DEVICES=3 \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --port 8100 \
    --enforce-eager \
    --load-format dummy \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

decode:
    UCX_LOG_LEVEL=info \
    NIXL_ROLE="RECVER" \
    CUDA_VISIBLE_DEVICES=4 \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --port 8200 \
    --enforce-eager \
    --load-format dummy \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

proxy:
    python examples/disagg_proxy_server.py --port 8192

send_request:
  curl -X POST http://localhost:8192/v1/completions \
    -H "Content-Type: application/json" \
    -d '{ \
      "model": "meta-llama/Llama-3.2-1B-Instruct", \
      "prompt": "Generate a curl command to send to an openai server hosted at local_host:8192 with this as the", \
      "max_tokens": 150, \
      "temperature": 0.7 \
    }'
