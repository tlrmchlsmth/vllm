CUDA_VISIBLE_DEVICES=4 \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --disable-log-requests \
    --enforce-eager \
    --port 8300 \

#    --gpu-memory-utilization 0.6 \
#    --max-model-len 8192 \
#    --max-num-batched-tokens 8192 \
#    --max-num-partial-prefills 8 \   ######## THIS WILL FALL BACK TO V0
