# SPDX-License-Identifier: Apache-2.0

import time

import zmq
from lmcache.integration.vllm.vllm_adapter import close_lmcache_engine

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

if __name__ == "__main__":

    # ZMQ socket for synchronization
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("ipc://localhost:44444")

    context = "Hi " * 1000
    context2 = "Hey " * 500
    prompts = [
        context + "Hello, my name is",
        context + "The capital of France is",
        context2 + "Your name is",
        context2 + "The capital of China is",
    ]

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.6,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        kv_transfer_config=KVTransferConfig.from_cli(
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer", '
            '"kv_connector_extra_config": {}}'
        )  #, max_model_len=2048, max_num_batched_tokens=2048)
    )

    # Prefiller should start processing after the decoder is ready
    msg = socket.recv()

    # 1ST generation (prefill instance)
    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    # Tell the decoder that prefilling is done
    socket.send(b"")

    new_prompts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
        print(
            f"Prompt: ....{prompt[-10:]!r}, Generated text: {generated_text!r}"
        )

    # Write new_prompts to output.txt
    with open("output.txt", "w") as f:
        for prompt in new_prompts:
            f.write(prompt + "\n")
    print(f"Saved {len(new_prompts)} prompts to output.txt")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        close_lmcache_engine()
