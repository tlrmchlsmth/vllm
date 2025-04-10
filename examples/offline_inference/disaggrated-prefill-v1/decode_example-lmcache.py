# SPDX-License-Identifier: Apache-2.0

import time

import zmq
from lmcache.integration.vllm.vllm_adapter import close_lmcache_engine
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

if __name__ == "__main__":

    # ZMQ socket for synchronization
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("ipc://localhost:44444")

    # Read prompts from output.txt
    prompts = []
    try:
        with open("output.txt") as f:
            for line in f:
                prompts.append(line.strip())
        print(f"Loaded {len(prompts)} prompts from output.txt")
    except FileNotFoundError:
        print("Error: output.txt file not found")
        exit(-1)

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.6,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        tensor_parallel_size=1,
        kv_transfer_config=KVTransferConfig.from_cli(
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer",'
            '"kv_connector_extra_config": {}}')
    )  #, max_model_len=2048, max_num_batched_tokens=2048)

    # Tell the prefiller that the llm is ready
    socket.send(b"ready")
    # Wait for the prefiller to finish
    print("Waiting for the prefiller to finish!")
    msg = socket.recv()
    print("Sleep for another 3 seconds!")
    for i in tqdm(range(3)):
        time.sleep(1)

    # 1ST generation (prefill instance)
    outputs = llm.generate(prompts, sampling_params)

    new_prompts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
        print(
            f"Prompt: ....{prompt[-10:]!r}, Generated text: {generated_text!r}"
        )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        close_lmcache_engine()
