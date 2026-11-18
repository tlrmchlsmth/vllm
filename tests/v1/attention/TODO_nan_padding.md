# NaN Padding Test Coverage Gaps

## TODO

### Shared experts (DeepseekV2MoE)
DeepSeek-R1 has shared experts that run in parallel with routed MoE.
The shared expert output is added to routed expert output
(`final_hidden_states += shared_output` in deepseek_v2.py:386-388).
If the shared expert computation on NaN-padded hidden_states produces
NaN, it corrupts the final output even if the MoE path is clean.

Currently the shared expert MLP is tested in isolation
(`test_deepseek_dense_mlp_nvfp4_cudagraph_nan_padding`), but the
combined path through `DeepseekV2MoE.forward()` is not tested because
it requires `get_ep_group()` and distributed setup.

### Multi-layer accumulation
The tests run a single decoder layer. Production runs 61 layers in
sequence with residual connections. Even if each layer individually
handles padding, floating-point noise from near-boundary effects could
compound across layers. A 2-3 layer sequential test would catch
accumulation bugs.

### Block table with stale padding entries
The tests carefully zero-pad the block table. In production, block
table entries for padding requests might contain stale block indices
from previous iterations pointing to random KV cache blocks. If the
attention kernel follows those indices, it reads from unrelated cache
blocks.

### Varying CUDA graph bucket sizes
Production captures graphs at multiple batch sizes. When batch size
drops (e.g., 32→8), the graph captured at 32 replays with 24 padding
tokens whose buffers contain results from the previous 32-token
iteration. The tests always capture and replay at a single size.

### LM head + sampler interaction
The lm_head linear projection runs on the full padded hidden_states.
If the lm_head output for padding positions is NaN and any kernel
reads beyond logits_indices, it could corrupt. Worth a smoke test.

### FlashInfer NVLink one-sided all2all backend
Not tested because it requires vLLM's `get_ep_group()` distributed
setup (not just `torch.distributed`). This is one of the production
all2all backends (`--all2all-backend flashinfer_nvlink_one_sided`).

### enable_pdl=False experiment
The TRTLLM MLA decode kernel (`nvjet_sm100_tst_*`) cannot be
instrumented by compute-sanitizer. PDL (Programmatic Dependent Launch)
issues in these kernels would go undetected. Adding `enable_pdl=False`
to the `trtllm_batch_decode_with_kv_cache_mla` call in
`flashinfer_mla.py:190` could help isolate timing-dependent NaN.
