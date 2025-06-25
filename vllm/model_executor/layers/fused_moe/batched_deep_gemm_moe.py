# SPDX-License-Identifier: Apache-2.0
import importlib.util
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)

    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)

    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,

    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,

    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,

    # Stride for counts (elements)
    stride_counts_e,

    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,

    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK)
    cols = cols.to(tl.int64)
    mask_h = cols < BLOCK

    t = tl.zeros([], tl.int64)
    while t < n_tokens:
        base_i_offset = (e * stride_i_e + t * stride_i_t +
                         g * GROUP_SIZE * stride_i_h)
        base_yq_offset = (e * stride_yq_e + t * stride_yq_t +
                          g * GROUP_SIZE * stride_yq_h)
        base_ys_offset = e * stride_ys_e + t * stride_ys_t + g * stride_ys_g

        mask = mask_h
        x = tl.load(input_ptr + base_i_offset + cols * stride_i_h,
                    mask=mask,
                    other=0.0).to(tl.float32)
        y2 = tl.load(input_ptr + base_i_offset + H * stride_i_h +
                     cols * stride_i_h,
                     mask=mask,
                     other=0.0).to(tl.float32)

        x = x * (1.0 / (1.0 + tl.exp(-x)))
        y = x * y2

        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + cols * stride_yq_h, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset, y_s)

        t += 1


def silu_mul_fp8_quant_deep_gemm(
    y: torch.Tensor,  # (E, T, 2*H) float32
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    group_size: int = 128,
    eps: float = 1e-10,
):
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales

    y has shape (E, T, 2*H). The first half of the last dimension is 
    silu-activated, multiplied by the second half, then quantized into FP8.

    Returns `(y_q, y_s)` where
    * `y_q` is the FP8 tensor of shape `(E, T, H)`, same layout as `y[..., :H]`.
    * `y_s` has shape `(E, T, H // group_size)` and strides `(T*G, 1, T)`
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = H // group_size
    assert H % group_size == 0, "H must be divisible by group_size"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E, \
        "tokens_per_expert must be shape (E,)"
    tokens_per_expert = tokens_per_expert.to(device=y.device,
                                             dtype=torch.int32)

    # allocate outputs
    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    # strides (elements)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    # desired scale strides (elements): (T*G, 1, T)
    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    y_s = torch.empty_strided((E, T, G),
                              (stride_ys_e, stride_ys_t, stride_ys_g),
                              dtype=torch.float32,
                              device=y.device)

    stride_cnt_e = tokens_per_expert.stride()[0]

    # static grid over experts and H-groups.
    # A loop inside the kernel handles the token dim
    grid = (E * G, )

    f_info = torch.finfo(fp8_dtype)
    fp8_max = f_info.max
    fp8_min = f_info.min

    _silu_mul_fp8_quant_deep_gemm[grid](
        y,
        y_q,
        y_s,
        tokens_per_expert,
        H,
        group_size,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yq_e,
        stride_yq_t,
        stride_yq_h,
        stride_ys_e,
        stride_ys_t,
        stride_ys_g,
        stride_cnt_e,
        eps,
        fp8_min,
        fp8_max,
        BLOCK=group_size,
        num_warps=4,
    )

    return y_q, y_s


@triton.jit
def _silu_and_mul_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr + expert_id * stride_output_scale_0 + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE):
        gate = tl.load(input_ptr_offs + token_index * stride_input_1, mask=offs_in_d < size_n, other=0.0).to(tl.float32)
        up = tl.load(input_ptr_offs + token_index * stride_input_1 + size_n, mask=offs_in_d < size_n, other=0.0)
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(output_ptr.dtype.element_ty)
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
        )


def silu_and_mul_masked_post_quant_fwd(
    input: torch.Tensor,  masked_m: torch.Tensor, quant_group_size: int = 128,
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
    output_scale [expert_num token_num_paddded, hidden_dim // 2 // 128] dtype float32
    quant_group_size  int,
    masked_m shape [expert_num],
    """

    assert input.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    E, T, H = input.shape
    G = H // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    output = torch.empty((E, T, H), dtype=fp8_dtype, device=input.device)

    # desired scale strides (elements): (T*G, 1, T)
    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    output_scale = torch.empty_strided((E, T, G),
                              (stride_ys_e, stride_ys_t, stride_ys_g),
                              dtype=torch.float32,
                              device=input.device)

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N % quant_group_size == 0

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    _silu_and_mul_post_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
    )
    return output, output_scale


class BatchedDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    # The Deep Gemm kernels only support block size of 128
    DEEPGEMM_BLOCK_SHAPE = 128

    def __init__(self, max_num_tokens: int, world_size: int, dp_size: int,
                 block_shape: list[int]):
        """
        max_num_tokens: Maximum number of tokens from a DP Rank
        world_size: Number of EP ranks
        dp_size: Number of data-parallel ranks
        block_shape: Block quantization block shape
        """
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.dp_size = dp_size
        self.block_shape = block_shape

        assert (len(self.block_shape) == 2 and all(
            [v == self.DEEPGEMM_BLOCK_SHAPE for v in self.block_shape]))

    def supports_chunking(self) -> bool:
        return False

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        assert a.dim() == 2
        # FIXME (varun): We should be able to dispatch only from the leader
        # DP ranks in the case of TP > 1. At the moment, all the Ranks
        # end up sending their tokens. This needs to be fixed.
        num_dispatchers = self.world_size
        num_experts = local_num_experts
        max_num_tokens = a.size(
            0) if self.max_num_tokens is None else self.max_num_tokens
        workspace13 = (num_experts, max_num_tokens * num_dispatchers,
                       max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, (N // 2))
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_num_tokens: Optional[torch.Tensor],
    ):
        import deep_gemm as dg
        assert hidden_states.ndim == 3

        a1q = hidden_states
        _, N, K = w1.size()

        assert w2.size(1) == K

        E, max_num_tokens, N, K, top_k_num = mk._moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        # (from deepgemm docs) : A value hint (which is a value on CPU)
        # for the M expectation of each batch, correctly setting this value
        # may lead to better performance.
        expected_m = max_num_tokens

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked((a1q, a1q_scale),
                                                 (w1, w1_scale),
                                                 out=workspace1,
                                                 masked_m=expert_num_tokens,
                                                 expected_m=expected_m)

        assert expert_num_tokens is not None
#        a2q, a2q_scale = silu_mul_fp8_quant_deep_gemm(workspace1,
#                                                      expert_num_tokens)
        a2q, a2q_scale = silu_and_mul_masked_post_quant_fwd(workspace1,
                                                            expert_num_tokens)

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked((a2q, a2q_scale),
                                                 (w2, w2_scale),
                                                 out=output,
                                                 masked_m=expert_num_tokens,
                                                 expected_m=expected_m)
