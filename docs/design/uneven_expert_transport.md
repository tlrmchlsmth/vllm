# Uneven expert ownership with uniform transport slots

The modular MoE path can preserve uneven expert ownership while using an
all-to-all library that requires equal expert capacity on every rank. This applies
to DeepEP high-throughput, low-latency and v2, MoRI high-throughput and low-latency,
NIXL EP, and FlashInfer NVLink two-sided (`flashinfer_all2allv` is an alias).

For `N` experts and `R` EP ranks, communication reserves `ceil(N / R)` slots per
rank. The router and checkpoint loader continue to use the original expert IDs.
For example, ten experts across three ranks retain ownership of 4/3/3 experts:

| Rank | Model expert IDs | Transport slots | Unused slot |
| --- | --- | --- | --- |
| 0 | 0–3 | 0–3 | None |
| 1 | 4–6 | 4–6 | 7 |
| 2 | 7–9 | 8–10 | 11 |

`UnevenExpertPrepareAndFinalize` translates IDs before dispatch and combine.
Standard-format dispatch results are translated back into model IDs before
expert computation. Batched dispatch results drop the unused expert rows before
computation; combine restores those rows with zeros. Checkpoint weights, router
logits, and routing probabilities are not padded. Round-robin placement uses the
same adapter when the backend already supports that placement.

Divisible configurations bypass the transport adapter. This change also enables
Triton and Marlin with FlashInfer two-sided for divisible configurations and
normalizes FlashInfer sentinel IDs for expert-map kernels. Uneven configurations add
tensor operations for ID translation. Batched backends also allocate and copy
into a padded combine buffer on ranks with fewer experts. These costs need
multi-GPU measurement before production use.

## Scope and constraints

- At least one real expert per EP rank is required.
- EPLB still requires an evenly divisible physical expert count. Transport slots
  are neither logical experts nor EPLB replicas. Logical expert counts need not
  divide evenly: existing redundant replicas can make the physical count
  divisible, for example ten logical experts plus two replicas on three ranks.
- Standard-format compute kernels must support an explicit uneven expert map.
  Kernel selection excludes implementations that assume `rank * local_count`
  ownership. Triton, Marlin, DeepGEMM, CUTLASS FP8, OAI Triton, and AITER adapters
  opt into this capability; batched compute consumes local expert rows directly.
- Model-specific initialization assertions and other backend constraints (GPU
  topology, supported rank counts, hidden sizes, quantization) still apply.

## Validation

CPU contract tests cover ownership preservation, invalid IDs, transport capacity,
batched padding, and async hook ordering:

```bash
.venv/bin/python -m pytest --noconftest tests/distributed/test_expert_placement.py \
  tests/kernels/moe/test_uneven_prepare_finalize.py -q
```

The existing MoE layer suite includes a two-GPU numerical comparison for every
backend family, using seven experts and both decode-sized and prefill-sized inputs:

```bash
.venv/bin/python -m pytest tests/kernels/moe/test_moe_layer.py \
  -k uneven_experts -v
```

Run on NVIDIA hardware with the corresponding transport libraries and on AMD
hardware with MoRI/AITER. GPU numerical tests, CUDA-graph/DBO coverage, model
evaluations, and performance measurements remain required before merging.
