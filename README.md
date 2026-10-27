# warp_decode

An **unofficial re-implementation** of Cursor's *warp decode* MoE inference
technique ([blog](https://cursor.com/blog/warp-decode)), targeting **blockwise FP8 W8A8
precision on NVIDIA Hopper GPUs (H20 / H100)**.

Cursor's original work targets Blackwell and MXFP8; the public blog describes
the high-level idea (organize the kernel around outputs instead of experts,
give each warp one output neuron, avoid all staging) but not the concrete
kernel shape. This repository is an independent implementation that arrives at
similar conclusions on Hopper by working the design out from a roofline
argument. Performance tuning focuses on the eight Qwen MoE shape/TP
configurations listed below, at batch sizes 1–8.

The two MoE kernels in this repository are:

| Kernel                          | Fused FP8 kernel operation                                                      |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `moe_gate_up_3d_batched`        | gate + up projections, `silu(gate) * up` activation, all in one launch          |
| `moe_down_3d_batched`           | down projection with topk-weighted sum-reduce, in one launch                    |

With BF16 inputs, the complete FFN uses four kernel launches, or three when
activation quantization is fused into down for selected small shapes.
Both take unsorted `topk_ids` directly, so **no `moe_align_block_size` /
scatter / permute is needed**. Router scoring and top-k selection are excluded.

---

## Performance on Hopper

### Setup

- **GPU:** NVIDIA H20 (Hopper, sm_90), with dynamic clocks.
- **Software:** PyTorch 2.13.0+cu130, CUDA toolkit 13.0.88, Triton 3.7.1,
  SGLang 0.5.21, sglang-kernel 0.4.7.
- **Shapes:** eight single-rank configurations from Qwen3.5/3.6 and Qwen3-Next,
  listed below. Inputs and FP8 weights are synthetic; model names specify
  shapes, without loading model checkpoints.
- **Reference:** SGLang's standard Triton FP8 configuration lookup and
  `fused_experts` end-to-end path. The installed tuning configurations and
  default fallbacks are used unchanged and recorded in each benchmark JSON.
- **Routing:** distinct experts per token (top-k=8 or 10), with normalized FP32 routing weights.
- **Timing:** Warm CUDA Graph replay, with 16 calls per graph and 100 replays
  per sample timed by CUDA Events. Results use five rounds for each of two seeds.
  Weights and routing are fixed
  within each timed case; repeated replay can benefit from caching.
- **Precision:** E4M3 activations and weights, FP32 accumulators. Activations
  use one scale per row and 128 elements; weights use one scale per 128×128 block.

### Results

The benchmark measures dynamic activation quantization, gate/up, SiLU, down,
and expert combination in one call. Both implementations receive precomputed
top-k IDs and weights. Router computation, shared experts, and TP communication
are excluded.

This round measures the eight named cases in benchmark/cases.py at batches 1–8.

Speedup is SGLang latency / warp_decode latency. Of 64 shape/batch cases, **41 exceed 1.30x** using the median of all ten samples per implementation; **40 exceed 1.30x in both seeds**. The 30% speedup target is not met for every case.
The geometric mean is 1.499x against SGLang and 1.032x against the version frozen at the start of this round.

The table shows speedup; the last column covers all batches 1–8. See
[all 64 latency results](benchmark/results/qwen_h20_fp8.csv), including before/after
timings, and [benchmark metadata](benchmark/results/qwen_h20_fp8_metadata.json).

| Model | TP | inter/rank | B=1 | B=2 | B=4 | B=8 | B=1–8 range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5/3.6-35B-A3B | 1 | 512 | 1.33x | 1.17x | 1.35x | 1.06x | 1.06–1.35x |
| Qwen3.5/3.6-35B-A3B | 2 | 256 | 1.54x | 1.26x | 1.28x | 1.35x | 1.26–1.54x |
| Qwen3.5-122B-A10B | 2 | 512 | 1.30x | 1.25x | 1.18x | 0.96x | 0.96–1.30x |
| Qwen3.5-122B-A10B | 4 | 256 | 1.48x | 1.35x | 1.27x | 1.24x | 1.24–1.48x |
| Qwen3.5-397B-A17B | 4 | 256 | 1.57x | 1.35x | 1.28x | 1.05x | 1.05–1.57x |
| Qwen3.5-397B-A17B | 8 | 128 | 3.61x | 2.96x | 2.77x | 2.23x | 2.23–3.61x |
| Qwen3-Next-80B-A3B | 1 | 512 | 2.72x | 2.47x | 1.90x | 1.70x | 1.70–2.72x |
| Qwen3-Next-80B-A3B | 2 | 256 | 1.64x | 1.55x | 1.44x | 1.30x | 1.30–1.64x |

Target counts use unrounded ratios.

---

## Design & optimizations

The blog gives one useful headline — *"assign each warp to one output value,
stream weights, avoid staging"*. Both kernels retain this output-oriented layout
and apply FP8 block scales within the dot-product loop.

### `moe_gate_up_3d_batched`

**Shape.** `X:[B, hidden]` × `W:[E, 2*inter, hidden]` → `[B*topk, inter]`,
where the output is `silu(gate) * up` fused.

**Kernel structure:**

- The grid assigns output groups to each `(token, topk_slot)` pair.
- Each warp computes two or four neurons. Selected shape/batch paths make
  hidden, inter, expert count, and top-k compile-time constants while preserving
  the original grid mapping. Other shapes retain the generic path.
- Each lane loads 16 FP8 activation values from global memory and reuses
  the decoded values across its gate/up weight rows, without shared-memory staging.
- Weight loads use vectorized `uint4` reads with the L1 `evict_first` hint.
  For B<=3, the next K block's gate/up weights for the warp's first output
  are prefetched into registers.
- Each vector stays within one 128-element scale group. FP8 pairs are widened
  to FP32, accumulated, and multiplied by the activation and weight scales.
  For hidden=4096, inter=128, E=512, top-k=10 and B<=6, combined scale factors
  stay in warp registers and are shared by shuffle instructions.
- Gate/up results round to BF16 before FP32 `silu(g) * u`; the final BF16
  write-out happens at the end in a single lane.

**Why this specific layout:**

- Multiple outputs per warp reuse decoded activations and block scales.
- Each warp performs its own reduction, without a CTA barrier.

### `moe_down_3d_batched`

**Shape.** `buffer:[B*topk, inter]` × `W:[E, hidden, inter]` → `[B, hidden]`,
with per-slot `topk_scale` folded in and a sum-reduce over `topk`.

**Kernel structure:**

- **2D grid** over `(token, output-group)` avoids dynamic division and modulo.
  Decoded FP8 inputs are reused across four or eight output weight rows.
- For B<=8, `inter=128/256` uses 8-lane reduction groups. Each warp handles
  one expert and several output groups; the CTA combines the expert partials
  in FP32 through shared memory. Top-k=8/10 uses static reduction bounds and
  shared-memory sizes, with four outputs per group at B<=3 and eight at B>=4.
- The `inter=512` paths for hidden=3072/E=256/top-k=8 and
  hidden=2048/E=512/top-k=10 specialize the full shape for B<=8. At B<=2,
  each warp handles one expert and eight outputs; at B>=3, each warp
  accumulates experts serially across four outputs.
- Selected small `inter=128/256` cases quantize BF16 inputs inside down,
  removing a separate quantization launch while preserving the FP8 values and
  scales. This fused path retains four outputs per reduction group.
- Other shapes retain full-warp reductions, with experts parallelized for small
  batches and accumulated within each warp for larger batches.
- The original `hidden=2048, inter=512, E=256, topk=8` path is retained,
  including its B<=2 parallel expert reduction and B=6–8 L1 cache policy.
- FP8 operands are widened before FP32 multiplication and accumulation;
  activation and weight block scales are applied within the reduction.
- Independent FP32 accumulators (one per output `h`) live in registers per
  lane; the topk-weighted sum folds `topk_scale` in FP32 before the final
  BF16 conversion.
- The output and reduction loops are `#pragma unroll`-ed.

### Numerical accuracy

Both kernels use E4M3 activations and weights with FP32 accumulators.
Dequantization is `value = fp8_value * scale`. BF16 inputs are dynamically
quantized per row and 128 elements, with `scale = max(abs(group), 1e-10) / 448`.
The BF16 SiLU output is requantized to FP8 before the down dot products.
The standalone quantizer uses a warp REDUX maximum for its nonnegative magnitudes.
The down kernel applies `topk_scale` and sums across experts in FP32 before
one final BF16 conversion.

The implementation passed independent FP32-reference, CUDA Graph, CUDA memory,
and race checks. Native FFN validation requires NRMSE < 0.006. SGLang is checked
against independent FP32 projections using its own quantization and per-expert
BF16 rounding, with NRMSE < 0.02. Errors against the common reference, including
failures of its original 0.02 threshold, are also retained in benchmark JSON.

Both kernels support PyTorch's current CUDA stream and CUDA Graph capture;
warp_decode does not currently use Programmatic Dependent Launch (PDL).

---

## Build & test

### Requirements

- Linux, a CUDA toolkit compatible with the installed PyTorch build, and a
  Hopper GPU (sm_90). The kernels were tested on H20 with CUDA 13.0.
- PyTorch (with matching CUDA build) — the kernels are JIT-compiled via
  `torch.utils.cpp_extension.load`, no separate build step is needed.
- For benchmarks: `sglang` and `sglang-kernel`.

### Install & first run

```bash
# From the repo root — the kernels are JIT-compiled on the first import.
python tests/test_fp8_blockwise.py --case qwen35_36_35b_tp1

# With SGLang and sglang-kernel installed:
python tests/test_benchmark_reference.py
```

The initial import triggers an `nvcc` build under `build/`. Subsequent imports
reuse the cached extension.

Set `WARP_DECODE_VERBOSE=1` to see the `nvcc` invocation on rebuild:

```bash
WARP_DECODE_VERBOSE=1 python tests/test_fp8_blockwise.py --case qwen35_36_35b_tp1
```

### Benchmark

```bash
# List the eight shared cases.
python benchmark/bench_qwen.py --list-cases

# Validate or benchmark all eight cases at B=1–8.
python benchmark/bench_qwen.py --mode test
python benchmark/bench_qwen.py

# Run a single case using the same name as the correctness tests.
python benchmark/bench.py --case qwen35_122b_tp4
```

The benchmark and correctness tests share the cases in `benchmark/cases.py`.
The benchmark prints the SGLang version and records selected configurations
in the output JSON. SGLang logs any configuration fallback. Both implementations
use existing FP8 weights and their FP32 block scales directly.

### Python API

```python
from warp_decode import warp_decode as wd

# gate_up + silu_and_mul (fused)
buf = wd.moe_gate_up_3d_batched(
    x,               # [B, hidden]         bf16
    gate_up_w,       # [E, 2*inter, hidden] fp8 E4M3
    gate_up_scale,   # [E, 2*inter/128, hidden/128] fp32
    topk_ids,        # [B, topk]           int32
)                    # -> [B*topk, inter]  bf16

# down + topk-weighted sum-reduce (fused)
out = wd.moe_down_3d_batched(
    buf,             # [B*topk, inter]     bf16
    down_w,          # [E, hidden, inter]  fp8 E4M3
    down_scale,      # [E, hidden/128, inter/128] fp32
    topk_ids,        # [B, topk]           int32
    topk_scale,      # [B, topk]           fp32
)                    # -> [B, hidden]      bf16

# Complete FFN, including both activation quantization steps
out = wd.moe_ffn_3d_batched(
    x, gate_up_w, down_w, gate_up_scale, down_scale, topk_ids, topk_scale,
)
```

Prequantized E4M3 activations are also accepted with their matching FP32
scale as the last argument (`x_scale` for gate/up and FFN, `buffer_scale` for down). `quantize_fp8_blockwise(x)` returns FP8 values
and scales for a BF16 input. Weights and their scales are supplied directly.

### Shape constraints

`hidden` and per-rank `inter` must be positive multiples of 128;
`1 <= topk <= min(num_experts, 32)`. Shapes are inferred from the supplied weights
and routing tensors. The tested model configurations are:

| Model | hidden | intermediate | Experts | top-k | TP → per-rank inter |
| --- | ---: | ---: | ---: | ---: | --- |
| [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json) / [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json) | 2048 | 512 | 256 | 8 | 1→512, 2→256 |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json) | 3072 | 1024 | 256 | 8 | 2→512, 4→256 |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) | 4096 | 1024 | 512 | 10 | 4→256, 8→128 |
| [Qwen3-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) | 2048 | 512 | 512 | 10 | 1→512, 2→256 |

Both kernels assume the standard blockwise FP8 weight layouts used by
sglang's fused_moe: E4M3 weights `[E,N,K]`, FP32 scales `[E,N/128,K/128]`.
All tensors must be contiguous and on the same CUDA device. FP8 data pointers
must be 16-byte aligned; BF16 quantizer inputs need 8-byte alignment. Scales
must be finite and positive; routing IDs must lie in `[0,num_experts)`.

---

## Acknowledgements

- [Cursor's warp decode blog](https://cursor.com/blog/warp-decode) —
  the original idea.
- [sglang](https://github.com/sgl-project/sglang) — the reference
  implementation used for correctness and performance comparisons.
