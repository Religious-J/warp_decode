# warp_decode

An **unofficial re-implementation** of Cursor's *warp decode* MoE inference
technique ([blog](https://cursor.com/blog/warp-decode)), targeting **BF16
precision on NVIDIA Hopper GPUs (H20 / H100)**.

Cursor's original work targets Blackwell and MXFP8; the public blog describes
the high-level idea (organize the kernel around outputs instead of experts,
give each warp one output neuron, avoid all staging) but not the concrete
kernel shape. This repository is an independent implementation that arrives at
similar conclusions on Hopper by working the design out from a roofline
argument, and adds a few Hopper-specific optimizations that were needed to
actually turn the idea into a measured win.

The two kernels in this repository are:

| Kernel                          | Fused operation                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `moe_gate_up_3d_batched`        | gate + up projections, `silu(gate) * up` activation, all in one launch          |
| `moe_down_3d_batched`           | down projection with topk-weighted sum-reduce, in one launch                    |

Both take unsorted `topk_ids` directly, so **no `moe_align_block_size` /
scatter / permute is needed** — routing overhead is zero.

---

## Reproducing Cursor's claim on Hopper

### Setup

- **GPU:** NVIDIA H20 (Hopper, sm_90). HBM3 peak ≈ 4 TB/s.
- **Model shape:** Qwen3-235B-A22B under TP=8
  - `hidden = 4096`
  - `intermediate_size_per_expert = 1536`, TP-sharded to `inter = 192`
  - `num_experts = 128`, `topk = 8`
- **Reference:** the sglang fused-MoE path from
  `tests/bench_bf16_fuse_moe.py` (`moe_align_block_size` → triton
  `invoke_fused_moe_kernel` gate_up → `silu_and_mul` → triton
  `invoke_fused_moe_kernel` down → topk-weighted sum-reduce).
- **Timing:** CUDA-graph capture + replay, 30 iterations after 5 warmup runs.
- **All inputs BF16, accumulators FP32.**

### Results

Times are per-call, in milliseconds. `speedup = sglang_ms / warp_decode_ms`.

**gate_up + `silu_and_mul`:**

| batch | tokens | warp_decode | sglang triton | speedup |
| ----: | -----: | ----------: | ------------: | ------: |
|     1 |      8 |      0.0078 |        0.0222 |   2.84x |
|     2 |     16 |      0.0144 |        0.0351 |   2.43x |
|     4 |     32 |      0.0334 |        0.0465 |   1.39x |
|     8 |     64 |      0.0541 |        0.0680 |   1.26x |
|    16 |    128 |      0.0890 |        0.0912 |   1.02x |
|    32 |    256 |      0.1594 |        0.2015 |   1.26x |
|    48 |    384 |      0.2223 |        0.2305 |   1.04x |
|    64 |    512 |      0.2840 |        0.2300 |   0.81x |

**down + topk sum-reduce:**

| batch | tokens | warp_decode | sglang triton | speedup |
| ----: | -----: | ----------: | ------------: | ------: |
|     1 |      8 |      0.0101 |        0.0158 |   1.56x |
|     2 |     16 |      0.0118 |        0.0208 |   1.76x |
|     4 |     32 |      0.0196 |        0.0301 |   1.53x |
|     8 |     64 |      0.0384 |        0.0474 |   1.23x |
|    16 |    128 |      0.0664 |        0.0696 |   1.05x |

**End-to-end MoE FFN (gate_up + down):**

| batch | tokens | warp_decode | sglang triton | speedup |
| ----: | -----: | ----------: | ------------: | ------: |
|     1 |      8 |      0.0207 |        0.0411 |   1.98x |
|     2 |     16 |      0.0370 |        0.0562 |   1.52x |
|     4 |     32 |      0.0555 |        0.0731 |   1.32x |
|     8 |     64 |      0.0974 |        0.1126 |   1.16x |
|    16 |    128 |      0.1580 |        0.1727 |   1.09x |
|    32 |    256 |      0.2853 |        0.2898 |   1.02x |

### What this shows

**Cursor's claim reproduces on Hopper.** For decode-heavy batches
(`B * topk / E_local < 1`, i.e. per-expert token count below the ridge-point
of the roofline), the output-centric design gives a consistent speedup — up
to **~2x on batch 1** and staying ahead of the reference until roughly
`B = 32`. Above `B ≈ 48` the arithmetic intensity of an expert-centric grouped
GEMM starts to dominate and the reference wins, which is exactly the regime
where Cursor also recommends the traditional expert-centric path.

The crossover point predicted by roofline analysis is
`B_crit ≈ E_local / topk = 128 / 8 = 16`. In practice we see the e2e crossover
around `B = 32`, slightly beyond the pure-roofline prediction because the
warp_decode kernels also eliminate the routing/permute overhead that the
reference has to pay unconditionally.

---

## Design & optimizations

The blog gives one useful headline — *"assign each warp to one output value,
stream weights, avoid staging"* — but leaves out the details that actually
determine performance on Hopper. Direct translation of the headline gives a
kernel that is correct but far from optimal (we measured this — see below).
Two structural changes were needed for each of the two kernels.

### `moe_gate_up_3d_batched`

**Shape.** `X:[B, hidden]` × `W:[E, 2*inter, hidden]` → `[B*topk, inter]`,
where the output is `silu(gate) * up` fused.

**Kernel structure:**
- 2D grid: `(B*topk, inter / WARPS_PER_CTA)`, `blockDim = 256` = 8 warps
- Each CTA handles one `(token, topk_slot)` pair.
- The 8 warps in a CTA together cover 8 adjacent output neurons.
- All 8 warps in a CTA share the **same** `x[token]` row → staged once into
  shared memory, then reused 8× on-chip.
- Each warp streams two weight rows (`gate` and `up` for its neuron) directly
  from HBM using vectorized `uint4` loads (`VEC = 8` bf16 per lane per iter),
  and accumulates the two dot products in registers.
- Final `silu(g) * u` and the bf16 write-out happen at the end in a single
  lane.

**Why this specific layout:**
- The reduction axis is `hidden = 4096`, long enough that a single warp gives
  a sufficiently deep dot-product pipeline.
- 8 warps per CTA is the largest choice that keeps the shared-memory copy of
  `x[token]` small (`hidden * 2B` = 8 KB) while giving every SM enough
  warp-level parallelism to hide the weight-loading latency.
- No cross-warp synchronization on the reduction — each warp is entirely
  independent after the initial `x` load.

### `moe_down_3d_batched`

**Shape.** `buffer:[B, topk, inter]` × `W:[E, hidden, inter]` → `[B, hidden]`,
with per-slot `topk_scale` folded in and a sum-reduce over `topk`.

**This kernel is where the naive translation of Cursor's headline hurts.**
The obvious mirror of the gate_up design — 1 warp per output `h`, stage
`buffer[token, :, :]` in shared memory, sum across topk in a single warp —
turns out to be significantly slower than the sglang triton reference for any
`B > 4`. The reasons are structural:

1. The reduction axis is `inter = 192`, only ~3 lane-loop iterations. A
   single dot product is not enough to hide HBM latency by itself.
2. There is no cross-warp reuse of the input — every warp reads its own set
   of weight rows, and staging the small `buffer` in smem just adds
   `__syncthreads` and register pressure without any real reuse win.

The design that actually works on Hopper is very different:

**Kernel structure (v2):**
- **1D grid** over `(token, h-group)`. Each warp owns `OUTS_PER_WARP = 8`
  adjacent output dims for one token — *not* a single `h`.
- No shared memory at all. The 3 KB per-token buffer fits comfortably in the
  L1/L2 hierarchy on its own; explicit staging just costs.
- For each of the `topk` experts, the warp reads one `bf16x2` element of the
  buffer and reuses it against **8 different weight rows** (one for each of
  its output dims), doing 8 `__hmul2` dot-product updates before consuming
  the next buffer element.
- 8 independent FP32 accumulators (one per output `h`) live in registers per
  lane; the topk-weighted sum folds `topk_scale` in FP32 before the final
  BF16 conversion.
- Both inner loops (over `inter/2` and over the 8 outputs) are fully
  `#pragma unroll`-ed so the compiler can hoist all the loads and overlap
  them with the FMAs.

**Result of these two changes** (Qwen3 TP=8, batch 16):

| version                                        | down time | vs sglang |
| ---------------------------------------------- | --------: | --------: |
| v1: 1 warp / output, smem-staged buffer        |  0.143 ms |     0.49× |
| v2, `OUTS_PER_WARP = 2`, no smem               |  0.097 ms |     0.72× |
| v2, `OUTS_PER_WARP = 8`, fully unrolled (now)  |  0.066 ms |     1.05× |

The `OUTS_PER_WARP` sweep on H20 showed 8 is the sweet spot: 4 leaves
input-side arithmetic intensity on the table, 16 pushes register pressure
past the point where occupancy falls off and the B=1 CTA count halves.

### Numerical accuracy

Both kernels keep activations in BF16 and accumulate in FP32 the whole way.
In particular, the down kernel folds `topk_scale` into the accumulator in
FP32 *before* the final BF16 conversion, so the topk-weighted sum does not
suffer an intermediate rounding step — which mirrors the accuracy argument in
Cursor's original blog.

Correctness is checked against a pure-PyTorch BF16 reference (see
`warp_decode/tests/`). All shapes we support match to within BF16 precision
(`atol=5e-2, rtol=5e-2`, but observed `max_abs_diff ≈ 5e-4` on Qwen3
shapes — dominated by BF16 output quantization, not accumulator error).

---

## Build & test

### Requirements

- Linux with CUDA 12.x + a Hopper GPU (sm_90). Tested on **H20 + CUDA 12.9**.
- PyTorch (with matching CUDA build) — the kernels are JIT-compiled via
  `torch.utils.cpp_extension.load`, no separate build step is needed.
- For the benchmark script only: `sglang` and `sgl-kernel` (used as the
  reference implementation).

### Install & first run

```bash
# From the repo root — the kernels are JIT-compiled on the first import.
python warp_decode/tests/test_gate_up_correctness.py
python warp_decode/tests/test_down_correctness.py
```

The initial import triggers an `nvcc` build under
`warp_decode/build/`. Subsequent imports reuse the cached extension.

Set `WARP_DECODE_VERBOSE=1` to see the `nvcc` invocation on rebuild:

```bash
WARP_DECODE_VERBOSE=1 python warp_decode/tests/test_gate_up_correctness.py
```

### Benchmark

```bash
# All three modes, Qwen3-235B TP=8 defaults, batch 1..64
python warp_decode/benchmark/bench.py

# One mode only, custom batch set
python warp_decode/benchmark/bench.py --mode e2e --batches 1,8,16,32,64

# Custom shape (e.g. TP=4)
python warp_decode/benchmark/bench.py --tp-size 4 --batches 1..32
```

### Python API

```python
from warp_decode import warp_decode as wd

# gate_up + silu_and_mul (fused)
buf = wd.moe_gate_up_3d_batched(
    x,               # [B, hidden]         bf16
    gate_up_w,       # [E, 2*inter, hidden] bf16
    topk_ids,        # [B, topk]           int32
)                    # -> [B*topk, inter]  bf16

# down + topk-weighted sum-reduce (fused)
out = wd.moe_down_3d_batched(
    buf,             # [B*topk, inter]     bf16
    down_w,          # [E, hidden, inter]  bf16
    topk_ids,        # [B, topk]           int32
    topk_scale,      # [B, topk]           fp32
    B,               # int
)                    # -> [B, hidden]      bf16
```

### Shape constraints

| kernel   | constraint                                                     |
| -------- | -------------------------------------------------------------- |
| gate_up  | `hidden % 256 == 0` (32 lanes × VEC=8), `inter % 8 == 0`       |
| down     | `inter % 64 == 0` (32 lanes × VEC=2), `hidden % 8 == 0`        |

Both kernels assume the standard weight layouts used by sglang's fused_moe.

---

## Future work

**Consumer GPUs with multi-precision support (sm_120 and similar).**
Local / on-device deployment is exactly the setting where small-batch,
latency-bound decode kernels are most valuable — batch sizes are often 1–4,
and every millisecond of TTFT/ITL is user-visible. Extending warp_decode to
GeForce-class Blackwell (sm_120) with the full precision matrix (BF16, FP16,
FP8, MXFP8, INT8, INT4) would let a single-GPU workstation run frontier MoEs
at interactive latency. Concrete next steps:
- On-the-fly dequant inside the warp loop — the parallelism layout does not
  change, only the innermost read+convert.
- Group / per-channel / MX scale handling that stays in registers to avoid
  extra smem traffic.
- Autotune the same shape knobs (`WARPS_PER_CTA`, `OUTS_PER_WARP`) for
  smaller SM counts and different L2/HBM ratios.

**Same idea on non-NVIDIA accelerators.**
The roofline argument that motivates warp_decode is architecture-agnostic:
for `B * topk / E_local < ridge_point`, any expert-centric grouped GEMM is
wasting work, and an output-centric kernel that avoids all routing/staging
should win. Ports worth trying:
- **AMD (CDNA3 / RDNA):** HIP is close enough to CUDA that the two kernels
  should translate almost mechanically; the CDNA3 memory hierarchy has a
  lower ridge point, which extends the useful batch range further.
- **Ascend NPU (Cube / Vector cores):** the equivalent of the "one warp per
  output" mapping is one Vector core per output group, feeding the Cube
  core with weight tiles. The same structural argument — skip routing,
  fold the topk-weighted sum inside the compute unit — should transfer.
- **Other tile-based accelerators (Trainium, TPU, etc.):** any device where
  small-batch decode is memory-bound is a candidate.


---

## Acknowledgements

- [Cursor's warp decode blog](https://cursor.com/blog/warp-decode) —
  the original idea.
- [sglang](https://github.com/sgl-project/sglang) — the reference
  implementation used for correctness and performance comparisons.
