"""Benchmark warp_decode kernels against a sglang triton reference.

Three modes (default: all):
  gate_up : warp_decode.moe_gate_up_3d_batched  vs  triton gate_up + silu_and_mul
  down    : warp_decode.moe_down_3d_batched      vs  triton down + sum-reduce
  e2e     : gate_up + down (warp_decode)         vs  full sglang MoE pipeline

Run from the repo root:
    python warp_decode/benchmark/bench.py --batches 1,2,4,8,16,32,64
"""

import argparse
import sys
from pathlib import Path

# Make the `warp_decode` package importable regardless of CWD.
# Repo layout: <repo>/warp_decode/benchmark/bench.py  ->  parents[2] = <repo>
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import torch
import triton.language as tl

from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
from sgl_kernel import silu_and_mul

from warp_decode import warp_decode as wd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--intermediate-size", type=int, default=1536,
                   help="moe_intermediate_size per expert BEFORE TP split")
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--mode", choices=["gate_up", "down", "e2e", "all"], default="all")
    p.add_argument("--batches", type=str, default="1..64",
                   help='e.g. "1..64", "16,32,64,128", "16..512:16"')
    return p.parse_args()


def parse_batches(spec):
    if ":" in spec:
        rng, step = spec.split(":")
        a, b = rng.split("..")
        return list(range(int(a), int(b) + 1, int(step)))
    if ".." in spec:
        a, b = spec.split("..")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_inputs(B, hidden, inter, E_local, topk, device="cuda"):
    dtype = torch.bfloat16
    x = torch.randn(B, hidden, dtype=dtype, device=device)
    gate_up_w  = torch.randn(E_local, 2 * inter, hidden, dtype=dtype, device=device)
    down_w     = torch.randn(E_local, hidden, inter,    dtype=dtype, device=device)
    topk_ids   = torch.randint(0, E_local, (B, topk), dtype=torch.int32, device=device)
    raw        = torch.rand(B, topk, dtype=torch.float32, device=device)
    topk_scale = raw / raw.sum(dim=1, keepdim=True)
    return x, gate_up_w, down_w, topk_ids, topk_scale


# ---------------------------------------------------------------------------
# Bench helper (CUDA graph)
# ---------------------------------------------------------------------------

def bench_cuda_graph(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=stream):
            fn()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def sglang_bf16_config(total_m, E):
    if total_m <= E:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=1, num_warps=4, num_stages=3)
    return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8, num_warps=4, num_stages=3)


def _alloc_routing_buffers(topk_ids, E_local, bm, device):
    if topk_ids.numel() < E_local + 1:
        max_padded = topk_ids.numel() * bm
    else:
        max_padded = topk_ids.numel() + (E_local + 1) * (bm - 1)
    max_m_blocks = (max_padded + bm - 1) // bm
    return dict(
        max_padded          = max_padded,
        sorted_ids          = torch.empty(max_padded,            dtype=torch.int32, device=device),
        expert_ids          = torch.empty(max_m_blocks,          dtype=torch.int32, device=device),
        num_tokens_post_pad = torch.empty(1,                     dtype=torch.int32, device=device),
        cumsum_buf          = torch.empty(E_local + 2,           dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# gate_up bench
# ---------------------------------------------------------------------------

def bench_gate_up_wd(x, gate_up_w, topk_ids, warmup, iters):
    def fn():
        wd.moe_gate_up_3d_batched(x, gate_up_w, topk_ids)
    return bench_cuda_graph(fn, warmup, iters)


def bench_gate_up_sgl(x, gate_up_w, topk_ids, topk_scale, E_local, warmup, iters):
    B, hidden = x.shape
    inter = gate_up_w.shape[1] // 2
    topk = topk_ids.shape[1]
    total = B * topk

    config = sglang_bf16_config(total, E_local)
    bm = config["BLOCK_SIZE_M"]
    R = _alloc_routing_buffers(topk_ids, E_local, bm, x.device)

    cache1 = torch.empty(R["max_padded"], 2 * inter, dtype=torch.bfloat16, device=x.device)
    cache2 = torch.empty(R["max_padded"], inter,     dtype=torch.bfloat16, device=x.device)

    def fn():
        sgl_moe_align_block_size(
            topk_ids, E_local + 1, bm,
            R["sorted_ids"], R["expert_ids"], R["num_tokens_post_pad"],
            R["cumsum_buf"], True,
        )
        invoke_fused_moe_kernel(
            x, gate_up_w, None, cache1,
            None, None, None,
            topk_scale, topk_ids,
            R["sorted_ids"], R["expert_ids"], R["num_tokens_post_pad"],
            False, topk, config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False, block_shape=None,
        )
        silu_and_mul(cache1, cache2)

    return bench_cuda_graph(fn, warmup, iters)


# ---------------------------------------------------------------------------
# down bench
# ---------------------------------------------------------------------------

def bench_down_wd(buffer, down_w, topk_ids, topk_scale, warmup, iters):
    B = topk_scale.shape[0]
    def fn():
        wd.moe_down_3d_batched(buffer, down_w, topk_ids, topk_scale, B)
    return bench_cuda_graph(fn, warmup, iters)


def bench_down_sgl(down_w, topk_ids, topk_scale, E_local, warmup, iters):
    """Mirrors step 3+4 of the reference pipeline (down GEMM + topk sum-reduce).

    The sglang triton down kernel indexes its A input via `sorted_ids`, so we
    allocate `cache2` at max padded size. Values don't matter for timing.
    """
    B = topk_scale.shape[0]
    hidden = down_w.shape[1]
    inter  = down_w.shape[2]
    topk = topk_ids.shape[1]
    total = B * topk

    config = sglang_bf16_config(total, E_local)
    bm = config["BLOCK_SIZE_M"]
    R = _alloc_routing_buffers(topk_ids, E_local, bm, down_w.device)

    cache2 = torch.empty(R["max_padded"], inter, dtype=torch.bfloat16, device=down_w.device)
    cache3 = torch.empty(B, topk, hidden,        dtype=torch.bfloat16, device=down_w.device)
    out    = torch.empty(B, hidden,              dtype=torch.bfloat16, device=down_w.device)

    def fn():
        sgl_moe_align_block_size(
            topk_ids, E_local + 1, bm,
            R["sorted_ids"], R["expert_ids"], R["num_tokens_post_pad"],
            R["cumsum_buf"], True,
        )
        invoke_fused_moe_kernel(
            cache2, down_w, None, cache3,
            None, None, None,
            topk_scale, topk_ids,
            R["sorted_ids"], R["expert_ids"], R["num_tokens_post_pad"],
            True,  # mul_routed_weight
            1,     # top_k=1: sorted rows map 1:1 into cache3[B, topk, hidden]
            config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False, block_shape=None,
        )
        torch.sum(cache3, dim=1, out=out)

    return bench_cuda_graph(fn, warmup, iters)


# ---------------------------------------------------------------------------
# e2e bench
# ---------------------------------------------------------------------------

def bench_e2e_wd(x, gate_up_w, down_w, topk_ids, topk_scale, warmup, iters):
    B = x.shape[0]
    def fn():
        buf = wd.moe_gate_up_3d_batched(x, gate_up_w, topk_ids)
        wd.moe_down_3d_batched(buf, down_w, topk_ids, topk_scale, B)
    return bench_cuda_graph(fn, warmup, iters)


def bench_e2e_sgl(x, gate_up_w, down_w, topk_ids, topk_scale, E_local, warmup, iters):
    B, hidden = x.shape
    inter = gate_up_w.shape[1] // 2
    topk = topk_ids.shape[1]
    total = B * topk

    config = sglang_bf16_config(total, E_local)
    bm = config["BLOCK_SIZE_M"]
    R = _alloc_routing_buffers(topk_ids, E_local, bm, x.device)

    cache1 = torch.empty(R["max_padded"], 2 * inter, dtype=torch.bfloat16, device=x.device)
    cache2 = torch.empty(R["max_padded"], inter,     dtype=torch.bfloat16, device=x.device)
    cache3 = torch.empty(B, topk, hidden,            dtype=torch.bfloat16, device=x.device)
    out    = torch.empty(B, hidden,                  dtype=torch.bfloat16, device=x.device)

    def fn():
        sgl_moe_align_block_size(
            topk_ids, E_local + 1, bm,
            R["sorted_ids"], R["expert_ids"], R["num_tokens_post_pad"],
            R["cumsum_buf"], True,
        )
        invoke_fused_moe_kernel(
            x, gate_up_w, None, cache1,
            None, None, None,
            topk_scale, topk_ids,
            R["sorted_ids"], R["expert_ids"], R["num_tokens_post_pad"],
            False, topk, config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False, block_shape=None,
        )
        silu_and_mul(cache1, cache2)
        invoke_fused_moe_kernel(
            cache2, down_w, None, cache3,
            None, None, None,
            topk_scale, topk_ids,
            R["sorted_ids"], R["expert_ids"], R["num_tokens_post_pad"],
            True, 1, config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False, block_shape=None,
        )
        torch.sum(cache3, dim=1, out=out)

    return bench_cuda_graph(fn, warmup, iters)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_mode(name, B_list, args, hidden, inter, E_local, topk, wd_call, sgl_call):
    print(f"\n=== {name} ===")
    hdr = (f"{'batch':>6}  {'toks':>6}  "
           f"{'wd(ms)':>10}  {'sgl(ms)':>10}  {'speedup':>8}")
    print(hdr)
    print("-" * len(hdr))

    for B in B_list:
        x, gate_up_w, down_w, ids, scale = make_inputs(
            B, hidden, inter, E_local, topk
        )
        wd_ms  = wd_call (x, gate_up_w, down_w, ids, scale, args.warmup, args.iters)
        sgl_ms = sgl_call(x, gate_up_w, down_w, ids, scale, args.warmup, args.iters)

        print(f"{B:>6}  {B*topk:>6}  "
              f"{wd_ms:>10.4f}  {sgl_ms:>10.4f}  {sgl_ms/wd_ms:>7.2f}x")


def main():
    args = parse_args()
    hidden = args.hidden_size
    inter  = args.intermediate_size // args.tp_size
    E_local = args.num_experts     # matches sglang bench: TP shards only inter
    topk = args.topk
    B_list = parse_batches(args.batches)

    torch.manual_seed(0)

    dev = torch.cuda.get_device_properties(0)

    print(f"\ndevice        : {dev.name}")
    print(f"config        : hidden={hidden} inter={inter} E_local={E_local} topk={topk}")
    print(f"weights       : gate_up={E_local*2*inter*hidden*2/1e6:.1f} MB, "
          f"down={E_local*hidden*inter*2/1e6:.1f} MB")

    modes = ["gate_up", "down", "e2e"] if args.mode == "all" else [args.mode]

    # signature adapters: (x, gate_up_w, down_w, ids, scale, warmup, iters) -> ms
    def _wd_gu (x, gu, dn, ids, sc, w, i): return bench_gate_up_wd (x, gu, ids, w, i)
    def _sgl_gu(x, gu, dn, ids, sc, w, i): return bench_gate_up_sgl(x, gu, ids, sc, E_local, w, i)

    def _wd_dn (x, gu, dn, ids, sc, w, i):
        # Use a real gate_up output so the L2 state during the down bench is
        # representative of the e2e path.
        buf = wd.moe_gate_up_3d_batched(x, gu, ids)
        return bench_down_wd(buf, dn, ids, sc, w, i)

    def _sgl_dn(x, gu, dn, ids, sc, w, i):
        return bench_down_sgl(dn, ids, sc, E_local, w, i)

    def _wd_e2e (x, gu, dn, ids, sc, w, i): return bench_e2e_wd (x, gu, dn, ids, sc,           w, i)
    def _sgl_e2e(x, gu, dn, ids, sc, w, i): return bench_e2e_sgl(x, gu, dn, ids, sc, E_local, w, i)

    if "gate_up" in modes:
        run_mode("gate_up  (warp_decode vs triton gate_up + silu_and_mul)",
                 B_list, args, hidden, inter, E_local, topk, _wd_gu, _sgl_gu)
    if "down" in modes:
        run_mode("down     (warp_decode vs triton down + sum-reduce)",
                 B_list, args, hidden, inter, E_local, topk, _wd_dn, _sgl_dn)
    if "e2e" in modes:
        run_mode("e2e      (warp_decode gate_up+down vs full sglang pipeline)",
                 B_list, args, hidden, inter, E_local, topk, _wd_e2e, _sgl_e2e)


if __name__ == "__main__":
    main()
