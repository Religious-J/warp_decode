"""Validate the benchmark's real SGLang FP8 FFN, including graph replay."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from warp_decode import warp_decode as wd
from warp_decode.benchmark.bench import (
    initialize_sglang, MoeRunnerConfig, StandardTopKOutput, fused_experts,
    sglang_per_token_group_quant_fp8,
)
from fp8_reference import make_weights, make_inputs, gate_reference, down_reference, ffn_reference, metrics
from test_fp8_blockwise import graph_test


def sglang_reference(x, gu, dn, gs, ds, ids, route):
    # Keep SGLang's quantized inputs, independently evaluate the projections.
    q, xs = sglang_per_token_group_quant_fp8(x, 128)
    buf = gate_reference(q, gu, xs, gs, ids)
    q2, s2 = sglang_per_token_group_quant_fp8(buf, 128)
    return down_reference(q2, dn, s2, ds, ids, route)


def main():
    initialize_sglang()
    torch.backends.cuda.matmul.allow_tf32 = False
    gu, dn, gs, ds = make_weights()
    runner = MoeRunnerConfig(num_experts=256, num_local_experts=256, top_k=8,
                             inplace=False, gate_up_interleaved=False)
    for b in [1, 16, 64]:
        x, ids, route = make_inputs(b)
        topk = StandardTopKOutput(route, ids, None)
        def fn():
            return fused_experts(x, gu, dn, topk, runner, use_fp8_w8a8=True,
                                 w1_scale=gs, w2_scale=ds, block_shape=[128,128])
        # BF16 GEMM/SiLU rounding can cross FP8 requantization boundaries.
        # Allow 1% for the full SGLang path; native stage limits are separate.
        accuracy = metrics(fn(), sglang_reference(x, gu, dn, gs, ds, ids, route), .01)
        metrics(wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, ids, route),
                ffn_reference(x, gu, dn, gs, ds, ids, route))
        def mutate():
            x.mul_(-1.03)
            ids.copy_(ids.roll(1, -1)); route.copy_(route.roll(1, -1))
            gs.mul_(1.001); ds.mul_(.999)
        graph_test(fn, mutate)
        metrics(fn(), sglang_reference(x, gu, dn, gs, ds, ids, route), .01)
        print('PASS SGLang FP8 reference + graph', b, accuracy, flush=True)
    print('All SGLang FP8 reference tests passed', flush=True)


if __name__ == '__main__':
    main()
