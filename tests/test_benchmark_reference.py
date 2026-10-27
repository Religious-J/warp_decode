"""Validate the benchmark's real SGLang FP8 FFN, including graph replay."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from warp_decode import warp_decode as wd
from warp_decode.benchmark.bench import (
    initialize_sglang, MoeRunnerConfig, StandardTopKOutput, fused_experts,
    sglang_reference,
)
from fp8_reference import make_weights, make_inputs, ffn_reference, metrics
from test_fp8_blockwise import graph_test


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
        # Keep the test's original 1% limit and record the common-reference error.
        actual = fn()
        common = ffn_reference(x, gu, dn, gs, ds, ids, route)
        common_accuracy = metrics(actual, common, .02, check=False)
        print("SGLang/common reference", b, common_accuracy, flush=True)
        accuracy = metrics(
            actual, sglang_reference(x, gu, dn, gs, ds, ids, route), .01,
        )
        metrics(wd.moe_ffn_3d_batched(x, gu, dn, gs, ds, ids, route), common)
        def mutate():
            x.mul_(-1.03)
            ids.copy_(ids.roll(1, -1)); route.copy_(route.roll(1, -1))
            gs.mul_(1.001); ds.mul_(.999)
        graph_test(fn, mutate)
        actual = fn()
        graph_common_accuracy = metrics(
            actual, ffn_reference(x, gu, dn, gs, ds, ids, route), .02, check=False,
        )
        print("Graph SGLang/common reference", b, graph_common_accuracy, flush=True)
        metrics(actual, sglang_reference(x, gu, dn, gs, ds, ids, route), .01)
        print('PASS SGLang FP8 reference + graph', b, accuracy, flush=True)
    print('All SGLang FP8 reference tests passed', flush=True)


if __name__ == '__main__':
    main()
