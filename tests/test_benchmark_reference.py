"""Check the benchmark's staged BF16 reference against standard SGLang Triton."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from warp_decode.benchmark.bench import (
    initialize_sglang,
    make_inputs,
    make_gate_up_sgl,
    make_down_sgl,
    make_e2e_sgl,
)


def run_case(B, E, topk, hidden, inter):
    torch.manual_seed(B + topk)
    x, gu, dn, ids, scale = make_inputs(B, hidden, inter, E, topk)
    x.mul_(0.1)
    gu.mul_(0.05)
    dn.mul_(0.05)
    assert all(len(set(row)) == topk for row in ids.tolist())

    gate_up = make_gate_up_sgl(x, gu, ids, scale)
    buffer = gate_up()
    down = make_down_sgl(buffer, dn, ids, scale)
    staged = down()
    e2e = make_e2e_sgl(x, gu, dn, ids, scale)
    standard = e2e()
    torch.testing.assert_close(staged, standard, atol=0, rtol=0)

    # Independent FP32 reference, retaining the BF16 FFN activation boundary.
    selected_gu = gu[ids.flatten().long()].float()
    selected_x = x[:, None, :].expand(-1, topk, -1).reshape(-1, hidden).float()
    projections = torch.bmm(selected_gu, selected_x.unsqueeze(-1)).squeeze(-1)
    gate, up = projections.chunk(2, dim=-1)
    ref_buffer = (torch.nn.functional.silu(gate) * up).bfloat16()
    projected = torch.bmm(dn[ids.flatten().long()].float(),
                          ref_buffer.float().unsqueeze(-1)).squeeze(-1)
    ref = (projected.view(B, topk, hidden) * scale[:, :, None]).sum(1).bfloat16()
    diff = standard.float() - ref.float()
    normalized_rmse = diff.norm() / ref.float().norm().clamp_min(1e-20)
    assert normalized_rmse.item() < 0.02, normalized_rmse.item()

    # Check both replay and updated inputs/routing using the real model shape.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            e2e()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = e2e()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, standard, atol=0, rtol=0)
    x.mul_(0.5)
    ids.copy_((ids + 1) % E)
    updated = e2e()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, updated, atol=0, rtol=0)
    print(f"B={B} E={E} topk={topk} H={hidden} I={inter}: "
          f"staged==standard, FP32 NRMSE={normalized_rmse.item():.5f}, graph passed")


def main():
    initialize_sglang()
    cases = [(1, 4, 1, 256, 64), (3, 8, 2, 512, 128)]
    cases += [(B, 128, 8, 4096, 192) for B in [1, 8, 16, 32, 64]]
    cases += [(4, 128, 8, 4096, 384)]
    for case in cases:
        run_case(*case)
    print("all 8 standard-reference cases passed")


if __name__ == "__main__":
    main()
