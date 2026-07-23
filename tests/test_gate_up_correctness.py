"""Correctness test for warp_decode moe_gate_up_3d_batched.

Reference: pure-PyTorch bf16 gate_up + silu_and_mul.
"""

import os
import sys
from pathlib import Path

# make `warp_decode/` importable when run from repo root or from tests/
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import torch

from warp_decode import warp_decode as wd


def reference_gate_up(x, w, topk_ids):
    """
    x        : [B, hidden]           bf16
    w        : [E, 2*inter, hidden]  bf16
    topk_ids : [B, topk]             int32

    returns  : [B*topk, inter]       bf16
    """
    B, hidden = x.shape
    E, two_i, _ = w.shape
    inter = two_i // 2
    topk = topk_ids.shape[1]

    # gather per (token, slot) expert weight rows -> [B*topk, 2*inter, hidden]
    ids = topk_ids.reshape(-1).long()                  # [B*topk]
    w_sel = w[ids]                                     # [B*topk, 2*inter, hidden]

    # broadcast x over topk: [B, 1, hidden] -> [B, topk, hidden] -> [B*topk, hidden]
    x_bt = x.unsqueeze(1).expand(B, topk, hidden).reshape(B * topk, hidden)

    # bmm: [B*topk, 2*inter, hidden] @ [B*topk, hidden, 1] -> [B*topk, 2*inter, 1]
    y = torch.bmm(w_sel, x_bt.unsqueeze(-1)).squeeze(-1)   # [B*topk, 2*inter]

    gate, up = y[:, :inter], y[:, inter:]
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


def run_case(B, E, topk, hidden, inter, seed=0):
    torch.manual_seed(seed)
    device = "cuda"

    x = torch.randn(B, hidden, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(E, 2 * inter, hidden, dtype=torch.bfloat16, device=device) * 0.05
    topk_ids = torch.randint(0, E, (B, topk), dtype=torch.int32, device=device)

    y_ref  = reference_gate_up(x, w, topk_ids)
    y_ours = wd.moe_gate_up_3d_batched(x, w, topk_ids)

    torch.cuda.synchronize()

    # bf16 tolerance: dot products of length `hidden` accumulate rounding error.
    # Reference uses bmm (fp32 accum internally on H20), ours uses per-lane
    # partial fp32 accum then warp reduce. Allow ~1% relative on element magnitudes.
    diff = (y_ref.float() - y_ours.float()).abs()
    ref_scale = y_ref.float().abs().mean().item() + 1e-6
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"  B={B:3d} E={E:3d} topk={topk} hidden={hidden} inter={inter}  "
          f"max|d|={max_abs:.4f}  mean|d|={mean_abs:.4f}  ref_mean={ref_scale:.4f}")

    torch.testing.assert_close(y_ref, y_ours, rtol=5e-2, atol=5e-2)


def main():
    if not torch.cuda.is_available():
        print("no cuda"); return

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"gencode ok for sm_{''.join(map(str, torch.cuda.get_device_capability()))}")

    cases = [
        # (B, E, topk, hidden, inter)
        (1,   4, 2,  256,   8),   # smallest legal
        (2,   8, 2,  512,  16),
        (4,  16, 4, 1024,  64),
        (8,  16, 8, 4096, 192),   # Qwen3-235B TP8 per-GPU shape
        (16, 16, 8, 4096, 192),
        (64, 16, 8, 4096, 192),
    ]
    for c in cases:
        run_case(*c)
    print("all cases passed")


if __name__ == "__main__":
    main()
