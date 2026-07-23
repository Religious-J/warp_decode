"""Correctness test for warp_decode moe_down_3d_batched."""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import torch

from warp_decode import warp_decode as wd


def reference_down(buffer, w, topk_ids, topk_scale):
    """
    buffer     : [B*topk, inter]     bf16
    w          : [E, hidden, inter]  bf16
    topk_ids   : [B, topk]           int32
    topk_scale : [B, topk]           fp32

    returns    : [B, hidden]         bf16
    """
    B, topk = topk_ids.shape
    E, hidden, inter = w.shape

    ids = topk_ids.reshape(-1).long()                # [B*topk]
    w_sel = w[ids]                                   # [B*topk, hidden, inter]

    # y = W_sel @ buffer  -> [B*topk, hidden, 1]
    y = torch.bmm(w_sel, buffer.unsqueeze(-1)).squeeze(-1)  # [B*topk, hidden]
    y = y.view(B, topk, hidden).float()

    scaled = y * topk_scale.unsqueeze(-1)                    # [B, topk, hidden]
    out = scaled.sum(dim=1).to(torch.bfloat16)
    return out


def run_case(B, E, topk, hidden, inter, seed=0):
    torch.manual_seed(seed)
    device = "cuda"

    buffer = torch.randn(B * topk, inter, dtype=torch.bfloat16, device=device) * 0.1
    w      = torch.randn(E, hidden, inter, dtype=torch.bfloat16, device=device) * 0.05
    topk_ids = torch.randint(0, E, (B, topk), dtype=torch.int32, device=device)
    raw = torch.rand(B, topk, dtype=torch.float32, device=device)
    topk_scale = raw / raw.sum(dim=1, keepdim=True)

    y_ref  = reference_down(buffer, w, topk_ids, topk_scale)
    y_ours = wd.moe_down_3d_batched(buffer, w, topk_ids, topk_scale, B)

    torch.cuda.synchronize()

    diff = (y_ref.float() - y_ours.float()).abs()
    ref_scale = y_ref.float().abs().mean().item() + 1e-6
    print(f"  B={B:3d} E={E:3d} topk={topk} hidden={hidden} inter={inter}  "
          f"max|d|={diff.max().item():.5f}  mean|d|={diff.mean().item():.5f}  "
          f"ref_mean={ref_scale:.5f}")

    torch.testing.assert_close(y_ref, y_ours, rtol=5e-2, atol=5e-2)


def main():
    if not torch.cuda.is_available():
        print("no cuda"); return

    print(f"device: {torch.cuda.get_device_name()}")

    cases = [
        # (B, E, topk, hidden, inter)   inter must be multiple of 64
        (1,   4, 2,   256,  64),
        (2,   8, 4,   512,  64),
        (4,  16, 4,  1024, 128),
        (8,  16, 8,  4096, 192),   # Qwen3-235B TP=8
        (16, 16, 8,  4096, 192),
        (64, 16, 8,  4096, 192),
    ]
    for c in cases:
        run_case(*c)
    print("all cases passed")


if __name__ == "__main__":
    main()
