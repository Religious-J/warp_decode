"""JIT builder for warp_decode kernels.

Usage:
    from warp_decode import warp_decode as wd
    out = wd.moe_gate_up_3d_batched(x, w, topk_ids)
"""

import os
from pathlib import Path

from torch.utils.cpp_extension import load

_this_dir = Path(__file__).resolve().parent
_csrc     = _this_dir / "csrc"

_sources = [
    str(_csrc / "binding.cpp"),
    str(_csrc / "moe_gate_up_3d_batched.cu"),
    str(_csrc / "moe_down_3d_batched.cu"),
]

_extra_cuda_cflags = [
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_90,code=sm_90",  # H20 / H100
]

_build_dir = _this_dir / "build"
_build_dir.mkdir(exist_ok=True)

warp_decode = load(
    name="warp_decode_ext",
    sources=_sources,
    extra_cflags=["-O3"],
    extra_cuda_cflags=_extra_cuda_cflags,
    build_directory=str(_build_dir),
    verbose=os.environ.get("WARP_DECODE_VERBOSE", "0") == "1",
)
