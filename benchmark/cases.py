"""Named Qwen FP8 shapes shared by correctness tests and benchmarks.

Dimensions describe one TP rank. This module uses only the Python standard
library so listing cases and validating CLI arguments never initializes CUDA.
"""
import json

DEFAULT_BATCHES = "1,2,3,4,5,6,7,8"
DEFAULT_CASE = "qwen35_36_35b_tp1"
CASES = {
    "qwen35_36_35b_tp1": dict(hidden=2048, inter=512, experts=256, topk=8, tp=1),
    "qwen35_36_35b_tp2": dict(hidden=2048, inter=256, experts=256, topk=8, tp=2),
    "qwen35_122b_tp2": dict(hidden=3072, inter=512, experts=256, topk=8, tp=2),
    "qwen35_122b_tp4": dict(hidden=3072, inter=256, experts=256, topk=8, tp=4),
    "qwen35_397b_tp4": dict(hidden=4096, inter=256, experts=512, topk=10, tp=4),
    "qwen35_397b_tp8": dict(hidden=4096, inter=128, experts=512, topk=10, tp=8),
    "qwen3_next_80b_tp1": dict(hidden=2048, inter=512, experts=512, topk=10, tp=1),
    "qwen3_next_80b_tp2": dict(hidden=2048, inter=256, experts=512, topk=10, tp=2),
}
SHAPE_FIELDS = ("hidden", "inter", "experts", "topk", "tp")


def print_cases():
    print(json.dumps(CASES, indent=2))


def add_shape_arguments(parser):
    parser.add_argument("--case", choices=tuple(CASES),
                        help="named shape; cannot be combined with explicit shape flags")
    parser.add_argument("--list-cases", action="store_true",
                        help="list the eight named shapes and exit without CUDA imports")
    defaults = CASES[DEFAULT_CASE]
    for field in SHAPE_FIELDS:
        detail = {"inter": "intermediate dimension per TP rank",
                  "tp": "TP metadata; inter is already per rank"}.get(field, field)
        parser.add_argument("--" + field, type=int, default=None,
                            help=f"{detail} (default: {defaults[field]})")


def resolve_shape_arguments(parser, args):
    supplied = [field for field in SHAPE_FIELDS if getattr(args, field) is not None]
    if args.case and supplied:
        parser.error("--case cannot be combined with explicit shape flags: " +
                     ", ".join("--" + field for field in supplied))
    defaults = CASES[args.case or DEFAULT_CASE]
    for field in SHAPE_FIELDS:
        if getattr(args, field) is None:
            setattr(args, field, defaults[field])
    if args.hidden <= 0 or args.inter <= 0 or args.hidden % 128 or args.inter % 128:
        parser.error("hidden and inter must be positive multiples of 128")
    if args.experts < 1 or not 1 <= args.topk <= min(args.experts, 32) or args.tp < 1:
        parser.error("experts and tp must be positive, with 1 <= topk <= min(experts, 32)")
    return args


def parse_batches(parser, value, *, allow_zero=False):
    try:
        batches = [int(part) for part in value.split(",")]
    except ValueError:
        parser.error("batches must be comma-separated integers")
    minimum = 0 if allow_zero else 1
    if not batches or any(batch < minimum for batch in batches):
        parser.error("batches must be " + ("non-negative" if allow_zero else "positive"))
    if len(batches) != len(set(batches)):
        parser.error("batches must not contain duplicates")
    return batches
