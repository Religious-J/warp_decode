"""Run the eight named Qwen cases through the existing benchmark or tests.

Examples:
  python benchmark/bench_qwen.py --dry-run
  python benchmark/bench_qwen.py --mode test
  python benchmark/bench_qwen.py --cases qwen35_36_35b_tp2 --seeds 20261023
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import CASES, DEFAULT_BATCHES, parse_batches, print_cases

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("bench", "test"), default="bench")
    p.add_argument("--cases", default="all", help="all, or comma-separated names from --list-cases")
    p.add_argument("--list-cases", action="store_true")
    p.add_argument("--batches", default=DEFAULT_BATCHES)
    p.add_argument("--seeds", default="20261023,20261024")
    p.add_argument("--rounds", type=int, default=5, help="benchmark timing rounds")
    p.add_argument("--iters", type=int, default=100, help="benchmark graph replays")
    p.add_argument("--calls-per-graph", type=int, default=16)
    p.add_argument("--sanitizer", action="store_true",
                   help="test mode: pass through the existing sanitizer-compatible test option")
    p.add_argument("--output-dir", type=Path, default=ROOT/"benchmark/results/qwen_focus")
    p.add_argument("--dry-run", action="store_true", help="print commands without imports or execution")
    args = p.parse_args(argv)
    if args.list_cases:
        print_cases()
        p.exit()
    args.cases = list(CASES) if args.cases == "all" else args.cases.split(",")
    if not args.cases or any(name not in CASES for name in args.cases):
        p.error("unknown case; use --list-cases for the supported names")
    if len(args.cases) != len(set(args.cases)):
        p.error("cases must not contain duplicates")
    parse_batches(p, args.batches, allow_zero=args.mode == "test")
    try:
        args.seeds = [int(value) for value in args.seeds.split(",")]
    except ValueError:
        p.error("seeds must be comma-separated integers")
    if len(args.seeds) != len(set(args.seeds)):
        p.error("seeds must not contain duplicates")
    if min(args.rounds, args.iters, args.calls_per_graph) < 1:
        p.error("rounds, iters and calls-per-graph must be positive")
    if args.sanitizer and args.mode != "test":
        p.error("--sanitizer is only valid with --mode test")
    return args


def make_plan(args):
    script = ROOT/("benchmark/bench.py" if args.mode == "bench" else "tests/test_fp8_blockwise.py")
    directory = args.output_dir.resolve()/args.mode
    runs = []
    for name in args.cases:
        for seed in args.seeds:
            output = directory/f"{name}_seed{seed}.json"
            command = [sys.executable, str(script), "--case", name,
                       "--batches", args.batches, "--seed", str(seed), "--output", str(output)]
            if args.mode == "bench":
                command += ["--rounds", str(args.rounds), "--iters", str(args.iters),
                            "--calls-per-graph", str(args.calls_per_graph)]
            elif args.sanitizer:
                command += ["--sanitizer"]
            runs.append(dict(case=name, seed=seed, shape=CASES[name], command=command,
                             output=str(output), log=str(output.with_suffix(".log"))))
    return directory, runs


def main():
    args = parse_args()
    directory, runs = make_plan(args)
    plan = dict(mode=args.mode, batches=args.batches, runs=runs,
                source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), ROOT/"benchmark/cases.py",
                                 Path(runs[0]["command"][1]))})
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"metadata.json").write_text(json.dumps(plan, indent=2)+"\n")
    status = []
    for run in runs:
        print(f"{args.mode}: {run['case']} seed={run['seed']}", flush=True)
        with Path(run["log"]).open("w") as log:
            process = subprocess.run(run["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status.append(dict(case=run["case"], seed=run["seed"], returncode=process.returncode,
                           passed=process.returncode == 0, output=run["output"], log=run["log"]))
        (directory/"status.json").write_text(json.dumps(status, indent=2)+"\n")
        print("PASS" if process.returncode == 0 else "FAIL", run["log"], flush=True)
    return 0 if all(item["passed"] for item in status) else 1


if __name__ == "__main__":
    sys.exit(main())
