#!/usr/bin/env python3
"""Capture profiling traces (py-spy flamegraphs or Linux perf data) for Raptors.

The helper focuses on the Phase 1 diagnostics targets from `docs/numpy_parity_plan.md`:
float32 axis-0 reductions and scale kernels under both threaded and single-threaded
execution.  It can also run arbitrary commands under the selected profiler.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


ROOT = Path(__file__).resolve().parent.parent


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required tool '{name}' not found on PATH.")


def default_axis0_command(simd_mode: str, warmup: int, repeats: int) -> List[str]:
    script = ROOT / "benchmarks" / "run_axis0_suite.py"
    return [
        sys.executable,
        str(script),
        "--simd-mode",
        simd_mode,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]


def default_scale_command(simd_mode: str, warmup: int, repeats: int) -> List[str]:
    script = ROOT / "scripts" / "compare_numpy_raptors.py"
    return [
        sys.executable,
        str(script),
        "--shape",
        "2048x2048",
        "--dtype",
        "float32",
        "--operations",
        "scale",
        "--simd-mode",
        simd_mode,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]


def run_pyspy(cmd: Sequence[str], output: Path, env: dict[str, str]) -> None:
    ensure_tool("py-spy")
    output.parent.mkdir(parents=True, exist_ok=True)
    pyspy_cmd = [
        "py-spy",
        "record",
        "--output",
        str(output),
        "--format",
        "flamegraph",
        "--rate",
        "250",
        "--",
        *cmd,
    ]
    print(f"[py-spy] running: {' '.join(shlex.quote(arg) for arg in pyspy_cmd)}")
    subprocess.run(pyspy_cmd, check=True, env=env)
    print(f"py-spy flamegraph written to {output}")


def run_perf(
    cmd: Sequence[str],
    data_output: Path,
    flamegraph_output: Path | None,
    env: dict[str, str],
    sample_freq: int,
    perf_flamegraph: Path | None,
) -> None:
    ensure_tool("perf")
    data_output.parent.mkdir(parents=True, exist_ok=True)
    perf_cmd = [
        "perf",
        "record",
        "-F",
        str(sample_freq),
        "-g",
        "-o",
        str(data_output),
        "--",
        *cmd,
    ]
    print(f"[perf] running: {' '.join(shlex.quote(arg) for arg in perf_cmd)}")
    subprocess.run(perf_cmd, check=True, env=env)
    print(f"perf data file written to {data_output}")

    if flamegraph_output is None:
        return

    if perf_flamegraph is None:
        ensure_tool("inferno-flamegraph")
        perf_flamegraph = Path(shutil.which("inferno-flamegraph") or "")

    script_cmd = ["perf", "script"]
    print(f"[perf] generating flamegraph -> {flamegraph_output}")
    flamegraph_output.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(
        script_cmd,
        stdout=subprocess.PIPE,
        env=env,
    ) as perf_script:
        assert perf_script.stdout is not None  # for mypy
        flamegraph_cmd = [
            str(perf_flamegraph),
            "--color",
            "hot",
        ]
        with open(flamegraph_output, "wb") as handle:
            proc = subprocess.Popen(
                flamegraph_cmd,
                stdin=perf_script.stdout,
                stdout=handle,
                env=env,
            )
        perf_script.stdout.close()
        perf_script.wait()
        proc.wait()
    print(f"perf flamegraph written to {flamegraph_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=["axis0", "scale", "custom"],
        default="axis0",
        help="Which preconfigured benchmark to profile (default: axis0).",
    )
    parser.add_argument(
        "--tool",
        choices=["py-spy", "perf"],
        default="py-spy",
        help="Profiler to invoke (default: py-spy).",
    )
    parser.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="Custom command to profile (requires --operation custom).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (flamegraph SVG for py-spy, perf data for perf).",
    )
    parser.add_argument(
        "--flamegraph-output",
        type=Path,
        help="Optional flamegraph SVG path when using perf.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Override RAPTORS_THREADS for the profiled command.",
    )
    parser.add_argument(
        "--simd-mode",
        choices=["auto", "force", "disable"],
        default="force",
        help="SIMD mode passed to built-in benchmark commands (default: force).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up iterations for built-in commands (default: 1).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Timed iterations for built-in commands (default: 7).",
    )
    parser.add_argument(
        "--perf-frequency",
        type=int,
        default=4000,
        help="Sampling frequency for perf (Hz, default: 4000).",
    )
    parser.add_argument(
        "--perf-flamegraph",
        type=Path,
        help="Path to the inferno-flamegraph binary (optional).",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    if args.operation == "custom":
        if not args.command:
            raise SystemExit("Provide --command when --operation custom is selected.")
        return list(args.command)
    if args.operation == "axis0":
        return default_axis0_command(args.simd_mode, args.warmup, args.repeats)
    if args.operation == "scale":
        return default_scale_command(args.simd_mode, args.warmup, args.repeats)
    raise SystemExit(f"Unsupported operation {args.operation!r}")


def main() -> int:
    args = parse_args()
    cmd = build_command(args)

    env = os.environ.copy()
    if args.threads is not None:
        env["RAPTORS_THREADS"] = str(args.threads)

    default_dir = ROOT / "benchmarks" / "profiles"
    default_dir.mkdir(parents=True, exist_ok=True)

    if args.tool == "py-spy":
        output = args.output or default_dir / f"{args.operation}-{args.simd_mode}.svg"
        run_pyspy(cmd, output, env)
    else:
        data_output = args.output or default_dir / f"{args.operation}-{args.simd_mode}.data"
        flamegraph_output = args.flamegraph_output
        perf_flamegraph = args.perf_flamegraph
        run_perf(
            cmd,
            data_output,
            flamegraph_output,
            env,
            args.perf_frequency,
            perf_flamegraph,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

