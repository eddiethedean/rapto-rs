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
from dataclasses import dataclass
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


def run_cprofile(cmd: Sequence[str], output: Path, env: dict[str, str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd_list = list(cmd)
    try:
        if cmd_list and Path(cmd_list[0]).resolve() == Path(sys.executable).resolve():
            cmd_list = cmd_list[1:]
    except FileNotFoundError:
        # Fall back to raw command if resolution fails (e.g., executable missing)
        pass
    cprofile_cmd = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        str(output),
        *cmd_list,
    ]
    print(f"[cProfile] running: {' '.join(shlex.quote(arg) for arg in cprofile_cmd)}")
    subprocess.run(cprofile_cmd, check=True, env=env)
    print(f"cProfile stats written to {output}")


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
        choices=["py-spy", "perf", "cprofile"],
        default="py-spy",
        help="Profiler to invoke (default: py-spy).",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help=(
            "Run a dispatch matrix covering threaded/single-threaded and SIMD forced/disabled "
            "combinations (writes outputs per scenario)."
        ),
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write profiler outputs when --matrix is used.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, simd_mode: str) -> List[str]:
    if args.operation == "custom":
        if not args.command:
            raise SystemExit("Provide --command when --operation custom is selected.")
        return list(args.command)
    if args.operation == "axis0":
        return default_axis0_command(simd_mode, args.warmup, args.repeats)
    if args.operation == "scale":
        return default_scale_command(simd_mode, args.warmup, args.repeats)
    raise SystemExit(f"Unsupported operation {args.operation!r}")


@dataclass
class Scenario:
    label: str
    threads: int | None
    simd_mode: str


def scenario_matrix(args: argparse.Namespace) -> List[Scenario]:
    default_threads = args.threads or int(os.environ.get("RAPTORS_THREADS", "10"))
    return [
        Scenario("threads", default_threads, "force"),
        Scenario("threads", default_threads, "disable"),
        Scenario("single", 1, "force"),
        Scenario("single", 1, "disable"),
    ]


def resolve_output(
    base_dir: Path,
    operation: str,
    tool: str,
    scenario: Scenario,
    explicit: Path | None = None,
) -> tuple[Path, Path | None]:
    if tool == "py-spy":
        suffix = "svg"
    elif tool == "perf":
        suffix = "data"
    else:
        suffix = "prof"
    threads_part = "tNA" if scenario.threads is None else f"t{scenario.threads}"
    stem = f"{operation}-{scenario.label}-{threads_part}-simd-{scenario.simd_mode}"
    output = (explicit or base_dir / f"{stem}.{suffix}").resolve()
    flamegraph = None
    if tool == "perf":
        flamegraph = output.with_suffix(".svg")
    return output, flamegraph


def main() -> int:
    args = parse_args()
    default_dir = ROOT / "benchmarks" / "profiles"
    default_dir.mkdir(parents=True, exist_ok=True)

    scenarios: List[Scenario]
    if args.matrix:
        scenarios = scenario_matrix(args)
    else:
        scenarios = [
            Scenario(
                "custom",
                args.threads,
                args.simd_mode,
            )
        ]

    for scenario in scenarios:
        cmd = build_command(args, scenario.simd_mode)
        env = os.environ.copy()
        if scenario.threads is not None:
            env["RAPTORS_THREADS"] = str(scenario.threads)
        python_pkg_dir = str(ROOT / "python")
        env["PYTHONPATH"] = (
            f"{python_pkg_dir}{os.pathsep}{env['PYTHONPATH']}"
            if "PYTHONPATH" in env and env["PYTHONPATH"]
            else python_pkg_dir
        )
        label_desc = (
            f"{args.operation} simd={scenario.simd_mode} "
            f"threads={scenario.threads if scenario.threads is not None else env.get('RAPTORS_THREADS', 'inherit')}"
        )
        print(f"[profile] Running {label_desc} with {args.tool}")

        base_dir = args.output_dir or default_dir
        if args.matrix:
            output, flamegraph_output = resolve_output(
                base_dir, args.operation, args.tool, scenario
            )
        else:
            if args.tool == "py-spy":
                suffix = "svg"
            elif args.tool == "perf":
                suffix = "data"
            else:
                suffix = "prof"
            stem = f"{args.operation}-{scenario.simd_mode}"
            output = (args.output or base_dir / f"{stem}.{suffix}").resolve()
            flamegraph_output = args.flamegraph_output if args.tool == "perf" else None

        if args.tool == "py-spy":
            run_pyspy(cmd, output, env)
        elif args.tool == "perf":
            data_output = output
            run_perf(
                cmd,
                data_output,
                flamegraph_output,
                env,
                args.perf_frequency,
                args.perf_flamegraph,
            )
        elif args.tool == "cprofile":
            run_cprofile(cmd, output, env)
        else:
            raise SystemExit(f"Unsupported profiler tool '{args.tool}'")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

