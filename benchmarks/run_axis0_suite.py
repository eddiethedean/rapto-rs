#!/usr/bin/env python3
"""Run focused axis-0 benchmarks for Raptors vs NumPy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


DEFAULT_SHAPES: List[int] = [512, 1024, 2048]
DEFAULT_DTYPES: List[str] = ["float32", "float64"]
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_numpy_raptors.py"


def run_case(
    shape: int,
    dtype: str,
    simd_mode: str,
    warmup: int,
    repeats: int,
) -> Dict[str, object]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_path = Path(tmp.name)
    tmp.close()

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--shape",
        f"{shape}x{shape}",
        "--dtype",
        dtype,
        "--operations",
        "mean_axis0",
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--simd-mode",
        simd_mode,
        "--output-json",
        str(tmp_path),
    ]

    env = os.environ.copy()
    env.setdefault("RAPTORS_THREADS", "8")
    subprocess.run(cmd, check=True, env=env)

    try:
        data = json.loads(tmp_path.read_text())
    finally:
        tmp_path.unlink(missing_ok=True)

    if not data:
        raise RuntimeError(f"No benchmark output captured for dtype={dtype}, shape={shape}")
    return data[0]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="*",
        type=int,
        default=DEFAULT_SHAPES,
        help="Square matrix sizes to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--dtypes",
        nargs="*",
        choices=DEFAULT_DTYPES,
        default=DEFAULT_DTYPES,
        help="Data types to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--simd-mode",
        choices=["force", "auto", "disable"],
        default="force",
        help="SIMD dispatch mode forwarded to compare_numpy_raptors.py",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up iterations before timing (default: %(default)s)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="Timed repeats per operation (default: %(default)s)",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to store aggregated benchmark results.",
    )

    args = parser.parse_args(argv)

    if not SCRIPT_PATH.exists():
        raise SystemExit(f"Unable to locate benchmark harness at {SCRIPT_PATH}")

    results: List[Dict[str, object]] = []
    for dtype in args.dtypes:
        for shape in args.shapes:
            case = run_case(shape, dtype, args.simd_mode, args.warmup, args.repeats)
            results.append(case)

    aggregate = {
        "simd_mode": args.simd_mode,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": results,
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(aggregate, indent=2) + "\n")

    for case in results:
        shape = "x".join(str(dim) for dim in case.get("shape", []))
        dtype = case.get("dtype")
        ops = case.get("operations", [])
        summary = next((op for op in ops if op.get("name") == "mean_axis0"), None)
        if summary:
            speedup = summary.get("speedup")
            raptors_ms = summary.get("raptors_mean_s", 0.0) * 1_000.0
            numpy_ms = summary.get("numpy_mean_s", 0.0) * 1_000.0
            print(
                f"[axis0] shape={shape:<9} dtype={dtype:<7} "
                f"raptors={raptors_ms:>7.3f} ms numpy={numpy_ms:>7.3f} ms speedup={speedup:>6.2f}x"
            )
            if dtype == "float32" and speedup is not None:
                cols = case.get("shape", [0, 0])[1] if len(case.get("shape", [])) > 1 else 0
                if cols <= 1024:
                    assert (
                        speedup >= 1.05
                    ), f"float32 axis-0 speedup regression: {shape} reported {speedup:.2f}x"
                else:
                    assert (
                        speedup >= 0.65
                    ), f"float32 axis-0 large-shape regression: {shape} reported {speedup:.2f}x"

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


