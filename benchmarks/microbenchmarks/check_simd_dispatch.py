#!/usr/bin/env python3
"""Probe Raptors SIMD dispatch selections across RAPTORS_SIMD modes.

The script runs the same workload under RAPTORS_SIMD=disable/auto/force in
child processes (to avoid cached detection) and writes the observed dispatch
levels to stdout or a JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

MODES = ("disable", "auto", "force")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        type=int,
        default=1024,
        help="Square matrix size to exercise dispatch (default: %(default)s).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Data type for the probe workload (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the dispatch snapshot JSON.",
    )
    parser.add_argument(
        "--worker",
        choices=MODES,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def run_worker(shape: int, dtype: str, mode: str) -> Dict[str, object]:
    import numpy as np  # type: ignore[import]
    import raptors  # type: ignore[import]

    if dtype == "float32":
        array = np.ones((shape, shape), dtype=np.float32)
    else:
        array = np.ones((shape, shape), dtype=np.float64)
    rust_array = raptors.from_numpy(array)

    # Exercise a few kernels to populate dispatch selections.
    rust_array.scale(1.0009765625)
    rust_array.mean_axis(0)
    other = rust_array.broadcast_add(rust_array)
    other.sum()

    info = raptors.threading_info()
    dispatch = info.get("simd_dispatch", {})
    return {
        "mode": mode,
        "shape": [shape, shape],
        "dtype": dtype,
        "simd_dispatch": dispatch,
        "simd_capabilities": info.get("simd_capabilities", {}),
    }


def spawn_probe(shape: int, dtype: str, mode: str) -> Dict[str, object]:
    env = os.environ.copy()
    if mode == "disable":
        env["RAPTORS_SIMD"] = "0"
    elif mode == "force":
        env["RAPTORS_SIMD"] = "1"
    else:
        env.pop("RAPTORS_SIMD", None)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--shape",
        str(shape),
        "--dtype",
        dtype,
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip())


def main() -> int:
    args = parse_args()
    if args.worker:
        result = run_worker(args.shape, args.dtype, args.worker)
        sys.stdout.write(json.dumps(result))
        sys.stdout.flush()
        return 0

    results: List[Dict[str, object]] = []
    for mode in MODES:
        print(f"[dispatch] probing mode={mode}")
        results.append(spawn_probe(args.shape, args.dtype, mode))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Wrote dispatch snapshot to {args.output}")
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

