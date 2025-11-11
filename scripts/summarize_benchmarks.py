#!/usr/bin/env python3
"""Summarize Raptors benchmark JSON dumps into a tabular view.

The script expects one or more JSON files emitted by the existing harnesses
(`scripts/compare_numpy_raptors.py`, `benchmarks/run_axis0_suite.py`) and
produces a CSV or pretty table containing speedups, absolute timings, and
variance.  This allows quick inspection of regressions before and after
optimizations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class BenchmarkRow:
    source: Path
    shape: Sequence[int]
    dtype: str
    operation: str
    numpy_ms: float
    raptors_ms: float
    raptors_std_ms: float
    speedup: float

    @property
    def shape_str(self) -> str:
        return "x".join(str(dim) for dim in self.shape)


def load_json(path: Path) -> Iterable[BenchmarkRow]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        cases = data.get("cases", [])
    else:
        cases = data

    for case in cases:
        shape = case.get("shape", [])
        dtype = case.get("dtype")
        operations = case.get("operations", [])
        for op in operations:
            yield BenchmarkRow(
                source=path,
                shape=shape,
                dtype=dtype,
                operation=op.get("name", ""),
                numpy_ms=float(op.get("numpy_mean_s", 0.0)) * 1000.0,
                raptors_ms=float(op.get("raptors_mean_s", 0.0)) * 1000.0,
                raptors_std_ms=float(op.get("raptors_std_s", 0.0)) * 1000.0,
                speedup=float(op.get("speedup", 0.0)),
            )


def summarize(rows: Iterable[BenchmarkRow]) -> List[BenchmarkRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.speedup,
            row.dtype,
            row.shape,
            row.operation,
            row.source.name,
        ),
    )


def write_csv(rows: Sequence[BenchmarkRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "shape",
                "dtype",
                "operation",
                "numpy_ms",
                "raptors_ms",
                "raptors_std_ms",
                "speedup",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.source.name,
                    row.shape_str,
                    row.dtype,
                    row.operation,
                    f"{row.numpy_ms:.3f}",
                    f"{row.raptors_ms:.3f}",
                    f"{row.raptors_std_ms:.3f}",
                    f"{row.speedup:.2f}",
                ]
            )


def print_table(rows: Sequence[BenchmarkRow], top: int | None = None) -> None:
    from textwrap import shorten

    header = (
        f"{'source':<20} {'shape':<12} {'dtype':<8} {'operation':<16} "
        f"{'numpy_ms':>10} {'raptors_ms':>12} {'± ms':>8} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    count = 0
    for row in rows:
        print(
            f"{shorten(row.source.name, width=20):<20} "
            f"{row.shape_str:<12} "
            f"{row.dtype:<8} "
            f"{row.operation:<16} "
            f"{row.numpy_ms:>10.3f} "
            f"{row.raptors_ms:>12.3f} "
            f"{row.raptors_std_ms:>8.3f} "
            f"{row.speedup:>8.2f}"
        )
        count += 1
        if top is not None and count >= top:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[
            Path("benchmarks/results/local_simd.json"),
            Path("benchmarks/results/local_scalar.json"),
            Path("benchmarks/results/axis0_local.json"),
        ],
        help="Benchmark JSON files to summarize (default: common results).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to write a CSV summary.",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Limit the number of rows printed to stdout.",
    )
    parser.add_argument(
        "--sub-one",
        action="store_true",
        help="Only report entries where Raptors is slower than NumPy (<1×).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: List[BenchmarkRow] = []
    for path in args.inputs:
        if not path.exists():
            continue
        rows.extend(load_json(path))

    if not rows:
        print("No benchmark entries found.")
        return 1

    rows = summarize(rows)

    if args.sub_one:
        rows = [row for row in rows if row.speedup < 1.0]

    if args.output_csv:
        write_csv(rows, args.output_csv)

    print_table(rows, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

