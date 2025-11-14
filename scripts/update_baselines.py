#!/usr/bin/env python3
"""Convert benchmark results to baseline format and update baseline files."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def convert_results_to_baseline(
    results_json: Path, slack_factor: float = 0.1, min_slack_ms: float = 0.01
) -> Dict[str, List[Dict]]:
    """Convert benchmark results JSON to baseline format, split by dtype and scalar mode."""
    with open(results_json, "r") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    simd_mode = data.get("metadata", {}).get("simd_mode", "auto")
    is_scalar = simd_mode == "disable"

    # Collect entries by dtype
    baselines: Dict[str, List[Dict]] = {}

    for case in cases:
        shape = tuple(case["shape"])
        dtype = case["dtype"]
        operations = case.get("operations", [])

        # Determine baseline key
        baseline_key = f"{dtype}_scalar" if is_scalar else dtype

        if baseline_key not in baselines:
            baselines[baseline_key] = []

        for op in operations:
            op_name = op["name"]
            raptors_mean_s = op["raptors_mean_s"]
            raptors_mean_ms = raptors_mean_s * 1000.0

            # Add slack: relative and absolute minimum
            max_ms = raptors_mean_ms * (1.0 + slack_factor)
            max_ms = max(max_ms, raptors_mean_ms + min_slack_ms)

            entry = {
                "shape": list(shape),
                "dtype": dtype,
                "operation": op_name,
                "max_raptors_ms": round(max_ms, 4),
            }

            baselines[baseline_key].append(entry)

    return baselines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simd-results",
        type=Path,
        required=True,
        help="JSON file from benchmark run with SIMD forced",
    )
    parser.add_argument(
        "--scalar-results",
        type=Path,
        required=True,
        help="JSON file from benchmark run with SIMD disabled",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(__file__).parent.parent / "benchmarks" / "baselines",
        help="Directory containing baseline files (default: %(default)s)",
    )
    parser.add_argument(
        "--slack",
        type=float,
        default=0.1,
        help="Relative slack factor (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--min-slack-ms",
        type=float,
        default=0.01,
        help="Minimum absolute slack in ms (default: 0.01)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing files",
    )

    args = parser.parse_args()

    # Load and convert both result files
    simd_baselines = convert_results_to_baseline(args.simd_results, args.slack, args.min_slack_ms)
    scalar_baselines = convert_results_to_baseline(args.scalar_results, args.slack, args.min_slack_ms)

    # Merge scalar into the appropriate keys
    all_baselines = {
        "float32": simd_baselines["float32"],
        "float64": simd_baselines["float64"],
        "float32_scalar": scalar_baselines["float32_scalar"],
        "float64_scalar": scalar_baselines["float64_scalar"],
    }

    # Sort entries for consistency (by shape, then operation)
    for key in all_baselines:
        all_baselines[key].sort(
            key=lambda x: (
                tuple(x["shape"]),
                x["dtype"],
                x["operation"],
            )
        )

    # Update baseline files
    baseline_files = {
        "float32": args.baseline_dir / "2d_float32.json",
        "float64": args.baseline_dir / "2d_float64.json",
        "float32_scalar": args.baseline_dir / "2d_float32_scalar.json",
        "float64_scalar": args.baseline_dir / "2d_float64_scalar.json",
    }

    for key, output_path in baseline_files.items():
        entries = all_baselines[key]
        if not entries:
            print(f"Warning: No entries for {key}, skipping {output_path.name}")
            continue

        if args.dry_run:
            print(f"\nWould update {output_path.name}:")
            print(f"  {len(entries)} entries")
            for entry in entries[:3]:
                print(f"    {entry}")
            if len(entries) > 3:
                print(f"    ... and {len(entries) - 3} more")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(entries, f, indent=2)
            print(f"Updated {output_path.name} with {len(entries)} entries")

    return 0


if __name__ == "__main__":
    sys.exit(main())

