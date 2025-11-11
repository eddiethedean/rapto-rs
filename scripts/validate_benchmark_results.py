#!/usr/bin/env python3
"""Validate benchmark result artefacts and compare them with baselines."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REQUIRED_OPERATION_FIELDS = (
    "name",
    "numpy_mean_s",
    "raptors_mean_s",
    "speedup",
)


@dataclass(frozen=True)
class Baseline:
    max_ms: float
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "benchmarks" / "results",
        help="Directory containing benchmark result JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=root / "benchmarks" / "baselines",
        help="Directory containing baseline JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--slack",
        type=float,
        default=0.05,
        help="Relative slack (fractional) allowed over the baseline (default: %(default)s)",
    )
    parser.add_argument(
        "--absolute-slack-ms",
        type=float,
        default=0.05,
        help="Minimum absolute slack in milliseconds (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="Treat missing baselines as warnings instead of errors",
    )
    return parser.parse_args()


def iter_cases(payload: object) -> Iterable[Dict[str, object]]:
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        cases = payload.get("cases")
        if isinstance(cases, list):
            yield from cases
        else:
            raise ValueError("Expected 'cases' list inside JSON object")
    else:
        raise ValueError("Unexpected JSON payload type")


def normalise_shape(shape: object) -> Tuple[int, ...]:
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"invalid shape field {shape!r}")
    return tuple(shape)


def load_baselines(
    directory: Path,
) -> Tuple[Dict[Tuple[Tuple[int, ...], str, str], Baseline], Dict[Tuple[Tuple[int, ...], str, str], Baseline]]:
    if not directory.exists():
        raise FileNotFoundError(f"baseline directory {directory} not found")

    general: Dict[Tuple[Tuple[int, ...], str, str], Baseline] = {}
    scalar: Dict[Tuple[Tuple[int, ...], str, str], Baseline] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: failed to parse JSON ({exc})") from exc

        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected list of baseline entries")
        for entry in payload:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: baseline entry must be an object")
            shape = normalise_shape(entry.get("shape"))
            dtype = entry.get("dtype")
            op_name = entry.get("operation")
            max_ms = entry.get("max_raptors_ms")
            if not isinstance(dtype, str) or not dtype:
                raise ValueError(f"{path}: invalid dtype field {dtype!r}")
            if not isinstance(op_name, str) or not op_name:
                raise ValueError(f"{path}: invalid operation field {op_name!r}")
            if not isinstance(max_ms, (int, float)) or max_ms <= 0.0:
                raise ValueError(f"{path}: invalid max_raptors_ms {max_ms!r}")
            target = scalar if path.stem.endswith("_scalar") else general
            target[(shape, dtype, op_name)] = Baseline(float(max_ms), path)
    return general, scalar


def validate_case_structure(case: Dict[str, object], source: Path) -> Tuple[Tuple[int, ...], str]:
    shape = normalise_shape(case.get("shape"))
    dtype = case.get("dtype")
    operations = case.get("operations")

    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"{source}: invalid dtype field {dtype!r}")
    if not isinstance(operations, list) or not operations:
        raise ValueError(f"{source}: missing operations for shape={shape}, dtype={dtype}")

    for op in operations:
        if not isinstance(op, dict):
            raise ValueError(f"{source}: operation entry is not an object")
        for field in REQUIRED_OPERATION_FIELDS:
            if field not in op:
                raise ValueError(f"{source}: operation missing '{field}' field")
        numpy_mean = op["numpy_mean_s"]
        raptors_mean = op["raptors_mean_s"]
        if not isinstance(numpy_mean, (int, float)) or numpy_mean < 0.0:
            raise ValueError(f"{source}: invalid numpy_mean_s {numpy_mean!r}")
        if not isinstance(raptors_mean, (int, float)) or raptors_mean < 0.0:
            raise ValueError(f"{source}: invalid raptors_mean_s {raptors_mean!r}")

    return shape, dtype


def check_against_baseline(
    shape: Tuple[int, ...],
    dtype: str,
    op: Dict[str, object],
    source: Path,
    baselines: Dict[Tuple[Tuple[int, ...], str, str], Baseline],
    slack: float,
    absolute_slack_ms: float,
    allow_missing: bool,
) -> Optional[str]:
    key = (shape, dtype, str(op["name"]))
    baseline = baselines.get(key)
    if baseline is None:
        if allow_missing:
            return f"[validate] warning: no baseline for {dtype} {shape} op={op['name']}"
        raise ValueError(f"{source}: missing baseline for dtype={dtype}, shape={shape}, op={op['name']}")

    raptors_ms = float(op["raptors_mean_s"]) * 1_000.0
    baseline_limit = baseline.max_ms * (1.0 + slack)
    baseline_limit = max(baseline_limit, baseline.max_ms + absolute_slack_ms)

    if raptors_ms > baseline_limit:
        raise ValueError(
            (
                f"{source}: {dtype} shape={shape} op={op['name']} "
                f"{raptors_ms:.3f} ms exceeds baseline {baseline.max_ms:.3f} ms "
                f"(slack {slack * 100:.1f}% or {absolute_slack_ms:.2f} ms) "
                f"[baseline: {baseline.source.name}]"
            )
        )
    return None


def main() -> int:
    args = parse_args()

    if not args.results_dir.exists():
        print(f"[validate] results directory {args.results_dir} not found", file=sys.stderr)
        return 1

    general_baselines, scalar_baselines = load_baselines(args.baseline_dir)
    if not general_baselines and not scalar_baselines:
        print(f"[validate] no baseline entries found in {args.baseline_dir}", file=sys.stderr)
        return 1

    json_files: List[Path] = sorted(args.results_dir.glob("*.json"))
    if not json_files:
        print(f"[validate] no JSON result files found in {args.results_dir}", file=sys.stderr)
        return 1

    warnings: List[str] = []
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}: failed to parse JSON ({exc})") from exc
        active_map = scalar_baselines if "scalar" in path.stem else general_baselines
        for case in iter_cases(payload):
            shape, dtype = validate_case_structure(case, path)
            for op in case["operations"]:
                maybe_warning = check_against_baseline(
                    shape,
                    dtype,
                    op,
                    path,
                    active_map,
                    slack=args.slack,
                    absolute_slack_ms=args.absolute_slack_ms,
                    allow_missing=args.allow_missing_baseline,
                )
                if maybe_warning:
                    warnings.append(maybe_warning)

    print(
        f"[validate] checked {len(json_files)} result file(s) in {args.results_dir} "
        f"against {len(general_baselines) + len(scalar_baselines)} baseline entries from {args.baseline_dir}"
    )
    for message in warnings:
        print(message, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
